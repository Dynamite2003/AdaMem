from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Callable, Mapping, Protocol


@dataclass(slots=True, frozen=True)
class StatePatch:
    slot: str
    value: str
    evidence: str
    subject: str = "user"
    status: str = "active"
    invalidates_value: str | None = None

    @property
    def key(self) -> str:
        return f"state.{self.subject}.{self.slot}"

    @property
    def content(self) -> str:
        if self.status == "unknown_current":
            invalidated = (
                f"\nInvalidated prior value: {self.invalidates_value}"
                if self.invalidates_value else ""
            )
            return (
                f"Current {self.subject} {self.slot}: unknown-current."
                f"{invalidated}\nState evidence: {self.evidence}"
            )
        return (
            f"Current {self.subject} {self.slot}: {self.value}.\n"
            f"State evidence: {self.evidence}"
        )


StateExtractor = Callable[[str, Mapping[str, object] | None], list[StatePatch]]


class LLMClientLike(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        ...


STATE_EXTRACTOR_SYSTEM = (
    "Extract only durable current-state updates from the observation. "
    "Return strict JSON with a top-level patches array. Each patch must have "
    "slot, value, and optional status, subject, invalidates_value. Use "
    "status unknown_current only when the observation invalidates an old value "
    "without giving a replacement. Do not infer unsupported facts."
)

STATE_EXTRACTOR_TEMPLATE = """\
Observation:
{content}

Return JSON only:
{{"patches":[{{"slot":"location","value":"Boston","status":"active","subject":"user"}}]}}
"""


class LLMStateExtractor:
    """LLM-backed state extractor with deterministic parsing.

    The client is injected so tests can use `MockLLMClient` and paper runs can
    record the provider/model separately from AdaMem's state-management logic.
    """

    def __init__(
        self,
        client: LLMClientLike,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        self.client = client
        self.max_tokens = max_tokens
        self.temperature = temperature

    def __call__(
        self,
        content: str,
        metadata: Mapping[str, object] | None = None,
    ) -> list[StatePatch]:
        if metadata and metadata.get("derived"):
            return []
        if _speaker(content) == "assistant":
            return []
        prompt = STATE_EXTRACTOR_TEMPLATE.format(content=content)
        raw = self.client.complete(
            prompt,
            system=STATE_EXTRACTOR_SYSTEM,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return parse_state_patch_payload(raw, evidence=content)


def build_state_extractor(name: str) -> StateExtractor:
    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"", "deterministic", "rules"}:
        return extract_state_patches
    if normalized in {"metadata_mock_llm", "mock_llm", "mock"}:
        return metadata_mock_llm_state_extractor
    if normalized in {"llm_json", "llm"}:
        raise RuntimeError(
            "state_extractor_name='llm_json' requires injecting LLMStateExtractor(client) into AdaMem"
        )
    raise ValueError(f"unknown state_extractor_name: {name}")


def metadata_mock_llm_state_extractor(
    content: str,
    metadata: Mapping[str, object] | None = None,
) -> list[StatePatch]:
    """Deterministic mock path for LLM extractor experiments.

    Test fixtures can put LLM-shaped JSON under `metadata["mock_state_patches"]`
    without using benchmark answers or judge labels. Runtime code should use
    `LLMStateExtractor` with a real client for API-backed extractor ablations.
    """

    if metadata and metadata.get("derived"):
        return []
    if _speaker(content) == "assistant":
        return []
    if not metadata:
        return []
    payload = metadata.get("mock_state_patches")
    if payload is None:
        return []
    return parse_state_patch_payload({"patches": payload}, evidence=content)


def parse_state_patch_payload(raw: str | Mapping[str, Any], *, evidence: str) -> list[StatePatch]:
    payload = _coerce_json_payload(raw)
    records = payload.get("patches") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        return []
    patches: list[StatePatch] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        slot = _clean_state_field(record.get("slot"))
        value = _clean_state_field(record.get("value"))
        if not slot or not value:
            continue
        status = _clean_state_field(record.get("status") or "active") or "active"
        if status not in {"active", "unknown_current"}:
            continue
        subject = _clean_state_field(record.get("subject") or "user") or "user"
        invalidates = (
            _clean_state_field(record.get("invalidates_value"))
            or _clean_state_field(record.get("invalidated_value"))
            or None
        )
        patches.append(
            StatePatch(
                slot=slot,
                value=value,
                evidence=evidence,
                subject=subject,
                status=status,
                invalidates_value=invalidates,
            )
        )
    return patches


def _coerce_json_payload(raw: str | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(raw, Mapping):
        return raw
    text = _strip_json_fences(str(raw).strip())
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _strip_json_fences(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _clean_state_field(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


STATE_DEPENDENCY_PREFIXES = {
    "location": (
        "commute.",
        "local.",
        "schedule.local.",
        "timezone.",
    ),
    "health.peanut_allergy.status": (
        "meal.",
        "restaurant.",
        "food.",
    ),
    "health.gluten_restriction.status": (
        "meal.",
        "restaurant.",
        "food.",
    ),
    "health.dairy_restriction.status": (
        "meal.",
        "restaurant.",
        "food.",
    ),
    "health.nut_allergy.status": (
        "meal.",
        "restaurant.",
        "food.",
    ),
    "health.shellfish_allergy.status": (
        "meal.",
        "restaurant.",
        "food.",
    ),
    "organization.employer": (
        "employment.",
        "workplace.",
    ),
}


LOCATION_PATTERNS = [
    re.compile(r"\b(?:i[’']?ve|i have)\s+been\s+(?:based|staying|living)\s+in\s+([A-Z][A-Za-z .'-]{1,40})", re.I),
    re.compile(r"\b(?:i[’']?m|i am)\s+(?:based|staying|living)\s+in\s+([A-Z][A-Za-z .'-]{1,40})", re.I),
    re.compile(r"\bhome base\s+is\s+([A-Z][A-Za-z .'-]{1,40})", re.I),
    re.compile(r"\b(?:my\s+)?(?:new\s+)?place\s+in\s+([A-Z][A-Za-z .'-]{1,40})", re.I),
    re.compile(r"\brelocated\s+to\s+([A-Z][A-Za-z .'-]{1,40})", re.I),
    re.compile(r"\bmoved\s+(?:into\s+a\s+place\s+in|to)\s+([A-Z][A-Za-z .'-]{1,40})", re.I),
]

LOCATION_UNKNOWN_CURRENT_PATTERNS = [
    re.compile(r"\b(?:i\s+)?no\s+longer\s+(?:live|am\s+based|am\s+staying)\s+in\s+([A-Z][A-Za-z .'-]{1,40})", re.I),
    re.compile(r"\b(?:i[’']?m|i am)\s+not\s+(?:based|staying|living)\s+in\s+([A-Z][A-Za-z .'-]{1,40})\s+anymore", re.I),
    re.compile(r"\bmoved\s+out\s+of\s+([A-Z][A-Za-z .'-]{1,40})", re.I),
    re.compile(r"\bleft\s+([A-Z][A-Za-z .'-]{1,40})\b", re.I),
]

LOCATION_STOP = {
    "and",
    "after",
    "around",
    "before",
    "but",
    "can",
    "for",
    "from",
    "if",
    "near",
    "now",
    "since",
    "so",
    "that",
    "the",
    "to",
    "where",
    "with",
}

LOCATION_QUERY_TERMS = {
    "area",
    "based",
    "city",
    "live",
    "located",
    "near",
    "nearby",
    "neighborhood",
    "neighborhoods",
    "place",
    "places",
    "relax",
    "spots",
    "staying",
}

LOCAL_LOCATION_CONTEXT_TERMS = {
    "around",
    "go",
    "near",
    "nearby",
    "option",
    "options",
    "place",
    "places",
    "recommend",
    "resource",
    "resources",
    "spot",
    "spots",
    "where",
}

LOCAL_GYM_PATTERNS = [
    re.compile(
        r"\b(?:my\s+)?local\s+gym\s+is\s+(?P<value>[A-Z][A-Za-z0-9 &.'-]{1,60})",
        re.I,
    ),
    re.compile(
        r"\b(?:i\s+)?use\s+(?P<value>[A-Z][A-Za-z0-9 &.'-]{1,60})\s+as\s+"
        r"(?:my\s+)?local\s+gym\b",
        re.I,
    ),
]

LOCAL_GYM_QUERY_TERMS = {
    "gym",
    "fitness",
}

BEVERAGE_VALUES = (
    "black coffee",
    "cappuccino",
    "coffee",
    "espresso",
    "green tea",
    "latte",
    "matcha",
    "tea",
)

BEVERAGE_PATTERNS = [
    re.compile(r"\bswitched\s+from\s+(?P<old>[A-Za-z ]{2,30})\s+to\s+(?P<value>[A-Za-z ]{2,30})", re.I),
    re.compile(r"\b(?:i\s+)?prefer\s+(?P<value>[A-Za-z ]{2,30})\s+now\b", re.I),
    re.compile(r"\b(?:my\s+)?(?:favorite|usual)\s+(?:drink|beverage|coffee\s+order|order)\s+is\s+(?P<value>[A-Za-z ]{2,30})", re.I),
]

BEVERAGE_UNKNOWN_CURRENT_PATTERNS = [
    re.compile(r"\b(?:i\s+)?no\s+longer\s+(?:prefer|drink|order|like)\s+(?P<value>[A-Za-z ]{2,30})", re.I),
    re.compile(
        r"\b(?P<value>[A-Za-z ]{2,30})\s+is\s+no\s+longer\s+my\s+"
        r"(?:favorite|usual)\s+(?:drink|beverage|coffee\s+order|order)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:my\s+)?(?:favorite|usual)\s+(?:drink|beverage|coffee\s+order|order)\s+"
        r"is\s+no\s+longer\s+(?P<value>[A-Za-z ]{2,30})",
        re.I,
    ),
]

BEVERAGE_QUERY_TERMS = {
    "beverage",
    "cafe",
    "coffee",
    "drink",
    "espresso",
    "latte",
    "matcha",
    "order",
    "tea",
}

TASK_STATUS_PATTERNS = [
    re.compile(
        r"\b(?:the\s+)?(?P<task>[A-Za-z][A-Za-z0-9 _/'-]{2,50}?)\s+status\s+is\s+(?P<status>blocked|done|complete|completed|resolved|open|pending|paused|cancelled|canceled)\b",
        re.I,
    ),
    re.compile(
        r"\bmarked\s+(?:the\s+)?(?P<task>[A-Za-z][A-Za-z0-9 _/'-]{2,50}?)\s+as\s+(?P<status>blocked|done|complete|completed|resolved|open|pending|paused|cancelled|canceled)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:the\s+)?(?P<task>[A-Za-z][A-Za-z0-9 _/'-]{2,50}?)\s+is\s+(?:now\s+)?(?P<status>blocked|done|complete|completed|resolved|open|pending|paused|cancelled|canceled)\b",
        re.I,
    ),
]

TASK_QUERY_TERMS = {
    "blocked",
    "complete",
    "completed",
    "done",
    "pending",
    "resolved",
    "status",
}

TASK_QUERY_SUBJECT_TERMS = {
    "deployment",
    "incident",
    "issue",
    "migration",
    "project",
    "request",
    "task",
    "ticket",
    "workflow",
}

SCHEDULE_PATTERNS = [
    re.compile(r"\b(?:i[’']?m|i am)\s+(?P<value>available|free)\s+(?P<when>on\s+[A-Za-z0-9 ,:'-]{2,50})", re.I),
    re.compile(r"\b(?:my\s+)?availability\s+is\s+(?P<when>[A-Za-z0-9 ,:'-]{2,50})", re.I),
    re.compile(r"\bi\s+can\s+meet\s+(?P<when>on\s+[A-Za-z0-9 ,:'-]{2,50})", re.I),
    re.compile(r"\bi\s+can'?t\s+meet\s+(?P<when>on\s+[A-Za-z0-9 ,:'-]{2,50})", re.I),
]

SCHEDULE_QUERY_TERMS = {
    "availability",
    "available",
    "calendar",
    "free",
    "meet",
    "meeting",
    "schedule",
    "time",
    "times",
}

DIETARY_ACTIVE_PATTERNS = [
    re.compile(r"\b(?:i[’']?m|i am)\s+allergic\s+to\s+(?P<item>[A-Za-z][A-Za-z -]{1,40})", re.I),
    re.compile(r"\b(?:i\s+have|my)\s+(?:a\s+)?(?P<item>[A-Za-z][A-Za-z -]{1,40})\s+allergy\b", re.I),
    re.compile(r"\bi\s+can(?:not|[’']?t)\s+eat\s+(?P<item>[A-Za-z][A-Za-z -]{1,40})", re.I),
    re.compile(r"\b(?:i\s+need|please\s+keep\s+me)\s+(?P<constraint>gluten[- ]free|dairy[- ]free|nut[- ]free|peanut[- ]free|vegan|vegetarian)\b", re.I),
]

DIETARY_CLEARED_PATTERNS = [
    re.compile(r"\b(?:i[’']?m|i am)\s+no\s+longer\s+allergic\s+to\s+(?P<item>[A-Za-z][A-Za-z -]{1,40})", re.I),
    re.compile(r"\bi\s+(?:can|may)\s+eat\s+(?P<item>[A-Za-z][A-Za-z -]{1,40})\s+(?:again|now)\b", re.I),
    re.compile(r"\b(?:my\s+)?(?P<item>[A-Za-z][A-Za-z -]{1,40})\s+allergy\s+(?:is\s+)?(?:cleared|resolved|gone)\b", re.I),
    re.compile(r"\b(?:cleared|resolved)\s+(?P<item>[A-Za-z][A-Za-z -]{1,40})\b", re.I),
    re.compile(r"\b(?:no\s+longer|not)\s+avoiding\s+(?P<item>[A-Za-z][A-Za-z -]{1,40})", re.I),
]

DIETARY_QUERY_TERMS = {
    "allergic",
    "allergy",
    "diet",
    "dietary",
    "dairy",
    "eat",
    "food",
    "gluten",
    "meal",
    "peanut",
    "peanuts",
    "restaurant",
    "safe",
    "shellfish",
}

RESOURCE_STATUS_PATTERNS = [
    re.compile(
        r"\b(?:my\s+|the\s+)?(?P<resource>[A-Za-z][A-Za-z0-9 _/'-]{2,50}?)\s+"
        r"(?:is|was)\s+(?:now\s+)?"
        r"(?P<status>active|available|blocked|disabled|enabled|expired|found|lost|renewed|revoked|rotated|unavailable|valid)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:my\s+|the\s+)?(?P<resource>[A-Za-z][A-Za-z0-9 _/'-]{2,50}?)\s+"
        r"has\s+been\s+"
        r"(?P<status>activated|deactivated|disabled|enabled|expired|renewed|revoked|rotated)\b",
        re.I,
    ),
    re.compile(
        r"\b(?P<status>activated|deactivated|disabled|enabled|renewed|revoked|rotated)\s+"
        r"(?:my\s+|the\s+)?(?P<resource>[A-Za-z][A-Za-z0-9 _/'-]{2,50})\b",
        re.I,
    ),
]

RESOURCE_UNKNOWN_CURRENT_PATTERNS = [
    re.compile(
        r"\b(?:my\s+|the\s+)?(?P<resource>[A-Za-z][A-Za-z0-9 _/'-]{2,50}?)\s+"
        r"(?:is|was)\s+no\s+longer\s+"
        r"(?P<status>active|available|blocked|disabled|enabled|expired|found|lost|renewed|revoked|rotated|unavailable|valid)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:my\s+|the\s+)?(?P<resource>[A-Za-z][A-Za-z0-9 _/'-]{2,50}?)\s+"
        r"(?:is|was)\s+not\s+"
        r"(?P<status>active|available|blocked|disabled|enabled|expired|found|lost|renewed|revoked|rotated|unavailable|valid)\s+anymore\b",
        re.I,
    ),
]

RESOURCE_QUERY_TERMS = {
    "access",
    "account",
    "api key",
    "badge",
    "credential",
    "credentials",
    "key",
    "license",
    "passport",
    "token",
    "visa",
}

RESOURCE_NOUNS = {
    "access",
    "account",
    "api",
    "badge",
    "card",
    "certificate",
    "credential",
    "credentials",
    "github",
    "key",
    "license",
    "passport",
    "token",
    "visa",
}

WORKFLOW_RULE_PATTERNS = [
    re.compile(
        r"\b(?:for|in)\s+(?P<workflow>[A-Za-z][A-Za-z0-9 _/'-]{2,50}?),?\s+"
        r"(?:the\s+)?(?P<rule>[A-Za-z][A-Za-z0-9 _/'-]{2,40}?)\s+"
        r"(?:rule|procedure|policy|runbook)\s+is\s+(?P<value>[A-Za-z0-9 _/'-]{2,80})",
        re.I,
    ),
    re.compile(
        r"\b(?:the\s+)?(?P<workflow>[A-Za-z][A-Za-z0-9 _/'-]{2,50}?)\s+"
        r"(?:workflow|runbook|procedure)\s+(?P<rule>rollback|deploy|release|incident|backup)\s+"
        r"(?:rule|step|policy)\s+is\s+(?P<value>[A-Za-z0-9 _/'-]{2,80})",
        re.I,
    ),
]

WORKFLOW_UNKNOWN_CURRENT_PATTERNS = [
    re.compile(
        r"\b(?:for|in)\s+(?P<workflow>[A-Za-z][A-Za-z0-9 _/'-]{2,50}?),?\s+"
        r"(?:the\s+)?(?P<rule>[A-Za-z][A-Za-z0-9 _/'-]{2,40}?)\s+"
        r"(?:rule|procedure|policy|runbook)\s+is\s+no\s+longer\s+"
        r"(?P<value>[A-Za-z0-9 _/'-]{2,80})",
        re.I,
    ),
    re.compile(
        r"\b(?:the\s+)?(?P<workflow>[A-Za-z][A-Za-z0-9 _/'-]{2,50}?)\s+"
        r"(?:workflow|runbook|procedure)\s+(?P<rule>rollback|deploy|release|incident|backup)\s+"
        r"(?:rule|step|policy)\s+is\s+not\s+(?P<value>[A-Za-z0-9 _/'-]{2,80})\s+anymore",
        re.I,
    ),
]

WORKFLOW_QUERY_TERMS = {
    "backup",
    "deploy",
    "deployment",
    "incident",
    "policy",
    "procedure",
    "release",
    "rollback",
    "runbook",
    "workflow",
}

RUNTIME_STATUS_PATTERNS = [
    re.compile(
        r"\b(?:the\s+)?(?P<system>[A-Za-z][A-Za-z0-9 _/'-]{2,50}?)\s+"
        r"(?:is|was)\s+(?:now\s+)?"
        r"(?P<status>available|blocked|degraded|down|fixed|healthy|offline|online|unavailable|up)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:the\s+)?(?P<system>[A-Za-z][A-Za-z0-9 _/'-]{2,50}?)\s+"
        r"has\s+been\s+(?P<status>degraded|fixed|restored|unblocked)\b",
        re.I,
    ),
]

RUNTIME_UNKNOWN_CURRENT_PATTERNS = [
    re.compile(
        r"\b(?:the\s+)?(?P<system>[A-Za-z][A-Za-z0-9 _/'-]{2,50}?)\s+"
        r"(?:is|was)\s+no\s+longer\s+"
        r"(?P<status>available|blocked|degraded|down|fixed|healthy|offline|online|unavailable|up)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:the\s+)?(?P<system>[A-Za-z][A-Za-z0-9 _/'-]{2,50}?)\s+"
        r"(?:is|was)\s+not\s+"
        r"(?P<status>available|blocked|degraded|down|fixed|healthy|offline|online|unavailable|up)\s+anymore\b",
        re.I,
    ),
]

RUNTIME_QUERY_TERMS = {
    "api endpoint",
    "cluster",
    "database",
    "db",
    "endpoint",
    "environment",
    "job",
    "queue",
    "runner",
    "service",
    "system",
    "tool",
}

RUNTIME_NOUNS = {
    "api",
    "cluster",
    "database",
    "db",
    "endpoint",
    "environment",
    "job",
    "queue",
    "runner",
    "service",
    "system",
    "tool",
}

ROLE_PATTERNS = [
    re.compile(
        r"\b(?:my\s+)?(?:current\s+)?(?:role|job\s+title|title|position)\s+is\s+"
        r"(?P<value>[A-Za-z][A-Za-z0-9 _/'-]{2,50})",
        re.I,
    ),
    re.compile(
        r"\b(?:my\s+)?(?:new\s+)(?:role|job\s+title|title|position)\s+is\s+"
        r"(?P<value>[A-Za-z][A-Za-z0-9 _/'-]{2,50})",
        re.I,
    ),
]

ROLE_UNKNOWN_CURRENT_PATTERNS = [
    re.compile(
        r"\b(?:i[’']?m|i am)\s+no\s+longer\s+(?:the\s+)?"
        r"(?P<value>[A-Za-z][A-Za-z0-9 _/'-]{2,50})",
        re.I,
    ),
    re.compile(
        r"\b(?:my\s+)?(?:role|job\s+title|title|position)\s+is\s+no\s+longer\s+"
        r"(?P<value>[A-Za-z][A-Za-z0-9 _/'-]{2,50})",
        re.I,
    ),
]

ROLE_QUERY_TERMS = {
    "position",
    "responsibilities",
    "responsibility",
    "role",
    "title",
}

ROLE_NOUNS = {
    "architect",
    "designer",
    "developer",
    "director",
    "engineer",
    "lead",
    "manager",
    "owner",
    "pm",
    "researcher",
}

MANAGER_PATTERNS = [
    re.compile(
        r"\b(?:my\s+)?(?:new\s+)?manager\s+is\s+(?P<person>[A-Z][A-Za-z .'-]{1,40})",
        re.I,
    ),
    re.compile(
        r"\b(?P<person>[A-Z][A-Za-z .'-]{1,40})\s+is\s+my\s+(?:new\s+)?manager\b",
        re.I,
    ),
]

MANAGER_UNKNOWN_CURRENT_PATTERNS = [
    re.compile(
        r"\b(?P<person>[A-Z][A-Za-z .'-]{1,40})\s+is\s+no\s+longer\s+my\s+manager\b",
        re.I,
    ),
    re.compile(
        r"\b(?:my\s+)?manager\s+is\s+no\s+longer\s+(?P<person>[A-Z][A-Za-z .'-]{1,40})",
        re.I,
    ),
]

MANAGER_QUERY_TERMS = {
    "manager",
    "reports to",
    "supervisor",
}

EMPLOYER_PATTERNS = [
    re.compile(
        r"\b(?:i\s+)?(?:work|worked)\s+(?:at|for)\s+"
        r"(?P<value>[A-Z][A-Za-z0-9 &.'-]{1,60})",
        re.I,
    ),
    re.compile(
        r"\b(?:my\s+)?(?:current\s+)?(?:employer|company|workplace|organization)\s+is\s+"
        r"(?P<value>[A-Z][A-Za-z0-9 &.'-]{1,60})",
        re.I,
    ),
    re.compile(
        r"\b(?:i\s+)?(?:joined|started\s+(?:at|with))\s+"
        r"(?P<value>[A-Z][A-Za-z0-9 &.'-]{1,60})",
        re.I,
    ),
]

EMPLOYER_UNKNOWN_CURRENT_PATTERNS = [
    re.compile(
        r"\b(?:i\s+)?no\s+longer\s+(?:work|worked)\s+(?:at|for)\s+"
        r"(?P<value>[A-Z][A-Za-z0-9 &.'-]{1,60})",
        re.I,
    ),
    re.compile(
        r"\b(?:i[’']?m|i am)\s+not\s+working\s+(?:at|for)\s+"
        r"(?P<value>[A-Z][A-Za-z0-9 &.'-]{1,60})\s+anymore\b",
        re.I,
    ),
    re.compile(
        r"\b(?:my\s+)?(?:employer|company|workplace|organization)\s+is\s+no\s+longer\s+"
        r"(?P<value>[A-Z][A-Za-z0-9 &.'-]{1,60})",
        re.I,
    ),
    re.compile(
        r"\b(?P<value>[A-Z][A-Za-z0-9 &.'-]{1,60})\s+is\s+no\s+longer\s+my\s+"
        r"(?:employer|company|workplace|organization)\b",
        re.I,
    ),
]

EMPLOYER_QUERY_TERMS = {
    "company",
    "employer",
    "organization",
    "workplace",
}

BENEFITS_PORTAL_PATTERNS = [
    re.compile(
        r"\b(?:my\s+)?benefits\s+(?:portal|site|system)\s+is\s+"
        r"(?P<value>[A-Z][A-Za-z0-9 &.'-]{1,60})",
        re.I,
    ),
    re.compile(
        r"\b(?:for\s+)?benefits(?:\s+enrollment)?,?\s+(?:use|uses|go\s+through)\s+"
        r"(?P<value>[A-Z][A-Za-z0-9 &.'-]{1,60})",
        re.I,
    ),
]

BENEFITS_PORTAL_QUERY_TERMS = {
    "benefit",
    "benefits",
    "enrollment",
    "portal",
}


def extract_state_patches(content: str, metadata: dict[str, object] | None = None) -> list[StatePatch]:
    """Deterministic state extractor used for API-free development.

    This is intentionally narrow. It establishes the state-aware memory
    interface without using benchmark labels or an LLM. Later paper experiments
    should replace or augment it with a documented extractor baseline.
    """

    if metadata and metadata.get("derived"):
        return []
    if _speaker(content) == "assistant":
        return []

    patches: list[StatePatch] = []
    location_added = False
    for pattern in LOCATION_PATTERNS:
        match = pattern.search(content)
        if not match:
            continue
        value = _clean_location(match.group(1))
        if value:
            patches.append(StatePatch(slot="location", value=value, evidence=content))
            location_added = True
            break
    if not location_added:
        invalidated_location = _extract_unknown_current_location(content)
        if invalidated_location:
            patches.append(
                StatePatch(
                    slot="location",
                    value="unknown-current",
                    evidence=content,
                    status="unknown_current",
                    invalidates_value=invalidated_location,
                )
            )
    if local_gym := _extract_local_gym(content):
        patches.append(StatePatch(slot="local.gym", value=local_gym, evidence=content))
    beverage_unknown_current = _extract_beverage_unknown_current(content)
    if beverage_unknown_current:
        patches.append(
            StatePatch(
                slot="preference.beverage",
                value="unknown-current",
                evidence=content,
                status="unknown_current",
                invalidates_value=beverage_unknown_current,
            )
        )
    elif beverage := _extract_beverage_value(content):
        patches.append(StatePatch(slot="preference.beverage", value=beverage, evidence=content))
    schedule = _extract_schedule_availability(content)
    if schedule:
        patches.append(StatePatch(slot="schedule.availability", value=schedule, evidence=content))
    task_status = _extract_task_status(content)
    if task_status:
        task, status = task_status
        patches.append(StatePatch(slot=f"task.{task}.status", value=status, evidence=content))
    dietary_status = _extract_dietary_status(content)
    if dietary_status:
        slot_suffix, value = dietary_status
        patches.append(StatePatch(slot=f"health.{slot_suffix}.status", value=value, evidence=content))
    resource_status = _extract_resource_status(content)
    if resource_status:
        resource, status = resource_status
        patches.append(StatePatch(slot=f"resource.{resource}.status", value=status, evidence=content))
    elif resource_unknown_current := _extract_resource_unknown_current(content):
        resource, status = resource_unknown_current
        patches.append(StatePatch(
            slot=f"resource.{resource}.status",
            value="unknown-current",
            evidence=content,
            status="unknown_current",
            invalidates_value=status,
        ))
    workflow_unknown_current = _extract_workflow_unknown_current(content)
    if workflow_unknown_current:
        workflow, rule, value = workflow_unknown_current
        patches.append(StatePatch(
            slot=f"workflow.{workflow}.{rule}",
            value="unknown-current",
            evidence=content,
            status="unknown_current",
            invalidates_value=value,
        ))
    elif workflow_rule := _extract_workflow_rule(content):
        workflow, rule, value = workflow_rule
        patches.append(StatePatch(slot=f"workflow.{workflow}.{rule}", value=value, evidence=content))
    runtime_status = _extract_runtime_status(content)
    if runtime_status:
        system, status = runtime_status
        patches.append(StatePatch(slot=f"runtime.{system}.status", value=status, evidence=content))
    elif runtime_unknown_current := _extract_runtime_unknown_current(content):
        system, status = runtime_unknown_current
        patches.append(StatePatch(
            slot=f"runtime.{system}.status",
            value="unknown-current",
            evidence=content,
            status="unknown_current",
            invalidates_value=status,
        ))
    role_unknown_current = _extract_role_unknown_current(content)
    if role_unknown_current:
        patches.append(StatePatch(
            slot="role.current",
            value="unknown-current",
            evidence=content,
            status="unknown_current",
            invalidates_value=role_unknown_current,
        ))
    elif role := _extract_role(content):
        patches.append(StatePatch(slot="role.current", value=role, evidence=content))
    manager_unknown_current = _extract_manager_unknown_current(content)
    if manager_unknown_current:
        patches.append(StatePatch(
            slot="relationship.manager",
            value="unknown-current",
            evidence=content,
            status="unknown_current",
            invalidates_value=manager_unknown_current,
        ))
    elif manager := _extract_manager(content):
        patches.append(StatePatch(slot="relationship.manager", value=manager, evidence=content))
    employer_unknown_current = _extract_employer_unknown_current(content)
    if employer_unknown_current:
        patches.append(StatePatch(
            slot="organization.employer",
            value="unknown-current",
            evidence=content,
            status="unknown_current",
            invalidates_value=employer_unknown_current,
        ))
    elif employer := _extract_employer(content):
        patches.append(StatePatch(slot="organization.employer", value=employer, evidence=content))
    if benefits_portal := _extract_benefits_portal(content):
        patches.append(StatePatch(
            slot="employment.benefits_portal",
            value=benefits_portal,
            evidence=content,
        ))
    return patches


def query_needs_state_readout(query: str) -> bool:
    return bool(query_relevant_state_slots(query))


def query_relevant_state_slots(query: str) -> list[str]:
    text = query.lower()
    slots: list[str] = []
    if _has_location_intent(text):
        slots.append("location")
    if _has_local_gym_intent(text):
        slots.append("local.gym")
    if _has_beverage_intent(text):
        slots.append("preference.beverage")
    if _has_schedule_intent(text):
        slots.append("schedule.availability")
    if _has_task_status_intent(text):
        slots.append("task.*.status")
    if _has_dietary_intent(text):
        slots.append("health.*.status")
    if _has_resource_intent(text):
        slots.append("resource.*.status")
    if _has_any_term(text, WORKFLOW_QUERY_TERMS):
        slots.append("workflow.*")
    if _has_runtime_intent(text):
        slots.append("runtime.*.status")
    if _has_role_intent(text):
        slots.append("role.current")
    if _has_manager_intent(text):
        slots.append("relationship.manager")
    if _has_employer_intent(text):
        slots.append("organization.employer")
    if _has_benefits_portal_intent(text):
        slots.append("employment.benefits_portal")
    return slots


def _has_local_location_intent(text: str) -> bool:
    if not _has_term(text, "local"):
        return False
    return _has_any_term(text, LOCAL_LOCATION_CONTEXT_TERMS)


def _has_location_intent(text: str) -> bool:
    if _has_any_term(text, {"live", "located", "based", "staying"}) and _has_self_location_subject(text):
        return True
    if _has_term(text, "city") and _has_any_term(text, {"right", "still", "where", "which", "what"}):
        return True
    if _has_any_phrase(text, {"near me", "nearby", "around me"}):
        return True
    if _has_local_location_intent(text):
        return True
    if _has_any_term(text, {"area", "areas", "neighborhood", "neighborhoods", "place", "places", "spots"}):
        return _has_any_term(text, {"focus", "go", "nearby", "option", "options", "recommend", "suggest", "where"})
    if _has_term(text, "near"):
        return _has_any_phrase(text, {"near me", "near the user", "nearby"}) or _has_any_term(
            text,
            {"current", "local", "recommend", "suggest"},
        )
    return False


def _has_local_gym_intent(text: str) -> bool:
    if _has_term(text, "local gym"):
        return True
    if not _has_any_term(text, LOCAL_GYM_QUERY_TERMS):
        return False
    return _has_any_term(text, {"go", "local", "near", "nearby", "use"})


def _has_self_location_subject(text: str) -> bool:
    if _has_any_phrase(text, {"my location", "user location", "user's location"}):
        return True
    return bool(
        re.search(r"\b(?:i|user)\b.{0,50}\b(?:live|located|based|staying)\b", text)
        or re.search(r"\b(?:live|located|based|staying)\b.{0,50}\b(?:me|user)\b", text)
    )


def _has_beverage_intent(text: str) -> bool:
    if _has_any_term(text, {"beverage", "drink", "espresso", "latte", "matcha", "tea"}):
        return True
    if _has_term(text, "cafe"):
        return _has_any_term(text, {"beverage", "drink", "order", "prefer", "preference", "usual"})
    if _has_term(text, "coffee"):
        return _has_any_term(text, {"beverage", "drink", "favorite", "order", "prefer", "preference", "usual"})
    return _has_term(text, "order") and _has_any_term(text, {"beverage", "cafe", "coffee", "drink", "tea"})


def _has_task_status_intent(text: str) -> bool:
    if _has_term(text, "status"):
        return True
    state_terms = {"blocked", "open", "pending", "resolved", "paused", "cancelled", "canceled"}
    if not _has_any_term(text, state_terms | {"complete", "completed", "done"}):
        return False
    if _has_any_phrase(text, {"how many", "total number", "number of"}):
        return False
    return _has_any_term(text, TASK_QUERY_SUBJECT_TERMS)


def _has_schedule_intent(text: str) -> bool:
    direct_terms = SCHEDULE_QUERY_TERMS - {"available", "free", "meet", "meeting", "time", "times"}
    if _has_any_term(text, direct_terms):
        return True
    if _has_any_term(text, {"available", "free"}):
        return _has_any_term(text, {"calendar", "meet", "meeting", "schedule", "time", "times"})
    if _has_any_term(text, {"meet", "meeting"}):
        return _has_any_term(text, {"available", "availability", "calendar", "can", "could", "free", "schedule", "time"})
    return _has_any_term(text, {"time", "times"}) and _has_any_term(
        text,
        {"available", "availability", "calendar", "free", "meet", "meeting", "schedule"},
    )


def _has_dietary_intent(text: str) -> bool:
    core_terms = {
        "allergic",
        "allergy",
        "dairy",
        "diet",
        "dietary",
        "gluten",
        "peanut",
        "peanuts",
        "safe",
        "shellfish",
        "vegan",
        "vegetarian",
    }
    if _has_any_term(text, core_terms):
        return True
    return _has_any_term(text, {"eat", "food", "meal", "restaurant"}) and _has_any_term(
        text,
        {"constraint", "diet", "dietary", "safe", "allergy", "allergic"},
    )


def _has_resource_intent(text: str) -> bool:
    strong_terms = RESOURCE_QUERY_TERMS - {"access"}
    if _has_any_term(text, strong_terms):
        return True
    return _has_term(text, "access") and _has_any_term(
        text,
        {"account", "api key", "badge", "credential", "credentials", "key", "license", "passport", "token", "visa"},
    )


def _has_runtime_intent(text: str) -> bool:
    strong_terms = RUNTIME_QUERY_TERMS - {"job", "service", "system", "tool"}
    if _has_any_term(text, strong_terms):
        return True
    status_terms = {"available", "blocked", "degraded", "down", "fixed", "healthy", "offline", "online", "status", "up"}
    return _has_any_term(text, {"job", "service", "system", "tool"}) and _has_any_term(text, status_terms)


def _has_role_intent(text: str) -> bool:
    if _has_any_term(text, ROLE_QUERY_TERMS):
        return True
    return _has_any_phrase(text, {"current job", "job title", "my job", "user job"})


def _has_manager_intent(text: str) -> bool:
    return _has_any_term(text, MANAGER_QUERY_TERMS)


def _has_employer_intent(text: str) -> bool:
    if _has_any_term(text, EMPLOYER_QUERY_TERMS):
        return True
    return re.search(r"\b(?:work|working|worked)\s+(?:at|for)\b", text) is not None


def _has_benefits_portal_intent(text: str) -> bool:
    if _has_term(text, "benefits portal"):
        return True
    if not _has_any_term(text, BENEFITS_PORTAL_QUERY_TERMS):
        return False
    return _has_any_term(text, {"enroll", "enrollment", "portal", "site", "system", "use"})


def _has_any_term(text: str, terms: set[str] | frozenset[str]) -> bool:
    return any(_has_term(text, term) for term in terms)


def _has_any_phrase(text: str, phrases: set[str] | frozenset[str]) -> bool:
    return any(_has_term(text, phrase) for phrase in phrases)


def _has_term(text: str, term: str) -> bool:
    escaped = re.escape(term.lower()).replace(r"\ ", r"\s+")
    return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text) is not None


def state_slot_matches_query(slot: str, relevant_slots: set[str]) -> bool:
    if slot in relevant_slots:
        return True
    for relevant in relevant_slots:
        if "*" not in relevant:
            continue
        prefix, suffix = relevant.split("*", 1)
        if slot.startswith(prefix) and slot.endswith(suffix):
            return True
    return False


def state_slot_depends_on(slot: str, changed_slot: str) -> bool:
    """Return whether `slot` should be invalidated by `changed_slot`.

    This is intentionally small and deterministic. It encodes dependency
    topology for state validity, not semantic retrieval relevance.
    """

    prefixes = STATE_DEPENDENCY_PREFIXES.get(changed_slot, ())
    return any(slot.startswith(prefix) for prefix in prefixes)


def _speaker(content: str) -> str | None:
    match = re.search(r"\]\s*([A-Za-z_]+):", content)
    if not match:
        return None
    return match.group(1).lower()


def _clean_location(raw: str) -> str:
    value = raw.strip(" .,!?:;[](){}\"'“”‘’")
    words = value.split()
    kept: list[str] = []
    for word in words:
        stripped = word.strip(" .,!?:;[](){}\"'“”‘’")
        if stripped.lower() in LOCATION_STOP:
            break
        kept.append(stripped)
    return " ".join(kept).strip()


def _extract_beverage_value(content: str) -> str | None:
    for pattern in BEVERAGE_PATTERNS:
        match = pattern.search(content)
        if not match:
            continue
        value = _clean_state_value(match.group("value"))
        if _is_known_beverage(value):
            return value
    return None


def _extract_beverage_unknown_current(content: str) -> str | None:
    for pattern in BEVERAGE_UNKNOWN_CURRENT_PATTERNS:
        match = pattern.search(content)
        if not match:
            continue
        value = _clean_state_value(match.group("value"))
        if _is_known_beverage(value):
            return value
    return None


def _extract_unknown_current_location(content: str) -> str | None:
    for pattern in LOCATION_UNKNOWN_CURRENT_PATTERNS:
        match = pattern.search(content)
        if not match:
            continue
        value = _clean_location(match.group(1))
        if value:
            return value
    return None


def _extract_local_gym(content: str) -> str | None:
    for pattern in LOCAL_GYM_PATTERNS:
        match = pattern.search(content)
        if not match:
            continue
        gym = _clean_organization_name(match.group("value"))
        if gym:
            return gym
    return None


def _extract_schedule_availability(content: str) -> str | None:
    for pattern in SCHEDULE_PATTERNS:
        match = pattern.search(content)
        if not match:
            continue
        when = _clean_state_value(match.group("when"))
        if not when:
            continue
        prefix = match.groupdict().get("value")
        if prefix:
            return f"{prefix.lower()} {when}"
        if "can't meet" in match.group(0).lower():
            return f"unavailable {when}"
        return when
    return None


def _extract_task_status(content: str) -> tuple[str, str] | None:
    for pattern in TASK_STATUS_PATTERNS:
        match = pattern.search(content)
        if not match:
            continue
        task = _slug(_clean_task_name(match.group("task")))
        status = _normalize_status(match.group("status"))
        if task and status:
            return task, status
    return None


def _extract_dietary_status(content: str) -> tuple[str, str] | None:
    for pattern in DIETARY_CLEARED_PATTERNS:
        match = pattern.search(content)
        if not match:
            continue
        slot_suffix, label = _dietary_slot(match.groupdict().get("item") or match.group(0))
        if slot_suffix:
            return slot_suffix, f"{label} cleared"
    for pattern in DIETARY_ACTIVE_PATTERNS:
        match = pattern.search(content)
        if not match:
            continue
        raw = match.groupdict().get("item") or match.groupdict().get("constraint") or match.group(0)
        slot_suffix, label = _dietary_slot(raw)
        if slot_suffix:
            return slot_suffix, f"{label} active"
    return None


def _extract_resource_status(content: str) -> tuple[str, str] | None:
    for pattern in RESOURCE_STATUS_PATTERNS:
        match = pattern.search(content)
        if not match:
            continue
        resource = _clean_resource_name(match.group("resource"))
        status = _normalize_resource_status(match.group("status"))
        if resource and status and _looks_like_resource(resource):
            return _slug(resource), status
    return None


def _extract_resource_unknown_current(content: str) -> tuple[str, str] | None:
    for pattern in RESOURCE_UNKNOWN_CURRENT_PATTERNS:
        match = pattern.search(content)
        if not match:
            continue
        resource = _clean_resource_name(match.group("resource"))
        status = _normalize_resource_status(match.group("status"))
        if resource and status and _looks_like_resource(resource):
            return _slug(resource), status
    return None


def _extract_workflow_rule(content: str) -> tuple[str, str, str] | None:
    for pattern in WORKFLOW_RULE_PATTERNS:
        match = pattern.search(content)
        if not match:
            continue
        workflow = _slug(_clean_state_phrase(match.group("workflow")))
        rule = _slug(_clean_state_phrase(match.group("rule")))
        value = _clean_state_phrase(match.group("value"))
        if workflow and rule and value:
            return workflow, rule, value
    return None


def _extract_workflow_unknown_current(content: str) -> tuple[str, str, str] | None:
    for pattern in WORKFLOW_UNKNOWN_CURRENT_PATTERNS:
        match = pattern.search(content)
        if not match:
            continue
        workflow = _slug(_clean_state_phrase(match.group("workflow")))
        rule = _slug(_clean_state_phrase(match.group("rule")))
        value = _clean_state_phrase(match.group("value"))
        if workflow and rule and value:
            return workflow, rule, value
    return None


def _extract_runtime_status(content: str) -> tuple[str, str] | None:
    for pattern in RUNTIME_STATUS_PATTERNS:
        match = pattern.search(content)
        if not match:
            continue
        system = _clean_state_phrase(match.group("system"))
        status = _normalize_runtime_status(match.group("status"))
        if system and status and _looks_like_runtime(system):
            return _slug(system), status
    return None


def _extract_runtime_unknown_current(content: str) -> tuple[str, str] | None:
    for pattern in RUNTIME_UNKNOWN_CURRENT_PATTERNS:
        match = pattern.search(content)
        if not match:
            continue
        system = _clean_state_phrase(match.group("system"))
        status = _normalize_runtime_status(match.group("status"))
        if system and status and _looks_like_runtime(system):
            return _slug(system), status
    return None


def _extract_role(content: str) -> str | None:
    for pattern in ROLE_PATTERNS:
        match = pattern.search(content)
        if not match:
            continue
        role = _clean_role_name(match.group("value"))
        if role and _looks_like_role(role):
            return role
    return None


def _extract_role_unknown_current(content: str) -> str | None:
    for pattern in ROLE_UNKNOWN_CURRENT_PATTERNS:
        match = pattern.search(content)
        if not match:
            continue
        role = _clean_role_name(match.group("value"))
        if role and _looks_like_role(role):
            return role
    return None


def _extract_manager(content: str) -> str | None:
    for pattern in MANAGER_PATTERNS:
        match = pattern.search(content)
        if not match:
            continue
        person = _clean_person_name(match.group("person"))
        if person:
            return person
    return None


def _extract_manager_unknown_current(content: str) -> str | None:
    for pattern in MANAGER_UNKNOWN_CURRENT_PATTERNS:
        match = pattern.search(content)
        if not match:
            continue
        person = _clean_person_name(match.group("person"))
        if person:
            return person
    return None


def _extract_employer(content: str) -> str | None:
    for pattern in EMPLOYER_PATTERNS:
        match = pattern.search(content)
        if not match:
            continue
        employer = _clean_organization_name(match.group("value"))
        if employer:
            return employer
    return None


def _extract_employer_unknown_current(content: str) -> str | None:
    for pattern in EMPLOYER_UNKNOWN_CURRENT_PATTERNS:
        match = pattern.search(content)
        if not match:
            continue
        employer = _clean_organization_name(match.group("value"))
        if employer:
            return employer
    return None


def _extract_benefits_portal(content: str) -> str | None:
    for pattern in BENEFITS_PORTAL_PATTERNS:
        match = pattern.search(content)
        if not match:
            continue
        portal = _clean_organization_name(match.group("value"))
        if portal:
            return portal
    return None


def _clean_state_value(raw: str) -> str:
    value = raw.strip(" .,!?:;[](){}\"'“”‘’").lower()
    words = value.split()
    kept: list[str] = []
    for word in words:
        stripped = word.strip(" .,!?:;[](){}\"'“”‘’").lower()
        if stripped in LOCATION_STOP:
            break
        kept.append(stripped)
    return " ".join(kept).strip()


def _clean_state_phrase(raw: str) -> str:
    value = raw.strip(" .,!?:;[](){}\"'“”‘’").lower()
    value = re.sub(r"^(?:the|a|an|my)\s+", "", value)
    return re.sub(r"\s+", " ", value).strip()


def _clean_resource_name(raw: str) -> str:
    value = raw.strip(" .,!?:;[](){}\"'“”‘’").lower()
    value = re.sub(r"^(?:the|a|an|my)\s+", "", value)
    words = value.split()
    kept: list[str] = []
    for word in words:
        stripped = word.strip(" .,!?:;[](){}\"'“”‘’").lower()
        if stripped in {"is", "was", "has", "now", "currently"}:
            break
        kept.append(stripped)
    return " ".join(kept).strip()


def _clean_role_name(raw: str) -> str:
    value = _clean_state_phrase(raw)
    words = value.split()
    kept: list[str] = []
    for word in words:
        stripped = word.strip(" .,!?:;[](){}\"'“”‘’").lower()
        if stripped in {"and", "but", "for", "now", "with"}:
            break
        kept.append(stripped)
    return " ".join(kept).strip()


def _clean_person_name(raw: str) -> str:
    value = raw.strip(" .,!?:;[](){}\"'“”‘’")
    words = value.split()
    kept: list[str] = []
    for word in words:
        stripped = word.strip(" .,!?:;[](){}\"'“”‘’")
        if stripped.lower() in {"and", "but", "for", "from", "now", "since", "so", "that", "the", "to", "with"}:
            break
        kept.append(stripped)
    return " ".join(kept).strip()


def _clean_organization_name(raw: str) -> str:
    value = raw.strip(" .,!?:;[](){}\"'“”‘’")
    words = value.split()
    kept: list[str] = []
    for word in words:
        stripped = word.strip(" .,!?:;[](){}\"'“”‘’")
        if stripped.lower() in {
            "and",
            "anymore",
            "because",
            "but",
            "for",
            "now",
            "since",
            "so",
            "where",
            "with",
        }:
            break
        kept.append(stripped)
    return " ".join(kept).strip()


def _dietary_slot(raw: str) -> tuple[str | None, str]:
    value = _clean_state_value(raw)
    if "peanut" in value:
        return "peanut_allergy", "peanut allergy"
    if "shellfish" in value:
        return "shellfish_allergy", "shellfish allergy"
    if "gluten" in value:
        return "gluten_restriction", "gluten restriction"
    if "dairy" in value or "milk" in value or "lactose" in value:
        return "dairy_restriction", "dairy restriction"
    if "nut" in value:
        return "nut_allergy", "nut allergy"
    if "vegan" in value:
        return "vegan_preference", "vegan preference"
    if "vegetarian" in value:
        return "vegetarian_preference", "vegetarian preference"
    return None, value


def _looks_like_resource(resource: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", resource.lower()))
    return bool(tokens & RESOURCE_NOUNS)


def _looks_like_runtime(system: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", system.lower()))
    return bool(tokens & RUNTIME_NOUNS)


def _looks_like_role(role: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", role.lower()))
    return bool(tokens & ROLE_NOUNS)


def _normalize_resource_status(raw: str) -> str:
    status = raw.strip().lower()
    if status == "activated":
        return "active"
    if status == "deactivated":
        return "disabled"
    return status


def _normalize_runtime_status(raw: str) -> str:
    status = raw.strip().lower()
    if status == "restored":
        return "online"
    if status == "unblocked":
        return "available"
    if status == "up":
        return "online"
    if status == "down":
        return "offline"
    return status


def _is_known_beverage(value: str) -> bool:
    return any(value == beverage or value.endswith(f" {beverage}") for beverage in BEVERAGE_VALUES)


def _clean_task_name(raw: str) -> str:
    value = raw.strip(" .,!?:;[](){}\"'“”‘’").lower()
    value = re.sub(r"^(?:the|a|an|my)\s+", "", value)
    words = value.split()
    kept: list[str] = []
    for word in words:
        stripped = word.strip(" .,!?:;[](){}\"'“”‘’").lower()
        if stripped in {"is", "now", "currently"}:
            break
        kept.append(stripped)
    return " ".join(kept).strip()


def _normalize_status(raw: str) -> str:
    status = raw.strip().lower()
    if status in {"complete", "completed", "done"}:
        return "completed"
    if status == "canceled":
        return "cancelled"
    return status


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug
