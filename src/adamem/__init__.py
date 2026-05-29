"""AdaMem: a minimal adaptive memory layer for LLM agents."""

from adamem.config import AdaMemConfig
from adamem.manager import AdaMem
from adamem.schema import MemoryItem, MemoryResult
from adamem.state import LLMStateExtractor, StatePatch
from adamem.store import JsonMemoryStore, MemoryStore

__all__ = [
    "AdaMem",
    "AdaMemConfig",
    "JsonMemoryStore",
    "LLMStateExtractor",
    "MemoryItem",
    "MemoryResult",
    "MemoryStore",
    "StatePatch",
]
