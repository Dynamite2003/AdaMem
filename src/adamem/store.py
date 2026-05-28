from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from adamem.schema import MemoryItem


class MemoryStore(Protocol):
    def all(self) -> list[MemoryItem]:
        ...

    def upsert(self, item: MemoryItem) -> None:
        ...

    def get(self, item_id: str) -> MemoryItem | None:
        ...

    def delete(self, item_id: str) -> None:
        ...


class InMemoryStore:
    def __init__(self) -> None:
        self._items: dict[str, MemoryItem] = {}

    def all(self) -> list[MemoryItem]:
        return list(self._items.values())

    def upsert(self, item: MemoryItem) -> None:
        self._items[item.id] = item

    def get(self, item_id: str) -> MemoryItem | None:
        return self._items.get(item_id)

    def delete(self, item_id: str) -> None:
        self._items.pop(item_id, None)


class JsonMemoryStore(InMemoryStore):
    """Tiny durable store for prototypes and tests.

    Production users can replace this with SQLite, Postgres, Redis, Graphiti,
    or any vector database by implementing the four-method MemoryStore protocol.
    """

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            self._items = {entry["id"]: MemoryItem(**entry) for entry in data}

    def upsert(self, item: MemoryItem) -> None:
        super().upsert(item)
        self._flush()

    def delete(self, item_id: str) -> None:
        super().delete(item_id)
        self._flush()

    def _flush(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump([asdict(item) for item in self.all()], handle, ensure_ascii=False, indent=2)
        tmp.replace(self.path)
