from __future__ import annotations

import argparse

from adamem.manager import AdaMem
from adamem.store import JsonMemoryStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny AdaMem prototype CLI")
    parser.add_argument("--store", default=".adamem/memory.json", help="JSON store path")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="Add a memory")
    add.add_argument("content")
    add.add_argument("--kind", default="observation")
    add.add_argument("--importance", type=float, default=0.5)
    add.add_argument("--key", default=None)

    ask = sub.add_parser("retrieve", help="Retrieve memory context")
    ask.add_argument("query")
    ask.add_argument("--top-k", type=int, default=6)
    ask.add_argument("--max-chars", type=int, default=1800)

    args = parser.parse_args()
    mem = AdaMem(store=JsonMemoryStore(args.store))

    if args.command == "add":
        metadata = {"memory_key": args.key} if args.key else None
        item = mem.observe(args.content, kind=args.kind, importance=args.importance, metadata=metadata)
        print(item.id)
    elif args.command == "retrieve":
        print(mem.context(args.query, top_k=args.top_k, max_chars=args.max_chars))


if __name__ == "__main__":
    main()
