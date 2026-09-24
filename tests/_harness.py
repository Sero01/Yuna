"""Minimal runner for the script-style tests: runs every `test_*` function in a module.

Async tests each get a fresh event loop (asyncio.run) and a 20 s time limit.
"""

import asyncio
import sys
import traceback

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING")


def run_module(namespace: dict) -> None:
    tests = [
        (name, fn)
        for name, fn in sorted(namespace.items())
        if name.startswith("test_") and callable(fn)
    ]
    failed = 0
    for name, fn in tests:
        try:
            result = fn()
            if asyncio.iscoroutine(result):

                async def limited(coro=result):
                    return await asyncio.wait_for(coro, 20)

                asyncio.run(limited())
            print(f"PASS {name}", flush=True)
        except Exception:
            failed += 1
            print(f"FAIL {name}", flush=True)
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed", flush=True)
    sys.exit(1 if failed else 0)
