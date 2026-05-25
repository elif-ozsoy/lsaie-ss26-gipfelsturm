#!/usr/bin/env python3
"""
Apply graceful-exit-on-sigterm changes to Megatron-LM training.py.

This script is the executable counterpart to 0002-graceful-exit-on-sigterm.patch.
It applies the same three changes using string replacement instead of git apply,
which avoids all patch-format whitespace sensitivity issues.

Usage: python3 0002-graceful-exit-on-sigterm.py <path-to-training.py>
"""

import sys

if len(sys.argv) != 2:
    sys.exit(f"Usage: {sys.argv[0]} <path-to-training.py>")

path = sys.argv[1]
src = open(path).read()

# ── Gap 1 & 2: finalize in-flight async checkpoint, then wrap SIGTERM save ──
old = (
    "            if args.save:\n"
    "                save_checkpoint_and_time("
)
new = (
    "            if args.save:\n"
    "                maybe_finalize_async_save(blocking=True)\n"
    "                ft_integration.on_checkpointing_start()\n"
    "                save_checkpoint_and_time("
)
n = src.count(old)
assert n == 1, f"Pattern 1 matched {n} times (expected 1) — training.py may have changed"
src = src.replace(old, new, 1)

old = (
    "                )\n"
    "            print_datetime('exiting program after receiving SIGTERM.')"
)
new = (
    "                )\n"
    "                ft_integration.on_checkpointing_end()\n"
    "            print_datetime('exiting program after receiving SIGTERM.')"
)
n = src.count(old)
assert n == 1, f"Pattern 2 matched {n} times (expected 1) — training.py may have changed"
src = src.replace(old, new, 1)

# ── Gap 3: pre-step SIGTERM check before launching a new train_step ──
old = (
    "        ft_integration.on_checkpointing_end(is_async_finalization=True)\n"
    "        # Update the timeout for all process groups after initialization"
)
new = (
    "        ft_integration.on_checkpointing_end(is_async_finalization=True)\n"
    "\n"
    "        if args.exit_signal_handler:\n"
    "            signal_handler = get_signal_handler()\n"
    "            if any(signal_handler.signals_received()):\n"
    "                print_datetime('exiting program after receiving SIGTERM (pre-step).')\n"
    "                should_exit = True\n"
    "                break\n"
    "\n"
    "        # Update the timeout for all process groups after initialization"
)
n = src.count(old)
assert n == 1, f"Pattern 3 matched {n} times (expected 1) — training.py may have changed"
src = src.replace(old, new, 1)

open(path, "w").write(src)
print(f"[patch 0002] graceful-exit changes applied to {path}")
