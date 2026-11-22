#!/usr/bin/env python3
"""Run one compilation slice, stopping all writers before a ccache checkpoint."""

import argparse
import math
import os
from pathlib import Path
import signal
import subprocess
import sys


CHECKPOINT_SECONDS = 110 * 60


def set_checkpoint(value: bool) -> None:
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        output.write(f"checkpoint={'true' if value else 'false'}\n")


def stop_tree(process) -> None:
    if os.name == "nt":
        process.stop_tree()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        # The leader may exit before compiler children, including children that
        # ignore SIGTERM. Always terminate the remaining process group as well.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=CHECKPOINT_SECONDS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("expected a build command after --")
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive")

    if os.name == "nt":
        from windows_build_process import WindowsBuildProcess
        process = WindowsBuildProcess(command)
    else:
        process = subprocess.Popen(command, start_new_session=True)
    checkpoint = False
    try:
        try:
            result = process.wait(timeout=args.seconds)
        except subprocess.TimeoutExpired:
            # A naturally failing build at the interval boundary is still a failure,
            # not permission to replace a healthy cache.
            result = process.poll()
            if result is None:
                stop_tree(process)
                checkpoint = True
                result = 0
        except BaseException:
            stop_tree(process)
            raise
    finally:
        if os.name == "nt":
            process.close()
    # Windows close also confirms that normal/failed builds left no writers.
    # No output is published when shutdown or handle cleanup fails.
    set_checkpoint(checkpoint)
    if checkpoint:
        print("Compilation slice ended; compiler tree stopped for ccache checkpoint.", flush=True)
    return result


if __name__ == "__main__":
    sys.exit(main())
