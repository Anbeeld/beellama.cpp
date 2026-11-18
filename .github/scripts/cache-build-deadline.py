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


def stop_tree(process: subprocess.Popen[bytes]) -> None:
    if os.name == "nt":
        # /T includes cmake, ninja/make, compiler and ccache descendants. Fail
        # closed if Windows cannot confirm that the tree was terminated.
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=True)
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

    options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    process = subprocess.Popen(command, **options)
    try:
        result = process.wait(timeout=args.seconds)
    except subprocess.TimeoutExpired:
        # A naturally failing build at the interval boundary is still a failure,
        # not permission to replace a healthy cache.
        result = process.poll()
        if result is None:
            stop_tree(process)
            set_checkpoint(True)
            print("Compilation slice ended; compiler tree stopped for ccache checkpoint.", flush=True)
            return 0
    except BaseException:
        stop_tree(process)
        raise
    set_checkpoint(False)
    return result


if __name__ == "__main__":
    sys.exit(main())
