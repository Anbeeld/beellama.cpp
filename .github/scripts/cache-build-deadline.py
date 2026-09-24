#!/usr/bin/env python3
"""Run a build until the job's cache-save deadline, then stop its process tree."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import time


TIMED_OUT_EXIT = 75


def mark_timed_out() -> int:
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        output.write("timed_out=true\n")
    print("Build deadline reached; compiler processes stopped before saving ccache.", flush=True)
    return TIMED_OUT_EXIT


def stop_tree(process: subprocess.Popen[bytes]) -> None:
    if os.name == "nt":
        # cmd / cmake / ninja / compiler children must all exit before cache archiving.
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=True)
    else:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        # The leader can exit before its compiler children; stop the whole group.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def main() -> int:
    command = sys.argv[1:]
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("Expected a build command")
    deadline = int(os.environ["CACHE_SAVE_DEADLINE"])
    remaining = deadline - time.time()
    if remaining <= 0:
        return mark_timed_out()

    options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    process = subprocess.Popen(command, **options)
    try:
        return process.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        stop_tree(process)
        return mark_timed_out()


if __name__ == "__main__":
    sys.exit(main())
