#!/usr/bin/env python3
"""Exercise the compilation-slice CLI with real, short-lived process trees."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github/scripts/cache-build-deadline.py"


class CompilationSliceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / "output"
        self.env = {**os.environ, "GITHUB_OUTPUT": str(self.output)}

    def run_slice(self, code, seconds="5"):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--seconds", seconds, "--", sys.executable, "-c", code],
            env=self.env, capture_output=True, text=True, timeout=20,
        )
        outputs = self.output.read_text() if self.output.exists() else ""
        return result, outputs

    def test_default_interval_is_110_minutes(self):
        spec = importlib.util.spec_from_file_location("cache_build_deadline", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.CHECKPOINT_SECONDS, 110 * 60)

    def test_success_is_complete(self):
        result, outputs = self.run_slice("print('compiled')")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("compiled", result.stdout)
        self.assertEqual(outputs, "checkpoint=false\n")

    def test_compiler_errors_are_not_checkpoints(self):
        for code in (1, 2, 75):
            with self.subTest(code=code):
                self.output.write_text("")
                result, outputs = self.run_slice(f"raise SystemExit({code})")
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertEqual(outputs, "checkpoint=false\n")

    def test_interval_stops_compiler_descendants_before_checkpoint(self):
        heartbeat = self.root / "heartbeat"
        child = (
            "import pathlib, time, signal; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            f"p = pathlib.Path({str(heartbeat)!r}); "
            "exec('while True:\\n p.write_text(str(time.time_ns()))\\n time.sleep(0.02)')"
        )
        parent = (
            "import subprocess, sys, time; "
            f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
            "time.sleep(60)"
        )
        result, outputs = self.run_slice(parent, "1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(outputs, "checkpoint=true\n")
        self.assertIn("compiler tree stopped for ccache checkpoint", result.stdout)
        self.assertTrue(heartbeat.exists(), "descendant must run before the interval expires")
        final = heartbeat.read_text()
        time.sleep(0.2)
        self.assertEqual(heartbeat.read_text(), final, "compiler child continued writing during checkpoint")

    def test_incremental_build_can_finish_after_checkpoint(self):
        state = self.root / "finished-object"
        code = (
            "from pathlib import Path; import time; "
            f"p = Path({str(state)!r}); "
            "done = p.exists(); p.write_text('finished'); "
            "time.sleep(0 if done else 60)"
        )
        first, outputs = self.run_slice(code, "0.5")
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(outputs, "checkpoint=true\n")
        self.assertTrue(state.exists())
        self.output.write_text("")
        second, outputs = self.run_slice(code)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(outputs, "checkpoint=false\n")

    def test_invalid_interval_cannot_launch_a_build(self):
        for seconds in ("0", "-1", "nan", "inf"):
            with self.subTest(seconds=seconds):
                result, outputs = self.run_slice("raise SystemExit(99)", seconds)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(outputs, "")

    def test_missing_command_is_rejected(self):
        result = subprocess.run([sys.executable, str(SCRIPT)], env=self.env, capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.output.exists())

    @unittest.skipUnless(os.name == "nt", "Windows Job containment")
    def test_completed_leader_leaves_no_writer_descendants(self):
        for exit_code in (0, 17):
            with self.subTest(exit_code=exit_code):
                heartbeat = self.root / f"heartbeat-{exit_code}"
                child = (
                    "import pathlib, time; "
                    f"p = pathlib.Path({str(heartbeat)!r}); "
                    "exec('for _ in range(500):\\n p.write_text(str(time.time_ns()))\\n time.sleep(0.02)')"
                )
                parent = (
                    "import pathlib, subprocess, sys, time; "
                    f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
                    f"p = pathlib.Path({str(heartbeat)!r}); "
                    "exec('for _ in range(200):\\n if p.exists(): break\\n time.sleep(0.01)'); "
                    f"raise SystemExit({exit_code})"
                )
                self.output.write_text("")
                start = time.monotonic()
                result, outputs = self.run_slice(parent)
                self.assertEqual(result.returncode, exit_code, result.stderr)
                self.assertEqual(outputs, "checkpoint=false\n")
                self.assertLess(time.monotonic() - start, 5, "waited for abandoned descendant")
                self.assertTrue(heartbeat.exists(), "descendant must run before leader exits")
                final = heartbeat.read_text()
                time.sleep(0.2)
                self.assertEqual(heartbeat.read_text(), final, "completed leader left an active writer")

    @unittest.skipUnless(os.name == "nt", "Windows Job containment")
    def test_timeout_needs_no_taskkill_on_path(self):
        self.env["PATH"] = str(self.root)
        result, outputs = self.run_slice("import time; time.sleep(60)", "0.1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(outputs, "checkpoint=true\n")

    @unittest.skipUnless(os.name == "nt", "Windows Job containment")
    def test_windows_command_line_cwd_environment_and_stdio_are_inherited(self):
        arguments = ["space separated", 'embedded\"quote', "trailing \\", "", "nonascii-\u03bb"]
        self.env["DEADLINE_TEST_VALUE"] = "inherited"
        code = (
            "import os, pathlib, sys; "
            f"assert sys.argv[1:] == {arguments!r}; "
            "assert os.environ['DEADLINE_TEST_VALUE'] == 'inherited'; "
            "assert pathlib.Path.cwd() == pathlib.Path(sys.argv[0]).resolve().parent; "
            "assert input() == 'stdin-data'; "
            "print('stdout-data'); print('stderr-data', file=sys.stderr)"
        )
        # A real script gives argv[0] a meaningful cwd for the inheritance check.
        child = self.root / "child script.py"
        child.write_text(code, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--seconds", "5", "--", sys.executable, str(child), *arguments],
            cwd=self.root, env=self.env, input="stdin-data\n", capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("stdout-data", result.stdout)
        self.assertIn("stderr-data", result.stderr)
        self.assertEqual(self.output.read_text(), "checkpoint=false\n")


@unittest.skipUnless(os.name == "nt", "Windows Job containment")
class WindowsJobTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "windows_build_process", SCRIPT.with_name("windows_build_process.py"))
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        spec = importlib.util.spec_from_file_location("cache_build_deadline", SCRIPT)
        self.cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.cli)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / "output"
        self.command = [sys.executable, "-c", "import time; time.sleep(60)"]

    def assert_handles_closed(self, handles):
        for handle in handles:
            self.assertEqual(self.module.kernel32.WaitForSingleObject(handle, 0), 0xFFFFFFFF)
            self.assertEqual(self.module.ctypes.get_last_error(), 6)  # ERROR_INVALID_HANDLE

    def test_assignment_precedes_execution_and_handles_close(self):
        marker = self.root / "executed"
        command = [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"]
        api = self.module.kernel32
        create = api.CreateProcessW
        assign = api.AssignProcessToJobObject
        resume = api.ResumeThread
        events = []
        handles = []

        def create_suspended(*args):
            self.assertTrue(args[5] & self.module.CREATE_SUSPENDED)
            self.assertTrue(args[5] & subprocess.CREATE_NEW_PROCESS_GROUP)
            result = create(*args)
            info = args[-1]._obj
            handles.extend((info.hProcess, info.hThread))
            events.append("created")
            return result

        def assign_before_execution(job, process):
            self.assertFalse(marker.exists())
            self.assertEqual(api.WaitForSingleObject(process, 0), 0x102)
            events.append("assigned")
            handles.append(job)
            return assign(job, process)

        def resume_after_assignment(thread):
            self.assertEqual(events, ["created", "assigned"])
            events.append("resumed")
            return resume(thread)

        with mock.patch.object(api, "CreateProcessW", side_effect=create_suspended), \
             mock.patch.object(api, "AssignProcessToJobObject", side_effect=assign_before_execution), \
             mock.patch.object(api, "ResumeThread", side_effect=resume_after_assignment):
            process = self.module.WindowsBuildProcess(command)
        try:
            self.assertEqual(process.wait(timeout=5), 0)
            self.assertTrue(marker.exists())
            self.assertEqual(process.poll(), 0)
        finally:
            process.close()
        self.assertEqual(events, ["created", "assigned", "resumed"])
        self.assert_handles_closed(handles)
        process.close()  # idempotent cleanup

    def test_setup_failures_never_execute_child_and_release_handles(self):
        api = self.module.kernel32
        for function, value in (
            ("CreateJobObjectW", None), ("SetInformationJobObject", 0),
            ("CreateProcessW", 0), ("AssignProcessToJobObject", 0),
            ("ResumeThread", 0xFFFFFFFF),
        ):
            with self.subTest(function=function):
                marker = self.root / function
                command = [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"]
                create = api.CreateProcessW
                create_job = api.CreateJobObjectW
                handles = []

                def capture_process(*args):
                    result = create(*args)
                    info = args[-1]._obj
                    handles.extend(h for h in (info.hProcess, info.hThread) if h)
                    return result

                def capture_job(*args):
                    handle = create_job(*args)
                    if handle:
                        handles.append(handle)
                    return handle

                close = api.CloseHandle
                process_handles = []
                def confirm_process_stopped(handle):
                    if process_handles and handle == process_handles[0]:
                        self.assertEqual(api.WaitForSingleObject(handle, 0), 0)
                    return close(handle)

                # Capture the process handle independently of which setup call fails.
                def capture_and_remember_process(*args):
                    result = capture_process(*args)
                    if args[-1]._obj.hProcess:
                        process_handles.append(args[-1]._obj.hProcess)
                    return result

                terminate = api.TerminateProcess
                with mock.patch.object(api, "CreateProcessW", side_effect=capture_and_remember_process), \
                     mock.patch.object(api, "CreateJobObjectW", side_effect=capture_job), \
                     mock.patch.object(api, "TerminateProcess", wraps=terminate) as kill, \
                     mock.patch.object(api, "CloseHandle", side_effect=confirm_process_stopped), \
                     mock.patch.object(api, function, return_value=value):
                    with self.assertRaises(OSError):
                        self.module.WindowsBuildProcess(command)
                if function == "AssignProcessToJobObject":
                    kill.assert_called_once()  # not yet contained: terminate suspended process directly
                self.assertFalse(marker.exists(), "setup failure allowed user code to execute")
                self.assert_handles_closed(handles)

    def test_interrupt_after_create_cleans_suspended_process(self):
        marker = self.root / "executed"
        command = [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"]
        api = self.module.kernel32
        create = api.CreateProcessW
        handles = []

        def create_then_interrupt(*args):
            self.assertTrue(create(*args))
            info = args[-1]._obj
            handles.extend((info.hProcess, info.hThread))
            raise KeyboardInterrupt

        terminate = api.TerminateProcess
        with mock.patch.object(api, "CreateProcessW", side_effect=create_then_interrupt), \
             mock.patch.object(api, "TerminateProcess", wraps=terminate) as kill:
            with self.assertRaises(KeyboardInterrupt):
                self.module.WindowsBuildProcess(command)
        kill.assert_called_once()
        self.assertFalse(marker.exists())
        self.assert_handles_closed(handles)

    def test_shutdown_errors_cannot_publish_checkpoint(self):
        for function in ("TerminateJobObject", "QueryInformationJobObject"):
            with self.subTest(function=function):
                self.output.unlink(missing_ok=True)
                process = self.module.WindowsBuildProcess(self.command)
                handles = [process._job, process._process]
                with mock.patch.dict(os.environ, {"GITHUB_OUTPUT": str(self.output)}), \
                     mock.patch.dict(sys.modules, {"windows_build_process": self.module}), \
                     mock.patch.object(self.module, "WindowsBuildProcess", return_value=process), \
                     mock.patch.object(sys, "argv", [str(SCRIPT), "--seconds", "0.05", "--", "unused"]), \
                     mock.patch.object(self.module.kernel32, function, return_value=0):
                    with self.assertRaises(OSError):
                        self.cli.main()
                self.assertFalse(self.output.exists(), "unconfirmed shutdown published an output")
                self.assert_handles_closed(handles)

    def test_nonempty_job_confirmation_is_bounded(self):
        process = self.module.WindowsBuildProcess(self.command)
        try:
            def always_active(job, info_class, information, length, returned):
                information._obj.ActiveProcesses = 1
                return 1

            start = time.monotonic()
            with mock.patch.object(self.module, "SHUTDOWN_SECONDS", 0.05), \
                 mock.patch.object(self.module.kernel32, "QueryInformationJobObject", side_effect=always_active):
                with self.assertRaisesRegex(TimeoutError, "checkpoint refused"):
                    process.stop_tree()
            self.assertLess(time.monotonic() - start, 1)
        finally:
            process.close()

    def test_interrupt_stops_descendants_without_checkpoint(self):
        heartbeat = self.root / "heartbeat"
        child = (
            "from pathlib import Path; import time; "
            f"p = Path({str(heartbeat)!r}); "
            "exec('while True:\\n p.write_text(str(time.time_ns()))\\n time.sleep(0.02)')"
        )
        parent = (
            "import subprocess, sys, time; "
            f"subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(60)"
        )
        process = self.module.WindowsBuildProcess([sys.executable, "-c", parent])
        try:
            deadline = time.monotonic() + 5
            while not heartbeat.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(heartbeat.exists(), "descendant did not start")
            wait = process.wait
            # Only interrupt the outer wait, not cleanup's real process wait.
            def interrupt_once(timeout=None):
                process.wait = wait
                raise KeyboardInterrupt

            with mock.patch.dict(os.environ, {"GITHUB_OUTPUT": str(self.output)}), \
                 mock.patch.dict(sys.modules, {"windows_build_process": self.module}), \
                 mock.patch.object(self.module, "WindowsBuildProcess", return_value=process), \
                 mock.patch.object(sys, "argv", [str(SCRIPT), "--", "unused"]):
                process.wait = interrupt_once
                with self.assertRaises(KeyboardInterrupt):
                    self.cli.main()
            self.assertFalse(self.output.exists())
            final = heartbeat.read_text()
            time.sleep(0.2)
            self.assertEqual(heartbeat.read_text(), final, "interrupted build left a writer")
        finally:
            process.close()

    def test_failure_at_timeout_boundary_is_not_checkpoint(self):
        process = mock.Mock()
        process.wait.side_effect = subprocess.TimeoutExpired(self.command, 1)
        process.poll.return_value = 23
        with mock.patch.dict(os.environ, {"GITHUB_OUTPUT": str(self.output)}), \
             mock.patch.dict(sys.modules, {"windows_build_process": self.module}), \
             mock.patch.object(self.module, "WindowsBuildProcess", return_value=process), \
             mock.patch.object(sys, "argv", [str(SCRIPT), "--", "unused"]):
            self.assertEqual(self.cli.main(), 23)
        self.assertEqual(self.output.read_text(), "checkpoint=false\n")
        process.stop_tree.assert_not_called()
        process.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
