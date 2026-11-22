"""Windows-only build process containment; no child executes outside the Job."""

import ctypes
from ctypes import wintypes as w
import math
import subprocess
import time


# Layouts and signatures from Windows SDK winnt.h, processthreadsapi.h and
# jobapi2.h. SIZE_T/ULONG_PTR must remain pointer-sized on 64-bit runners.
class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", w.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", w.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", w.DWORD),
        ("SchedulingClass", w.DWORD),
    ]


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
    )]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_longlong),
        ("TotalKernelTime", ctypes.c_longlong),
        ("ThisPeriodTotalUserTime", ctypes.c_longlong),
        ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
        ("TotalPageFaultCount", w.DWORD),
        ("TotalProcesses", w.DWORD),
        ("ActiveProcesses", w.DWORD),
        ("TotalTerminatedProcesses", w.DWORD),
    ]


class STARTUPINFOW(ctypes.Structure):
    _fields_ = [
        ("cb", w.DWORD), ("lpReserved", w.LPWSTR), ("lpDesktop", w.LPWSTR),
        ("lpTitle", w.LPWSTR), ("dwX", w.DWORD), ("dwY", w.DWORD),
        ("dwXSize", w.DWORD), ("dwYSize", w.DWORD),
        ("dwXCountChars", w.DWORD), ("dwYCountChars", w.DWORD),
        ("dwFillAttribute", w.DWORD), ("dwFlags", w.DWORD),
        ("wShowWindow", w.WORD), ("cbReserved2", w.WORD),
        ("lpReserved2", ctypes.POINTER(w.BYTE)),
        ("hStdInput", w.HANDLE), ("hStdOutput", w.HANDLE), ("hStdError", w.HANDLE),
    ]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("hProcess", w.HANDLE), ("hThread", w.HANDLE),
        ("dwProcessId", w.DWORD), ("dwThreadId", w.DWORD),
    ]


kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
for name, restype, argtypes in (
    ("CreateJobObjectW", w.HANDLE, [w.LPVOID, w.LPCWSTR]),
    ("SetInformationJobObject", w.BOOL, [w.HANDLE, ctypes.c_int, w.LPVOID, w.DWORD]),
    ("QueryInformationJobObject", w.BOOL, [w.HANDLE, ctypes.c_int, w.LPVOID, w.DWORD, w.LPDWORD]),
    ("AssignProcessToJobObject", w.BOOL, [w.HANDLE, w.HANDLE]),
    ("TerminateJobObject", w.BOOL, [w.HANDLE, w.UINT]),
    ("CreateProcessW", w.BOOL, [w.LPCWSTR, w.LPWSTR, w.LPVOID, w.LPVOID,
                              w.BOOL, w.DWORD, w.LPVOID, w.LPCWSTR,
                              ctypes.POINTER(STARTUPINFOW), ctypes.POINTER(PROCESS_INFORMATION)]),
    ("ResumeThread", w.DWORD, [w.HANDLE]),
    ("TerminateProcess", w.BOOL, [w.HANDLE, w.UINT]),
    ("WaitForSingleObject", w.DWORD, [w.HANDLE, w.DWORD]),
    ("GetExitCodeProcess", w.BOOL, [w.HANDLE, w.LPDWORD]),
    ("GetStdHandle", w.HANDLE, [w.DWORD]),
    ("GetCurrentProcess", w.HANDLE, []),
    ("DuplicateHandle", w.BOOL, [w.HANDLE, w.HANDLE, w.HANDLE,
                                ctypes.POINTER(w.HANDLE), w.DWORD, w.BOOL, w.DWORD]),
    ("CloseHandle", w.BOOL, [w.HANDLE]),
):
    function = getattr(kernel32, name)
    function.restype = restype
    function.argtypes = argtypes


CREATE_SUSPENDED = 0x00000004
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
SHUTDOWN_SECONDS = 30


def check(success):
    if not success:
        raise ctypes.WinError(ctypes.get_last_error())


def close_handles(handles):
    error = None
    for handle in handles:
        if handle and not kernel32.CloseHandle(handle) and error is None:
            error = ctypes.WinError(ctypes.get_last_error())
    if error is not None:
        raise error


class WindowsBuildProcess:
    """Minimal Popen-like interface with explicit, confirmed tree shutdown."""

    def __init__(self, command):
        self.args = command
        self.returncode = None
        self._job = None
        self._process = None
        self._thread = None
        self._stopped = False
        assigned = False
        std_handles = []
        try:
            try:
                self._job = kernel32.CreateJobObjectW(None, None)
                check(self._job)
                limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
                limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                check(kernel32.SetInformationJobObject(
                    self._job, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                    ctypes.byref(limits), ctypes.sizeof(limits)))

                startup = STARTUPINFOW()
                startup.cb = ctypes.sizeof(startup)
                startup.dwFlags = 0x00000100  # STARTF_USESTDHANDLES
                current = kernel32.GetCurrentProcess()
                for identifier, field in ((-10, "hStdInput"), (-11, "hStdOutput"), (-12, "hStdError")):
                    source = kernel32.GetStdHandle(identifier)
                    if source == ctypes.c_void_p(-1).value:
                        raise ctypes.WinError(ctypes.get_last_error())
                    if source:
                        duplicate = w.HANDLE()
                        check(kernel32.DuplicateHandle(
                            current, source, current, ctypes.byref(duplicate), 0, True,
                            0x00000002))  # DUPLICATE_SAME_ACCESS
                        std_handles.append(duplicate.value)
                        setattr(startup, field, duplicate.value)

                info = PROCESS_INFORMATION()
                # NULL environment/cwd inherit the helper's environment/cwd.
                # Keep the mutable command line alive through CreateProcessW.
                command_line = ctypes.create_unicode_buffer(subprocess.list2cmdline(command))
                try:
                    check(kernel32.CreateProcessW(
                        None, command_line, None, None, True,
                        CREATE_SUSPENDED | subprocess.CREATE_NEW_PROCESS_GROUP,
                        None, None, ctypes.byref(startup), ctypes.byref(info)))
                finally:
                    # Also retain the suspended child's handles if Python handles
                    # an interrupt immediately after the native call returns.
                    self._process, self._thread = info.hProcess, info.hThread
                    self.pid = info.dwProcessId
                check(kernel32.AssignProcessToJobObject(self._job, self._process))
                assigned = True
                # No user code (including descendant creation) runs before this.
                if kernel32.ResumeThread(self._thread) == 0xFFFFFFFF:
                    raise ctypes.WinError(ctypes.get_last_error())
                close_handles([self._thread])
                self._thread = None
            finally:
                close_handles(std_handles)
        except BaseException:
            try:
                if self._process:
                    if assigned:
                        self.stop_tree()
                    else:
                        # Assignment failed: the child is still suspended and
                        # cannot be terminated by closing our (empty) Job.
                        check(kernel32.TerminateProcess(self._process, 1))
                        self.wait(timeout=SHUTDOWN_SECONDS)
            finally:
                self._release_handles()
            raise

    def poll(self):
        if self.returncode is None:
            status = kernel32.WaitForSingleObject(self._process, 0)
            if status == 0:  # WAIT_OBJECT_0; exit code 259 can also be a real exit.
                code = w.DWORD()
                check(kernel32.GetExitCodeProcess(self._process, ctypes.byref(code)))
                self.returncode = code.value
            elif status != 0x00000102:  # WAIT_TIMEOUT
                raise ctypes.WinError(ctypes.get_last_error())
        return self.returncode

    def wait(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.poll() is None:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise subprocess.TimeoutExpired(self.args, timeout)
            # Short waits also permit Python KeyboardInterrupt handling.
            milliseconds = 100 if remaining is None else min(100, math.ceil(remaining * 1000))
            status = kernel32.WaitForSingleObject(self._process, milliseconds)
            if status not in (0, 0x00000102):
                raise ctypes.WinError(ctypes.get_last_error())
        return self.returncode

    def stop_tree(self):
        if self._stopped:
            return
        deadline = time.monotonic() + SHUTDOWN_SECONDS
        check(kernel32.TerminateJobObject(self._job, 1))
        while True:
            accounting = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION()
            check(kernel32.QueryInformationJobObject(
                self._job, JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
                ctypes.byref(accounting), ctypes.sizeof(accounting), None))
            if accounting.ActiveProcesses == 0:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Windows Job still has active processes; checkpoint refused")
            time.sleep(0.02)
        self.wait(timeout=max(0, deadline - time.monotonic()))
        self._stopped = True

    def _release_handles(self):
        handles = [self._thread, self._process, self._job]
        self._thread = self._process = self._job = None
        close_handles(handles)

    def close(self):
        if self._job:
            try:
                # Even a normally exited leader can leave writer descendants.
                self.stop_tree()
            finally:
                self._release_handles()
