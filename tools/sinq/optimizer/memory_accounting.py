"""Measured process/CUDA resource guards, including polling during expensive phases."""
from contextlib import contextmanager
import os
import threading
import time
from .resources import peak_rss_bytes


class ResourceLimitError(RuntimeError):
    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        super().__init__(str(diagnostic))


def _host_memory():
    try:
        import psutil
        process = psutil.Process()
        return dict(current_bytes=int(process.memory_info().rss), peak_bytes=peak_rss_bytes(),
                    source='psutil process RSS; platform process high-water working set')
    except ImportError:
        pass
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD),
                *[(n, ctypes.c_size_t) for n in ('peak', 'working', 'paged_peak', 'paged',
                    'nonpaged_peak', 'nonpaged', 'pagefile', 'pagefile_peak')]]
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        info = Counters(); info.cb = ctypes.sizeof(info)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(info), info.cb):
            raise OSError(ctypes.get_last_error(), 'cannot measure process working set')
        return dict(current_bytes=int(info.working), peak_bytes=int(info.peak), source='Windows GetProcessMemoryInfo')
    if os.path.exists('/proc/self/statm'):
        with open('/proc/self/statm', encoding='ascii') as stream:
            resident = int(stream.read().split()[1]) * os.sysconf('SC_PAGE_SIZE')
        return dict(current_bytes=resident, peak_bytes=peak_rss_bytes(), source='procfs statm RSS; getrusage peak')
    # macOS without psutil: ps supplies current RSS in KiB, not a peak estimate.
    import subprocess
    value = subprocess.check_output(['ps', '-o', 'rss=', '-p', str(os.getpid())], text=True)
    return dict(current_bytes=int(value.strip()) * 1024, peak_bytes=peak_rss_bytes(),
                source='ps process RSS; getrusage peak')


def host_capacity():
    """Available physical memory plus process RSS, never a workspace estimate."""
    current = _host_memory()['current_bytes']
    try:
        import psutil
        return dict(current_bytes=current, available_bytes=int(psutil.virtual_memory().available),
                    source='psutil available physical memory + measured process RSS - reserve')
    except ImportError:
        pass
    if os.name == 'nt':
        import ctypes
        class Status(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                *[(name, ctypes.c_ulonglong) for name in ('total', 'available', 'page_total',
                    'page_available', 'virtual_total', 'virtual_available', 'extended')]]
        status = Status(); status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError('cannot measure available physical memory')
        available = int(status.available)
        source = 'Windows GlobalMemoryStatusEx available physical + process working set - reserve'
    elif os.path.exists('/proc/meminfo'):
        with open('/proc/meminfo', encoding='ascii') as stream:
            fields = dict(line.split(':', 1) for line in stream)
        available = int(fields['MemAvailable'].split()[0]) * 1024
        source = 'procfs MemAvailable + process RSS - reserve'
    else:
        raise RuntimeError('available physical memory measurement requires psutil on this platform; install psutil or provide host_limit_bytes')
    return dict(current_bytes=current, available_bytes=available, source=source)


class ResourceMonitor:
    """Process RSS caps and CUDA allocator caps; driver free memory is separate.

    Polling cannot interrupt a running native kernel. An observed violation is
    retained and raised at the next check or phase exit, never silently discarded.
    """
    def __init__(self, policy, *, max_records=256):
        if type(max_records) is not int or max_records < 2:
            raise ValueError('resource record bound must be at least two')
        self.max_records = max_records
        self.policy = policy
        self.records = []
        self._lock = threading.Lock()
        self._active = False
        self._failure = None
        self._started = time.monotonic()
        self._phase_started = self._started
        self._peak_host = 0
        self._peak_cuda = {}

    def _sample(self):
        cuda = {}
        if any(d.type == 'cuda' for d in self.policy.devices):
            import torch
            for d in self.policy.devices:
                free, total = torch.cuda.mem_get_info(d)
                cuda[str(d)] = dict(allocated_bytes=torch.cuda.memory_allocated(d),
                    reserved_bytes=torch.cuda.memory_reserved(d), free_bytes=free, total_bytes=total,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(d),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(d),
                    source='torch CUDA allocator; cudaMemGetInfo device-global free (includes other processes)')
        host = _host_memory()
        try:
            capacity = host_capacity()
            host['available_physical_bytes'] = capacity['available_bytes']
            host['available_source'] = capacity['source']
        except (RuntimeError, OSError):
            host['available_physical_bytes'] = 'not_measured'
        return dict(host=host, cuda=cuda)

    def check(self):
        with self._lock:
            if self._failure is not None:
                raise self._failure
            sample = self._sample()
            sample['elapsed_seconds'] = time.monotonic() - self._started
            sample['phase'] = getattr(self, '_phase', None)
            sample['phase_elapsed_seconds'] = time.monotonic() - self._phase_started
            self._peak_host = max(self._peak_host, sample['host']['current_bytes'])
            sample['phase_peak_current_host_bytes'] = self._peak_host
            for name, record in sample['cuda'].items():
                peaks = self._peak_cuda.setdefault(name, {})
                for key in ('allocated_bytes', 'reserved_bytes'):
                    peaks[key] = max(peaks.get(key, 0), record[key])
            sample['phase_cuda_peaks'] = {name: dict(values) for name, values in self._peak_cuda.items()}
            # Two records per phase, bounded to the most recent 128 phases.
            # Polling updates the end/peak summary rather than growing output.
            if self._active and len(self.records) >= 2 and self.records[-1].get('phase_sequence') == self._phase_sequence:
                sample['phase_sequence'] = self._phase_sequence
                self.records[-1] = sample
            else:
                sample['phase_sequence'] = getattr(self, '_phase_sequence', 0)
                self.records.append(sample)
                if len(self.records) > self.max_records:
                    del self.records[:len(self.records)-self.max_records]
            m = self.policy.memory_policy
            violation = None
            if sample['host']['current_bytes'] > m['host_limit_bytes']:
                violation = 'host working set exceeds host_limit_bytes'
            available = sample['host'].get('available_physical_bytes')
            if isinstance(available, int) and available < m['reserve_bytes']:
                violation = 'available host physical memory below reserve'
            for name, record in sample['cuda'].items():
                cap = m['cuda_limit_bytes'].get(name, record['total_bytes'])
                if record['reserved_bytes'] + m['reserve_bytes'] > cap:
                    violation = f'{name} reserved bytes plus reserve exceed CUDA limit'
                if record['free_bytes'] < m['reserve_bytes']:
                    violation = f'{name} device free memory below reserve'
            if violation:
                self._failure = ResourceLimitError(dict(code='RESOURCE_LIMIT', reason=violation, measurement=sample))
                raise self._failure
            return sample

    @contextmanager
    def phase(self, name):
        with self._lock:
            if self._active:
                raise ValueError('resource phases cannot overlap on one monitor')
            self._active = True
            self._phase = name
            self._phase_sequence = getattr(self, '_phase_sequence', 0) + 1
            self._phase_started = time.monotonic()
            self._peak_host = 0
            self._peak_cuda = {}
        stop = threading.Event()
        def poll():
            while not stop.wait(.025):
                try:
                    self.check()
                except Exception as error:
                    with self._lock:
                        self._failure = error
                    return
        thread = None
        fractions = {}
        try:
            # Enforce explicit CUDA caps at the allocator, before graph work.
            # Driver/context allocations and other processes remain separately
            # measured by cudaMemGetInfo; they are not allocator-attributed.
            import torch
            for device in self.policy.devices:
                if device.type == 'cuda':
                    _, total = torch.cuda.mem_get_info(device)
                    cap = self.policy.memory_policy['cuda_limit_bytes'].get(str(device), total)
                    usable = cap - self.policy.memory_policy['reserve_bytes']
                    if usable <= 0:
                        raise ResourceLimitError(dict(code='INSUFFICIENT_CAPACITY', device=str(device), cap_bytes=cap))
                    fractions[device] = torch.cuda.get_per_process_memory_fraction(device)
                    torch.cuda.set_per_process_memory_fraction(min(fractions[device], usable / total), device)
                    torch.cuda.reset_peak_memory_stats(device)
            start = self.check()
            with self._lock:
                self.records.append(dict(start))
                if len(self.records) > self.max_records:
                    del self.records[:len(self.records)-self.max_records]
            thread = threading.Thread(target=poll, name='sinq-resource-monitor', daemon=True)
            thread.start()
            yield self
            self.check()
        finally:
            stop.set()
            if thread is not None:
                thread.join()
            for device, fraction in fractions.items():
                torch.cuda.set_per_process_memory_fraction(fraction, device)
            with self._lock:
                self._active = False
                self._phase = None
                failure = self._failure
            # A poll can fail between the exit check and join. Never publish
            # observations from that final phase after losing its diagnostic.
            if failure is not None:
                raise failure
