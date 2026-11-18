"""Session-owned authenticated inputs and canonical assembly inventories.

Reuse is valid only during mandatory Win32 deny-write/delete guard lifetimes.
There is deliberately no persisted stat-only digest cache or Linux flock claim.
"""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import gguf
from .resources import file_digest, observe_phase
from .schemas import read_json
from .trusted_reference import GuardedFile


class VerifiedInputs:
    def __init__(self, parent=None):
        self.parent=parent
        self.stack = None
        self.guards = {}
        self.digests = {}

    def __enter__(self):
        if self.stack is not None:raise RuntimeError('verified input session already active')
        self.stack = ExitStack()
        return self

    def fingerprint(self, path, expected=None):
        if self.stack is None:raise RuntimeError('verified input session not active')
        path = Path(path).absolute()
        if self.parent is not None and path in self.parent.guards:
            return self.parent.fingerprint(path,expected)
        if path not in self.guards:
            guard = self.stack.enter_context(GuardedFile(path))
            self.guards[path] = guard
            with observe_phase('verified_input_fresh_hash'):
                self.digests[path] = file_digest(path)
        self.guards[path].check()
        value = self.digests[path]
        if expected is not None and value != expected:raise ValueError('verified input SHA256 mismatch')
        return value

    def check(self):
        if self.stack is None:raise RuntimeError('verified input session not active')
        if self.parent is not None:self.parent.check()
        for guard in self.guards.values():guard.check()

    def __exit__(self,*exc):
        self.stack.__exit__(*exc)
        self.stack = None
        self.guards.clear();self.digests.clear()


class PreparedAssembly:
    def __init__(self, source, stock_baseline, *, threads=1, verified_inputs=None):
        self.source = Path(source).resolve()
        self.stock_baseline = Path(stock_baseline).resolve()
        self.threads = threads
        self.parent_inputs=verified_inputs
        self.stack = None

    def __enter__(self):
        from .baseline import load_baseline
        if self.stack is not None:raise RuntimeError('prepared assembly already active')
        self.stack = ExitStack()
        try:
            # Temporary storage must outlive maps and deny-delete input handles.
            storage=self.stack.enter_context(ExitStack())
            self.inputs = self.stack.enter_context(VerifiedInputs(self.parent_inputs))
            self.inputs.fingerprint(self.stock_baseline)
            manifest = read_json(self.stock_baseline,max_bytes=64<<20)
            self.baseline_path = (self.stock_baseline.parent/manifest['baseline_path']).resolve()
            self.snapshot_origin=None
            if self.baseline_path.stat().st_nlink>1:
                # Historical native baselines may have publication hard links.
                # Never relax alias rejection or mutate that shared object: make
                # a private byte copy, then freshly hash it under a mandatory
                # guard against the original immutable manifest's content claim.
                self.snapshot_origin=str(self.baseline_path)
                directory=storage.enter_context(tempfile.TemporaryDirectory(prefix='sinq-verified-stock-'))
                snapshot=Path(directory)/'stock.gguf'
                with observe_phase('prepared_private_stock_copy') as measured:
                    with self.baseline_path.open('rb') as src,snapshot.open('xb') as dst:
                        while block:=src.read(1<<20):
                            dst.write(block)
                            if measured is not None:
                                measured['bytes_read']=measured.get('bytes_read',0)+len(block)
                                measured['bytes_written']=measured.get('bytes_written',0)+len(block)
                self.baseline_path=snapshot
            # Guards precede every inventory scan; the normal manifest validator
            # still checks all provenance, recipe, metadata and tensor payloads.
            self.inputs.fingerprint(self.source, manifest['source_sha256'])
            self.inputs.fingerprint(self.baseline_path, manifest['baseline_sha256'])
            self.reader = gguf.GGUFReader(self.source)
            self.stack.callback(self.reader.data._mmap.close)
            self.stock_reader = gguf.GGUFReader(self.baseline_path)
            self.stack.callback(self.stock_reader.data._mmap.close)
            self.manifest,_ = load_baseline(self.stock_baseline,self.source,threads=self.threads,
                verified_inputs=self.inputs,baseline_override=self.baseline_path,readers=(self.reader,self.stock_reader))
            self.check()
            return self
        except BaseException:
            self.stack.close();self.stack=None
            raise

    def check(self, source=None, stock_baseline=None):
        if self.stack is None:raise RuntimeError('prepared assembly not active')
        if source is not None and Path(source).resolve()!=self.source:raise ValueError('prepared source mismatch')
        if stock_baseline is not None and Path(stock_baseline).resolve()!=self.stock_baseline:raise ValueError('prepared baseline mismatch')
        self.inputs.check()

    def __exit__(self,*exc):
        self.stack.__exit__(*exc)
        self.stack = None
