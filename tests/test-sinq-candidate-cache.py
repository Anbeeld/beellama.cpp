import sys, unittest, tempfile, hashlib, os, subprocess
from unittest import mock
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'gguf-py')]
from tools.sinq.optimizer.cache import CandidateStore

def provenance(name='blk.0.ffn_down.weight', width=64):
    h = hashlib.sha256(b'fixture').hexdigest()
    return dict(source_sha256=h, tensor_name=name, shape=[2,256], source_qtype='F16', qtype='Q5_0', input_sha256=h, recipe_sha256=h, imatrix_sha256=None, codec={'implementation':'ggml','chunk_rows':2}, sinq={'strip_width':width,'iterations':16,'strength':0.75,'algorithm':'sinkhorn_log','numerics':'f32-f16'}, generator={'generator':'sinq','strip_width':width,'iterations':16,'strength':0.75}, activation=None, producer={'source_sha256':h,'binary_sha256':h})

class CacheTest(unittest.TestCase):
    def test_roundtrip_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            s=CandidateStore(d)
            # Real Q5_0 layout: 22 bytes per 32 weights, all-zero scales/values.
            p=bytes(2*8*22); scales=np.ones(256,dtype='<f2').tobytes()
            key=s.put(provenance(),p,scales)
            self.assertEqual(s.get(key).payload.read_bytes(),p)
            self.assertEqual(s.put(provenance(),p,scales),key)
            s.get(key).payload.write_bytes(b'bad')
            with self.assertRaises(ValueError): s.get(key)
    def test_small_strip_widths(self):
        with tempfile.TemporaryDirectory() as d:
            store=CandidateStore(d)
            for width in (8,16):
                with self.subTest(width=width):
                    key=store.put(provenance(width=width),bytes(352),np.ones(256,dtype='<f2').tobytes())
                    self.assertEqual(store.get(key).provenance['sinq']['strip_width'],width)

    def test_native_identity_and_file_ingestion(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); s=CandidateStore(d/'cache'); payload=d/'packed'; scales=d/'scales'
            payload.write_bytes(bytes(352)); scales.write_bytes(np.ones(256,dtype='<f2').tobytes())
            key=s.put_files(provenance(),payload,scales)
            self.assertEqual(s.get(key).payload.read_bytes(),bytes(352))
            with self.assertRaisesRegex(ValueError,'provenance'):
                s.put(provenance(),bytes([1])*352,scales.read_bytes())
            with self.assertRaisesRegex(ValueError,'provenance mismatch'):
                s.get(key,expected={'qtype':'Q6_K'})
            with self.assertRaisesRegex(ValueError,'budget'):
                CandidateStore(d/'small',max_bytes=50).put(provenance(),bytes(352),scales.read_bytes())

    def test_public_identity_algorithm_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            s=CandidateStore(d);p=provenance();p['sinq']={**p['sinq'],'algorithm':'identity','strength':0}
            for scales in (np.full(256,2,dtype='<f2'),np.ones(256,dtype='<f2')):
                with self.assertRaises(ValueError):
                    s.put(p,bytes(352),scales.tobytes())

    def test_complete_generator_provenance_is_required(self):
        with tempfile.TemporaryDirectory() as d:
            for generator in ({'family':'ordinary'}, {'generator':'sinq'},
                              {'generator':'stock','strip_width':64}):
                p=provenance();p['generator']=generator
                with self.subTest(generator=generator),self.assertRaises(ValueError):
                    CandidateStore(d).put(p,bytes(352),np.ones(256,dtype='<f2').tobytes())

    def test_validation(self):
        with tempfile.TemporaryDirectory() as d:
            s=CandidateStore(d,max_bytes=1024)
            with self.assertRaises(ValueError): s.put(provenance(),b'bad',b'bad')
            bad=provenance(); bad['source_sha256']='wrong'
            with self.assertRaises(ValueError): s.put(bad,bytes(352),bytes(16))
            with self.assertRaises(ValueError): s.put(provenance(),bytes(352),np.zeros(256,dtype='<f2').tobytes())



class VerifiedSessionTest(unittest.TestCase):
    def put(self, store, name='blk.0.weight'):
        return store.put(provenance(name), bytes(352), np.ones(256,dtype='<f2').tobytes())

    def test_receipt_expected_and_independent_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            store=CandidateStore(d)
            with store.verified_session():
                key=self.put(store)
                first=store.get(key)
                first.provenance['shape'][0]=999
                self.assertEqual(store.get(key).provenance['shape'],[2,256])
                with self.assertRaisesRegex(ValueError,'provenance mismatch'):
                    store.get(key,expected={'qtype':'Q6_K'})
                with store.verified_session():
                    self.assertEqual(self.put(store),key)
            self.assertEqual(store.get(key).provenance['shape'],[2,256])

    def test_mutation_denied_or_detected_even_with_restored_timestamps(self):
        with tempfile.TemporaryDirectory() as d:
            store=CandidateStore(d)
            key=self.put(store)
            with store.verified_session():
                path=store.get(key).payload
                before=path.stat()
                try:
                    path.write_bytes(bytes([1])*352)
                    os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
                except PermissionError:
                    self.assertEqual(os.name,'nt')
                    self.assertEqual(store.get(key).payload.read_bytes(),bytes(352))
                else:
                    with self.assertRaisesRegex(ValueError,'checksum'):
                        store.get(key)
            path.write_bytes(bytes(352))

    def test_external_artifact_counted_and_not_deleted(self):
        with tempfile.TemporaryDirectory() as d:
            store=CandidateStore(d,max_bytes=10000)
            with store.verified_session():
                self.put(store)
                orphan=Path(d)/'orphan'
                orphan.write_bytes(bytes(10000))
                with self.assertRaisesRegex(ValueError,'budget'):
                    self.put(store,'blk.1.weight')
                self.assertTrue(orphan.exists())

    def test_lock_legacy_and_process_death(self):
        with tempfile.TemporaryDirectory() as d:
            legacy=Path(d)/'.write-lock'; legacy.write_bytes(b'')
            with self.assertRaises(RuntimeError):
                with CandidateStore(d).verified_session(): pass
            legacy.unlink()
            store=CandidateStore(d); key=self.put(store)
            orphan=Path(d)/'.pending-interrupted';orphan.write_bytes(b'interrupted')
            code="import sys,time;from pathlib import Path;root=Path(sys.argv[1]);sys.path[:0]=[str(root),str(root/'gguf-py')];from tools.sinq.optimizer.cache import CandidateStore;s=CandidateStore(sys.argv[2]);ctx=s.verified_session();ctx.__enter__();print('ready',flush=True);time.sleep(120)"
            process=subprocess.Popen([sys.executable,'-c',code,str(ROOT),d],stdout=subprocess.PIPE,text=True)
            try:
                self.assertEqual(process.stdout.readline().strip(),'ready')
                with self.assertRaises(RuntimeError):
                    with CandidateStore(d).verified_session(): pass
            finally:
                process.kill();process.wait();process.stdout.close()
            with store.verified_session():
                self.assertEqual(store.get(key).payload.read_bytes(),bytes(352))
                self.assertGreaterEqual(store.index_stats['files'],6)
            self.assertEqual(orphan.read_bytes(),b'interrupted')
            store.get(key).payload.write_bytes(bytes(352))

    def test_non_windows_fallback_and_guard_failure(self):
        with tempfile.TemporaryDirectory() as d:
            store=CandidateStore(d)
            with mock.patch('tools.sinq.optimizer.cache._WindowsIndex',side_effect=OSError('no guards')):
                with store.verified_session():
                    key=self.put(store)
                    before=store.index_stats['hash_calls']
                    store.get(key);store.get(key)
                    self.assertGreater(store.index_stats['hash_calls'],before)
                    self.assertGreater(store.index_stats['fallbacks'],0)
                    self.assertEqual(store.index_stats['receipt_hits'],0)

    def test_non_windows_branch_does_not_reuse_receipts(self):
        # Exercise capability selection on Windows too; real POSIX flock is
        # covered only when this suite runs on a POSIX host.
        import tools.sinq.optimizer.cache as cache
        class NoWindows:
            name='posix'
            def __getattr__(self, name): return getattr(os,name)
        with tempfile.TemporaryDirectory() as d:
            store=CandidateStore(d);key=self.put(store)
            with mock.patch.object(cache,'os',NoWindows()), mock.patch.object(cache,'_WriterLock'):
                with store.verified_session():
                    before=store.index_stats['hash_calls']
                    store.get(key);store.get(key)
                    self.assertEqual(store.index_stats['hash_calls']-before,4)
                    self.assertEqual(store.index_stats['receipt_hits'],0)
                    self.assertIn('platform',store.index_stats['fallback_reason'])

    def test_file_budget_includes_lock_and_unknown_files(self):
        with tempfile.TemporaryDirectory() as d:
            store=CandidateStore(d,max_files=4)
            with store.verified_session():
                with self.assertRaisesRegex(ValueError,'budget'): self.put(store)
            self.assertEqual(list((Path(d)/'tensors').glob('*')),[])
        with tempfile.TemporaryDirectory() as d:
            store=CandidateStore(d,max_files=7)
            with store.verified_session():
                self.put(store)
                (Path(d)/'orphan').write_bytes(b'x')
                with self.assertRaisesRegex(ValueError,'budget'): self.put(store,'blk.1.weight')
                self.assertTrue((Path(d)/'orphan').exists())

    def test_failed_publish_orphan_is_counted_on_retry(self):
        with tempfile.TemporaryDirectory() as d:
            store=CandidateStore(d,max_files=5)
            with store.verified_session():
                with mock.patch.object(store,'_hash',side_effect=OSError('injected readback failure')):
                    with self.assertRaisesRegex(OSError,'injected'): self.put(store)
                self.assertEqual(len(list((Path(d)/'blobs').iterdir())),1)
                with self.assertRaisesRegex(ValueError,'budget'): self.put(store)
                self.assertEqual(store.index_stats['files'],2)
                self.assertEqual(len(list((Path(d)/'blobs').iterdir())),1)

    def test_exception_exit_and_reentry(self):
        with tempfile.TemporaryDirectory() as d:
            store=CandidateStore(d)
            with self.assertRaisesRegex(RuntimeError,'test abort'):
                with store.verified_session():
                    key=self.put(store)
                    raise RuntimeError('test abort')
            self.assertEqual(store.index_stats['guarded_files'],0)
            with store.verified_session():
                self.assertEqual(store.get(key).provenance['shape'],[2,256])
                with self.assertRaises(RuntimeError): self.put(CandidateStore(d),'blk.2.weight')

    @unittest.skipUnless(os.name=='nt','mandatory Win32 guards required')
    def test_existing_write_handle_forces_full_validation(self):
        with tempfile.TemporaryDirectory() as d:
            store=CandidateStore(d);key=self.put(store)
            path=store.get(key).payload
            with path.open('r+b') as writer:
                with store.verified_session():
                    self.assertGreater(store.index_stats['fallbacks'],0)
                    store.get(key)
                    before=path.stat()
                    writer.write(bytes([1])*352);writer.flush()
                    os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
                    with self.assertRaisesRegex(ValueError,'checksum'): store.get(key)

    @unittest.skipUnless(os.name=='nt','mandatory Win32 guards required')
    def test_windows_receipts_census_and_close(self):
        with tempfile.TemporaryDirectory() as d:
            store=CandidateStore(d)
            with store.verified_session():
                key=self.put(store)
                scans=store.index_stats['census_scans']; hashes=store.index_stats['hash_calls']
                for i in range(10): store.get(key)
                self.assertEqual(store.index_stats['hash_calls'],hashes)
                self.assertEqual(store.index_stats['census_scans'],scans)
                self.assertGreaterEqual(store.index_stats['receipt_hits'],10)
                orphan=Path(d)/'orphan'; orphan.write_bytes(b'x')
                store.get(key)
                self.assertGreater(store.index_stats['census_scans'],scans)
                with self.assertRaises(PermissionError): orphan.write_bytes(b'y')
                payload=store.get(key).payload
            payload.write_bytes(bytes(352)); orphan.unlink()
            self.assertEqual(store.index_stats['guarded_files'],0)

if __name__=='__main__': unittest.main()
