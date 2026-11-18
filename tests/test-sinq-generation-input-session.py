"""Bounded generation input authentication and imatrix-reader lifetimes."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'gguf-py')]
import gguf
import numpy as np
from tools.sinq.optimizer import generate as generation,prepared,generation_state


def write_imatrix(path,names):
    writer=gguf.GGUFWriter(path,'qwen35')
    for name in names:
        writer.add_tensor(name+'.in_sum2',np.full(256,4,dtype='<f4'))
        writer.add_tensor(name+'.counts',np.array([2],dtype='<f4'))
    writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()


class ImportanceReaderTests(unittest.TestCase):
    def test_guarded_runtime_identity_preserves_original_dictionary(self):
        from tools.sinq.optimizer.implementation import native_runtime_fingerprint
        from tools.sinq.optimizer.resources import file_digest
        class Inputs:
            fingerprint=staticmethod(file_digest)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);binary=root/'bin'/'bridge.exe';binary.parent.mkdir()
            binary.write_bytes(b'bounded executable fixture')
            (binary.parent/'ggml.dll').write_bytes(b'bounded runtime fixture')
            (root/'lib').mkdir();(root/'lib'/'libllama.so.1').write_bytes(b'bounded shared library')
            (root/'lib'/'unrelated.so').write_bytes(b'excluded library')
            self.assertEqual(generation._runtime_fingerprint(binary,Inputs()),native_runtime_fingerprint({'bridge':binary}))

    def test_indexed_queries_share_reader_and_preserve_legacy_api(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'imatrix.gguf';names=['first','second'];write_imatrix(path,names)
            reader=gguf.GGUFReader(path)
            try:
                tensors={t.name:t for t in reader.tensors}
                with patch.object(gguf,'GGUFReader',side_effect=AssertionError('indexed query reopened imatrix')):
                    for name in names:
                        np.testing.assert_array_equal(generation._importance(path,name,256,reader=reader,tensors=tensors),np.full(256,2,dtype='<f4'))
                with self.assertRaisesRegex(ValueError,'lacks selected tensor'):
                    generation._importance(path,'missing',256,reader=reader,tensors=tensors)
                np.testing.assert_array_equal(generation._importance(path,names[0],256),np.full(256,2,dtype='<f4'))
                self.assertFalse(reader.data._mmap.closed)
            finally:reader.data._mmap.close()


@unittest.skipUnless(os.environ.get('SINQ_NATIVE_BIN'),'local native binaries not configured')
class GenerationSessionTests(unittest.TestCase):
    def setUp(self):
        spec=importlib.util.spec_from_file_location('generation_resume_fixture',ROOT/'tests/test-sinq-generation-resume.py')
        fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
        fixture.NativeGenerationResumeTests.setUp(self)
        self.config.pop('activation_manifest')
        self.config['resolved_candidates']=[s for s in self.config['resolved_candidates'] if s['generator']=='sinq']
        self.imatrix=self.d/'imatrix.gguf';write_imatrix(self.imatrix,self.names)
        self.config['native_quantizer']['imatrix_path']=str(self.imatrix)

    def test_source_hashes_bounded_per_invocation_and_one_imatrix_reader(self):
        source=Path(self.config['source_gguf']).resolve();real_hash=prepared.file_digest;real_reader=gguf.GGUFReader
        hashes=[];readers=[]
        def hashed(path):
            if Path(path).resolve()==source:hashes.append(source)
            return real_hash(path)
        def opened(path,*args,**kwargs):
            if Path(path).resolve()==self.imatrix.resolve():readers.append(path)
            return real_reader(path,*args,**kwargs)
        for resume in (None,self.output):
            hashes.clear();readers.clear()
            with patch.object(prepared,'file_digest',side_effect=hashed),patch.object(gguf,'GGUFReader',side_effect=opened):
                run=generation.generate(self.config,resume=resume)
            self.assertEqual(len(run['tensors']),2)
            self.assertEqual(len(readers),1)
            if os.name=='nt':self.assertEqual(len(hashes),1)
            else:self.assertEqual(len(hashes),0)  # Linux retains original full-file checks.

    @unittest.skipUnless(os.name=='nt','mandatory Windows guard integration')
    def test_source_write_denied_during_tensor_generation(self):
        real=generation.native_codec.request;attempts=[]
        def request(binary,value,directory):
            if value['operation']=='generate':
                with self.assertRaises(OSError):
                    with Path(self.config['source_gguf']).open('r+b') as stream:stream.write(b'bad')
                attempts.append(True)
            return real(binary,value,directory)
        with patch.object(generation.native_codec,'request',side_effect=request):generation.generate(self.config)
        self.assertEqual(len(attempts),2)

    @unittest.skipUnless(os.name=='nt','Windows unsupported-guard fallback')
    def test_unsupported_guard_uses_full_hashes_and_detects_mutation(self):
        real=generation.file_digest;hashes=[]
        def hashed(path):
            if Path(path).resolve()==Path(self.config['source_gguf']).resolve():hashes.append(path)
            return real(path)
        with patch.object(prepared,'GuardedFile',side_effect=OSError('unsupported volume')),patch.object(generation,'file_digest',side_effect=hashed):
            generation.generate(self.config)
        self.assertGreaterEqual(len(hashes),3)
        # No cached digest from the successful invocation survives into resume.
        source=Path(self.config['source_gguf'])
        with source.open('r+b') as stream:
            stream.seek(-1,2);last=stream.read(1);stream.seek(-1,2);stream.write(bytes([last[0]^1]))
        with patch.object(prepared,'GuardedFile',side_effect=OSError('unsupported volume')):
            with self.assertRaisesRegex(ValueError,'identity'):generation.generate(self.config,resume=self.output)

    def test_active_session_integrity_failure_is_not_a_fallback(self):
        with patch.object(generation_state,'_run_generation',side_effect=RuntimeError('guarded file identity changed')):
            with self.assertRaisesRegex(RuntimeError,'identity changed'):generation.generate(self.config)
        self.assertFalse(self.output.exists())


if __name__=='__main__':unittest.main()
