"""Guarded preparation preserves canonical output and rejects lifetime escapes."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'gguf-py')]
import gguf
from tools.sinq.optimizer import assembly, baseline
from tools.sinq.optimizer.resources import digest, file_digest

@unittest.skipUnless(os.name=='nt','mandatory immutable fast path is Windows-only')
class PreparedAssemblyTests(unittest.TestCase):
    def fixture(self,d):
        source=d/'source.gguf';stock=d/'stock.gguf'
        for p in (source,stock):
            w=gguf.GGUFWriter(p,'qwen35');w.add_tensor('weight',np.ones((2,256),dtype='<f4'))
            w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()
        recipe={'weight':'F32'}
        records,metadata=baseline._inspect(source,stock,recipe)
        m=dict(version=2,kind='sinq-stock-baseline',verification='native-reencode-exact',
            source_sha256=file_digest(source),baseline_sha256=file_digest(stock),baseline_path='stock.gguf',
            recipe=recipe,recipe_sha256=digest(recipe),imatrix_sha256=None,quantizer_sha256='a'*64,
            metadata_sha256=metadata,tensors=records)
        m['checksum']=digest(m);manifest=d/'baseline.json';manifest.write_text(json.dumps(m))
        return source,stock,manifest

    def test_one_inventory_scan_and_identical_published_bytes(self):
        from tools.sinq.optimizer.prepared import PreparedAssembly
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);source,stock,manifest=self.fixture(d)
            oracle=d/'oracle.gguf';expected=assembly.assemble(source,oracle,None,{},stock_baseline=manifest,native_validate=lambda _:None)
            with patch.object(baseline,'_inspect',wraps=baseline._inspect) as inspect:
                with PreparedAssembly(source,manifest) as prepared:
                    for i in range(3):
                        out=d/f'prepared-{i}.gguf'
                        actual=assembly.assemble(source,out,None,{},stock_baseline=manifest,prepared=prepared,native_validate=lambda _:None)
                        self.assertEqual(actual,expected);self.assertEqual(out.read_bytes(),oracle.read_bytes())
                    self.assertEqual(inspect.call_count,1)
                with self.assertRaisesRegex(RuntimeError,'active'):
                    assembly.assemble(source,d/'expired.gguf',None,{},stock_baseline=manifest,prepared=prepared,native_validate=lambda _:None)

    def test_normal_assemble_config_reuses_guarded_inventory(self):
        from tools.sinq.optimizer import stages,prepared as module
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);source,stock,manifest=self.fixture(d)
            lock=json.loads(manifest.read_text())
            run={'schema_version':2,'source_gguf':source.name,'source_sha256':lock['source_sha256'],
                'cache':'cache','recipe_sha256':lock['recipe_sha256'],'imatrix_sha256':None,'tensors':{},
                'stock_baseline':{'manifest':manifest.name,'manifest_sha256':file_digest(manifest),
                    'baseline_sha256':lock['baseline_sha256'],'recipe_sha256':lock['recipe_sha256']}}
            run['checksum']=digest(run);run_path=d/'run-manifest.json';run_path.write_text(json.dumps(run))
            config={'operation':'assemble','source_gguf':str(source),'candidate_manifest':str(run_path),
                'stock_baseline_manifest':str(manifest),'assignment':{},'native_loader':'mock-loader',
                'output_gguf':str(d/'configured.gguf'),'threads':1,
                'budgets':{'max_output_bytes':1<<20,'max_files':100,'max_tensors':1,'max_candidates':1}}
            oracle=d/'oracle-configured.gguf'
            expected=assembly.assemble(source,oracle,None,{},stock_baseline=manifest,native_validate=lambda _:None)
            with patch.object(module,'file_digest',wraps=module.file_digest) as fresh,patch.object(baseline,'_inspect',wraps=baseline._inspect) as inspect,patch.object(stages,'_validator',return_value=lambda _:None):
                result=stages.assemble_config(config)
                self.assertEqual(inspect.call_count,1)
                self.assertEqual(sum(Path(call.args[0])==source for call in fresh.call_args_list),1)
            self.assertEqual(result['output_sha256'],expected)
            self.assertEqual(Path(config['output_gguf']).read_bytes(),oracle.read_bytes())

    def test_parent_session_hashes_source_once_and_outlives_preparation(self):
        from tools.sinq.optimizer import prepared as module
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);source,stock,manifest=self.fixture(d)
            with patch.object(module,'file_digest',wraps=module.file_digest) as fresh:
                with module.VerifiedInputs() as inputs:
                    expected=inputs.fingerprint(source)
                    with module.PreparedAssembly(source,manifest,verified_inputs=inputs) as prepared:
                        self.assertEqual(prepared.inputs.fingerprint(source),expected)
                        prepared.check()
                    self.assertEqual(inputs.fingerprint(source),expected)
                    self.assertEqual(sum(Path(call.args[0])==source for call in fresh.call_args_list),1)
                    with self.assertRaises(OSError):
                        with source.open('r+b') as stream:stream.write(b'X')

    def test_hardlinked_historical_baseline_uses_verified_private_copy(self):
        from tools.sinq.optimizer.prepared import PreparedAssembly
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);source,stock,manifest=self.fixture(d)
            os.link(stock,d/'publication-alias.gguf')
            original=file_digest(stock)
            with PreparedAssembly(source,manifest) as prepared:
                self.assertEqual(prepared.snapshot_origin,str(stock.resolve()))
                self.assertNotEqual(prepared.baseline_path,stock)
                self.assertEqual(prepared.baseline_path.stat().st_nlink,1)
                self.assertEqual(file_digest(prepared.baseline_path),original)
                with self.assertRaises(OSError):
                    with prepared.baseline_path.open('r+b') as stream:stream.write(b'X')
                out=d/'snapshot-result.gguf'
                assembly.assemble(source,out,None,{},stock_baseline=manifest,prepared=prepared,native_validate=lambda _:None)
            self.assertEqual(file_digest(stock),original)

    def test_guarded_mutation_and_replacement_denied(self):
        from tools.sinq.optimizer.prepared import PreparedAssembly
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);source,stock,manifest=self.fixture(d)
            with PreparedAssembly(source,manifest) as prepared:
                for path in (source,stock,manifest):
                    with self.assertRaises(OSError):
                        with path.open('r+b') as stream:stream.write(b'X')
                    with self.assertRaises(OSError):path.rename(d/'renamed')
                prepared.check()
            with source.open('r+b') as stream:stream.write(b'X')
            with self.assertRaises(ValueError):
                with PreparedAssembly(source,manifest):pass

if __name__=='__main__':unittest.main()
