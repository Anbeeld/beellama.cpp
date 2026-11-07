"""Explicit replacement stock provenance, without relaxing original-control checks."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from tools.sinq.optimizer.resources import digest,file_digest

@unittest.skipUnless(os.name=='nt','mandatory Windows file guard')
class StockBaselineControl(unittest.TestCase):
    def setUp(self):
        spec=importlib.util.spec_from_file_location('control_validation',REPO/'tests/sinq-reference/experiment/final_control_validation.py')
        self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'baseline.json'
        self.baseline={'source_sha256':'a'*64,'baseline_sha256':'b'*64,'baseline_path':'stock.gguf'}
        self.write()
        self.model={'kind':'stock-baseline','stock_baseline_manifest_path':str(self.path),'stock_baseline_manifest_sha256':file_digest(self.path),'expected_sha256':'b'*64}
    def write(self):
        self.baseline['checksum']=digest({k:v for k,v in self.baseline.items() if k!='checksum'})
        self.path.write_text(json.dumps(self.baseline),encoding='utf-8')
    def test_valid(self):self.module.validate_stock_baseline_control(self.model,'a'*64)
    def test_changed_manifest_rejected(self):
        self.path.write_text(self.path.read_text()+' ')
        with self.assertRaises(ValueError):self.module.validate_stock_baseline_control(self.model,'a'*64)
    def test_bad_checksum_rejected(self):
        self.baseline['checksum']='c'*64;self.path.write_text(json.dumps(self.baseline))
        self.model['stock_baseline_manifest_sha256']=file_digest(self.path)
        with self.assertRaises(ValueError):self.module.validate_stock_baseline_control(self.model,'a'*64)
    def test_wrong_model_rejected(self):
        self.model['expected_sha256']='d'*64
        with self.assertRaises(ValueError):self.module.validate_stock_baseline_control(self.model,'a'*64)
    def test_wrong_source_rejected(self):
        with self.assertRaises(ValueError):self.module.validate_stock_baseline_control(self.model,'e'*64)
    def test_wrong_kind_rejected(self):
        self.model['kind']='stock'
        with self.assertRaises(ValueError):self.module.validate_stock_baseline_control(self.model,'a'*64)
if __name__=='__main__':unittest.main()
