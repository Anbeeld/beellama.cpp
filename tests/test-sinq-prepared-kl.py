"""Prepared native output boundary; real-model numerical parity is separate."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'gguf-py')]
from tools.sinq.optimizer.prepared_kl import token_metrics, PreparedGroupwiseEvaluator, validate_reference_evidence

class PreparedKLTests(unittest.TestCase):
    def output(self,root,rows):
        path=root/'tokens.tsv'
        with path.open('w',newline='') as stream:
            writer=csv.writer(stream,delimiter='\t')
            writer.writerow(['input_index','context_position','target_token','kld','base_nll','candidate_nll','top1_agree','candidate_top1'])
            writer.writerows(rows)
        return path

    def test_aligned_finite_metrics_and_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=self.output(Path(tmp),[[0,0,2,.1,3,4,1,2],[1,1,3,.3,5,6,0,2]])
            score,rows=token_metrics(p,[1,2,3],8)
            self.assertEqual(score,.2);self.assertEqual(len(rows),2)
            self.assertEqual(rows[1]['candidate_top1'],2)

    def test_alignment_count_nonfinite_and_invalid_probability_rejected(self):
        valid=[0,0,2,.1,3,4,1,2]
        with tempfile.TemporaryDirectory() as tmp:
            for column,value in [(0,1),(1,1),(2,3),(3,'nan'),(3,-.1),(4,'inf'),(5,'nan'),(6,.5)]:
                row=list(valid);row[column]=value
                with self.subTest(column=column,value=value),self.assertRaises(ValueError):
                    token_metrics(self.output(Path(tmp),[row]),[1,2],8)
            with self.assertRaises(ValueError):token_metrics(self.output(Path(tmp),[]),[1,2],8)
            with self.assertRaises(ValueError):token_metrics(self.output(Path(tmp),[valid,valid]),[1,2],8)

    def test_reference_origin_and_model_semantics_fail_closed(self):
        import copy
        gpu={'uuid':'GPU-origin','pci_bus_id':'0000:01:00.0'}
        policy={'gpu':gpu};header={'context':4,'vocabulary':8,'tokens':[1,2,3]}
        manifest={'schema_version':2,'policy':policy,'alignment':header,'source_sha256':'source',
            'tokenizer':'graph-and-tokenizer','execution_evidence':{'gpu':gpu,'offloaded_layers':33,'total_layers':33}}
        kwargs={'policy':policy,'header':header,'source_sha256':'source','tokenizer':'graph-and-tokenizer'}
        validate_reference_evidence(manifest,**kwargs)
        for field,value in [('schema_version',1),('source_sha256','wrong-source'),('tokenizer','wrong-tokenizer'),('alignment',{}),('policy',{})]:
            bad=copy.deepcopy(manifest);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_reference_evidence(bad,**kwargs)
        for field,value in [('gpu',{'uuid':'GPU-wrong'}),('offloaded_layers',0),('offloaded_layers',32),('total_layers',32)]:
            bad=copy.deepcopy(manifest);bad['execution_evidence'][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):validate_reference_evidence(bad,**kwargs)

    def test_no_silent_cpu_fallback(self):
        class Oracle:gpu=None
        with self.assertRaisesRegex(ValueError,'CUDA'):
            PreparedGroupwiseEvaluator(Oracle(),'unused','unused','unused',native_loader='unused')

if __name__=='__main__':unittest.main()
