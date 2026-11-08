"""Current optimizer contract tests, independent of model/checkpoint dependencies."""
import copy
import io
from contextlib import redirect_stdout
from unittest.mock import patch
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.sinq.optimizer.schemas import validate_config, load_config


class ConfigTests(unittest.TestCase):
    def config(self):
        return dict(schema_version=2,
            inputs=dict(source_gguf='source.gguf',native_bridge='bridge.exe'),
            quantization=dict(recipe={'blk.0.ffn_up.weight':'Q4_K'}),
            targets=dict(include=['blk.0.ffn_up.weight']),
            candidate_generators=[dict(type='sinq',expansion='cartesian',parameters=dict(
                strip_width=[32,64,128],iterations=[8,16],strength=[0.5,1]))],
            limits=dict(max_candidates_per_tensor=13,workspace_bytes=1<<26,max_artifact_bytes=1<<30,max_files=1000),
            output=dict(directory='output'))

    def test_explicit_cartesian_and_complete_identity(self):
        c=validate_config(self.config(),'generate')
        self.assertEqual(len(c['resolved_candidates']),12)
        self.assertEqual(len({json.dumps(x,sort_keys=True) for x in c['resolved_candidates']}),12)
        self.assertNotIn('activation_manifest',c)
        self.assertEqual({x['generator'] for x in c['resolved_candidates']},{'sinq'})

    def test_small_strip_widths(self):
        for width in (8,16):
            with self.subTest(width=width):
                c=self.config()
                c['candidate_generators'][0]['parameters']['strip_width']=[width]
                result=validate_config(c,'generate')
                self.assertEqual({x['strip_width'] for x in result['resolved_candidates']},{width})

    def test_native_workspace_limit_is_checked_before_execution(self):
        c=self.config();c['limits']['workspace_bytes']=1<<30
        validate_config(c,'generate')
        c['limits']['workspace_bytes']=(1<<30)+1
        with self.assertRaisesRegex(ValueError,'native.*workspace'):validate_config(c,'generate')
        score=dict(schema_version=2,inputs=dict(candidate_manifest='run.json',capture_manifest='capture.json',native_bridge='bridge.exe'),
            scoring=dict(kind='local_output_l1'),limits=dict(workspace_bytes=2<<30,max_artifact_bytes=1<<30,max_files=100,max_local_scores=4),output=dict(directory='scores'))
        with self.assertRaisesRegex(ValueError,'native.*workspace'):validate_config(score,'score')
        score['scoring']['kind']='blockwise_output_mse';score['inputs']['checkpoint']='checkpoint'
        del score['limits']['max_local_scores'];score['limits']['max_block_scores']=4
        validate_config(score,'score') # reference backends do not use the native workspace cap

    def test_product_must_be_explicit_and_budgeted(self):
        c=self.config();del c['candidate_generators'][0]['expansion']
        with self.assertRaises(ValueError):validate_config(c,'generate')
        c=self.config();c['limits']['max_candidates_per_tensor']=12
        with self.assertRaises(ValueError):validate_config(c,'generate') # stock also counts

    def test_stage_specific_limits_and_no_old_dispatch(self):
        c=self.config();c['operation']='generate'
        with self.assertRaises(ValueError):validate_config(c,'generate')
        c=self.config();c['limits']['max_fisher_vjps']=1
        with self.assertRaises(ValueError):validate_config(c,'generate')
        c=self.config();c['schema_version']=1
        with self.assertRaisesRegex(ValueError,'version'):validate_config(c,'generate')
        c=self.config();c['execution']={'scoring':{'device':'cuda'}}
        with self.assertRaisesRegex(ValueError,'only to score'):validate_config(c,'generate')

    def test_invalid_values_fail(self):
        for key,value in [('strip_width',[48]),('iterations',[12]),('strength',[float('nan')]),('bogus',[1])]:
            c=self.config();c['candidate_generators'][0]['parameters'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_config(c,'generate')
        c=self.config();c['candidate_generators'][0]['type']='plugin'
        with self.assertRaises(ValueError):validate_config(c,'generate')

    def test_cli_emits_stage_resource_and_identity_summary(self):
        from tools.sinq.optimizer import cli
        from tools.sinq.optimizer.resources import file_digest
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);source=d/'source';source.write_bytes(b'original');bridge=d/'bridge';bridge.write_bytes(b'native')
            c=self.config();c['inputs']={'source_gguf':str(source),'native_bridge':str(bridge)}
            path=d/'config.json';path.write_text(json.dumps(c));stdout=io.StringIO()
            with patch('tools.sinq.optimizer.generate.preflight',return_value={'status':'PLAN'}),redirect_stdout(stdout):
                self.assertEqual(cli.main(['plan','--config',str(path)]),0)
            summary=json.loads(stdout.getvalue())['stage_summary']
            self.assertEqual(summary['stage'],'plan')
            self.assertEqual(summary['input_hashes']['source_gguf'],file_digest(source))
            self.assertGreater(summary['process_peak_rss_bytes'],0)
            self.assertGreaterEqual(summary['elapsed_seconds'],0)

    def test_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.json';p.write_text('{"schema_version":2,"schema_version":2}')
            with self.assertRaisesRegex(ValueError,'duplicate'):load_config(p,'generate')

    def test_awq_requires_bound_captures(self):
        c=self.config();c['candidate_generators']=[dict(type='awq',expansion='cartesian',parameters=dict(
            strip_width=[64],iterations=[16],strength=[1],correction=['weight_activation'],alpha=[0.5],**{'lambda':[1]}))]
        with self.assertRaisesRegex(ValueError,'activation'):validate_config(c,'generate')

if __name__=='__main__':unittest.main()
