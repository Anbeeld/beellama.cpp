"""Strict grouped optimizer v2 and native-v1 activation pairing contracts."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.sinq.optimizer.schemas import load_config, validate_config, validate_pairing


def config():
    return dict(schema_version=2,
        inputs=dict(source_gguf='source.gguf',native_bridge='bridge',activation_manifest='train.json'),
        quantization=dict(recipe={'blk.0.attn_q.weight':'Q5_0'}),
        targets=dict(include=['blk.0.attn_q.weight']),
        candidate_generators=[dict(type='sinq',expansion='cartesian',parameters=dict(
            strip_width=[64,128],iterations=[4],strength=[.75])),
            dict(type='awq',expansion='cartesian',parameters=dict(strip_width=[64],iterations=[4],
                strength=[.75],correction=['weight_activation','activation_only'],alpha=[0,.5],**{'lambda':[0,1]}))],
        limits=dict(workspace_bytes=1<<20,max_artifact_bytes=1<<20,max_files=64,
            max_tensors=2,max_candidates_per_tensor=64),output=dict(directory='output'))


class ConfigTests(unittest.TestCase):
    def test_canonical_candidates_and_input_immutability(self):
        c=config();before=copy.deepcopy(c);r=validate_config(c,'generate')
        self.assertEqual(c,before)
        self.assertEqual(r['threads'],os.cpu_count() or 1)
        self.assertEqual(len(r['resolved_candidates']),10)
        self.assertEqual({p['generator'] for p in r['resolved_candidates']},{'sinq','awq'})
        self.assertEqual(len({json.dumps(p,sort_keys=True) for p in r['resolved_candidates']}),10)
        c['candidate_generators'].reverse()
        for g in c['candidate_generators']:
            for values in g['parameters'].values():values.reverse()
        self.assertEqual(r['resolved_candidates'],validate_config(c,'generate')['resolved_candidates'])

    def test_unknown_fields_at_every_level(self):
        for group in (None,'inputs','quantization','targets','limits','output'):
            c=config();target=c if group is None else c[group];target['typo']=1
            with self.subTest(group=group),self.assertRaises(ValueError):validate_config(c,'generate')
        for key in ('operation','search','generators','alphas','budgets','source_gguf','tensor_list'):
            c=config();c[key]={}
            with self.subTest(key=key),self.assertRaises(ValueError):validate_config(c,'generate')

    def test_nonfinite_types_and_ranges(self):
        for value in (float('nan'),float('inf'),-.1,1.1,True,'0.5'):
            c=config();c['candidate_generators'][1]['parameters']['alpha']=[value]
            with self.subTest(value=value),self.assertRaises(ValueError):validate_config(c,'generate')
        for value in (0,-1,True,1.5):
            c=config();c['limits']['max_files']=value
            with self.subTest(value=value),self.assertRaises(ValueError):validate_config(c,'generate')

    def test_expansion_budget_includes_stock_and_deduplicates(self):
        c=config();c['limits']['max_candidates_per_tensor']=10
        with self.assertRaises(ValueError):validate_config(c,'generate')
        c['limits']['max_candidates_per_tensor']=11
        self.assertEqual(len(validate_config(c,'generate')['resolved_candidates']),10)
        c['candidate_generators'].append(copy.deepcopy(c['candidate_generators'][0]))
        self.assertEqual(len(validate_config(c,'generate')['resolved_candidates']),10)

    def test_no_legacy_search_or_public_identity(self):
        for g in ({'type':'ordinary'},{'type':'sinq','expansion':'legacy','parameters':{}},
                  {'type':'sinq','configurations':[{'strip_width':64,'iterations':4,'strength':.75,'algorithm':'identity'}]}):
            c=config();c['candidate_generators']=[g]
            with self.subTest(generator=g),self.assertRaises(ValueError):validate_config(c,'generate')
        c=config();c['schema_version']=1
        with self.assertRaises(ValueError):validate_config(c,'generate')

    def test_awq_requires_capture_and_sinq_rejects_awq_parameters(self):
        c=config();del c['inputs']['activation_manifest']
        with self.assertRaises(ValueError):validate_config(c,'generate')
        c=config();c['candidate_generators']=[dict(type='sinq',configurations=[dict(
            strip_width=64,iterations=4,strength=.75,alpha=0)])]
        with self.assertRaises(ValueError):validate_config(c,'generate')

    def test_load_duplicate_keys_and_no_artifact_reads(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'config.json';c=config();c['output']['directory']=str(Path(d)/'absent')
            path.write_text(json.dumps(c),encoding='utf-8')
            self.assertEqual(load_config(path,'generate')['operation'],'generate')
            self.assertFalse(Path(c['output']['directory']).exists())
            path.write_text('{"schema_version":2,"schema_version":2}',encoding='utf-8')
            with self.assertRaises(ValueError):load_config(path,'generate')

    def test_stage_specific_configs_and_limits(self):
        limits=dict(workspace_bytes=1<<20,max_artifact_bytes=1<<20,max_files=64)
        cases=[('score',dict(inputs=dict(candidate_manifest='c',capture_manifest='x',native_bridge='b'),
                   scoring=dict(kind='local_output_l1'),limits={**limits,'max_local_scores':4},output=dict(directory='scores'))),
               ('select',dict(inputs=dict(source_gguf='s',candidate_manifest='c',native_loader='l'),
                   selection=dict(policy='manual',baseline={'a':'id'},groups={'g':['a']}),output=dict(directory='selected'))),
               ('assemble',dict(inputs=dict(source_gguf='s',candidate_manifest='c',native_loader='l'),
                   selection=dict(assignment={'a':'id'}),output=dict(gguf='assembled.gguf'))),
               ('validate',dict(inputs=dict(artifact_manifest='c')))]
        for stage,fields in cases:
            c=dict(schema_version=2,limits=limits);c.update(fields)
            with self.subTest(stage=stage):
                self.assertEqual(validate_config(c,stage)['operation'],stage)
                bad=copy.deepcopy(c);bad['limits']['max_candidates']=4
                with self.assertRaises(ValueError):validate_config(bad,stage)
        c=dict(schema_version=2,inputs=dict(candidate_manifest='c',capture_manifest='x',native_bridge='b'),
               scoring=dict(kind='local_output_l1'),limits=limits,output=dict(directory='scores'))
        with self.assertRaisesRegex(ValueError,'max_local_scores'):validate_config(c,'score')

    def test_plan_uses_same_generation_contract(self):
        self.assertEqual(validate_config(config(),'plan'),validate_config(config(),'generate'))
        with self.assertRaises(ValueError):validate_config(config(),'lock_baseline')


class PairingTests(unittest.TestCase):
    def fixture(self):
        train=dict(schema_version=1,architecture='qwen3_5',source_sha256='a'*64,
            source_revision='locked',sampling=dict(token_sha256='b'*64,corpus_sha256='c'*64))
        val=copy.deepcopy(train);val['sampling']['token_sha256']='d'*64
        def split(name,h,start):
            return dict(split_id=name,token_sha256=h,chunks=[dict(token_source_sha256='c'*64,
                start=start,end=start+8)],row_ids=[name+':0'])
        return dict(schema_version=2,source_sha256='a'*64,split_method='split_before_reservoir',
            train=split('train','b'*64,0),validation=split('validation','d'*64,8)),train,val

    def test_disjoint_and_native_activation_version(self):
        p,t,v=self.fixture();self.assertEqual(validate_pairing(p,t,v),p)
        self.assertEqual(t['schema_version'],1)
        p['schema_version']=1
        with self.assertRaises(ValueError):validate_pairing(p,t,v)

    def test_overlap_source_rows_and_split(self):
        for change in ('chunks','row_ids','split_id'):
            p,t,v=self.fixture();p['validation'][change]=copy.deepcopy(p['train'][change])
            with self.subTest(change=change),self.assertRaises(ValueError):validate_pairing(p,t,v)
        p,t,v=self.fixture();v['source_sha256']='e'*64
        with self.assertRaises(ValueError):validate_pairing(p,t,v)

    def test_chunk_provenance_and_partial_overlap(self):
        for field,value in [('start',7),('token_source_sha256','e'*64)]:
            p,t,v=self.fixture();p['validation']['chunks'][0][field]=value
            with self.assertRaises(ValueError):validate_pairing(p,t,v)
        p,t,v=self.fixture();p['validation']['token_sha256']='e'*64
        with self.assertRaises(ValueError):validate_pairing(p,t,v)

    def test_tiny_reservoir_explicit(self):
        p,t,v=self.fixture();p['split_method']='tiny_reservoir'
        self.assertEqual(validate_pairing(p,t,v)['split_method'],'tiny_reservoir')
        p['split_method']='reservoir'
        with self.assertRaises(ValueError):validate_pairing(p,t,v)


if __name__=='__main__':unittest.main()
