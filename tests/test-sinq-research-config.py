"""Strict research configuration and split-provenance contracts (no native dependencies)."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.sinq.research.schemas import load_config, validate_config, validate_pairing


def config():
    return dict(schema_version=1, operation='generate', source_gguf='source.gguf',
        activation_manifest='train.json', tensor_list=['blk.0.attn_q.weight'],
        native_quantizer=dict(recipe={'blk.0.attn_q.weight': 'q5_0'}, sinq_strip_width=128,
            sinq_iterations=16, sinq_strength=.75),
        generators=['ordinary', 'awq_weight_activation', 'awq_activation_only'],
        alphas=[0, .5], correction_strengths=[0, 1],
        search=dict(strategy='small_grid', local_objective='output_l1', selection_policy='in_sample',
            max_shortlist=2, refine_step=.05, tolerance=0), output_directory='output',
        native_bridge='bridge', budgets=dict(workspace_bytes=1048576, max_candidates=64,
            max_output_bytes=1048576, max_files=64, max_tensors=2, max_tokens=32,
            max_probes=8, max_evaluations=64))


class ConfigTests(unittest.TestCase):
    def test_canonical_candidates_and_ids(self):
        c = config(); original = copy.deepcopy(c)
        r = validate_config(c)
        self.assertEqual(c, original)
        self.assertEqual(r['threads'], os.cpu_count() or 1)
        self.assertEqual(len(r['resolved_candidates']), 4)
        self.assertEqual(r['resolved_candidates'][0]['family'], 'ordinary')
        self.assertEqual(len({x['stable_id'] for x in r['resolved_candidates']}), 4)
        c['alphas'].reverse(); c['correction_strengths'].reverse()
        self.assertEqual(r['resolved_candidates'], validate_config(c)['resolved_candidates'])

    def test_unknown_fields_at_every_level(self):
        for key in ('resolved_candidates', 'resource_bounds', 'typo'):
            c = config(); c[key] = []
            with self.assertRaises(ValueError): validate_config(c)
        for key in ('search', 'native_quantizer', 'budgets'):
            c = config(); c[key]['typo'] = 1
            with self.assertRaises(ValueError): validate_config(c)

    def test_nonfinite_types_and_ranges(self):
        for value in (float('nan'), float('inf'), -.1, 1.1, True, '0.5'):
            c = config(); c['alphas'] = [value]
            with self.assertRaises(ValueError): validate_config(c)
        for value in (0, -1, True, 1.5):
            c = config(); c['budgets']['max_tokens'] = value
            with self.assertRaises(ValueError): validate_config(c)

    def test_refinement_budget_is_conservative(self):
        c = config(); c['search']['strategy'] = 'coarse_then_refine'
        c['budgets']['max_candidates'] = 7
        with self.assertRaises(ValueError): validate_config(c)
        c['budgets']['max_candidates'] = 8
        self.assertEqual(validate_config(c)['resource_bounds']['candidates_per_tensor'], 8)

    def test_legacy_alpha_zero_is_not_ordinary(self):
        c = config(); c['search']['strategy'] = 'legacy'
        r = validate_config(c)
        self.assertEqual(len(r['resolved_candidates']), 21)
        self.assertEqual(r['resolved_candidates'][1]['family'], 'awq_weight_activation')
        self.assertEqual(r['resolved_candidates'][1]['alpha'], 0)

    def test_held_out_requires_pairing(self):
        c = config(); c['search']['selection_policy'] = 'held_out'
        with self.assertRaises(ValueError): validate_config(c)

    def test_method_mismatch_and_final_test_refusal(self):
        c = dict(schema_version=1, operation='score', method='fisher_vjp_mc',
            objective='output_l1', candidate_manifest='cache.json', capture_manifest='capture.json',
            output_directory='scores', budgets=config()['budgets'])
        with self.assertRaises(ValueError): validate_config(c)
        c = dict(schema_version=1, operation='select', source_gguf='source.gguf',
            candidate_manifest='cache.json', baseline={}, groups={'ffn':['a']},
            selection_manifest='selection.json', selection_split='final_test', evaluator='eval',
            output_directory='selected', budgets=config()['budgets'])
        with self.assertRaises(ValueError): validate_config(c)

    def test_load_duplicate_keys_and_no_output(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'config.json'; c = config(); c['output_directory'] = str(Path(d)/'absent')
            p.write_text(json.dumps(c), encoding='utf-8')
            self.assertEqual(load_config(p)['operation'], 'generate')
            self.assertFalse(Path(c['output_directory']).exists())
            p.write_text('{"schema_version":1,"schema_version":1}', encoding='utf-8')
            with self.assertRaises(ValueError): load_config(p)

    def test_other_operations_positive(self):
        common = dict(schema_version=1, budgets=config()['budgets'])
        operations = [
            dict(operation='score', method='local', objective='output_mse',
                candidate_manifest='c.json', capture_manifest='x.json', output_directory='scores'),
            dict(operation='score', method='blockwise', objective='output_l1',
                candidate_manifest='c.json', capture_manifest='x.json', output_directory='scores',
                checkpoint='checkpoint', native_bridge='bridge', context='reference_inputs'),
            dict(operation='score', method='fisher_vjp_mc', objective='fisher_quadratic',
                candidate_manifest='c.json', capture_manifest='x.json', output_directory='scores',
                checkpoint='checkpoint', native_bridge='bridge', context='reference_inputs', probes=4, seed=0),
            dict(operation='assemble', source_gguf='source.gguf', candidate_manifest='c.json',
                assignment={'a':'candidate'}, output_gguf='hybrid.gguf'),
            dict(operation='select', source_gguf='source.gguf', candidate_manifest='c.json',
                baseline={'a':'candidate'}, groups={'ffn':['a']}, selection_manifest='s.json',
                selection_split='selection', evaluator='eval', output_directory='selected'),
            dict(operation='validate', artifact_manifest='c.json'),
        ]
        for operation in operations:
            self.assertEqual(validate_config(dict(common, **operation))['operation'], operation['operation'])

    def test_recipe_types_imatrix_and_alias(self):
        for qtype in ('Q5_0', 'q2_0', 'iq1_s'):
            c = config(); c['native_quantizer']['recipe']['blk.0.attn_q.weight'] = qtype
            with self.assertRaises(ValueError): validate_config(c)
        c = config(); c['output_directory'] = c['source_gguf']
        with self.assertRaises(ValueError): validate_config(c)
        c = config(); c['budgets']['max_tensors'] = 1
        c['tensor_list'].append('blk.0.attn_k.weight')
        c['native_quantizer']['recipe']['blk.0.attn_k.weight'] = 'q5_0'
        with self.assertRaises(ValueError): validate_config(c)

    def test_advanced_held_out_is_rejected_until_capture_contract_exists(self):
        for method,objective in [('blockwise','output_mse'),('fisher_vjp_mc','fisher_quadratic')]:
            c=dict(schema_version=1,operation='score',method=method,objective=objective,
                candidate_manifest='c.json',capture_manifest='capture.json',output_directory='scores',
                checkpoint='checkpoint',native_bridge='bridge',context='reference_inputs',
                selection_policy='held_out',validation_manifest='validation.json',pairing_manifest='pairing.json',
                budgets=config()['budgets'])
            if method=='fisher_vjp_mc':c.update(probes=2,seed=1)
            with self.assertRaisesRegex(ValueError,'advanced.*held_out'):
                validate_config(c)

    def test_joint_advanced_assignments(self):
        c=dict(schema_version=1,operation='score',method='blockwise',objective='output_mse',
            candidate_manifest='c.json',capture_manifest='capture.json',output_directory='scores',
            checkpoint='checkpoint',native_bridge='bridge',context='reference_inputs',
            assignments={'joint':{'blk.0.ffn_up.weight':'a'*64,'blk.0.ffn_gate.weight':'b'*64}},
            budgets=config()['budgets'])
        self.assertEqual(validate_config(c)['assignments'],c['assignments'])
        c['method']='local'
        with self.assertRaises(ValueError):validate_config(c)

    def test_advanced_fisher_positions_are_bounded_and_unique(self):
        c=dict(schema_version=1,operation='score',method='fisher_vjp_mc',objective='fisher_quadratic',
            candidate_manifest='c.json',capture_manifest='capture.json',output_directory='scores',
            checkpoint='checkpoint',native_bridge='bridge',context='reference_inputs',probes=2,seed=17,
            fisher_positions=[[0,0],[0,10],[0,21],[0,31]],budgets=config()['budgets'])
        self.assertEqual(validate_config(c)['fisher_positions'],[[0,0],[0,10],[0,21],[0,31]])
        for positions in ([],[[0,0],[0,0]],[[True,1]],[[0,-1]],[[0,1,2]],'0,1'):
            bad=dict(c,fisher_positions=positions)
            with self.subTest(positions=positions),self.assertRaises(ValueError):validate_config(bad)
        bad=dict(c,method='blockwise',objective='output_mse')
        with self.assertRaises(ValueError):validate_config(bad)
        bad=dict(c,budgets={**c['budgets'],'max_probes':7})
        with self.assertRaises(ValueError):validate_config(bad)

    def test_complete_stock_baseline_operation_and_bindings(self):
        c=dict(schema_version=1,operation='lock_baseline',source_gguf='source.gguf',stock_gguf='stock.gguf',
            recipe={'norm.weight':'F32','token_embd.weight':'Q5_0'},quantizer='quantizer',native_loader='loader',
            output_manifest='baseline.json',budgets=config()['budgets'])
        self.assertEqual(validate_config(c)['operation'],'lock_baseline')
        c=config();c['stock_baseline_manifest']='baseline.json'
        self.assertEqual(validate_config(c)['stock_baseline_manifest'],'baseline.json')

    def test_operation_unknowns(self):
        for op in ('assemble', 'validate', 'select', 'score', 'generate_candidates'):
            c = config(); c['operation'] = op
            with self.assertRaises(ValueError): validate_config(c)


class PairingTests(unittest.TestCase):
    def fixture(self):
        train = dict(schema_version=1, architecture='qwen3_5', source_sha256='a'*64,
            source_revision='locked', sampling=dict(token_sha256='b'*64, corpus_sha256='c'*64))
        val = copy.deepcopy(train); val['sampling']['token_sha256'] = 'd'*64
        def split(name, digest, start):
            return dict(split_id=name, token_sha256=digest, chunks=[dict(token_source_sha256='c'*64,
                start=start, end=start+8)], row_ids=[name+':0'])
        pairing = dict(schema_version=1, source_sha256='a'*64, split_method='split_before_reservoir',
            train=split('train', 'b'*64, 0), validation=split('validation', 'd'*64, 8))
        return pairing, train, val

    def test_disjoint(self):
        p,t,v = self.fixture(); self.assertEqual(validate_pairing(p,t,v)['split_method'], 'split_before_reservoir')

    def test_overlap_source_rows_and_split(self):
        for change in ('chunks', 'row_ids', 'split_id'):
            p,t,v = self.fixture(); p['validation'][change] = copy.deepcopy(p['train'][change])
            with self.assertRaises(ValueError): validate_pairing(p,t,v)
        p,t,v = self.fixture(); v['source_sha256'] = 'e'*64
        with self.assertRaises(ValueError): validate_pairing(p,t,v)

    def test_chunk_provenance_and_partial_overlap(self):
        p,t,v = self.fixture(); p['validation']['chunks'][0]['start'] = 7
        with self.assertRaises(ValueError): validate_pairing(p,t,v)
        p,t,v = self.fixture(); p['validation']['chunks'][0]['token_source_sha256'] = 'e'*64
        with self.assertRaises(ValueError): validate_pairing(p,t,v)
        p,t,v = self.fixture(); p['validation']['token_sha256'] = 'e'*64
        with self.assertRaises(ValueError): validate_pairing(p,t,v)

    def test_tiny_reservoir_explicit(self):
        p,t,v = self.fixture(); p['split_method'] = 'tiny_reservoir'
        self.assertEqual(validate_pairing(p,t,v)['split_method'], 'tiny_reservoir')
        p['split_method'] = 'reservoir'
        with self.assertRaises(ValueError): validate_pairing(p,t,v)


if __name__ == '__main__': unittest.main()
