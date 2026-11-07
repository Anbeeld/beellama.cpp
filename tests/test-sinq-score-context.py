"""Canonical score context identity and persisted-artifact closure regressions."""
import copy
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'gguf-py')]
from tools.sinq.optimizer import stages
from tools.sinq.optimizer.resources import canonical_json,digest,file_digest
spec=importlib.util.spec_from_file_location('integrity_fixture',ROOT/'tests/test-sinq-optimizer-integrity.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)


class ScoreContextTests(unittest.TestCase):
    setUp=fixture.RepairTests.setUp
    limits=fixture.RepairTests.limits

    def config(self):
        bridge=self.d/'native-bridge';bridge.write_bytes(b'native-v1')
        return dict(method='local',objective='output_mse',scorer_name='local_output_mse',threads=1,
            selection_policy='in_sample',device='cpu',native_bridge=str(bridge),
            capture_manifest=str(self.capture),candidate_manifest=str(self.manifest),
            budgets=self.budgets,output_directory=str(self.d/'scores'))

    def score(self,config,partition=None):
        activations={'tensors':{name:object() for name in self.names}}
        with patch.object(stages,'_score_partition',return_value=partition),patch.object(stages.native_codec,'validate_activations',return_value=activations),patch.object(stages.native_codec,'rescore_many',side_effect=lambda binary,candidates,*args,**kwargs:[(key,dict(train_score=2.,validation_score=1.,encode_count=0)) for key,candidate in candidates]),patch.object(stages,'validate_pairing'):
            return stages._score(config,self.run,self.store,self.source)

    def replay(self,result):
        path=self.d/'replay.json';path.write_bytes(canonical_json(result))
        return stages._score_artifact(path,dict(candidate_manifest=str(self.manifest),budgets=self.budgets))

    def resign(self,result):
        result['checksum']=digest({k:v for k,v in result.items() if k!='checksum'})
        return result

    def test_serialized_context_replays_and_has_explicit_absent_values(self):
        result=self.score(self.config())
        context=result['scorer_context']
        self.assertEqual(result['context_identity'],digest(context))
        self.assertEqual(context['source_sha256'],file_digest(self.source))
        self.assertEqual(context['candidate_manifest_sha256'],file_digest(self.manifest))
        self.assertEqual(context['training_capture_sha256'],file_digest(self.capture))
        self.assertEqual(context['selection_policy'],'in_sample')
        for key in ('validation_manifest_sha256','pairing_manifest_sha256','data_partition_identity','reference_identity','execution_identity_sha256','execution_policy'):
            self.assertIn(key,context);self.assertIsNone(context[key])
        self.assertIsNotNone(context['backend_identity'])
        self.assertIsNotNone(context['precision_identity'])
        self.assertEqual(context['device_identity'],{'devices':['cpu'],'device_identifiers':None})
        self.assertEqual(self.replay(result),result)
        config=self.config();config['output_directory']=str(self.d/'repeat')
        self.assertEqual(self.score(config)['context_identity'],result['context_identity'])

    def test_every_local_context_input_changes_identity(self):
        config=self.config();validation=self.d/'validation.json';pairing=self.d/'pairing.json'
        validation.write_bytes(b'{}');pairing.write_bytes(b'{}')
        config.update(selection_policy='held_out',validation_manifest=str(validation),pairing_manifest=str(pairing))
        baseline=self.score(config,partition={'calibration':'a','validation':'b'})
        def check(changed,partition=None):
            changed['output_directory']=str(self.d/('score-'+str(len(list(self.d.glob('score-*'))))))
            self.assertNotEqual(self.score(changed,partition or {'calibration':'a','validation':'b'})['context_identity'],baseline['context_identity'])
        for key,value in [('selection_policy','in_sample'),('threads',2),('objective','output_l1')]:
            with self.subTest(key=key):
                changed=dict(config,**{key:value})
                if key=='objective':changed['scorer_name']='local_output_l1'
                check(changed)
        for key in ('capture_manifest','candidate_manifest','validation_manifest','pairing_manifest','native_bridge'):
            with self.subTest(key=key):
                path=Path(config[key]);original=path.read_bytes();path.write_bytes(original+b' ')
                try:check(dict(config))
                finally:path.write_bytes(original)
        check(dict(config),partition={'calibration':'a','validation':'changed'})
        context=baseline['scorer_context']
        changed=stages._scorer_context(config,dict(self.run,source_sha256='f'*64),context['data_partition_identity'],stages._score_input_hashes(config))
        self.assertNotEqual(digest(changed),baseline['context_identity'])

    def test_local_capture_revalidation_prevents_publication(self):
        config=self.config()
        initial={'tensors':{name:object() for name in self.names},'data_identity':'original'}
        changed={**initial,'data_identity':'changed'}
        with patch.object(stages,'_score_partition',return_value=None),patch.object(stages.native_codec,'validate_activations',side_effect=[initial,changed]),patch.object(stages.native_codec,'rescore_many',side_effect=lambda binary,candidates,*args,**kwargs:[(key,dict(train_score=2.,validation_score=None,encode_count=0)) for key,candidate in candidates]):
            with self.assertRaisesRegex(ValueError,'capture.*changed'):
                stages._score(config,self.run,self.store,self.source)
        self.assertFalse(Path(config['output_directory']).exists())

    def test_reader_rejects_resigned_internal_inconsistency(self):
        baseline=self.score(self.config())
        mutations=[lambda r:r['scorer_context'].__setitem__('selection_policy','held_out'),
            lambda r:r.__setitem__('context_identity','f'*64),
            lambda r:r.__setitem__('scorer_configuration_identity','f'*64),
            lambda r:r.__setitem__('data_partition_identity',{'wrong':True}),
            lambda r:r.__setitem__('method','blockwise'),
            lambda r:r['records'][0].__setitem__('context_identity','f'*64),
            lambda r:r['records'][0].__setitem__('capture_sha256','f'*64),
            lambda r:r['records'][0].__setitem__('objective','wrong'),
            lambda r:r['records'][0].__setitem__('scorer_configuration_identity','f'*64)]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                value=copy.deepcopy(baseline);mutate(value)
                with self.assertRaisesRegex(ValueError,'context'):self.replay(self.resign(value))
        value=copy.deepcopy(baseline);value['scorer_context'].pop('validation_manifest_sha256')
        value['context_identity']=digest(value['scorer_context'])
        with self.assertRaisesRegex(ValueError,'context'):self.replay(self.resign(value))

    def test_advanced_reference_precision_device_and_controls_bind_identity(self):
        config=self.config();config.update(method='fisher_vjp_mc',objective='fisher_quadratic',scorer_name='fisher_vjp_mc',
            context='reference_inputs',probes=1,seed=0,reference_dtype='fp32')
        policy=dict(reference_dtype='fp32',execution_dtype='torch.float32',parameter_storage_policy='uniform-fp32-v1',
            devices=['cpu'],device_identifiers={},versions={'torch':'fixture'},deterministic={'algorithms':True},memory_policy={'host_limit_bytes':1024})
        def make_context(config=config,policy=policy,reference_extra=None):
            reference=dict(source_gguf_sha256=self.source_hash,execution_policy=policy,checkpoint_sha256='a'*64,**(reference_extra or {}))
            identity=hashlib.sha256(json.dumps(reference,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            return stages._scorer_context(config,self.run,None,stages._score_input_hashes(config),reference=reference,execution_policy=policy,execution_identity=identity)
        context=make_context();baseline=digest(context)
        for key,value in [('context','selected_quantized_inputs'),('probes',2),('seed',1),('fisher_positions',[[0,1]]),
                ('reference_dtype','bf16'),('device','cuda:0'),('cuda_devices',['cuda:0','cuda:1']),('memory_policy',{'host_limit_bytes':2048})]:
            with self.subTest(key=key):self.assertNotEqual(digest(make_context(config=dict(config,**{key:value}))),baseline)
        for key,value in [('reference_dtype','bf16'),('execution_dtype','torch.bfloat16'),('parameter_storage_policy','bf16-bulk-exact-fp32-qwen-norms-v1'),
                ('devices',['cuda:0']),('device_identifiers',{'cuda:0':{'uuid':'fixture'}}),('versions',{'torch':'different'}),
                ('deterministic',{'algorithms':False}),('memory_policy',{'host_limit_bytes':2048})]:
            with self.subTest(key=key):self.assertNotEqual(digest(make_context(policy=dict(policy,**{key:value}))),baseline)
        changed_reference=copy.deepcopy(context['reference_identity']);changed_reference['checkpoint_sha256']='b'*64
        changed_context=stages._scorer_context(config,self.run,None,stages._score_input_hashes(config),reference=changed_reference,
            execution_policy=policy,execution_identity=hashlib.sha256(json.dumps(changed_reference,sort_keys=True,separators=(',',':')).encode()).hexdigest())
        self.assertNotEqual(digest(changed_context),baseline)
        result=dict(schema_version=2,method=config['method'],objective=config['objective'],records=[],
            scorer_context=context,context_identity=baseline,scorer_configuration_identity=digest(context['scorer_configuration']),
            candidate_manifest_sha256=file_digest(self.manifest),data_partition_identity=None,
            reference_identity=context['reference_identity'],execution_policy=context['execution_policy'],execution_identity_sha256=context['execution_identity_sha256'])
        self.assertEqual(self.replay(self.resign(result)),result)
        for key in ('reference_identity','execution_policy','execution_identity_sha256'):
            changed=copy.deepcopy(result);changed[key]=None
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'context'):self.replay(self.resign(changed))
        # Even a rehashed artifact cannot describe a reference bound to a different policy/source.
        for key,value in [('source_gguf_sha256','b'*64),('execution_policy',{})]:
            changed=copy.deepcopy(result);changed['scorer_context']['reference_identity'][key]=value
            changed['context_identity']=digest(changed['scorer_context'])
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'context'):self.replay(self.resign(changed))
        runtime=self.d/'ggml-context-fixture.dll';runtime.write_bytes(b'runtime-v1')
        self.assertNotEqual(digest(make_context()),baseline)
        with patch('tools.sinq.optimizer.implementation.implementation_fingerprint',return_value={'checksum':'f'*64}):
            self.assertNotEqual(digest(make_context()),baseline)

    def test_selection_resume_is_bound_to_score_artifact_hash(self):
        result=self.score(self.config());scores=self.d/'selection-scores.json';scores.write_bytes(canonical_json(result))
        loader=self.d/'loader';loader.write_bytes(b'loader')
        output=self.d/'selection-run'
        config=dict(policy='per_group_score',baseline={name:self.ids[name]['stock'] for name in self.names},
            groups={name:[name] for name in self.names},locks={},candidate_manifest=str(self.manifest),
            scores=str(scores),source_gguf=str(self.source),native_loader=str(loader),threads=1,
            output_directory=str(output),budgets=self.budgets)
        def assemble(source,path,*args,**kwargs):
            Path(path).write_bytes(b'model');return file_digest(path)
        with patch('tools.sinq.optimizer.assembly.assemble',side_effect=assemble),patch.object(stages,'_validator',return_value=lambda path:None):
            original=stages._select(config,self.run,self.store,self.source)
            self.assertEqual(stages._select(config,self.run,self.store,self.source,resume=output),original)
            score_config=self.config();score_config.update(threads=2,output_directory=str(self.d/'changed-scores'))
            changed=self.score(score_config);scores.write_bytes(canonical_json(changed))
            with self.assertRaisesRegex(ValueError,'identity'):stages._select(config,self.run,self.store,self.source,resume=output)

    def test_pipeline_selection_identity_binds_score_artifact(self):
        from tools.sinq.optimizer.run_state import selection_identity
        from tools.sinq.optimizer.schemas import validate_config
        raw=fixture.RepairTests.selection_config(self);config=validate_config(raw,'select')
        evaluator=stages._json(config['evaluator'])
        score_config=self.config();result=self.score(score_config)
        scores=self.d/'pipeline-scores.json';scores.write_bytes(canonical_json(result))
        config['stages']=[dict(type='filter',scores=str(scores),keep=1,context='frozen'),dict(type='select',method='groupwise_kl')]
        baseline=selection_identity(config,evaluator,{'fixture':True})
        self.assertEqual(selection_identity(config,evaluator,{'fixture':True}),baseline)
        score_config.update(threads=2,output_directory=str(self.d/'pipeline-changed-scores'))
        scores.write_bytes(canonical_json(self.score(score_config)))
        self.assertNotEqual(selection_identity(config,evaluator,{'fixture':True})['checksum'],baseline['checksum'])

    def test_nested_assignment_records_are_bound(self):
        config=self.config();config['assignments']={'joint':{name:self.ids[name]['stock'] for name in self.names}}
        result=self.score(config)
        self.assertEqual(self.replay(result),result)
        result['records'][0]['tensor_scores'][0]['context_identity']='f'*64
        with self.assertRaisesRegex(ValueError,'context'):self.replay(self.resign(result))


if __name__=='__main__':unittest.main()
