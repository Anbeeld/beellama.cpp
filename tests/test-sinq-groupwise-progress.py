import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.sinq.optimizer.selector import Group, State, coordinate_descent
from tools.sinq.optimizer.resources import canonical_json, digest


class GroupwiseProgressTests(unittest.TestCase):
    def test_full_248_groups_13_choices(self):
        assignment = {f't{i:03}': '00' for i in range(248)}
        groups = [Group(n, (n,), tuple({n: f'{i:02}'} for i in range(13))) for n in assignment]
        calls = []
        def evaluate(state):
            calls.append(state.state_id)
            return 3000 - sum(int(v) for _, v in state.assignment)
        result = coordinate_descent(State.create(assignment, {}), groups, evaluate,
                                    max_evaluations=2977, max_passes=1,
                                    recipe='T340', evaluator_policy_identity='cuda-fixture')
        progress = result.progress
        self.assertEqual(progress['status'], 'COMPLETE_FIRST_PASS')
        self.assertEqual(progress['eligible_groups'], 248)
        self.assertEqual(progress['groups_completed'], 248)
        self.assertEqual(progress['proposals_scored'], 2976)
        self.assertEqual(progress['evaluations'], 2977)
        self.assertEqual(len(set(calls)), 2977)
        self.assertEqual(progress['recipe'], 'T340')
        self.assertEqual(progress['evaluator_policy_identity'], 'cuda-fixture')
        for group in progress['passes'][0]['per_group']:
            self.assertTrue(group['completed'])
            self.assertEqual(len(group['candidates']), 13)
            self.assertEqual(sum(c['status'] == 'current_assignment' for c in group['candidates']), 1)
            self.assertTrue(all(c['state_id'] for c in group['candidates']))

    def test_failure_resume_and_atomic_progress(self):
        initial = State.create({'x': '0', 'locked': '0'}, {})
        groups = [Group('x', ('x',), tuple({'x': str(i)} for i in range(13))),
                  Group('locked', ('locked',), ({'locked': '1'},))]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'checkpoint.json'
            progress_path = Path(directory) / 'progress.json'
            calls = []
            def evaluate(state):
                value = dict(state.assignment)['x']
                calls.append(state.state_id)
                if value == '5':
                    raise TimeoutError('fixture')
                return 20 - int(value)
            kwargs = dict(checkpoint=checkpoint, progress_path=progress_path,
                          max_passes=1, locks={'locked': '0'})
            with self.assertRaises(TimeoutError):
                coordinate_descent(initial, groups, evaluate, **kwargs)
            failed = json.loads(progress_path.read_text())
            self.assertEqual(failed['status'], 'FAILED')
            self.assertEqual(failed['eligible_groups'], 1)
            self.assertEqual(failed['locked_groups'], 1)
            self.assertEqual(failed['groups_completed'], 0)
            self.assertFalse(failed['passes'][0]['per_group'][0]['completed'])
            committed = [e['state'] for e in json.loads(checkpoint.read_text())['trace'] if e['status'] == 'evaluated']
            resumed_calls = []
            def resumed(state):
                resumed_calls.append(state.state_id)
                return 20 - int(dict(state.assignment)['x'])
            result = coordinate_descent(initial, groups, resumed, **kwargs)
            self.assertTrue(set(committed).isdisjoint(resumed_calls))
            direct = coordinate_descent(initial, groups, resumed, max_passes=1, locks={'locked': '0'})
            self.assertEqual(result.state, direct.state)
            events = json.loads(checkpoint.read_text())['trace']
            scored = [e['state'] for e in events if e['status'] == 'evaluated']
            self.assertEqual(len(scored), len(set(scored)))
            self.assertEqual(json.loads(progress_path.read_text()), result.progress)
            self.assertEqual(result.progress['proposals_scored'], 12)
            self.assertEqual(result.progress['proposals_deduplicated_exact'], 0)
            self.assertEqual([e for e in result.trace if e['status'] == 'accepted'],
                             [e for e in direct.trace if e['status'] == 'accepted'])
            again_calls = []
            again = coordinate_descent(initial, groups, lambda s: again_calls.append(s) or 0, **kwargs)
            self.assertFalse(again_calls)
            self.assertEqual(result.state, again.state)
            self.assertEqual(result.progress, again.progress)

    def test_budget_and_per_pass_ties(self):
        initial = State.create({'a': '0', 'b': '0'}, {})
        groups = [Group('a', ('a',), ({'a': '1'}, {'a': '2'})), Group('b', ('b',), ({'b': '1'},))]
        scores = {('0', '0'): 10, ('1', '0'): 5, ('2', '0'): 5,
                  ('1', '1'): 2, ('0', '1'): 11, ('2', '1'): 3}
        def evaluate(state):
            return scores[tuple(v for _, v in state.assignment)]
        short = coordinate_descent(initial, groups, evaluate, max_evaluations=2)
        self.assertEqual(short.progress['status'], 'INCOMPLETE_BUDGET')
        self.assertEqual(short.progress['groups_completed'], 0)
        self.assertFalse(short.progress['passes'][0]['completed'])
        full = coordinate_descent(initial, groups, evaluate, max_evaluations=6)
        self.assertEqual(dict(full.state.assignment), {'a': '1', 'b': '1'})
        self.assertEqual(full.progress['status'], 'COMPLETE_CONVERGED')
        self.assertEqual(len(full.progress['passes']), 2)
        self.assertEqual([p['accepted_changes'] for p in full.progress['passes']], [2, 0])
        self.assertEqual(full.progress['groups_completed'], 2)
        candidates = full.progress['passes'][1]['per_group'][1]['candidates']
        self.assertEqual({c['status'] for c in candidates}, {'current_assignment', 'exact_score_cache'})

    def test_checkpoint_above_old_limit_and_policy_binding(self):
        # Two complete assignment copies make this checkpoint exceed 16 MiB.
        initial = State.create({'x': 'a' * (9 << 20)}, {})
        groups = [Group('x', ('x',), ())]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'checkpoint.json'
            kwargs = dict(checkpoint=checkpoint, max_passes=1, recipe='T425',
                          evaluator_policy_identity='policy-a')
            result = coordinate_descent(initial, groups, lambda s: 1, **kwargs)
            self.assertGreater(checkpoint.stat().st_size, 16 << 20)
            resumed = coordinate_descent(initial, groups, lambda s: self.fail('cached initial re-evaluated'), **kwargs)
            self.assertEqual(result.state, resumed.state)
            kwargs['evaluator_policy_identity'] = 'policy-b'
            with self.assertRaisesRegex(ValueError, 'provenance'):
                coordinate_descent(initial, groups, lambda s: 1, **kwargs)

    def test_legacy_v2_checkpoint_replay(self):
        initial = State.create({'x': '0'}, {})
        groups = [Group('x', ('x',), ({'x': '1'},))]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'checkpoint.json'
            kwargs = dict(checkpoint=checkpoint, max_passes=1)
            expected = coordinate_descent(initial, groups, lambda s: 2-int(dict(s.assignment)['x']), **kwargs)
            saved = json.loads(checkpoint.read_text())
            saved.pop('group_progress')
            for event in saved['trace']:
                event.pop('pass', None)
                if event['status'] == 'evaluated':
                    event.pop('group', None)
            saved.pop('checksum')
            saved['checksum'] = digest(saved)
            checkpoint.write_bytes(canonical_json(saved))
            result = coordinate_descent(initial, groups, lambda s: self.fail('legacy cached state re-evaluated'), **kwargs)
            self.assertEqual(result.state, expected.state)
            self.assertEqual(result.progress['groups_completed'], 1)
            self.assertEqual(result.progress['proposals_deduplicated_exact'], 1)

    def test_all_locked_and_exact_budget_completion(self):
        initial = State.create({'a': '0'}, {})
        groups = [Group('a', ('a',), ({'a': '1'},))]
        locked = coordinate_descent(initial, groups, lambda s: 1, locks={'a': '0'}, max_evaluations=1)
        self.assertEqual(locked.progress['eligible_groups'], 0)
        self.assertEqual(locked.progress['locked_groups'], 1)
        self.assertEqual(locked.progress['status'], 'COMPLETE_CONVERGED')
        exact = coordinate_descent(initial, groups, lambda s: 1, max_evaluations=2)
        self.assertEqual(exact.progress['status'], 'COMPLETE_CONVERGED')
        self.assertFalse(exact.budget_exhausted)
        self.assertEqual(exact.progress['groups_completed'], 1)


if __name__ == '__main__':
    unittest.main()
