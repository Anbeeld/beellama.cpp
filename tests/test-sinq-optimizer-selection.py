import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.sinq.optimizer.selector import State,Group,per_group_score_select,filter_groups,coordinate_descent


class ScoreSelectionTests(unittest.TestCase):
    def records(self):
        return [dict(assignment={'a':c},score=s,context_identity='ctx',scorer_configuration_identity='metric',
            objective='output_mse',direction='minimize',scope='group') for c,s in [('stock',4),('w32',2),('w128',1)]]

    def test_generic_minimum_and_filter(self):
        initial=State.create({'a':'stock'},{'source':'fixture'})
        groups=[Group('layer',('a',),({'a':'w32'},{'a':'w128'}))]
        result=per_group_score_select(initial,groups,self.records(),context_identity='ctx')
        self.assertEqual(dict(result.assignment),{'a':'w128'})
        filtered=filter_groups(initial,groups,self.records(),context_identity='ctx',keep=1)
        self.assertEqual(filtered[0].alternatives,({'a':'w128'},))

    def test_locks_and_ties(self):
        state=State.create({'a':'stock'},{})
        groups=[Group('a',('a',),({'a':'w32'},{'a':'w128'}))]
        records=self.records()
        self.assertEqual(per_group_score_select(state,groups,records,context_identity='ctx',locks={'a':'stock'}),state)
        for r in records:r['score']=1
        self.assertEqual(per_group_score_select(state,groups,records,context_identity='ctx'),state)

    def test_kl_cannot_mutate_locks(self):
        state=State.create({'a':'stock','b':'stock'},{})
        groups=[Group('a',('a',),({'a':'w32'},)),Group('b',('b',),({'b':'w128'},))]
        calls=[]
        def evaluate(s):
            a=dict(s.assignment);calls.append(a)
            self.assertEqual(a['a'],'stock')
            return 1 if a['b']=='stock' else 0
        result=coordinate_descent(state,groups,evaluate,locks={'a':'stock'})
        self.assertEqual(dict(result.state.assignment),{'a':'stock','b':'w128'})
        self.assertEqual(len(calls),2)

    def test_wrong_context_metric_missing_nonfinite_fail(self):
        state=State.create({'a':'stock'},{})
        groups=[Group('a',('a',),({'a':'w32'},{'a':'w128'}))]
        for mutate in [lambda r:r[0].update(context_identity='other'),lambda r:r[0].update(objective='output_l1'),
                       lambda r:r.pop(),lambda r:r[0].update(score=float('nan'))]:
            records=self.records();mutate(records)
            with self.subTest(records=records),self.assertRaises(ValueError):
                per_group_score_select(state,groups,records,context_identity='ctx')

    def test_mixed_locked_group_and_stale_lock_fail(self):
        state=State.create({'a':'stock','b':'stock'},{})
        groups=[Group('ab',('a','b'),({'a':'other','b':'other'},))]
        with self.assertRaises(ValueError):filter_groups(state,groups,[],context_identity='ctx',keep=1,locks={'a':'stock'})
        with self.assertRaises(ValueError):filter_groups(state,groups,[],context_identity='ctx',keep=1,locks={'a':'other'})

if __name__=='__main__':unittest.main()
