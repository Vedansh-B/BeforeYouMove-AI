#!/usr/bin/env python
"""Quick test of planning module"""

from planning import plan_current_position
from eval_material import MaterialEvaluator
import chess

board = chess.Board()
evaluator = MaterialEvaluator()

print('Testing planning module...')
plans = plan_current_position(board, evaluator, depth=2, our_branching=5, opp_branching=3, top_k=3)
print(f'Generated {len(plans)} plans')
for plan in plans:
    print(f'  {plan.rank}. {plan.san_line} | leaf_eval={plan.leaf_eval:.0f} | delta={plan.delta:.0f}')
print('✓ Planning works!')
