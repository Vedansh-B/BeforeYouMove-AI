import chess
from chess_model import ChessModel
from eval_material import MaterialEvaluator
from planning import plan_current_position

# Test planning integration
model = ChessModel()
evaluator = MaterialEvaluator()

print("Testing planning integration from main modules...")
plans = plan_current_position(model.board, evaluator, depth=3, our_branching=5, opp_branching=4, top_k=3)

if plans:
    print(f"✓ Generated {len(plans)} plans")
    for plan in plans[:2]:
        print(f"  {plan.rank}. {plan.san_line} | Δ={plan.delta:+.0f}")
else:
    print("✗ No plans generated")
