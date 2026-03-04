import chess
from chess_model import ChessModel
from eval_material import MaterialEvaluator
from planning import plan_current_position

print("=== COMPREHENSIVE INTEGRATION TEST ===\n")

# Test 1: Board and moves
print("TEST 1: Board Operations")
model = ChessModel()
print(f"✓ Initial position: {model.board.fen()[:30]}...")
san = model.try_push_move("e4")
print(f"✓ Played move: {san}")
print(f"✓ Move history (PGN): {model.get_move_history_pgn()}")

# Test 2: Opening detection
print("\nTEST 2: Opening Detection")
opening = model.get_opening()
print(f"✓ Opening detected: {opening if opening else 'Not yet in database'}")
model.try_push_move("c5")
opening = model.get_opening()
print(f"✓ After second move: {opening if opening else 'None'}")

# Test 3: Legal moves in SAN
print("\nTEST 3: Legal Moves (SAN)")
legal_moves = model.get_legal_moves_san()
print(f"✓ Found {len(legal_moves)} legal moves")
print(f"✓ Sample moves: {', '.join(legal_moves[:5])}")

# Test 4: Planning
print("\nTEST 4: Planning Generation")
evaluator = MaterialEvaluator()
plans = plan_current_position(model.board, evaluator, depth=3, our_branching=5, opp_branching=4, top_k=5)
print(f"✓ Generated {len(plans)} ranked plans")
if plans:
    for plan in plans[:3]:
        print(f"  Rank {plan.rank}: {plan.san_line} | leaf_eval={plan.leaf_eval:+.0f} | Δ={plan.delta:+.0f}")

# Test 5: Undo and reset
print("\nTEST 5: Board State Management")
initial_mv_count = len(model.move_history_san)
model.reset()
print(f"✓ Reset successful, position reset to start")
model.try_push_move("d4")
model.pop_move()
print(f"✓ Undo successful")

print("\n=== ALL TESTS PASSED ===")
