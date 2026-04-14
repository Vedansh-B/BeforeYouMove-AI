import chess

from eval_material import MaterialEvaluator
from probabilistic.inference import estimate_first_move_successes


def test_king_walk_is_discouraged():
    evaluator = MaterialEvaluator()
    safe_board = chess.Board("rnbq1rk1/ppppbppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQ1RK1 w - - 4 6")
    king_walk_board = chess.Board("rnbq1rk1/ppppbppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQ - 4 6")
    king_walk_board.push_san("Ke2")

    assert evaluator.evaluate(safe_board) > evaluator.evaluate(king_walk_board)


def test_probabilistic_success_uses_best_available_move_margin():
    evaluator = MaterialEvaluator()
    board = chess.Board("4k3/8/8/8/8/8/3n4/3QK3 w - - 0 1")

    results = estimate_first_move_successes(
        board=board,
        evaluator=evaluator,
        horizon=0,
        top_k=5,
        opponent_top_k=1,
        simulations=1,
        success_threshold=0.2,
    )

    assert results
    assert results[0].success_prob == 1.0
    assert any(result.avg_delta < 0 for result in results)
