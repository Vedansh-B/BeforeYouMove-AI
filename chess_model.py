"""Chess model wrapper around python-chess."""

import chess
from typing import List, Tuple, Optional, Dict


class ChessModel:
    """Manages chess board state and move validation."""

    def __init__(self):
        """Initialize with starting position."""
        self.board = chess.Board()
        self.move_stack = []  # Save history for undo

    def load_fen(self, fen: str) -> bool:
        """Load a FEN string. Returns True if valid."""
        try:
            self.board = chess.Board(fen)
            self.move_stack = []
            return True
        except ValueError:
            return False

    def reset(self) -> None:
        """Reset to starting position."""
        self.board = chess.Board()
        self.move_stack = []

    def get_board_fen(self) -> str:
        """Return current position as FEN."""
        return self.board.fen()

    def make_move(self, move_str: str) -> bool:
        """
        Make a move from UCI (e2e4) or SAN (Nf3, e4, O-O).
        Returns True if move was legal and made, False otherwise.
        """
        try:
            move = self._parse_move(move_str)
            if move is None or move not in self.board.legal_moves:
                return False
            
            self.move_stack.append(self.board.copy())
            self.board.push(move)
            return True
        except Exception:
            return False

    def undo_move(self) -> bool:
        """Undo last move. Returns True if successful."""
        if self.move_stack:
            self.board = self.move_stack.pop()
            return True
        return False

    def _parse_move(self, move_str: str) -> Optional[chess.Move]:
        """Parse move from UCI or SAN. Returns Move or None."""
        move_str = move_str.strip()
        
        # Try UCI first
        try:
            return chess.Move.from_uci(move_str)
        except ValueError:
            pass
        
        # Try SAN
        try:
            return self.board.parse_san(move_str)
        except (ValueError, chess.IllegalMoveError):
            pass
        
        return None

    def get_legal_moves_uci(self) -> List[str]:
        """Return all legal moves in UCI notation."""
        return [move.uci() for move in self.board.legal_moves]

    def get_legal_moves_san(self) -> List[str]:
        """Return all legal moves in SAN notation."""
        return [self.board.san(move) for move in self.board.legal_moves]

    def get_last_move_uci(self) -> str:
        """Return last move in UCI, or empty string if none."""
        if self.board.move_stack:
            return self.board.move_stack[-1].uci()
        return ""

    def get_position_info(self) -> Dict[str, any]:
        """Return dict with position metadata."""
        return {
            "side_to_move": "White" if self.board.turn else "Black",
            "in_check": self.board.is_check(),
            "is_checkmate": self.board.is_checkmate(),
            "is_stalemate": self.board.is_stalemate(),
            "is_game_over": self.board.is_game_over(),
        }

    def get_piece_at(self, square: int) -> Optional[chess.Piece]:
        """Get piece at square (0-63), or None if empty."""
        return self.board.piece_at(square)

    def get_legal_moves_from_square(self, from_square: int) -> List[int]:
        """Get list of destination squares for legal moves from given square."""
        destinations = []
        for move in self.board.legal_moves:
            if move.from_square == from_square:
                destinations.append(move.to_square)
        return destinations

    def square_to_coords(self, square: int) -> Tuple[int, int]:
        """Convert square index (0-63) to (row, col) for display (0-based)."""
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        return (7 - rank, file)

    def coords_to_square(self, row: int, col: int) -> int:
        """Convert (row, col) to square index (0-63)."""
        rank = 7 - row
        file = col
        return chess.square(file, rank)
