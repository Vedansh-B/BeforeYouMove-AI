"""Chess model wrapper around python-chess with SAN-first notation."""

import chess
from typing import List, Tuple, Optional, Dict
import os


class OpeningDetector:
    """Detects openings using ECO data with longest-prefix matching."""

    def __init__(self, eco_file: str = "data/eco_san.tsv"):
        """Initialize with ECO data file."""
        self.openings = {}  # map: tuple of SAN moves -> (eco_code, name)
        self.max_depth = 0
        self._load_eco_data(eco_file)

    def _load_eco_data(self, eco_file: str) -> None:
        """Load ECO data from TSV file."""
        if not os.path.exists(eco_file):
            return

        try:
            with open(eco_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("\t")
                    if len(parts) < 3:
                        continue
                    eco_code, name, san_moves_str = parts[0], parts[1], parts[2]
                    
                    # Parse SAN moves
                    san_moves = tuple(san_moves_str.split())
                    self.openings[san_moves] = (eco_code, name)
                    self.max_depth = max(self.max_depth, len(san_moves))
        except Exception:
            pass

    def detect(self, move_history_san: List[str]) -> Optional[Tuple[str, str]]:
        """
        Detect opening from SAN move history (longest-prefix match).
        Returns (eco_code, name) or None.
        """
        for depth in range(min(len(move_history_san), self.max_depth), 0, -1):
            key = tuple(move_history_san[:depth])
            if key in self.openings:
                return self.openings[key]
        return None


class ChessModel:
    """Manages chess board state with SAN-first notation."""

    def __init__(self):
        """Initialize with starting position."""
        self.board = chess.Board()
        self.move_history_san: List[str] = []
        self.move_stack = []  # Save full board copies for undo
        self.opening_detector = OpeningDetector()

    def load_fen(self, fen: str) -> bool:
        """Load a FEN string. Returns True if valid."""
        try:
            self.board = chess.Board(fen)
            self.move_history_san = []
            self.move_stack = []
            return True
        except ValueError:
            return False

    def reset(self) -> None:
        """Reset to starting position."""
        self.board = chess.Board()
        self.move_history_san = []
        self.move_stack = []

    def try_push_move(self, move_text: str) -> Optional[str]:
        """
        Try to push a move from SAN or UCI text.
        Returns the SAN notation of the move if successful, None otherwise.
        """
        move_text = move_text.strip()
        
        # Try SAN first
        try:
            move = self.board.parse_san(move_text)
            if move in self.board.legal_moves:
                san = self.board.san(move)
                self.move_stack.append(self.board.copy())
                self.board.push(move)
                self.move_history_san.append(san)
                return san
        except (ValueError, chess.IllegalMoveError):
            pass
        
        # Try UCI as fallback
        try:
            move = chess.Move.from_uci(move_text)
            if move in self.board.legal_moves:
                san = self.board.san(move)
                self.move_stack.append(self.board.copy())
                self.board.push(move)
                self.move_history_san.append(san)
                return san
        except ValueError:
            pass
        
        return None

    def push_move_obj(self, move: chess.Move) -> str:
        """Push a Move object. Returns its SAN notation."""
        san = self.board.san(move)
        self.move_stack.append(self.board.copy())
        self.board.push(move)
        self.move_history_san.append(san)
        return san

    def pop_move(self) -> bool:
        """Pop last move. Returns True if successful."""
        if self.move_stack and self.move_history_san:
            self.board = self.move_stack.pop()
            self.move_history_san.pop()
            return True
        return False

    def get_board_fen(self) -> str:
        """Return current position as FEN."""
        return self.board.fen()

    def get_legal_moves_san(self) -> List[str]:
        """Return all legal moves in SAN notation."""
        return [self.board.san(move) for move in self.board.legal_moves]

    def get_legal_moves_san_with_uci(self) -> List[Tuple[str, str]]:
        """Return legal moves as (san, uci) tuples."""
        return [(self.board.san(move), move.uci()) for move in self.board.legal_moves]

    def get_last_move_san(self) -> str:
        """Return last move in SAN, or empty string if none."""
        if self.move_history_san:
            return self.move_history_san[-1]
        return ""

    def get_move_history_san(self) -> List[str]:
        """Return full move history in SAN."""
        return self.move_history_san.copy()

    def get_move_history_pgn(self) -> str:
        """Return move history in PGN format (1. e4 e5 2. Nf3 ...)."""
        pgn_text = ""
        for i, move in enumerate(self.move_history_san):
            if i % 2 == 0:  # White move
                pgn_text += f"{i // 2 + 1}. {move} "
            else:  # Black move
                pgn_text += f"{move} "
        return pgn_text.strip()

    def get_position_info(self) -> Dict[str, any]:
        """Return dict with position metadata."""
        return {
            "side_to_move": "White" if self.board.turn else "Black",
            "in_check": self.board.is_check(),
            "is_checkmate": self.board.is_checkmate(),
            "is_stalemate": self.board.is_stalemate(),
            "is_game_over": self.board.is_game_over(),
        }

    def get_opening(self) -> str:
        """Return opening name (e.g. 'E4 — Italian Game') or '(unknown)'."""
        result = self.opening_detector.detect(self.move_history_san)
        if result:
            eco, name = result
            return f"{eco} — {name}"
        return "(unknown)"

    def get_piece_at(self, square: int) -> Optional[chess.Piece]:
        """Get piece at square (0-63), or None if empty."""
        return self.board.piece_at(square)

    def get_legal_moves_from_square(self, from_square: int) -> List[int]:
        """Get destination squares for legal moves from given square."""
        destinations = []
        for move in self.board.legal_moves:
            if move.from_square == from_square:
                destinations.append(move.to_square)
        return destinations

    def get_last_move_squares(self) -> Tuple[Optional[int], Optional[int]]:
        """Return (from_square, to_square) of last move, or (None, None)."""
        if self.board.move_stack:
            move = self.board.move_stack[-1]
            return (move.from_square, move.to_square)
        return (None, None)

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
