"""Chess UI using Tkinter."""

import tkinter as tk
from tkinter import messagebox
from chess_model import ChessModel
from eval_material import MaterialEvaluator
import chess
import logging


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class ChessWorkbenchApp:
    """Tkinter GUI for chess workbench."""

    # Board rendering
    SQUARE_SIZE = 60
    BOARD_SIZE = 8
    LIGHT_COLOR = "#f0d9b5"
    DARK_COLOR = "#b58863"
    HIGHLIGHT_COLOR = "#baca44"
    LEGAL_MOVE_COLOR = "#7fc97f"

    # Unicode pieces: positive=white, negative=black
    PIECES = {
        1: "♙", -1: "♟",
        2: "♘", -2: "♞",
        3: "♗", -3: "♝",
        4: "♖", -4: "♜",
        5: "♕", -5: "♛",
        6: "♔", -6: "♚",
    }

    def __init__(self, root: tk.Tk):
        """Initialize the UI."""
        self.root = root
        self.root.title("Before You Move - Chess Workbench")
        self.root.geometry("1200x800")

        self.model = ChessModel()
        self.evaluator = MaterialEvaluator()
        
        self.selected_square = None
        self.legal_destinations = set()

        self._create_ui()
        self._refresh_all()
        
        logger.info("Application started")

    def _create_ui(self) -> None:
        """Build the UI layout."""
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # -- LEFT: Board --
        board_frame = tk.Frame(main_frame)
        board_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.canvas = tk.Canvas(
            board_frame,
            width=self.SQUARE_SIZE * self.BOARD_SIZE,
            height=self.SQUARE_SIZE * self.BOARD_SIZE,
            bg="gray20",
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # -- RIGHT: Controls & Info --
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5)

        # FEN section
        tk.Label(right_frame, text="FEN", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(5, 2))
        self.fen_input = tk.Entry(right_frame, width=50)
        self.fen_input.pack(anchor=tk.W, pady=(0, 5))

        button_frame = tk.Frame(right_frame)
        button_frame.pack(anchor=tk.W, pady=(0, 10))
        tk.Button(button_frame, text="Load FEN", command=self._load_fen).pack(side=tk.LEFT, padx=2)
        tk.Button(button_frame, text="Copy FEN", command=self._copy_fen).pack(side=tk.LEFT, padx=2)

        # Move section
        tk.Label(right_frame, text="Make Move (UCI/SAN)", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 2))
        self.move_input = tk.Entry(right_frame, width=30)
        self.move_input.pack(anchor=tk.W, pady=(0, 5))
        self.move_input.bind("<Return>", lambda e: self._play_move_from_input())

        tk.Button(right_frame, text="Play Move", command=self._play_move_from_input).pack(anchor=tk.W, pady=(0, 10))

        # Control buttons
        control_frame = tk.Frame(right_frame)
        control_frame.pack(anchor=tk.W, pady=(10, 10))
        tk.Button(control_frame, text="Undo", command=self._undo, width=8).pack(side=tk.LEFT, padx=2)
        tk.Button(control_frame, text="Reset", command=self._reset, width=8).pack(side=tk.LEFT, padx=2)

        # Status section
        tk.Label(right_frame, text="Position Status", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 5))
        self.status_label = tk.Label(right_frame, text="", justify=tk.LEFT, font=("Courier", 9))
        self.status_label.pack(anchor=tk.NW)

        # Evaluation section
        tk.Label(right_frame, text="Evaluation", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 5))
        self.eval_label = tk.Label(right_frame, text="", font=("Courier", 9))
        self.eval_label.pack(anchor=tk.W)

        # Legal moves section
        tk.Label(right_frame, text="Legal Moves (UCI)", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 5))
        
        moves_scroll = tk.Scrollbar(right_frame)
        moves_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.moves_listbox = tk.Listbox(right_frame, height=15, width=30, yscrollcommand=moves_scroll.set)
        self.moves_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        moves_scroll.config(command=self.moves_listbox.yview)

    def _on_canvas_click(self, event: tk.Event) -> None:
        """Handle click on board canvas."""
        col = event.x // self.SQUARE_SIZE
        row = event.y // self.SQUARE_SIZE

        if not (0 <= col < self.BOARD_SIZE and 0 <= row < self.BOARD_SIZE):
            return

        square = self.model.coords_to_square(row, col)

        # If no piece selected, try to select this square
        if self.selected_square is None:
            piece = self.model.get_piece_at(square)
            if piece and piece.color == self.model.board.turn:
                self.selected_square = square
                self.legal_destinations = set(self.model.get_legal_moves_from_square(square))
                logger.info(f"Selected {chess.square_name(square)}")
                self._refresh_board()
            return

        # If same square clicked, deselect
        if square == self.selected_square:
            self.selected_square = None
            self.legal_destinations = set()
            logger.info("Deselected")
            self._refresh_board()
            return

        # Try to move
        if square in self.legal_destinations:
            from_name = chess.square_name(self.selected_square)
            to_name = chess.square_name(square)
            move_uci = from_name + to_name

            # Check for promotion
            piece = self.model.get_piece_at(self.selected_square)
            if piece.piece_type == chess.PAWN:
                if (piece.color == chess.WHITE and chess.square_rank(square) == 7) or \
                   (piece.color == chess.BLACK and chess.square_rank(square) == 0):
                    # Promotion -- default to queen
                    move_uci += "q"

            if self.model.make_move(move_uci):
                logger.info(f"Move played: {move_uci}")
                self.move_input.delete(0, tk.END)
                self.selected_square = None
                self.legal_destinations = set()
                self._refresh_all()
            else:
                messagebox.showerror("Error", f"Move {move_uci} failed")
        else:
            messagebox.showwarning("Invalid", f"Cannot move to {chess.square_name(square)}")

    def _play_move_from_input(self) -> None:
        """Play move from text input."""
        move_str = self.move_input.get().strip()
        if not move_str:
            messagebox.showwarning("Input", "Enter a move (e.g., e2e4 or Nf3)")
            return

        if self.model.make_move(move_str):
            logger.info(f"Move played: {move_str}")
            self.move_input.delete(0, tk.END)
            self.selected_square = None
            self.legal_destinations = set()
            self._refresh_all()
        else:
            messagebox.showerror("Illegal", f"Cannot play: {move_str}")

    def _load_fen(self) -> None:
        """Load FEN from input field."""
        fen = self.fen_input.get().strip()
        if not fen:
            messagebox.showwarning("Input", "Enter a FEN string")
            return

        if self.model.load_fen(fen):
            logger.info(f"FEN loaded")
            self.selected_square = None
            self.legal_destinations = set()
            self._refresh_all()
            messagebox.showinfo("Success", "FEN loaded")
        else:
            messagebox.showerror("Error", "Invalid FEN string")

    def _copy_fen(self) -> None:
        """Copy current FEN to clipboard."""
        fen = self.model.get_board_fen()
        self.root.clipboard_clear()
        self.root.clipboard_append(fen)
        logger.info("FEN copied to clipboard")
        messagebox.showinfo("Copy", "FEN copied to clipboard")

    def _undo(self) -> None:
        """Undo last move."""
        if self.model.undo_move():
            logger.info("Move undone")
            self.selected_square = None
            self.legal_destinations = set()
            self._refresh_all()
        else:
            messagebox.showinfo("Undo", "Nothing to undo")

    def _reset(self) -> None:
        """Reset to starting position."""
        self.model.reset()
        logger.info("Position reset to start")
        self.selected_square = None
        self.legal_destinations = set()
        self._refresh_all()

    def _refresh_all(self) -> None:
        """Refresh board, status, evals, and moves list."""
        self._refresh_board()
        self._refresh_status()
        self._refresh_eval()
        self._refresh_moves()
        self.fen_input.delete(0, tk.END)
        self.fen_input.insert(0, self.model.get_board_fen())

    def _refresh_board(self) -> None:
        """Redraw the chess board."""
        self.canvas.delete("all")

        # Draw squares and pieces
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                x1 = col * self.SQUARE_SIZE
                y1 = row * self.SQUARE_SIZE
                x2 = x1 + self.SQUARE_SIZE
                y2 = y1 + self.SQUARE_SIZE

                square = self.model.coords_to_square(row, col)

                # Determine color
                is_light = (row + col) % 2 == 0
                color = self.LIGHT_COLOR if is_light else self.DARK_COLOR

                if square == self.selected_square:
                    color = self.HIGHLIGHT_COLOR
                elif square in self.legal_destinations:
                    color = self.LEGAL_MOVE_COLOR

                # Draw square
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")

                # Draw piece
                piece = self.model.get_piece_at(square)
                if piece:
                    symbol = self.PIECES.get(
                        piece.piece_type if piece.color else -piece.piece_type
                    )
                    fg = "white" if piece.color else "black"
                    self.canvas.create_text(
                        x1 + self.SQUARE_SIZE // 2,
                        y1 + self.SQUARE_SIZE // 2,
                        text=symbol,
                        font=("Arial", 40, "bold"),
                        fill=fg,
                    )

        # Draw coordinates
        for i in range(self.BOARD_SIZE):
            file_char = chr(ord("a") + i)
            self.canvas.create_text(
                i * self.SQUARE_SIZE + self.SQUARE_SIZE // 2,
                self.SQUARE_SIZE * self.BOARD_SIZE + 10,
                text=file_char,
                font=("Arial", 12),
                fill="white",
            )
            rank_char = str(8 - i)
            self.canvas.create_text(
                -15,
                i * self.SQUARE_SIZE + self.SQUARE_SIZE // 2,
                text=rank_char,
                font=("Arial", 12),
                fill="white",
            )

    def _refresh_status(self) -> None:
        """Update status label."""
        info = self.model.get_position_info()
        side = info["side_to_move"]
        last_move = self.model.get_last_move_uci()

        status_text = f"Side to Move: {side}\n"
        status_text += f"Last Move: {last_move if last_move else '(none)'}\n"

        if info["is_checkmate"]:
            status_text += "Status: CHECKMATE"
        elif info["is_stalemate"]:
            status_text += "Status: STALEMATE"
        elif info["in_check"]:
            status_text += "Status: CHECK"
        else:
            status_text += "Status: Playing"

        self.status_label.config(text=status_text)

    def _refresh_eval(self) -> None:
        """Update evaluation label."""
        score = self.evaluator.evaluate(self.model.board)
        self.eval_label.config(text=f"Material: {score:+.0f} cp")

    def _refresh_moves(self) -> None:
        """Update legal moves listbox."""
        self.moves_listbox.delete(0, tk.END)
        moves = self.model.get_legal_moves_uci()
        for move in moves:
            self.moves_listbox.insert(tk.END, move)


def main():
    """Launch the application."""
    root = tk.Tk()
    app = ChessWorkbenchApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
