"""Chess UI using Tkinter with PNG piece assets and auto-planning."""

import tkinter as tk
from tkinter import messagebox
from chess_model import ChessModel
from eval_material import MaterialEvaluator
from planning import plan_current_position
from probabilistic import estimate_first_move_successes
import chess
import logging
import os
import time
from PIL import Image, ImageTk


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class ChessWorkbenchApp:
    """Tkinter GUI for chess workbench with PNG piece assets."""

    # Board rendering (80x80 squares, 72x72 pieces centered)
    SQUARE_SIZE = 80
    PIECE_SIZE = 72
    PIECE_OFFSET = (SQUARE_SIZE - PIECE_SIZE) // 2  # 4px padding
    BOARD_SIZE = 8
    
    LIGHT_COLOR = "#f0d9b5"
    DARK_COLOR = "#b58863"
    HIGHLIGHT_COLOR = "#baca44"
    LEGAL_MOVE_DOT_COLOR = "#5b5b5b"
    LAST_MOVE_COLOR = "#baca4460"  # Semi-transparent (won't work in tkinter, fallback to solid)

    # Unicode pieces fallback
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
        self.root.geometry("1600x800")

        self.model = ChessModel()
        self.evaluator = MaterialEvaluator()
        
        self.selected_square = None
        self.legal_destinations = set()
        self.current_plans = []  # Store generated plans
        self.current_prob_results = []  # Store probabilistic first-move results
        
        # Asset loading
        self.piece_images = {}  # Cached PhotoImage objects
        self.use_unicode = False
        self._load_piece_assets()

        self._create_ui()
        self._refresh_all()
        
        logger.info("Application started")

    def _load_piece_assets(self) -> None:
        """Load PNG piece assets. Falls back to unicode if missing."""
        piece_files = {
            (chess.PAWN, chess.WHITE): "assets/white/white-pawn.png",
            (chess.KNIGHT, chess.WHITE): "assets/white/white-knight.png",
            (chess.BISHOP, chess.WHITE): "assets/white/white-bishop.png",
            (chess.ROOK, chess.WHITE): "assets/white/white-rook.png",
            (chess.QUEEN, chess.WHITE): "assets/white/white-queen.png",
            (chess.KING, chess.WHITE): "assets/white/white-king.png",
            (chess.PAWN, chess.BLACK): "assets/black/black-pawn.png",
            (chess.KNIGHT, chess.BLACK): "assets/black/black-knight.png",
            (chess.BISHOP, chess.BLACK): "assets/black/black-bishop.png",
            (chess.ROOK, chess.BLACK): "assets/black/black-rook.png",
            (chess.QUEEN, chess.BLACK): "assets/black/black-queen.png",
            (chess.KING, chess.BLACK): "assets/black/black-king.png",
        }

        all_loaded = True
        for (piece_type, color), filepath in piece_files.items():
            if os.path.exists(filepath):
                try:
                    img = Image.open(filepath).convert("RGBA")
                    # Verify size (should be 72x72)
                    if img.size != (72, 72):
                        logger.warning(f"Piece {filepath} is {img.size}, expected 72x72")
                    photo = ImageTk.PhotoImage(img)
                    self.piece_images[(piece_type, color)] = photo
                except Exception as e:
                    logger.warning(f"Failed to load {filepath}: {e}")
                    all_loaded = False
            else:
                logger.warning(f"Piece asset not found: {filepath}")
                all_loaded = False

        if not all_loaded:
            logger.info("Using unicode pieces as fallback")
            self.use_unicode = True

    def _create_ui(self) -> None:
        """Build the UI layout with board, controls, and planning panels."""
        main_frame = tk.Frame(self.root, bg="white")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # -- SECTION 1: Board + Move Controls (Left) --
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5)

        # Board
        board_frame = tk.Frame(left_frame, bg="white")
        board_frame.pack(side=tk.TOP, fill=tk.BOTH, padx=5)

        tk.Label(board_frame, text="Chess Board", font=("Times New Roman", 12, "bold"), bg="white").pack(anchor=tk.W, pady=(0, 5))

        self.canvas = tk.Canvas(
            board_frame,
            width=self.SQUARE_SIZE * self.BOARD_SIZE,
            height=self.SQUARE_SIZE * self.BOARD_SIZE,
            bg="gray20",
            highlightthickness=0,
        )
        self.canvas.pack(padx=5, pady=5)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # Move controls (below board)
        controls_frame = tk.Frame(left_frame, bg="white")
        controls_frame.pack(side=tk.TOP, fill=tk.BOTH, padx=5, pady=(10, 0))

        # FEN section
        tk.Label(controls_frame, text="FEN", font=("Times New Roman", 10, "bold"), bg="white").pack(anchor=tk.W, pady=(5, 2))
        self.fen_input = tk.Entry(controls_frame, width=45)
        self.fen_input.pack(anchor=tk.W, pady=(0, 5))

        button_frame = tk.Frame(controls_frame, bg="white")
        button_frame.pack(anchor=tk.W, pady=(0, 5))
        tk.Button(button_frame, text="Load FEN", command=self._load_fen, width=10).pack(side=tk.LEFT, padx=2)
        tk.Button(button_frame, text="Copy FEN", command=self._copy_fen, width=10).pack(side=tk.LEFT, padx=2)

        # Move section
        tk.Label(controls_frame, text="Make Move (SAN/UCI)", font=("Times New Roman", 10, "bold"), bg="white").pack(anchor=tk.W, pady=(5, 2))
        self.move_input = tk.Entry(controls_frame, width=20)
        self.move_input.pack(anchor=tk.W, pady=(0, 5))
        self.move_input.bind("<Return>", lambda e: self._play_move_from_input())

        tk.Button(controls_frame, text="Play Move", command=self._play_move_from_input).pack(anchor=tk.W, pady=(0, 10))

        # Control buttons
        control_btn_frame = tk.Frame(controls_frame, bg="white")
        control_btn_frame.pack(anchor=tk.W, pady=(5, 10))
        tk.Button(control_btn_frame, text="Undo", command=self._undo, width=8).pack(side=tk.LEFT, padx=2)
        tk.Button(control_btn_frame, text="Reset", command=self._reset, width=8).pack(side=tk.LEFT, padx=2)

        # -- SECTION 2: Info Panels (Right - Upper) --
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        # Info upper (opening, status, eval, history)
        info_frame = tk.Frame(right_frame)
        info_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=False, padx=5)

        tk.Label(info_frame, text="Opening", font=("Times New Roman", 10, "bold")).pack(anchor=tk.W, pady=(5, 2))
        self.opening_label = tk.Label(info_frame, text="(unknown)", font=("Courier", 9), fg="#555")
        self.opening_label.pack(anchor=tk.W, pady=(0, 10))

        tk.Label(info_frame, text="Move History (PGN)", font=("Times New Roman", 10, "bold")).pack(anchor=tk.W, pady=(5, 2))
        self.history_label = tk.Label(info_frame, text="", font=("Courier", 8), justify=tk.LEFT, wraplength=220)
        self.history_label.pack(anchor=tk.NW, pady=(0, 10))

        tk.Label(info_frame, text="Position Status", font=("Times New Roman", 10, "bold")).pack(anchor=tk.W, pady=(5, 2))
        self.status_label = tk.Label(info_frame, text="", justify=tk.LEFT, font=("Courier", 9))
        self.status_label.pack(anchor=tk.NW, pady=(0, 10))

        tk.Label(info_frame, text="Material Evaluation", font=("Times New Roman", 10, "bold")).pack(anchor=tk.W, pady=(5, 2))
        self.eval_label = tk.Label(info_frame, text="", font=("Courier", 10, "bold"))
        self.eval_label.pack(anchor=tk.W, pady=(0, 10))

        # Legal moves section
        tk.Label(info_frame, text="Legal Moves (SAN)", font=("Times New Roman", 10, "bold")).pack(anchor=tk.W, pady=(5, 2))
        
        moves_scroll = tk.Scrollbar(info_frame)
        moves_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.moves_listbox = tk.Listbox(info_frame, height=8, width=22, yscrollcommand=moves_scroll.set, font=("Courier", 8))
        self.moves_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        moves_scroll.config(command=self.moves_listbox.yview)

        # -- SECTION 3: Planning Panel (Right - Lower) --
        planning_frame = tk.Frame(right_frame, relief=tk.SUNKEN, bd=1)
        planning_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=5, pady=(10, 0))

        tk.Label(planning_frame, text="PLANNING (Automated)", font=("Times New Roman", 10, "bold")).pack(anchor=tk.W, pady=(5, 5), padx=5)

        # Planning controls
        controls_subframe = tk.Frame(planning_frame)
        controls_subframe.pack(anchor=tk.W, fill=tk.X, padx=5, pady=5)

        # Depth
        tk.Label(controls_subframe, text="Depth (ply):", font=("Times New Roman", 9)).pack(side=tk.LEFT, padx=(0, 5))
        self.depth_spinbox = tk.Spinbox(controls_subframe, from_=1, to=6, width=3, font=("Times New Roman", 9))
        self.depth_spinbox.delete(0, tk.END)
        self.depth_spinbox.insert(0, "4")
        self.depth_spinbox.pack(side=tk.LEFT, padx=(0, 15))

        # Top K
        tk.Label(controls_subframe, text="Top K:", font=("Times New Roman", 9)).pack(side=tk.LEFT, padx=(0, 5))
        self.topk_spinbox = tk.Spinbox(controls_subframe, from_=1, to=20, width=3, font=("Times New Roman", 9))
        self.topk_spinbox.delete(0, tk.END)
        self.topk_spinbox.insert(0, "5")
        self.topk_spinbox.pack(side=tk.LEFT, padx=(0, 15))

        branching_subframe = tk.Frame(planning_frame)
        branching_subframe.pack(anchor=tk.W, fill=tk.X, padx=5, pady=5)

        # Our branching
        tk.Label(branching_subframe, text="Our Branch:", font=("Times New Roman", 9)).pack(side=tk.LEFT, padx=(0, 5))
        self.our_branch_spinbox = tk.Spinbox(branching_subframe, from_=1, to=20, width=3, font=("Times New Roman", 9))
        self.our_branch_spinbox.delete(0, tk.END)
        self.our_branch_spinbox.insert(0, "6")
        self.our_branch_spinbox.pack(side=tk.LEFT, padx=(0, 15))

        # Opp branching
        tk.Label(branching_subframe, text="Opp Branch:", font=("Times New Roman", 9)).pack(side=tk.LEFT, padx=(0, 5))
        self.opp_branch_spinbox = tk.Spinbox(branching_subframe, from_=1, to=20, width=3, font=("Times New Roman", 9))
        self.opp_branch_spinbox.delete(0, tk.END)
        self.opp_branch_spinbox.insert(0, "4")
        self.opp_branch_spinbox.pack(side=tk.LEFT, padx=(0, 0))

        # Generate Plans button
        tk.Button(
            planning_frame,
            text="Generate Plans",
            command=self._generate_plans,
            font=("Times New Roman", 10, "bold"),
            bg="#4CAF50",
            fg="white",
            width=20,
        ).pack(anchor=tk.W, padx=5, pady=5)

        # Plans listbox
        plans_scroll = tk.Scrollbar(planning_frame)
        plans_scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 5))

        self.plans_listbox = tk.Listbox(
            planning_frame,
            height=10,
            width=25,
            yscrollcommand=plans_scroll.set,
            font=("Courier", 8),
        )
        self.plans_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))
        self.plans_listbox.bind("<<ListboxSelect>>", self._on_plan_selected)
        plans_scroll.config(command=self.plans_listbox.yview)

        # Plan preview section
        preview_frame = tk.Frame(planning_frame)
        preview_frame.pack(anchor=tk.W, fill=tk.X, padx=5, pady=(5, 5))

        tk.Label(preview_frame, text="Plan Preview:", font=("Times New Roman", 9, "bold")).pack(anchor=tk.W)
        self.preview_label = tk.Label(preview_frame, text="", font=("Courier", 8), justify=tk.LEFT)
        self.preview_label.pack(anchor=tk.NW)

        # -- SECTION 4: Probabilistic Inference Panel --
        prob_frame = tk.Frame(planning_frame, relief=tk.GROOVE, bd=1)
        prob_frame.pack(anchor=tk.W, fill=tk.BOTH, expand=True, padx=5, pady=(6, 5))

        tk.Label(
            prob_frame,
            text="Probabilistic Inference",
            font=("Times New Roman", 10, "bold"),
        ).pack(anchor=tk.W, padx=5, pady=(5, 4))

        prob_controls_1 = tk.Frame(prob_frame)
        prob_controls_1.pack(anchor=tk.W, fill=tk.X, padx=5, pady=(0, 3))

        tk.Label(prob_controls_1, text="Candidate Top N:", font=("Times New Roman", 9)).pack(side=tk.LEFT, padx=(0, 4))
        self.prob_topn_spinbox = tk.Spinbox(prob_controls_1, from_=1, to=30, width=4, font=("Times New Roman", 9))
        self.prob_topn_spinbox.delete(0, tk.END)
        self.prob_topn_spinbox.insert(0, "8")
        self.prob_topn_spinbox.pack(side=tk.LEFT, padx=(0, 10))

        tk.Label(prob_controls_1, text="Simulations:", font=("Times New Roman", 9)).pack(side=tk.LEFT, padx=(0, 4))
        self.prob_sim_spinbox = tk.Spinbox(prob_controls_1, from_=1, to=500, width=5, font=("Times New Roman", 9))
        self.prob_sim_spinbox.delete(0, tk.END)
        self.prob_sim_spinbox.insert(0, "50")
        self.prob_sim_spinbox.pack(side=tk.LEFT, padx=(0, 10))

        tk.Label(prob_controls_1, text="Horizon:", font=("Times New Roman", 9)).pack(side=tk.LEFT, padx=(0, 4))
        self.prob_horizon_spinbox = tk.Spinbox(prob_controls_1, from_=1, to=12, width=4, font=("Times New Roman", 9))
        self.prob_horizon_spinbox.delete(0, tk.END)
        self.prob_horizon_spinbox.insert(0, "4")
        self.prob_horizon_spinbox.pack(side=tk.LEFT, padx=(0, 0))

        prob_controls_2 = tk.Frame(prob_frame)
        prob_controls_2.pack(anchor=tk.W, fill=tk.X, padx=5, pady=(0, 5))

        tk.Label(prob_controls_2, text="Opp Top K:", font=("Times New Roman", 9)).pack(side=tk.LEFT, padx=(0, 4))
        self.prob_opp_topk_spinbox = tk.Spinbox(prob_controls_2, from_=1, to=10, width=4, font=("Times New Roman", 9))
        self.prob_opp_topk_spinbox.delete(0, tk.END)
        self.prob_opp_topk_spinbox.insert(0, "3")
        self.prob_opp_topk_spinbox.pack(side=tk.LEFT, padx=(0, 10))

        tk.Label(prob_controls_2, text="Success Threshold (pawns):", font=("Times New Roman", 9)).pack(side=tk.LEFT, padx=(0, 4))
        self.prob_threshold_spinbox = tk.Spinbox(
            prob_controls_2,
            from_=-5.0,
            to=5.0,
            increment=0.1,
            width=5,
            font=("Times New Roman", 9),
        )
        self.prob_threshold_spinbox.delete(0, tk.END)
        self.prob_threshold_spinbox.insert(0, "0.5")
        self.prob_threshold_spinbox.pack(side=tk.LEFT, padx=(0, 0))

        tk.Button(
            prob_frame,
            text="Estimate Move Success",
            command=self._estimate_move_success,
            font=("Times New Roman", 10, "bold"),
            bg="#2a6fbe",
            fg="white",
            width=24,
        ).pack(anchor=tk.W, padx=5, pady=(0, 5))

        prob_results_frame = tk.Frame(prob_frame)
        prob_results_frame.pack(anchor=tk.W, fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        prob_scroll = tk.Scrollbar(prob_results_frame)
        prob_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.prob_listbox = tk.Listbox(
            prob_results_frame,
            height=8,
            width=64,
            yscrollcommand=prob_scroll.set,
            font=("Courier", 8),
        )
        self.prob_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.prob_listbox.bind("<<ListboxSelect>>", self._on_prob_result_selected)
        prob_scroll.config(command=self.prob_listbox.yview)

        self.prob_preview_label = tk.Label(prob_frame, text="", font=("Courier", 8), justify=tk.LEFT)
        self.prob_preview_label.pack(anchor=tk.NW, padx=5, pady=(0, 5))

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

            result = self.model.try_push_move(move_uci)
            if result:
                logger.info(f"Move played: {result}")
                self.move_input.delete(0, tk.END)
                self.selected_square = None
                self.legal_destinations = set()
                self._refresh_all()
            else:
                messagebox.showerror("Error", f"Move {move_uci} failed")
        else:
            messagebox.showwarning("Invalid", f"Cannot move to {chess.square_name(square)}")

    def _play_move_from_input(self) -> None:
        """Play move from text input (SAN preferred, UCI fallback)."""
        move_str = self.move_input.get().strip()
        if not move_str:
            messagebox.showwarning("Input", "Enter a move (e.g., e4, Nf3, or e2e4)")
            return

        result = self.model.try_push_move(move_str)
        if result:
            logger.info(f"Move played: {result}")
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
        if self.model.pop_move():
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

    def _generate_plans(self) -> None:
        """Generate plans using the planner."""
        try:
            depth = int(self.depth_spinbox.get())
            topk = int(self.topk_spinbox.get())
            our_branch = int(self.our_branch_spinbox.get())
            opp_branch = int(self.opp_branch_spinbox.get())

            logger.info(
                f"Planning: depth={depth}, topk={topk}, our_branch={our_branch}, opp_branch={opp_branch}"
            )

            # Generate plans
            self.current_plans = plan_current_position(
                self.model.board,
                self.evaluator,
                depth=depth,
                our_branching=our_branch,
                opp_branching=opp_branch,
                top_k=topk,
            )

            # Populate listbox
            self.plans_listbox.delete(0, tk.END)
            for plan in self.current_plans:
                display_text = f"{plan.rank}. {plan.san_line} | leaf={plan.leaf_eval:+.0f} | Δ={plan.delta:+.0f}"
                self.plans_listbox.insert(tk.END, display_text)

            logger.info(f"Generated {len(self.current_plans)} plans")

            # Clear preview
            self.preview_label.config(text="")

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid planning parameters: {e}")
        except Exception as e:
            messagebox.showerror("Planning Error", f"Planning failed: {e}")
            logger.error(f"Planning error: {e}")

    def _on_plan_selected(self, event: tk.Event) -> None:
        """Handle plan selection in listbox."""
        selection = self.plans_listbox.curselection()
        if not selection:
            return

        idx = selection[0]
        if idx >= len(self.current_plans):
            return

        plan = self.current_plans[idx]

        # Preview the plan: make a copy of the board and play all moves
        preview_board = self.model.board.copy(stack=True)

        try:
            for uci_move in plan.uci_line:
                move = chess.Move.from_uci(uci_move)
                preview_board.push(move)

            # Show preview
            preview_text = f"Plan {plan.rank}: {plan.san_line}\n"
            preview_text += f"Resulting FEN:\n{plan.leaf_fen}\n"
            preview_text += f"Leaf Eval: {plan.leaf_eval:+.0f} cp | Delta: {plan.delta:+.0f} cp"

            self.preview_label.config(text=preview_text)

        except Exception as e:
            logger.error(f"Error previewing plan: {e}")
            self.preview_label.config(text=f"Error: {e}")

    def _estimate_move_success(self) -> None:
        """Estimate success probabilities for top candidate first moves."""
        try:
            top_n = int(self.prob_topn_spinbox.get())
            simulations = int(self.prob_sim_spinbox.get())
            horizon = int(self.prob_horizon_spinbox.get())
            opponent_top_k = int(self.prob_opp_topk_spinbox.get())
            success_threshold = float(self.prob_threshold_spinbox.get())
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid probabilistic parameters: {e}")
            return

        try:
            start = time.time()
            logger.info(
                "Estimate Move Success requested: candidate_top_n=%d, simulations=%d, horizon=%d, opponent_top_k=%d, success_threshold=%+.2f",
                top_n,
                simulations,
                horizon,
                opponent_top_k,
                success_threshold,
            )

            self.current_prob_results = estimate_first_move_successes(
                board=self.model.board,
                evaluator=self.evaluator,
                horizon=horizon,
                top_k=top_n,
                opponent_top_k=opponent_top_k,
                simulations=simulations,
                success_threshold=success_threshold,
            )

            self.prob_listbox.delete(0, tk.END)
            for idx, result in enumerate(self.current_prob_results, start=1):
                row = (
                    f"{idx}. {result.san:<6} | P(success)={result.success_prob:.2f} "
                    f"| avg_delta={result.avg_delta:+.1f} | avg_leaf={result.avg_leaf_score:+.1f}"
                )
                self.prob_listbox.insert(tk.END, row)

            elapsed = time.time() - start
            if self.current_prob_results:
                best = self.current_prob_results[0]
                logger.info(
                    "Probabilistic panel complete: candidate_moves=%d, simulations_per_move=%d, runtime=%.2fs, best=%s P=%.2f",
                    len(self.current_prob_results),
                    simulations,
                    elapsed,
                    best.san,
                    best.success_prob,
                )
            else:
                logger.info(
                    "Probabilistic panel complete: candidate_moves=0, simulations_per_move=%d, runtime=%.2fs",
                    simulations,
                    elapsed,
                )

            self.prob_preview_label.config(text="")
        except Exception as e:
            logger.error(f"Probabilistic inference error: {e}")
            messagebox.showerror("Inference Error", f"Probabilistic inference failed: {e}")

    def _on_prob_result_selected(self, event: tk.Event) -> None:
        """Preview selected probabilistic first move on a copied board."""
        selection = self.prob_listbox.curselection()
        if not selection:
            return

        idx = selection[0]
        if idx >= len(self.current_prob_results):
            return

        result = self.current_prob_results[idx]
        preview_board = self.model.board.copy(stack=True)

        try:
            move = chess.Move.from_uci(result.uci)
            san = preview_board.san(move)
            preview_board.push(move)
            immediate_eval_cp = self.evaluator.evaluate(preview_board)

            preview_text = f"Preview move: {san} ({result.uci})\n"
            preview_text += f"Resulting FEN:\n{preview_board.fen()}\n"
            preview_text += f"Immediate eval (White perspective): {immediate_eval_cp:+.0f} cp"
            self.prob_preview_label.config(text=preview_text)
        except Exception as e:
            logger.error(f"Error previewing probabilistic move: {e}")
            self.prob_preview_label.config(text=f"Error: {e}")

    def _refresh_all(self) -> None:
        """Refresh all UI elements."""
        self._refresh_board()
        self._refresh_status()
        self._refresh_eval()
        self._refresh_moves()
        self._refresh_history()
        self._refresh_opening()
        self.fen_input.delete(0, tk.END)
        self.fen_input.insert(0, self.model.get_board_fen())
        self.current_prob_results = []
        if hasattr(self, "prob_listbox"):
            self.prob_listbox.delete(0, tk.END)
        if hasattr(self, "prob_preview_label"):
            self.prob_preview_label.config(text="")

    def _refresh_board(self) -> None:
        """Redraw the chess board with PNG assets or unicode fallback."""
        self.canvas.delete("all")

        # Get last move squares for highlighting
        last_from, last_to = self.model.get_last_move_squares()

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

                # Last move squares (subtle overlay)
                if square == last_from or square == last_to:
                    color = "#d4af37" if is_light else "#9d8b52"  # Subtle gold tint

                # Selected square (bright border)
                if square == self.selected_square:
                    color = self.HIGHLIGHT_COLOR

                # Draw square
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

                # Draw piece
                piece = self.model.get_piece_at(square)
                if piece:
                    if self.use_unicode:
                        # Unicode fallback
                        symbol = self.PIECES.get(
                            piece.piece_type if piece.color else -piece.piece_type
                        )
                        fg = "white" if piece.color else "black"
                        self.canvas.create_text(
                            x1 + self.SQUARE_SIZE // 2,
                            y1 + self.SQUARE_SIZE // 2,
                            text=symbol,
                            font=("Times New Roman", 48, "bold"),
                            fill=fg,
                        )
                    else:
                        # PNG asset
                        photo = self.piece_images.get((piece.piece_type, piece.color))
                        if photo:
                            self.canvas.create_image(
                                x1 + self.PIECE_OFFSET,
                                y1 + self.PIECE_OFFSET,
                                image=photo,
                                anchor=tk.NW,
                            )

                # Draw legal destination dot
                if square in self.legal_destinations:
                    cx = x1 + self.SQUARE_SIZE // 2
                    cy = y1 + self.SQUARE_SIZE // 2
                    dot_size = 8
                    self.canvas.create_oval(
                        cx - dot_size, cy - dot_size,
                        cx + dot_size, cy + dot_size,
                        fill=self.LEGAL_MOVE_DOT_COLOR,
                        outline=""
                    )

        # Draw board coordinates
        for i in range(self.BOARD_SIZE):
            file_char = chr(ord("a") + i)
            self.canvas.create_text(
                i * self.SQUARE_SIZE + self.SQUARE_SIZE // 2,
                self.SQUARE_SIZE * self.BOARD_SIZE + 12,
                text=file_char,
                font=("Times New Roman", 11, "bold"),
                fill="#666",
            )
            rank_char = str(8 - i)
            self.canvas.create_text(
                -12,
                i * self.SQUARE_SIZE + self.SQUARE_SIZE // 2,
                text=rank_char,
                font=("Times New Roman", 11, "bold"),
                fill="#666",
            )

    def _refresh_status(self) -> None:
        """Update status label."""
        info = self.model.get_position_info()
        side = info["side_to_move"]
        last_move = self.model.get_last_move_san()

        status_text = f"Side to Move: {side}\n"
        status_text += f"Last Move: {last_move if last_move else '(none)'}\n"

        if info["is_checkmate"]:
            status_text += "Status: ♔ CHECKMATE ♔"
        elif info["is_stalemate"]:
            status_text += "Status: STALEMATE"
        elif info["in_check"]:
            status_text += "Status: ⚠ CHECK ⚠"
        else:
            status_text += "Status: Playing"

        self.status_label.config(text=status_text)

    def _refresh_eval(self) -> None:
        """Update evaluation label."""
        score = self.evaluator.evaluate(self.model.board)
        if score > 0:
            self.eval_label.config(text=f"White +{score:.0f} cp", fg="#387e3f")
        elif score < 0:
            self.eval_label.config(text=f"Black +{-score:.0f} cp", fg="#387e3f")
        else:
            self.eval_label.config(text="Balanced", fg="#555")

    def _refresh_moves(self) -> None:
        """Update legal moves listbox (SAN)."""
        self.moves_listbox.delete(0, tk.END)
        moves = self.model.get_legal_moves_san()
        for move in moves:
            self.moves_listbox.insert(tk.END, move)

    def _refresh_history(self) -> None:
        """Update move history display (PGN format)."""
        pgn = self.model.get_move_history_pgn()
        self.history_label.config(text=pgn if pgn else "(no moves)")

    def _refresh_opening(self) -> None:
        """Update opening display."""
        opening = self.model.get_opening()
        self.opening_label.config(text=opening)


def main():
    """Launch the application."""
    root = tk.Tk()
    app = ChessWorkbenchApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
