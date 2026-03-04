# Before You Move - Chess Workbench

Interactive chess workbench with automated planning, PNG pieces, SAN notation, and ECO opening detection.

**Day 1:** Visual board + opening detection  
**Day 2:** Automated planning with ranked multi-move lines (this update)  
**Later:** Probabilistic inference + deep learning  

## Features

### Board & Interaction
- **Visual Board**: 8×8 with 72×72 PNG pieces (or unicode fallback)
- **Click-to-Move**: Select piece, highlight legal destinations, click to move
- **Move Input**: SAN (e4, Nf3, O-O) or UCI (e2e4)
- **Move History**: PGN-style display (1. e4 e5 2. Nf3 Nc6)
- **Opening Detection**: ECO database with longest-prefix matching

### Planning (NEW)
- **Automated Multi-Move Planning**: Generate and rank candidate multi-move lines
- **Depth-Limited Search**: Configurable depth (ply), default 4
- **Move Ordering**: Captures > Checks > Promotions > others (for speed)
- **Branching Control**: Limit our moves and opponent responses separately
- **Top K Results**: Display top ranked plans with material evaluation
- **Plan Preview**: Click a plan to see resulting FEN and leaf evaluation
- **Extensible Evaluation**: Currently uses material balance (easy to swap in neural nets later)

## Setup

### Install Dependencies
```bash
pip install -r requirements.txt
```

Installs: `python-chess`, `Pillow`

### Run
```bash
python main.py
```

## Usage

### Making Moves
1. **Click-to-Move**: Click piece → legal squares highlighted → click destination
2. **Text Input**: Type `e4`, `Nf3`, `O-O` (SAN) or `e2e4` (UCI) → click Play Move or press Enter
3. **Undo/Reset**: Undo last move or reset to start

### Planning

The **PLANNING** panel on the right side lets you:

1. **Set Planning Parameters**:
   - **Depth**: Search depth in half-moves (ply). Default 4 = 2 full moves for each side.
   - **Top K**: Number of best plans to return. Default 5.
   - **Our Branch**: Max moves to consider for our side. Default 6.
   - **Opp Branch**: Max moves to consider for opponent. Default 4.

2. **Generate Plans**:
   - Click "Generate Plans" button
   - The planner searches the game tree with strong pruning
   - Runtime ~0.5–2s depending on position and depth

3. **View Results**:
   - Plans displayed in listbox: rank, SAN line, leaf eval (cp), delta from current eval
   - Click a plan to preview: see resulting FEN and leaf evaluation

### Example: Planning from Starting Position
- Depth=4, Top K=5, Our Branch=5, Opp Branch=3
- Example plans might be:
  ```
  1. e4 c5 Nf3 d6     | leaf_eval=+0.0 | Δ=+0.0
  2. Nf3 c5 d4 cxd4  | leaf_eval=+0.0 | Δ=+0.0
  3. d4 d5 c4 e6      | leaf_eval=+0.0 | Δ=+0.0
  ...
  ```
- Material eval is flat in the opening, but capturing lines will score higher
- When you have an endgame with material imbalance, ranking becomes more obvious

## File Structure

```
before-you-move/
├── assets/                    # PNG piece sprites (72×72)
│   ├── white/
│   │   ├── white-pawn.png
│   │   └── ...
│   └── black/
├── data/
│   └── eco_san.tsv            # ECO opening database
├── chess_model.py             # Board, opening detection
├── chess_ui.py                # Tkinter GUI + planning panel
├── eval_material.py           # Material evaluator (Evaluator interface)
├── planning.py                # Planner, move ordering, search (NEW)
├── main.py                    # Entrypoint
├── requirements.txt
└── README.md
```

## Architecture

### planning.py
- **Planner**: Depth-limited search with move ordering
  - `generate_plans()`: Main entry; returns top K ranked PlanResult objects
  - `_search()`: Recursive depth-first search with branching limits
  - `_get_sorted_moves()`: Move ordering by heuristic (captures, checks, etc.)
  
- **PlanResult**: Dataclass containing:
  - `san_line`: Space-separated SAN moves
  - `leaf_eval`: Evaluation of final position
  - `delta`: Leaf eval minus root eval (for interpretability)
  - `leaf_fen`: FEN at end of plan

### chess_ui.py (Updated)
- Planning panel with controls, listbox, preview section
- `_generate_plans()`: Calls planner and displays results
- `_on_plan_selected()`: Previews selected plan

### eval_material.py
- `Evaluator`: Abstract interface for future AI models
- `MaterialEvaluator`: Simple piece counting (easy to swap later)

## How Planning Works

1. **Move Ordering** (speed optimization):
   - Captures first: avoids blunders, captures material
   - Checks second: forcing moves
   - Promotions third: high-value moves
   - Rest: alpha-beta pruning helps here

2. **Branching Control**:
   - Limits our candidate moves (default 6)
   - Limits opponent responses (default 4)
   - Keeps search tree manageable: ~6 × 4 × 6 × 4 = ~576 leaf nodes at depth 4

3. **Evaluation**:
   - Each leaf position evaluated by MaterialEvaluator
   - Ranking from root perspective: White higher-is-better, Black lower-is-better
   - Delta shows improvement vs. current position

## Performance

Typical runtimes (depth 4, defaults):
- Simple positions (no captures): ~0.2s
- Positions with captures: ~0.5–1s
- Deep searches or complex positions: 1–3s

To speed up: decrease depth, branching, or top K.

## Development Notes

- **No external engines**: Pure Python, no Stockfish dependencies
- **SAN everywhere**: All user-facing notation is SAN
- **Extensible**: Swap `MaterialEvaluator` for neural net later
- **Logging**: Terminal shows planning stats (nodes visited, time)
- **No board corruption**: All searches use `board.copy(stack=True)` safe push/pop

## Next Steps

When ready:
1. **Swap evaluator**: Implement a learned eval net (same Evaluator interface)
2. **Add probabilistic**: Opponent move distribution instead of uniform branching
3. **Deep learning**: Policy/value networks to guide search
4. All three components fit together via the existing architecture!

## Testing

Included `test_planning.py` for quick verification:
```bash
python test_planning.py
```

---

**CISC 352 Term Project | March 2026**
