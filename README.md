# Before You Move - Chess Workbench

Interactive chess workbench with automated planning, probabilistic first-move inference, PNG pieces, SAN notation, and ECO opening detection.

**Day 1:** Visual board + opening detection  
**Day 2:** Automated planning with ranked multi-move lines  
**Day 3:** Probabilistic first-move inference (this update)  
**Later:** Deep learning  

## Features

### Board & Interaction
- **Visual Board**: 8x8 with 72x72 PNG pieces (or unicode fallback)
- **Click-to-Move**: Select piece, highlight legal destinations, click to move
- **Move Input**: SAN (e4, Nf3, O-O) or UCI (e2e4)
- **Move History**: PGN-style display (1. e4 e5 2. Nf3 Nc6)
- **Opening Detection**: ECO database with longest-prefix matching

### Planning
- **Automated Multi-Move Planning**: Generate and rank candidate multi-move lines
- **Depth-Limited Search**: Configurable depth (ply), default 4
- **Move Ordering**: Captures > Checks > Promotions > others (for speed)
- **Branching Control**: Limit our moves and opponent responses separately
- **Top K Results**: Display top ranked plans with material evaluation
- **Plan Preview**: Click a plan to see resulting FEN and leaf evaluation

### Probabilistic Inference (NEW)
- **First-Move Near-Best Estimation**: Estimates whether a candidate move stays within a chosen margin of the best rollout outcome
- **Strong Opponent Model**:
  - Generate all legal replies
  - Score from opponent perspective
  - Keep top-K replies
  - Weighted sampling favoring stronger replies (default rank bias 0.6 / 0.3 / 0.1)
- **Success Definition**:
  - Roll out each candidate first move from the root-player perspective
  - Compute the best expected leaf score among the candidate first moves
  - `success = 1` if `leaf_score >= best_candidate_expected_score - success_threshold`
  - Default threshold: `0.5` pawns, meaning "within half a pawn of the best candidate"
- **Configurable Controls**:
  - Candidate top N
  - Simulations
  - Horizon
  - Opponent top K
  - Near-best margin
- **Output per first move**:
  - SAN, UCI, near-best probability, average leaf score, average delta, simulation count
- **Safe Preview**:
  - Clicking a result previews move/FEN/eval on a copied board without mutating the real game state

## Setup

### Install Dependencies
```bash
pip install -r requirements.txt
```

Installs: `python-chess`, `Pillow`, `torch`

### Run
```bash
python main.py
```

## Usage

### Making Moves
1. **Click-to-Move**: Click piece -> legal squares highlighted -> click destination
2. **Text Input**: Type `e4`, `Nf3`, `O-O` (SAN) or `e2e4` (UCI) -> click Play Move or press Enter
3. **Undo/Reset**: Undo last move or reset to start

### Planning
1. Set depth/top-k/branching controls.
2. Click **Generate Plans**.
3. Click any plan for FEN + eval preview.

### Probabilistic Inference
1. Set controls in **Probabilistic Inference** panel:
   - Candidate Top N (default 8)
   - Simulations (default 50)
   - Horizon (default 4 ply after first move)
   - Opp Top K (default 3)
   - Near-Best Margin (default 0.5 pawns)
2. Click **Estimate Move Success**.
3. Read ranked SAN-first results such as:
   ```
   1. Nf3   | P(near-best)=0.74 | avg_delta=+0.8 | avg_leaf=+1.2
   2. e4    | P(near-best)=0.61 | avg_delta=+0.5 | avg_leaf=+0.9
   3. d4    | P(near-best)=0.44 | avg_delta=+0.2 | avg_leaf=+0.6
   ```
4. Click a row to preview the move on a copied board (resulting FEN + immediate eval).

Terminal logs include:
- number of candidate moves
- simulations per move
- runtime
- best move by near-best probability

## File Structure

```text
BeforeYouMove-AI/
|-- assets/
|   |-- white/
|   `-- black/
|-- data/
|   `-- eco_san.tsv
|-- chess_model.py
|-- chess_ui.py
|-- eval_material.py
|-- planning/
|   |-- __init__.py
|   `-- planner.py
|-- probabilistic/
|   |-- __init__.py
|   `-- inference.py
|-- main.py
|-- requirements.txt
`-- README.md
```

## Architecture Notes

### `eval_material.py`
- Evaluator returns centipawns from **White** perspective.
- Uses material as the base signal, then adds lightweight heuristics for bishop pair, mobility, castling rights, and middlegame king safety.

### `planning/`
- Provides heuristic move ordering and ranked multi-ply plan generation.
- Uses separate branching limits for our turns and the opponent's turns.

### `probabilistic/`
- `MoveProbResult` dataclass for per-first-move estimates.
- `estimate_first_move_successes(...)` main API.
- `sample_strong_opponent_reply(...)` top-K weighted opponent sampling.
- `run_rollout_after_first_move(...)` short rollout and leaf scoring.
- Explicit root-perspective conversion handles both White-to-move and Black-to-move roots correctly.
- "Success" means finishing within a chosen margin of the best candidate move found, not simply improving on the current evaluation.

### `deep_learning/`
- Board encoding includes piece planes plus side-to-move, castling-rights, and material-balance context planes.
- Those features give the model access to obvious strategic signals that plain piece maps often miss.
- Training now supports:
  - concatenating multiple CSVs
  - optional horizontal-mirror augmentation
  - a hybrid CNN + scalar-feature model
  - richer data generation modes including engine-guided and mixed sampling

## Improving The Neural Model

The original neural evaluator was limited by two things:
- it only saw piece placement
- it was trained on a very small dataset

The updated pipeline addresses both.

1. Generate more labeled positions.
```bash
python deep_learning/generate_dataset.py --output_csv data/chess_positions_large.csv --num_positions 5000 --stockfish_path path/to/stockfish --mode mixed --pgn_path path/to/games.pgn --depth 10
```

2. Train on all available CSVs with augmentation.
```bash
python deep_learning/train.py --csv data/chess_positions.csv --extra_csvs data/chess_positions_large.csv --epochs 25 --batch_size 64 --lr 0.001 --augment_mirror
```

3. Replace the saved weights at `deep_learning/chess_value_model.pt`.

The model now learns from:
- board planes
- side to move
- castling rights
- explicit piece counts for both sides
- material totals and balance
- mobility
- check status
- game progress signals

## Sample Test Positions (FEN)

1. **Open tactical development (Black to move)**  
   FEN: `r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 2 3`  
   Expect stronger development/defensive Black replies to produce better success probability than loosening pawn moves.

2. **King-and-pawn endgame (White to move)**  
   FEN: `8/8/3k4/3P4/3K4/8/8/8 w - - 0 1`  
   Expect candidate moves that preserve/push the passed pawn and maintain king support to rank highest.

## Development Notes

- No external engine dependency is required to run the app itself.
- Stockfish is only needed for optional dataset generation and neural-model retraining.
- SAN is primary user-facing notation.
- Rollouts use safe board copies and push/pop patterns to avoid board corruption.
- Runs with `python main.py`.

---

**CISC 352 Term Project | March 2026**
