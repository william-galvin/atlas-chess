### Installation
```
pip install atlas-chess
```

### Usage
```python
from atlas_chess import Board, MoveGenerator

move_generator = MoveGenerator()
board = Board("2r2rk1/pp1q1pp1/1n2p2p/3pP1NQ/P1nP1B2/2P4P/2R2PP1/5RK1 w - - 5 27")
move_generator(board)
>>> ['g1h1', 'g1h2', ..., 'h5f7']
```