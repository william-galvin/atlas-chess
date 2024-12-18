import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))

import numpy as np

from atlas_chess import Board
from src.util import reflect

def test_reflect():
    board = Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
    bitboard = np.array(board.bitboard(), dtype=np.uint8).reshape(12, 8, 8)
    assert board.turn() == "black"
    flipped = reflect(bitboard)
    print(flipped)
    assert flipped[10][7][3] == 1