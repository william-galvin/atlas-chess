import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))

def test_img_draw():
    from src.util import show_board, move_squares_to_one_hot
    from atlas_chess import Board
    import numpy as np

    moves = np.zeros(4096)
    moves[move_squares_to_one_hot(0, 8)] = .5
    moves[move_squares_to_one_hot(1, 10)] = 1

    show_board(
        board=Board("r1bqkbnr/pp1ppppp/1np5/3P4/2P5/5N2/PP2PPPP/RNBQKB1R w KQkq - 0 1"),
        path="images/board.png", 
        images_path="images",
        square_size=85,
        masked=None,
        moves=moves
    )
