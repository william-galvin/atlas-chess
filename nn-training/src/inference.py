from atlas_chess import Board, MoveGenerator
from util import idx_to_move, load_model, reflect_move

import yaml
import torch

def get_best_move(board, nn, move_generator):
    reflect = False
    if board.turn() == "black":
        board.reflect()
        reflect = True

    legal_moves = move_generator(board)

    position = torch.tensor(board.bitboard(), dtype=torch.float32).reshape(12, 8, 8)
    position = torch.permute(position, (1, 2, 0)).reshape(-1, 64, 12)
    
    _, _, moves = nn(position)
    scores, idxs = moves.sort(descending=True)
    print(scores)
    
    for idx in idxs.flatten().tolist():
        if idx_to_move(idx) in legal_moves:
            best_move = idx_to_move(idx)
            break
    else:
        raise ValueError("no legal moves")
    
    if reflect:
        best_move = reflect_move(best_move)

    # TODO:
    # - check if promotion
    # - check if castling
    return best_move

def main():
    with open("runs/run-1/config.yaml") as f:
        config = yaml.safe_load(f)
    nn = load_model(config, "runs")
    nn.eval()

    fen = "2r2rk1/pp1q1pp1/1n2p2p/3pP1NQ/P1nP1B2/2P4P/R4PP1/5RK1 b - - 6 27"
    board = Board(fen)
    move_generator = MoveGenerator()

    print(get_best_move(board, nn, move_generator))

    



if __name__ == "__main__":
    main()