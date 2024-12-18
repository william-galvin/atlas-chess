
if __name__ == '__main__':
    from atlas_chess import Board, MoveGenerator
    import torch
    from util import load_model, mask_board, show_board, piece, get_config, get_move_idx
    import yaml
    import sys

    torch.set_printoptions(precision=3, sci_mode=False, linewidth=200)

    config = get_config()
    nn = load_model(config)
    nn.eval()

    # fen = "2r2rk1/pp1q1pp1/1n2p2p/3pP1NQ/P1nP1B2/2P4P/2R2PP1/5RK1 w - - 5 27"
    fen = "r4rk1/1p3pp1/4p3/3pP1pQ/2nP1B2/2P4P/2q2PP1/5RK1 w - - 0 31"
    # fen = "3r3r/1kp1b2p/1pn2pp1/p3p3/P3P2N/2P1B1P1/1P2KP1P/R2R4 w - - 0 17"

    board = Board(fen).bitboard()

    legal = torch.zeros(4096)
    for move in MoveGenerator()(Board(fen)):
        legal[get_move_idx(move)] = 1

    board_tensor = torch.permute(torch.tensor([board], dtype=int), (2, 1, 0)).reshape(-1, 64, 12)

    board_hat, legal_moves_hat, moves_hat = nn(board_tensor)
    # moves_hat = nn(board_tensor)

    print(moves_hat, moves_hat.mean())
    softmaxed = torch.nn.functional.softmax(moves_hat[:, legal == 1])
    k = 0

    played_moves_hat = torch.nn.functional.softmax(moves_hat)
    vec = torch.zeros(4096)
    for i in range(4096):
        if legal[i] == 1:
            vec[i] = softmaxed[0, k] * 10
            k += 1

    show_board(
        board=board,
        path="foo-board.png",
        masked=None,
        moves=vec
    )

    print(vec[legal == 1])

    exit(0)

    
    for m in range(64):
        masked_board, mask = mask_board(board_tensor, mask_idxs=[m], mask_percentage=None)
        # show_board(
        #     board=board, 
        #     path="foo-board.png",
        #     masked=[i for i in range(64) if not mask[0][i]]
        # )

        masked_target = board_tensor[~mask]

        if board_tensor[~mask].max().item() == 0:
            continue

        print("*" * 10, m)
        board_hat, legal_moves_hat = nn(masked_board)
        
        masked_pred = board_hat[~mask][..., :12]

        print("piece\tactual\texpected")
        for i in range(12):
            print(f"{piece(i)}\t{round(board_hat[~mask][:, i].item(), 4)}\t{board_tensor[~mask][:, i].item()}")

        top1 = (masked_target.argmax(dim=1) == masked_pred.argmax(dim=1)).mean(dtype=torch.float32).item()
        top3 = (masked_target.argmax(dim=1).unsqueeze(1) == 
                    torch.topk(masked_pred, k=3).indices).any(dim=1).mean(dtype=torch.float32).item()
        
        if top1 == 0.0:
            print("Not top 1")
            print("expected: ", masked_target.argmax(dim=1))
            print("actual: ", masked_pred.argmax(dim=1))
            print(masked_pred)