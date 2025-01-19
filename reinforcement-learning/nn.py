import os
import json
import matplotlib.pyplot as plt
import scipy.stats as stats
import torch
import numpy as np
from atlas_chess import mcts_search, Nnue, Board

class NNUE(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.l1 = torch.nn.Linear(40_960, 256, bias=False)
        self.l2 = torch.nn.Linear(256, 64, bias=False)
        self.l3 = torch.nn.Linear(128, 8, bias=False)
        self.l4 = torch.nn.Linear(8, 1, bias=False)

    def forward(self, x: torch.Tensor):
        x1, x2 = x.split(40_960, dim=-1)
        x1, x2 = x1.to_sparse_coo(), x2.to_sparse_coo()

        x1 = leaky_relu(layer_norm(self.l1(x1)))
        x2 = leaky_relu(layer_norm(self.l1(x2)))

        x1 = leaky_relu(layer_norm(self.l2(x1)))
        x2 = leaky_relu(layer_norm(self.l2(x2)))

        x = torch.concatenate([x1, x2], dim=-1)
        x = leaky_relu(layer_norm(self.l3(x)))

        return self.l4(x)
    
    def update_weights(self, board):
        n = Nnue()
        n.set_weights(
            w256=self.l1.weight.detach().numpy(), 
            w64=self.l2.weight.detach().numpy(),
            w8=self.l3.weight.detach().numpy(),
            w1=self.l4.weight.detach().numpy(),
        )
        board.enable_nnue(n)

class MCTS(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.nnue = NNUE()

    def forward(self, game_results):
        batch = []
        n_moves = []
        for ((fen,), moves, _, _) in game_results:
            board = Board(fen)
            features = Nnue.features(board)
            batch.append(features)

            n_moves.append(len(moves))
            for ((move,), _) in moves:
                board.push(move)
                batch.append(Nnue.features(board))
                board.pop()

        batch_results = self.nnue.forward(torch.tensor(np.array(batch)))
        forward_results = torch.tensor([])
        for n in n_moves:
            board_eval = torch.nn.functional.tanh(0.1 * batch_results)[0]
            move_scores = -batch_results[1:n+1] # negative because move evaluated from other color
            move_scores = torch.nn.functional.softmax(move_scores.T)[0]
            batch_results = batch_results[n+1:]
            
            forward_results = torch.hstack([forward_results, move_scores, board_eval])

        assert batch_results.shape[0] == 0
        return forward_results


class InfiniteGamesDataset(torch.utils.data.IterableDataset):
    def __init__(self, nn):
        self.nn = nn

    def __iter__(self):
        while True:
            board = Board()
            self.nn.nnue.update_weights(board)
            yield play_game(board)

    
leaky_relu = lambda x: torch.max(0.05 * x, x)

def layer_norm(x: torch.Tensor):
    mu = torch.mean(x, dim=1)[:, None]
    sigma = std_dev(x, mu)
    return (x - mu) / sigma[:, None]

def std_dev(x: torch.Tensor, mu: torch.Tensor):
    return (((x - mu) ** 2).sum(dim=-1) / x.shape[-1]).sqrt()

def play_game(board, max_moves=75, c=3, taus=(1, 0.1), tau_switch=30, depth=800):
    results = []
    move_count = 0
    while board.winner() is None and move_count < max_moves:
        tau = taus[0] if move_count < tau_switch else taus[1]
        moves = mcts_search(board, c, tau, depth)
        best_move = sorted(moves, key=lambda x: x[1], reverse=True)[0][0]
        results.append((board.fen(), moves, best_move))
        board.push(best_move)
        move_count += 1

    win_str = board.winner()
    if win_str == "white":
        winner = 1
    elif win_str == "black":
        winner = -1
    elif win_str == "tie" or win_str is None:
        winner = 0
    else:
        raise ValueError(win_str)
    
    # winner is from perspective of side to move
    return_val = []
    for fen, m, best in results:
        return_val.append((fen, m, best, winner))
        winner *= -1

    return return_val

def pgn(results):
    """
    Returns UCI pgn string.
    Note many pgn views don't support this. 
    Chess.com is one that does support it
    """
    value = ""
    i = 0
    for (_, _, (move,), _) in results:
        if i % 2 == 0:
            value += str(i // 2 + 1) + ". "
        value += move + " "
        i += 1
    return value

def record_metrics(new_metrics):
    if os.path.exists("metrics/metrics.json"):
        metrics = json.loads(open("metrics/metrics.json").read())
    else:
        metrics = {}
    for k, v in new_metrics.items():
        if k not in metrics:
            metrics[k] = []
        metrics[k].append(v)
    with open("metrics/metrics.json", "w") as f:
        json.dump(metrics, f)

    for k, v in metrics.items():
        plt.plot(v)
        plt.savefig(f"metrics/{k}.png")
        plt.clf()
    
def get_stockfish_correlation(nn):
    sf = []
    us = []
    with open("test_data/random_evals_1000.csv") as f:
        lines = f.readlines()
    for line in lines:
        fen, sf_eval = line.split(",")
        sf.append(int(sf_eval))
        board = Board(fen)
        color = 1 if board.turn() == "white" else -1
        nn.nnue.update_weights(board)
        us.append(board.forward() * color)

    plt.scatter(x=sf, y=us)
    plt.xlabel("stockfish")
    plt.ylabel("atlas")

    pearson, p = stats.pearsonr(sf, us)
    plt.savefig("metrics/sf_correlation.png")
    plt.clf()
    return pearson
    
    

def main():
    nn = MCTS()
    torch.compile(nn)

    total_params = sum(p.numel() for p in nn.parameters() if p.requires_grad)
    print(f"--- trainable parameters: {total_params:,} ---")

    board = Board()
    nn.nnue.update_weights(board)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(nn.parameters(), lr=0.00005, weight_decay=0.05)

    dataset = InfiniteGamesDataset(nn)
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=8)

    for i, game in enumerate(dataloader):
        y = []
        for (_, moves, _, result) in game:
            moves = [eval.item() for (m, eval) in moves]
            y += moves + [result.item()]
        y = torch.tensor(y)
        y_hat = nn(game)

        loss = loss_fn(y, y_hat)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            record_metrics({"loss": loss.item(), "stockfish_correlation": get_stockfish_correlation(nn)})
            print()
            print(pgn(game))
            print(f"game number: {i}")

        print(".", end="", flush=True)

if __name__ == "__main__":
    main()