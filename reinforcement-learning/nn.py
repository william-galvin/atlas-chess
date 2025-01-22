import os
import json
import matplotlib.pyplot as plt
import scipy.stats as stats
import torch
import numpy as np
import gc
from tqdm import tqdm
from atlas_chess import mcts_search, Nnue, Board

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

torch.set_default_dtype(torch.float32) 

class NNUE(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.l1 = torch.nn.Linear(40_960, 256, bias=False)
        self.l2 = torch.nn.Linear(256, 64, bias=False)
        self.l3 = torch.nn.Linear(128, 8, bias=False)
        self.l4 = torch.nn.Linear(8, 1, bias=False)

    def forward(self, x: torch.Tensor):
        x = x.reshape(-1, 40_960 * 2)

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
            w256=self.l1.weight.detach().cpu().numpy(), 
            w64=self.l2.weight.detach().cpu().numpy(),
            w8=self.l3.weight.detach().cpu().numpy(),
            w1=self.l4.weight.detach().cpu().numpy(),
        )
        board.enable_nnue(n)

class MCTS(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.nnue = NNUE()

    def forward(self, X):
        raw = self.nnue(X)

        board_eval = torch.nn.functional.tanh(0.1 * raw[:, 0])
        move_scores = (-raw[:, 1:]).softmax(dim=-1) # negative because move evaluated from other color

        raw[:, 0] = board_eval
        raw[:, 1:] = move_scores
        
        return raw


class InfiniteGamesDataset(torch.utils.data.IterableDataset):
    def __init__(self, nn):
        self.nn = nn
        self.n_games = 1
        self.n_ties = 0
        self.n_positions = 0
        self.pgn = ""

    def __iter__(self):
        while True:
            board = Board()
            self.nn.nnue.update_weights(board)
            try:
                game = play_game(
                    board, 
                    depth=np.random.randint(200, 800), 
                    max_moves=300
                )

                self.n_games += 1
                self.n_ties += abs(game[0][-1])
                for (position, moves, chosen, winner) in game:
                    self.n_positions += 1
                    board = Board(position)
                    X = torch.tensor(Nnue.features(board).reshape(1, 40_960 * 2))
                    Y = torch.tensor([winner], dtype=torch.float32)
                    for move, prob in moves:
                        board.push(move)
                        X = torch.cat([X, torch.tensor(Nnue.features(board).reshape(1, 40_960 * 2))], dim=0)
                        Y = torch.cat([Y, torch.tensor([prob], dtype=torch.float32)], dim=0)
                        board.pop()
                    yield X, Y
                
                self.pgn = pgn(game)

            except Exception as e:
                print("[WARNING] Game failed to complete: ", str(e))
                pass
    
leaky_relu = lambda x: torch.max(0.05 * x, x)

def layer_norm(x: torch.Tensor):
    mu = torch.mean(x, dim=1)[:, None]
    sigma = std_dev(x, mu)
    return (x - mu) / sigma[:, None]

def std_dev(x: torch.Tensor, mu: torch.Tensor):
    return (((x - mu) ** 2).sum(dim=-1) / x.shape[-1]).sqrt()

def play_game(board, max_moves=75, c=1.5, taus=(1, 0.1), tau_switch=30, depth=800):
    results = []
    move_count = 0
    while board.winner() is None and move_count < max_moves:
        tau = taus[0] if move_count < tau_switch else taus[1]

        moves = mcts_search(board, c, tau, depth)

        scores = np.array([score for (m, score) in moves])
        # perturb for better sampling
        scores += np.random.rand(*scores.shape) * 0.15

        probs = scores / np.linalg.norm(scores, 1)
        idx = np.random.multinomial(1, probs).nonzero()[0][0]

        normalized_moves = [(m, p) for ((m, score), p) in zip(moves, probs)]

        chosen = moves[idx][0]
        results.append((board.fen(), normalized_moves, chosen))
        board.push(chosen)

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

    gc.collect()
    return return_val

def pgn(results):
    """
    Returns UCI pgn string.
    Note many pgn views don't support this. 
    Chess.com is one that does support it
    """
    value = ""
    i = 0
    for (_, _, move, _) in results:
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
    nn.eval()
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
    nn.train()
    return pearson
    
def batch_collate(batch):
    return (
        torch.cat([X for (X, Y) in batch], dim=0),
        torch.cat([Y for (X, Y) in batch], dim=0)
    )

def main():
    if not os.path.exists("metrics"):
        os.mkdir("metrics")
    if os.path.exists("metrics/metrics.json"):
        os.remove("metrics/metrics.json")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"*** device is {device} ***")
    torch.set_default_device(device)

    nn = MCTS()
    torch.compile(nn)

    total_params = sum(p.numel() for p in nn.parameters() if p.requires_grad)
    print(f"--- trainable parameters: {total_params:,} ---")

    board = Board()
    nn.nnue.update_weights(board)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(nn.parameters(), lr=0.00005, weight_decay=0.025)

    dataset = InfiniteGamesDataset(nn)
    print("Num CPUs found:", os.cpu_count())
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=8,
        batch_size=8,
        collate_fn=batch_collate,
        multiprocessing_context='fork' if device == torch.device("mps") else None
        # num_workers=os.cpu_count(),
    )

    for (X, Y) in tqdm(dataloader):
        y_hat = nn(X)

        Y = Y.reshape(-1)
        y_hat = y_hat.reshape(-1)

        loss = loss_fn(Y, y_hat)
        loss.backward()
        optimizer.step()

        if dataset.n_games % 100 == 0:
            record_metrics({"loss": loss.item(), "not-ties": round(1 - (dataset.n_ties / dataset.n_games), 3)})
            if dataset.n_games % 1000 == 900:
                record_metrics({"stockfish_correlation": get_stockfish_correlation(nn)})

            if dataset.n_games % 2000 == 0:
                torch.save(nn.state_dict(), "nn.checkpoint")
        
        if dataset.n_games % 20 == 0:
            print()
            print(dataset.pgn)
            print(f"game number: {dataset.n_games}, % of games not ties: {round(1 - (dataset.n_ties / dataset.n_games), 3)}")
        
if __name__ == "__main__":
    main()