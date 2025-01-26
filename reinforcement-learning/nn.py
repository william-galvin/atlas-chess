import os
import json
import matplotlib.pyplot as plt
import scipy.stats as stats
import torch
import numpy as np
import gc
import time
import torch.distributed
import torch.multiprocessing.spawn
import atexit
from tqdm import tqdm
from atlas_chess import mcts_search, Nnue, Board
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
    def __init__(self, nn, workers=None):
        self.nn = nn
        self.n_positions = 0

        self.workers = workers or os.cpu_count() // 2
        self.queue = torch.multiprocessing.Manager().Queue()
        self.stats = torch.multiprocessing.Manager().dict()
        self.stats["n_games"] = 0
        self.stats["n_ties"] = 0
        # pool = torch.multiprocessing.Pool(processes=self.workers)

        # print(f"made pool with {self.workers} workers")
        # for worker in range(self.workers):
        #     pool.apply_async(
        #         InfiniteGamesDataset.generate, 
        #         (self.nn, self.queue, self.workers, self.stats)
        #     )

    def generate(nn, queue, workers, statistics):
        torch.set_num_threads(1)
        while True:
            if queue.qsize() > workers * 300:
                time.sleep(1)
                continue

            board = Board()
            nn.nnue.update_weights(board)
            
            try:
                game = play_game(
                    board, 
                    depth=np.random.randint(200, 800), 
                    max_moves=300
                )

                statistics["n_games"] += 1
                statistics["n_ties"] += abs(game[0][-1])
                statistics["last_pgn"] = pgn(game)

                for (position, moves, chosen, winner) in game:
                    board = Board(position)
                    X = torch.tensor(Nnue.features(board).reshape(1, 40_960 * 2))
                    Y = torch.tensor([winner], dtype=torch.float32)
                    for move, prob in moves:
                        board.push(move)
                        X = torch.cat([X, torch.tensor(Nnue.features(board).reshape(1, 40_960 * 2))], dim=0)
                        Y = torch.cat([Y, torch.tensor([prob], dtype=torch.float32)], dim=0)
                        board.pop()
                    queue.put((X, Y))

            except Exception as e:
                print("[WARNING] Game failed to complete: ", str(e))
                pass


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

                self.stats["n_games"] += 1
                self.stats["n_ties"] += abs(game[0][-1])
                self.stats["last_pgn"] = pgn(game)

                for (position, moves, chosen, winner) in game:
                    board = Board(position)
                    X = torch.tensor(Nnue.features(board).reshape(1, 40_960 * 2))
                    Y = torch.tensor([winner], dtype=torch.float32)
                    for move, prob in moves:
                        board.push(move)
                        X = torch.cat([X, torch.tensor(Nnue.features(board).reshape(1, 40_960 * 2))], dim=0)
                        Y = torch.cat([Y, torch.tensor([prob], dtype=torch.float32)], dim=0)
                        board.pop()
                    yield (X, Y)

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

        scores = np.array([score if score == score else 0.0 for (m, score) in moves]) # check for nans?
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
    c = []
    with open("test_data/random_evals_1000.csv") as f:
        lines = f.readlines()
    for line in lines:
        fen, sf_eval = line.split(",")
        sf.append(int(sf_eval))
        board = Board(fen)
        color = 1 if board.turn() == "white" else -1
        c.append(max(0, color))
        nn.nnue.update_weights(board)
        us.append(board.forward() * color)

    plt.scatter(x=sf, y=us, c=c)
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

def get_norm(nn, grad=False):
    total_norm = 0
    for p in nn.parameters():
        if grad:
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
            else:
                continue  # Skip if no gradient is available
        else:
            param_norm = p.detach().norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def train(rank, world_size, device):
    torch.distributed.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    torch.set_num_threads(1)

    nn = MCTS()
    nn.to(device)
    torch.compile(nn)

    ddp_nn = torch.nn.parallel.DistributedDataParallel(nn)

    if rank == 0:
        total_params = sum(p.numel() for p in nn.parameters() if p.requires_grad)
        print(f"--- trainable parameters: {total_params:,} ---")

    dataset = InfiniteGamesDataset(nn, workers=1)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(nn.parameters(), lr=0.00005, betas=(0.9, 0.95), weight_decay=0.01)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=2,
        batch_size=32,
        collate_fn=batch_collate,
    )

    for (X, Y) in tqdm(dataloader):
        if torch.isnan(X).any() or torch.isnan(Y).any():
            print("Warning: data has NaNs")
            continue

        optimizer.zero_grad()

        X, Y = X.to(device), Y.to(device)
        y_hat = ddp_nn(X)

        Y = Y.reshape(-1)
        y_hat = y_hat.reshape(-1)

        loss = loss_fn(Y, y_hat)
        if torch.isnan(loss):
            print("[WARNING] loss is NaN")
            continue
        loss.backward()

        torch.nn.utils.clip_grad_norm_(nn.parameters(), max_norm=1)
        optimizer.step()

        if rank == 0:
            if dataset.stats["n_games"] % 10 == 0:
                record_metrics({
                    "loss": loss.item(),
                    "not-ties": round(1 - (dataset.stats["n_ties"] / dataset.stats["n_games"]), 3),
                    "grad-norm": get_norm(nn, grad=True),
                    "weight-norm": get_norm(nn, grad=False),
                })
                if dataset.stats["n_games"] % 100 == 0:
                    record_metrics({"stockfish_correlation": get_stockfish_correlation(nn)})

                if dataset.stats["n_games"] % 100 == 0:
                    torch.save(nn.state_dict(), "nn.checkpoint")
            
            if dataset.stats["n_games"] % 20 == 0:
                print()
                print(dataset.stats["last_pgn"])
                print(f"game number: {dataset.stats['n_games']}, % of games not ties: {round(1 - (dataset.stats['n_ties'] / dataset.stats['n_games']), 3)}")
        
        torch.distributed.barrier()        

def main():
    if not os.path.exists("metrics"):
        os.mkdir("metrics")
    if os.path.exists("metrics/metrics.json"):
        os.remove("metrics/metrics.json")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"--- device is {device} ---")
    torch.set_default_device(device)

    torch.multiprocessing.set_start_method("spawn", force=True)

    cpus = os.cpu_count()
    world_size = cpus // 2
    print(f"Num CPUs found: {cpus}, using {world_size} world size")

    torch.multiprocessing.spawn(train, args=(world_size, device), nprocs=world_size, join=True)


def cleanup():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

atexit.register(cleanup)

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    main()