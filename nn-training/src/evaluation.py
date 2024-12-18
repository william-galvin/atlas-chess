import os
import json 
import torch

import matplotlib.pyplot as plt
import numpy as np
import torch.functional

from util import mask_board, format_dict, piece, get_piece_class_weights

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
torch.set_default_device(device)

binary_cross_entropy = torch.nn.BCEWithLogitsLoss(reduction="mean")
weighted_binary_cross_entropy = torch.nn.BCEWithLogitsLoss(reduction="mean", weight=torch.tensor([4096]))
# weighted_binary_cross_entropy8912 = \
#     torch.nn.BCEWithLogitsLoss(reduction="mean", weight=torch.tensor([8192 * 4]))
cross_entropy = torch.nn.CrossEntropyLoss()

def _process_batch_pretrain(board, legal_moves, played_moves, model, epoch, eval):
    board = torch.permute(board, (0, 2, 3, 1)).reshape(-1, 64, 12)

    masked_board, mask = mask_board(board, mask_percentage=np.random.randint(3, 8)/64)

    board_hat, legal_moves_hat, played_moves_hat = model(masked_board)

    masked_target = board[~mask]
    masked_pred = board_hat[~mask][..., :12]

    loss_mask = binary_cross_entropy(masked_pred, masked_target)

    loss_legal_moves = \
        weighted_binary_cross_entropy(legal_moves_hat, legal_moves) / 1024


    legal_mask = legal_moves == 1
    played_moves_hat[~legal_mask] = -torch.inf
    played_labels = played_moves.argmax(dim=1)

    loss_played_moves = \
        cross_entropy(played_moves_hat, played_labels)
        # weighted_binary_cross_entropy8912(played_moves_hat[legal_mask], played_moves[legal_mask]) / (2048 * 4)
    
    loss = loss_mask + \
           loss_legal_moves + \
           loss_played_moves * ((1 + epoch) / 10)

    empty_squares = masked_target.max(dim=1)[0] != 0

    masked_target = masked_target[empty_squares].reshape(-1, 12)
    masked_pred = masked_pred[empty_squares].reshape(-1, 12)

    if masked_target.shape[0] == 0:
        top1 = None 
        top3 = None 
    else:
        top1 = (masked_target.argmax(dim=1) == masked_pred.argmax(dim=1)).mean(dtype=torch.float32).item()
        top3 = (masked_target.argmax(dim=1).unsqueeze(1) == 
                    torch.topk(masked_pred, k=3).indices).any(dim=1).mean(dtype=torch.float32).item()

    if eval:
        move_acc = [] 
        for b in range(legal_moves.shape[0]):
            vec = legal_moves[b]
            n_moves = int(vec.sum().item())
            topk, topk_idxs = legal_moves_hat[b].topk(k=n_moves)
            move_acc.append(vec[topk_idxs].mean().item())
        move_acc = np.mean(move_acc)
        
        piece_accuracy = {}
        for p in range(12):
            mask = masked_target[:, p] == 1

            if mask.sum() == 0:
                piece_accuracy[piece(p)] = None 
            else:
                piece_accuracy[piece(p)] = (masked_pred[mask].argmax(dim=1) == p).mean(dtype=torch.float32).item()


        
        target_indices = torch.argmax(played_moves, dim=1) 
        sorted_indices = torch.argsort(played_moves_hat, dim=1, descending=True)  
        target_ranks = (sorted_indices == target_indices.unsqueeze(1)).nonzero(as_tuple=True)[1]
        scores = 1.0 / (target_ranks.float() + 1.0)
        played_move_acc = scores.mean().item()

        # played_move_acc = \
        #     (played_moves.argmax(dim=1) == played_moves_hat.argmax(dim=1)).mean(dtype=torch.float32).item()

        # for b in range(played_moves.shape[0]):
        #     mask = legal_moves[b] == 1

        #     vec = played_moves[b, mask]

        #     n_moves = int(vec.sum().item())
        #     topk, topk_idxs = played_moves_hat[b][mask].topk(k=n_moves)

        #     played_move_acc.append(vec[topk_idxs].mean().item())
        # played_move_acc = np.mean(played_move_acc)
    else:
        move_acc = None 
        played_move_acc = None
        piece_accuracy = {piece(p): None for p in range(12)}

    return loss, {
        "loss": {
            "mask_loss": loss_mask.item(), 
            "legal_loss": loss_legal_moves.item(),
            "played_loss": loss_played_moves.item(),
            "loss": loss.item(),
        },
        "accuracy": {
            "top1": top1,
            "top3": top3,
            "legal_move": move_acc,
            "played_move": played_move_acc,
        },
        "_piece-accuracy": piece_accuracy,
    }


def _process_batch(board, legal_moves, played_moves, model, eval = False):
    board = torch.permute(board, (0, 2, 3, 1)).reshape(-1, 64, 12)

    played_moves_hat = model(board)
    mask = legal_moves == 1

    loss = weighted_binary_cross_entropy(played_moves_hat[mask], played_moves[mask])

    if eval:
        move_acc = [] 
        for b in range(played_moves.shape[0]):
            mask = legal_moves[b] == 1

            vec = played_moves[b, mask]

            n_moves = int(vec.sum().item())
            topk, topk_idxs = played_moves_hat[b][mask].topk(k=n_moves)

            move_acc.append(vec[topk_idxs].mean().item())
        move_acc = np.mean(move_acc)
    else:
        move_acc = None

    return loss, {
        "loss": {
            "loss": loss.item(),
        },
        "accuracy": {
            "move": move_acc,
        },
    }



def process_batch(board, legal_moves, played_moves, model, epoch, pretrain, eval = False):
    if pretrain:
        return _process_batch_pretrain(board, legal_moves, played_moves, model, epoch, eval)
    else:
        return _process_batch(board, legal_moves, played_moves, model, eval)


def get_validation_metrics(model, data, pretrain):
    means = {}
    for batch in data:
        loss, metrics = process_batch(*batch, model, 1, pretrain, eval=True)
        for group, metrics_group in metrics.items():
            if group not in means:
                means[group] = {}
            for k, v in metrics_group.items():
                if k not in means[group]:
                    means[group][k] = []
                means[group][k].append(v)
    return {
        group: {
            k: np.mean([x for x in v if x is not None]) for k, v in metrics.items()
        }
        for group, metrics in means.items()
    }


def do_validation(
    run,
    metrics,
    pbar,
    model,
    validation_data,
    pretrain,
    runs_path="../runs"
):
    pbar.set_description(format_dict(metrics), refresh=False)

    if not os.path.exists(f"{runs_path}/{run}"):
        os.mkdir(f"{runs_path}/{run}")
        
    if os.path.exists(f"{runs_path}/{run}/metrics.json"):
        with open(f"{runs_path}/{run}/metrics.json", "r") as f:
            running_metrics = json.load(f)
    else:
        running_metrics = {}

    model.eval()
    validation_metrics = get_validation_metrics(model, validation_data, pretrain)
    model.train()

    for (group, metrics_group) in metrics.items():
        if group not in running_metrics:
            running_metrics[group] = {}
        for k, v in metrics_group.items():
            if k.endswith("_valid"):
                continue
            if k not in running_metrics[group]:
                running_metrics[group][k] = []
                running_metrics[group][f"{k}_valid"] = []
            
            if v is not None:
                running_metrics[group][k].append(v)
            if validation_metrics[group][k] is not None:
                running_metrics[group][f"{k}_valid"].append(validation_metrics[group][k])

            plt.plot(running_metrics[group][k], label=f"{k}_train")
            plt.plot(running_metrics[group][f"{k}_valid"], label=f"{k}_valid")
        plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"{runs_path}/{run}/{group}.png")
        plt.clf()

    with open(f"{runs_path}/{run}/metrics.json", "w") as f:
        json.dump(running_metrics, f)


def do_test(run, model, data, pretrain, runs_path="../runs"):
    if not pretrain:
        return

    model.eval()

    confusion = {x: [] for x in range(13)}
    all_legal_move_coeffs = []
    all_played_move_coeffs = []

    for (board, legal_moves, played_moves) in data:
        board = torch.permute(board, (0, 2, 3, 1)).reshape(-1, 64, 12)
        masked_board, mask = mask_board(board, mask_percentage=.05)
        board_hat, legal_moves_hat, played_moves_hat = model(masked_board)

        all_legal_move_coeffs += legal_moves_hat.flatten().tolist()
        all_played_move_coeffs += played_moves_hat[legal_moves == 1].flatten().tolist()
    
        masked_target = board[~mask]
        masked_pred = board_hat[~mask][:, :12]

        for b in range(masked_target.shape[0]):
            real = masked_target[b]
            hat = torch.nn.functional.sigmoid(masked_pred[b])

            if torch.max(real) > 0:
                confusion[torch.argmax(real).item() + 1].append(hat.detach().cpu().numpy())
            else:
                confusion[0].append(hat.detach().cpu().numpy())

    model.train()            

    confusion = {k: np.mean(np.vstack(v), axis=0) for k, v in confusion.items() if v != [] }
    arr = np.zeros((13, 12))
    for k in confusion:
        arr[k] = confusion[k]

    plt.imshow(arr, cmap="hot")
    plt.savefig(f"{runs_path}/{run}/confusion.png")
    plt.clf()

    all_legal_move_coeffs = torch.clamp(torch.tensor(all_legal_move_coeffs, device="cpu"), 0, 1)
    plt.hist(all_legal_move_coeffs, bins=np.arange(0, 1.05, .05), log=True)
    plt.savefig(f"{runs_path}/{run}/legal_moves.png")
    plt.clf()

    all_played_move_coeffs = torch.clamp(torch.tensor(all_played_move_coeffs, device="cpu"), -10, 10)
    plt.hist(all_played_move_coeffs, bins=50, log=True)
    plt.savefig(f"{runs_path}/{run}/played_moves.png")
    plt.clf()

