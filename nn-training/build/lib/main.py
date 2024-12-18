from dataloader import ChessDataset
from torch.utils.data import DataLoader
from torch.optim import *
import torch
from tqdm import tqdm

from util import load_model, checkpoint, get_config
from evaluation import do_validation, process_batch, do_test

if __name__ == "__main__":
    config = get_config()
    run_name = config["run_name"]
    pretrain = config["pretrain"]

    batch_size = config["batch_size"]

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"*** device is {device} ***")
    torch.set_default_device(device)

    train = ChessDataset("../data/data.hdf5", "train")
    test = ChessDataset("../data/data.hdf5", "test", max_n=500)
    valid = ChessDataset("../data/data.hdf5", "valid", max_n=500)
    
    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
    test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=False)
    valid_dataloader = DataLoader(valid, batch_size=batch_size, shuffle=False)

    nn = load_model(config)
    # if not pretrain:
    #     nn.requires_grad_(False)
    #     nn.best_move_guesser.requires_grad_(True)

    optimizer_class = globals()[config["optimizer"]]
    optimizer = optimizer_class(params=nn.parameters(), **config["optimizer_kwargs"])

    lr = config["optimizer_kwargs"]["lr"]
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr/100, max_lr=lr*2, step_size_up=10_000)

    mask_losses = []
    legal_losses = []
    losses = []

    for epoch in range(config["epochs"]):
        print(f"**** starting epoch {epoch + 1} ****")
        # if epoch == 4 and not pretrain:
        #     nn.requires_grad_(True)
        for i, batch in enumerate(pbar := tqdm(train_dataloader, miniters=100)):

            optimizer.zero_grad()

            loss, metrics = process_batch(*batch, nn, pretrain=pretrain, epoch=epoch)

            loss.backward()
            optimizer.step()
            scheduler.step()

            if (i + 1) % config["valid_frequency"] == 0:
                do_validation(
                    run=run_name,
                    metrics=metrics,
                    pbar=pbar,
                    model=nn,
                    validation_data=valid_dataloader,
                    pretrain=pretrain
                )
                
            if (i + 1) % config["checkpoint_frequency"] == 0:
                do_test(run_name, nn, test_dataloader, pretrain)
                checkpoint(run_name, nn)

        
    # TODO 
    #   - pip install (automatically make the pgn-extract tool)
    #   - evaluation suite
    #       - accuracy by piece type over time
    #           - In eval, make another graph for piece type
    #       - accuracy by move number over time
    #           - In test, make plot move number vs mean accuracy up to 100 moves
    #   - memorize small dataset    
    #   - learning rate scheduler (cyclical? sin? reduce on plateau? Warmup?)
    #   - Train only on unique positions
    #       - And redo train/test/eval splits to reflect that
    #       - Maybe store positions in DB not as fens?
    #       - When duplicate positions, there may be many move labels -- combine them, legal move style
    #       - When training resumes, don't start from beginning
    #           - key into the DB with idx, not fen?