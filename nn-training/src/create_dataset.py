from dataloader import IterableChessDataset
import numpy as np
from tqdm import tqdm
import torch
import h5py

from util import position_dt, entry_to_np, np_to_entry, get_config

def make_data(config, runs_path="../runs", max_items = None):
    data = IterableChessDataset(config["train_data"], numpy=True)
    p_test, p_valid = config["test_data"], config["valid_data"]

    n = len(data)

    with h5py.File("../data/data.hdf5", "w") as f:
        f.create_dataset("train", (1,), dtype=position_dt, maxshape=(None,))
        f.create_dataset("test", (1,), dtype=position_dt, maxshape=(None,))
        f.create_dataset("valid", (1,), dtype=position_dt, maxshape=(None,))

        counts = {
            "train": 0,
            "test": 0,
            "valid": 0
        }

        j = 0
        for data_item in tqdm(data, total=max_items if max_items is not None else n):

            j += 1
            if (max_items is not None and j > max_items):
                break

            position, move, legal_move_vec, bitboard, fen = data_item
            
            db = np.random.choice(["train", "test", "valid"], p=[1 - p_test - p_valid, p_test, p_valid])
            
            i = counts[db]
            if i % 10_000 == 0:
                f[db].resize((i + 10_000,))

            f[db][i - 1] = entry_to_np(fen, move, legal_move_vec)

            counts[db] += 1

        print(counts)

        f["train"].resize((counts["train"],))
        f["test"].resize((counts["test"],))
        f["valid"].resize((counts["valid"],))

            
    

    # i = 0
    # for data in tqdm(data, total=max_items if max_items is not None else len(data)):
    #     i += 1
    #     if (max_items is not None and i > max_items):
    #         break
    #     position, move, legal_move_vec, bitboard, fen = data
    #     fen = " ".join(fen.split()[:-2]) + " - -"

    #     mem_db = np.random.choice([train_db, test_db, valid_db], p=[1 - p_test - p_valid, p_test, p_valid])

    #     if not fen in mem_db:
    #         mem_db[fen] = torch.zeros(64 * 64, dtype=torch.float32), legal_move_vec
        
    #     played, legal = mem_db[fen]
    #     mem_db[fen] = played.logical_or(move), legal

    # train_buff = np.zeros(len(train_db), dtype=position_dt)
    # test_buff = np.zeros(len(test_db), dtype=position_dt)
    # valid_buff = np.zeros(len(valid_db), dtype=position_dt)

    # for np_buff, mem_db in [(train_buff, train_db), (test_buff, test_db), (valid_buff, valid_db)]: 
    #     for i, (fen, (played, legal)) in enumerate(mem_db.items()):
    #         np_buff[i] = entry_to_np(fen, played.numpy(), legal.bool().numpy())
    
    # for buff, name in zip([train_buff, test_buff, valid_buff], ["train", "test", "valid"]):
    #     np.save(f"../data/{name}", buff)
    #     print(f"Saved {name} with {len(buff)} items")
    

if __name__ == "__main__":
    make_data(get_config(), max_items=15_000_000)
