import numpy as np
from torch.utils.data import IterableDataset, Dataset, get_worker_info
import torch
import h5py

from util import iter_positions, count_positions, get_move_idx, bitboard_from_position, np_to_entry

from atlas_chess import Board

class ChessDataset(Dataset):
    """
    Provides random access to chess positions
    """
    def __init__(self, data_path, db, max_n = None) -> None:
        super().__init__()

        f = h5py.File(data_path, "r")
        self.data = f[db]

        self.max_n = max_n


    def __len__(self):
        if self.max_n is not None:
            return self.max_n
        return len(self.data) - 1
    

    def __getitem__(self, index):
        fen, legal, played = np_to_entry(self.data[index])
        items = (
            torch.tensor(Board.bitboard_from_fen(fen), dtype=torch.float32).reshape(12, 8, 8),
            torch.tensor(legal, dtype=torch.float32),
            torch.tensor(played, dtype=torch.float32)
        )
        return items

class IterableChessDataset(IterableDataset):
    """
    Data set of chess positions where the label is the 
    1-hot encoded move played from the position, and the data
    is a 12x8x8 bitboard.
    """
    def __init__(
            self, 
            zip_pgn_path: str, 
            from_white: bool = True, 
            pretraining: bool = True, 
            n: float = float("inf"),
            numpy: bool = False
        ):
        """
        Initializes a data set from a zipped pgn archive, for example, those found
        at https://database.nikonoel.fr/lichess_elite_lichess_elite_2020-09.zip

        zip_pgn_path: local path to zip archive
        from_white: if true, side to move is always white (i.e., black moves are mirrored)
        pretraining: if true, returns vector of legal moves with each data item
        n: max number of items to take
        """
        super().__init__()
        self.zip_pgn_path = zip_pgn_path
        self.from_white = from_white

        self.pretraining = pretraining

        self.n = n

        self.len = min(n, self.get_length())

        self.numpy = numpy

    
    def __len__(self):
        return self.len
    

    def get_length(self):
        return count_positions(self.zip_pgn_path)

    
    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            raise ValueError("Parallel data loading not supported")
        for i, item in enumerate(iter_positions(
            self.zip_pgn_path,
            legal_moves=self.pretraining,
            from_white=self.from_white
        )):
            if i > self.n:
                return
            
            if not self.pretraining:
                board, move, side, fen = item
            else:
                board, move, side, fen, legal_moves = item

            position = np.array(board, dtype=np.float32).reshape(12, 8, 8)

            move_idx = get_move_idx(move)    
            if self.numpy: 
                move = np.zeros(64 * 64, dtype=bool)
            else:
                move = torch.zeros(64 * 64, dtype=torch.float32)
            move[move_idx] = 1

            if not self.pretraining:
                yield position, move
            else:
                if self.numpy:
                    legal_move_vec = np.zeros(64 * 64, dtype=bool)
                else:
                    legal_move_vec = torch.zeros(64 * 64, dtype=torch.float32)
                
                for legal_move in legal_moves:
                    move_idx = get_move_idx(legal_move)     
                    legal_move_vec[move_idx] = 1

                yield position, move, legal_move_vec, bitboard_from_position(position), fen