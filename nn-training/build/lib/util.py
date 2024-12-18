import zipfile
import io
import os
import subprocess
import itertools
import random
import sys 
import yaml

import numpy as np

import torch
import math

from PIL import Image, ImageDraw

from atlas_chess import Board, MoveGenerator   

from nn import ChessNN, weights_init

move_generator = MoveGenerator()


def piece(n):
    return (
        "wP",
        "wN",
        "wB",
        "wR",
        "wQ", 
        "wK",
        "bP",
        "bN",
        "bB",
        "bR",
        "bQ", 
        "bK",
    )[n]


def iter_pgns(path: str, num_games: int = 100):
    """
    Yields a chunk of pgns from num_games
    """
    abs_path = os.path.abspath(path)
    root = os.path.basename(abs_path).split(".")[0]

    with zipfile.ZipFile(abs_path, "r") as z:
        with io.TextIOWrapper(z.open(f"{root}.pgn"), encoding="utf-8") as f:
            delims = 0
            buffer = io.StringIO()

            for line in f:
                buffer.write(line)

                if line == "\n":
                    delims += 1

                if delims == 2 * num_games:
                    delims = 0
                    yield buffer.getvalue()
                    buffer.seek(0)
                    buffer.truncate(0)

            yield buffer.getvalue()

def iter_positions(
        path: str, 
        num_games: int = 100, 
        legal_moves: bool = False, 
        shuffle: bool = True, 
        from_white: bool = True
    ):
    """
    yields a tuple of (position, move, side-to-move), 
    and optionally legal moves
    """
    for pgn in iter_pgns(path, num_games):
        result = subprocess.run(
            [
                "../pgn-extract/pgn-extract",
                "-Wuci",
                "--notags",
                "--noresults"
            ],
            input=pgn,
            encoding="utf-8",
            capture_output=True
        )

        for game in result.stdout.split("\n\n"):
            result = []
            board = Board()
            for i, (move1, move2) in enumerate(itertools.pairwise(game.split())):
                try:
                    board.push(move1.lower())

                    reflected = False
                    if from_white and board.turn() == "black":
                        reflected = True
                        board.reflect()
                        move2 = reflect_move(move2)

                    if legal_moves:
                        result.append((board.bitboard(), move2.lower(), board.turn(), board.fen(), move_generator(board)))
                    else:
                        result.append((board.bitboard(), move2.lower(), board.turn(), board.fen()))
                    
                    if reflected:
                        board.reflect()

                except ValueError:
                    continue
            if shuffle:
                random.shuffle(result)
            yield from result



def reflect_move(move):
    file1, file2 = move[1], move[3]
    file1, file2 = 9 - int(file1), 9 - int(file2)
    
    return move[0] + str(file1) + move[2] + str(file2) + move[4:]


def count_positions(path):
    count = 0
    for pgn_chunk in iter_pgns(path):
        for pgn in pgn_chunk.split("\n\n"):
            if not pgn.startswith("1."):
                continue
            count += pgn.count(" ") // 2 - 1
    return count


def move_squares_to_one_hot(from_square, to_square):
    return from_square * 64 + to_square


def one_hot_to_move_squares(one_hot):
    return one_hot // 64, one_hot % 64

def idx_to_move(idx):
    from_sq, to_sq = one_hot_to_move_squares(idx)

    from_file = chr(ord('a') + (from_sq % 8))  
    from_rank = str((from_sq // 8) + 1)

    to_file = chr(ord('a') + (to_sq % 8))  
    to_rank = str((to_sq // 8) + 1)
    
    return from_file + from_rank + to_file + to_rank


def get_move_idx(move):
    # promotion
    if len(move) != 4:
        move = move[:-1]

    from_rank = int(move[1]) - 1
    to_rank = int(move[3]) - 1
    
    from_square = (ord(move[0]) - ord("a")) + 8 * from_rank
    to_square = (ord(move[2]) - ord("a")) + 8 * to_rank

    move_idx = move_squares_to_one_hot(from_square, to_square)
    return move_idx


def bitboard_from_position(position):
    return position.astype(bool).reshape(12, 64).tolist()


def mask_board(board, mask_percentage=0.1, mask_idxs=None):
    """
    Randomly mask a percentage of the squares on the board, or
    mask a specific set of indexes.
    
    Args:
        board (torch.Tensor): Tensor of shape (batch_size, 64, 12).
        mask_percentage (float): The fraction of squares to mask (0.1 = 10%).
    
    Returns:
        masked_board (torch.Tensor): Tensor with some squares masked.
        mask (torch.Tensor): Tensor of shape (batch_size, 64) indicating which squares were masked.
    """
    if (mask_idxs is not None and mask_percentage is not None):
        raise ValueError("mask_idxs or mask_percentage must be None")

    batch_size, seq_len, dim = board.shape

    if mask_percentage is not None:
        num_squares_to_mask = int(mask_percentage * seq_len)
        
        # Create a mask tensor to indicate masked positions
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        for b in range(batch_size):
            masked_indices = torch.randperm(seq_len)[:num_squares_to_mask]
            mask[b, masked_indices] = 0  # Set mask to 0 for masked squares
    else:
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        masked_indices = torch.tensor(mask_idxs)
        mask[:, masked_indices] = 0

    # Apply mask to board (set masked values to zero or a special token)
    masked_board = board.clone()
    masked_board[~mask] = 0  # You can use a special value if needed

    return masked_board, mask


def show_board(
    board,
    path,
    images_path = "../images",
    square_size=60,
    masked = None,
    moves = None,
):
    img = draw_chessboard(square_size=square_size)

    bitboard  = board.bitboard() if not (isinstance(board, list) or isinstance(board, np.ndarray)) else board

    idx = 0
    for side in ("w", "b"):
        for piece in ("P", "N", "B", "R", "Q", "K"):
            bits = bitboard[idx]
            for i, val in enumerate(bits):
                if val:
                    square_position = (i // 8, i % 8)
                    piece_image_path = f"{images_path}/{side}{piece}.svg.png"
                    piece_img = Image.open(piece_image_path).resize((square_size, square_size))

                    top_left_x = square_position[1] * square_size
                    top_left_y = (7 - square_position[0]) * square_size
                    
                    img.paste(piece_img, (top_left_x, top_left_y), piece_img)

            idx += 1

    if masked is not None:
        for i in masked:
            row, col = (i // 8, i % 8)
            top_left_x = col * square_size
            top_left_y = (7 - row) * square_size
            bottom_right_x = top_left_x + square_size
            bottom_right_y = top_left_y + square_size

            draw = ImageDraw.Draw(img)
            draw.rectangle([top_left_x, top_left_y, bottom_right_x, bottom_right_y], fill=(100, 100, 100))

    if moves is not None:
        max_move = max(moves)
        for move, strength in enumerate(moves):
            from_square, to_square = one_hot_to_move_squares(move)

            from_row, from_col = (from_square // 8, from_square % 8)
            to_row, to_col = (to_square // 8, to_square % 8)
            from_mid_x = (from_col + .5) * square_size
            from_mid_y = (7 - from_row + .5) * square_size
            to_mid_x = (to_col + .5) * square_size
            to_mid_y = (7 - to_row + .5) * square_size

            img = arrowed_line(
                img,
                (from_mid_x, from_mid_y),
                (to_mid_x, to_mid_y),
                int(10 * strength),
                color= (0, 255, 255) if strength == max_move else (255, 0, 0)
            )

    img.save(path) 



def arrowed_line(im, ptA, ptB, width=1, color=(255, 0, 0)):
    """Draw line from ptA to ptB with arrowhead at ptB"""

    if width == 0:
        return im

    draw = ImageDraw.Draw(im)
    draw.line((ptA,ptB), width=width, fill=color)

    x0, y0 = ptA
    x1, y1 = ptB

    xb = 0.95 * (x1 - x0) + x0
    yb = 0.95 * (y1 - y0) + y0

    ptB_prime = (1.1 * (x1 - x0) + x0, 1.1 * (y1 - y0) + y0) 

    if x0 == x1:
       vtx0 = (xb - 2 * width, yb)
       vtx1 = (xb + 2 * width, yb)
    elif y0 == y1:
       vtx0 = (xb, yb + 2 * width)
       vtx1 = (xb, yb - 2 * width)
    else:
       alpha = math.atan2(y1 - y0, x1 - x0) - 90 * math.pi / 180
       a = 2 * width * math.cos(alpha)
       b = 2 * width * math.sin(alpha)
       vtx0 = (xb + a, yb + b)
       vtx1 = (xb - a, yb - b)

    draw.polygon([vtx0, vtx1, ptB_prime], fill=color)
    return im


def draw_chessboard(square_size=60, light_color=(240, 217, 181), dark_color=(181, 136, 99)):
    """
    Draws an 8x8 chessboard using Pillow.
    
    Args:
        square_size (int): Size of each square in pixels. Default is 60x60 pixels.
        light_color (tuple): RGB color for the light squares.
        dark_color (tuple): RGB color for the dark squares.
    
    Returns:
        Image object representing the chessboard.
    """
    # Define the size of the board (8 squares by 8 squares)
    board_size = 8 * square_size
    chessboard = Image.new("RGB", (board_size, board_size))

    draw = ImageDraw.Draw(chessboard)

    # Draw the chessboard by alternating colors
    for row in range(8):
        for col in range(8):
            # Calculate the top-left corner of each square
            top_left_x = col * square_size
            top_left_y = row * square_size
            bottom_right_x = top_left_x + square_size
            bottom_right_y = top_left_y + square_size

            # Alternate between light and dark colors
            if (row + col) % 2 == 0:
                fill_color = light_color
            else:
                fill_color = dark_color

            # Draw the square
            draw.rectangle([top_left_x, top_left_y, bottom_right_x, bottom_right_y], fill=fill_color)

    return chessboard


def load_model(config, runs_path="../runs"):
    run = config["run_name"]
    nn = ChessNN(
        num_layers=config["num_layers"],
        num_layers_decoder=config["num_layers_decoder"],
        dropout=config["dropout"],
        dim_feedforward=config["dim_feedforward"],
        pretraining=config["pretrain"]
    )

    
    if (os.path.exists(f"{runs_path}/{run}/nn.pt")):
        weights = torch.load(f"{runs_path}/{run}/nn.pt", weights_only=True, map_location=torch.device('cpu'))
        nn.load_state_dict(weights)
    elif not config["pretrain"]:
        print("--- loading pretrained model ---")
        weights = torch.load(f"{runs_path}/{run}/pretrained_nn.pt", weights_only=True)
        nn.load_state_dict(weights, strict=False)

    nn.train()

    total_params = sum(p.numel() for p in nn.parameters() if p.requires_grad)
    print(f"--- trainable parameters: {total_params:,} ---")

    return nn

def checkpoint(run, model, runs_path="../runs"):
    torch.save(model.state_dict(), f"{runs_path}/{run}/nn.pt")


def format_dict(map):
    result = {}
    for group, metrics in map.items():
        if group.startswith("_"):
            continue
        for k, v in metrics.items():
            result[k] = round(v, 4) if isinstance(v,float) else v
    return str(result)


def get_piece_class_weights():
    class_counts = torch.tensor([8, 2, 2, 2, 1, 1, 8, 2, 2, 2, 1, 1])
    class_weights = 1.0 / class_counts.float()
    return class_weights / class_weights.sum()


def get_config():
    return yaml.safe_load(open(sys.argv[1]).read())


position_dt = np.dtype([
    ("fen", "S85"),
    ("legal_moves", "S512"),
    ("played_moves", "S512")
])
def entry_to_np(fen, moves_played, legal_moves):
    data = np.zeros(1, position_dt)

    data["fen"] = fen.encode()
    
    data["legal_moves"] = np.packbits(legal_moves).tobytes()
    data["played_moves"] = np.packbits(moves_played).tobytes()

    return data


def np_to_entry(entry):
    return (
        entry["fen"].decode(),
        np.unpackbits(np.frombuffer(entry["legal_moves"], dtype=np.uint8), count=4096),
        np.unpackbits(np.frombuffer(entry["played_moves"], dtype=np.uint8), count=4096),
    )
