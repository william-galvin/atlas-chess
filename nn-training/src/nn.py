from torch.nn import (
    Module,
    Embedding,
    TransformerEncoder,
    TransformerEncoderLayer,
    LayerNorm,
    RMSNorm,
    Linear,
    Flatten,
    Sequential,
)

import torch

def weights_init(m):
    if isinstance(m, Linear):
        torch.nn.init.kaiming_uniform_(m.weight)

class ChessNN(Module):
    def __init__(
            self, 
            num_layers: int, 
            num_layers_decoder: int, 
            dropout: float,
            dim_feedforward: int,
            pretraining: bool = False, 
            *args, 
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.pretraining = pretraining

        self.positional_encoder = ChessPositionalEncoder()

        encoder_layer = TransformerEncoderLayer(
            d_model=14,
            nhead=2,
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers, norm=LayerNorm(14))
        self.flatten = Flatten()

        
        self.best_move_guesser = Sequential(
            TransformerEncoder(encoder_layer, num_layers=num_layers_decoder, norm=LayerNorm(14)),
            Linear(14, 64),
        )
    
        self.pretrain_unmasker = TransformerEncoder(encoder_layer, num_layers=3, norm=LayerNorm(14))
        self.legal_move_guesser = Linear(14, 64)
    
    def forward(self, x):
        """
        Takes a 64x12 input, adds 2D positional encoding, 
        and passes through transformer
        """
        x = self.positional_encoder(x)
        x = self.transformer_encoder(x)

        return self.pretrain_unmasker(x), self.flatten(self.legal_move_guesser(x)), self.flatten(self.best_move_guesser(x))



class ChessPositionalEncoder(Module):
    """
    Encoding a 64x12 board creates a 64x14 board, 
    with one extra dimension for rank and file each
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.row_embedding = Embedding(8, 1)
        self.col_embedding = Embedding(8, 1)

    
    def forward(self, x):
        """
        Input: x is a tensor of shape (batch_size, 64, 12), representing the 64 board squares
        Output: Tensor with positional encodings (2 dimensions for row/col) added
        """
        batch_size, seq_len, dim = x.shape


        rows = torch.arange(8).repeat(8)
        cols = torch.arange(8).unsqueeze(1).repeat(1, 8).flatten() 
        
        row_encoding = self.row_embedding(rows).unsqueeze(0)  
        col_encoding = self.col_embedding(cols).unsqueeze(0)
        
        pos_encoding = torch.cat([row_encoding, col_encoding], dim=-1)  # Shape: (1, 64, 2)
        
        pos_encoding = pos_encoding.expand(batch_size, -1, -1)  # Shape: (batch_size, 64, 2)

        x = torch.cat([x, pos_encoding], dim=-1)  # Shape: (batch_size, 64, 14)

        return x