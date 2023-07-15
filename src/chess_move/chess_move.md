# Chess-Move
A chess move is represented with a single 16-bit unsigned integer.
The bits 0-5 represent the square *FROM*, and the bits 6-11 represent the bits *TO*. The remaining 
bits 12-16 represent additional information about the move (castling, en-passant, promotions)

For the from and to fields, numbers 0-63 are used to index squares on the board. For the additional 
information field, 0 denotes no extra information, 1 denotes an en passant capture, 2 denotes a pawn jump, 
3 denotes the move is a castle, and  4, 5, 6 and 7 denote promotions to knight, bishop, rook, and 
queen, respectively.