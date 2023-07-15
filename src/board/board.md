# Board 
The spots on the board of each piece are represented as bits in `u64`s. 

For other information, the `info` field as a `u64`:
- Bit 1: 1 if white to move else 0
- Bits 2, 3: 1 if white has king and queen side castling rights
- Bits 4, 5: 1 if black has king and queen side castling rights
- Bits 6-13: 1 if white can take en passant to that file
- Bits 14-21: 1 if black can take en passant to that file

## TODO

- Implement `push_move` and `pop_move` (or does this belong
in a game manager class?)
    - Don't need to keep track of 12 new number for each move!
- Implement fmt display and debug stuff