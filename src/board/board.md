# Board 
The spots on the board of each piece are represented as bits in `u64`s. 

For other information, the `info` field as a `u64`:
- Bit 1: 1 if white to move else 0
- Bits 2, 3: 1 if white has king and queen side castling rights
- Bits 4, 5: 1 if black has king and queen side castling rights
- Bits 6-13: 1 if white can take en passant to that file
- Bits 14-21: 1 if black can take en passant to that file

# Pushing and Popping Moves
Since the `info` field is not backwards-recoverable, we will 
keep a stack of `info` that will be pushed to and popped from 
exactly once when making or unmaking a move. 

We will also keep stacks for each of the 12 piece types. When
the positions of a piece type changes, we will push a new `u64`
bitboard onto its stack.

Since multiple piece types may change every move (castling
, promotion, capture), we will also keep a stack of which
pieces moved. 

Thus, when a pushing a move, the algorithm is this: 
```
1. Calculate the new info field, set self.info and push
   into the info stack
   
2. Find which piece types moved. Push into their stacks. 
   Push into the move_type stack. Update self.pieces.
```

And the process for going back one move:
``` 
1. self.info = info_stack.pop()

2. For type in move_type.pop():
      self.pieces[type] = peices_stack[type].pop()
```

Note that the current board state is *not* stored on the 
stacks. Thus, the initial board has no previous positions.