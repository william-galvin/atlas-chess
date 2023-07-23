# Move Generator 

We are going to use **pseudo-legal** move generation. This
means that we generate moves without considering whether 
they put us in check, and then retroactively remove the ones that do.

```
moves = get_pseudo_legal_moves()
for move in moves:
    push(move)
    if in_check():
        moves.remove(move)
    pop(move)
```

We can quickly push and pop thanks to our implementation of `board`. We then need to be able 
to tell if a given position is in check; if so, the pseudo-legal move was not legal. (This 
check-checking functionality will be useful for search, too).

We will calculate the pseudo-legal king and knight moves the same way, since they 
have a very finite number of square they go directly to; we take these squares and 
mask out those that have similar colored pieces. 

We will similarly use an intuitive approach for calculating pawn moves: checking 
if jump moves are legal, the existence of capturable pieces (en passant included), 
promotion moves, etc. In fact, we can (and do) hardcode all the precomputed pawn checks 
we need, keying on color and square and going to moves to check.
```Rust
match pawn_file {
    on_left_edge => {...},
    on_right_edge => {...},
    _ => {}
}

match pawn_rank {
    2 => {...}, 
    7 => {...}
}
```

For sliding pieces, we can similarly leverage precompuation using magic bitboards. 
Essentially:
```
for each sliding piece (color does not matter):
    for each possible starting square:
        calculate unblocked sliding attack
        for each configuration of blockers:
            calculate blocked sliding attack
            make key with magic bitboard
            store table[key] = blocked attack
```
Then, at runtime, we recalculate the key by taking the position of the piece and 
the position of the blockers (all other pieces on the board) to look up our blocked attack.
Finally, we mask out the moves of our blocked attack that would be capturing our own pieces.

# TODO:
- Perf test from many positions + benchmark with build release
- Benchmark with flamegraph (need to do on mac or digital ocean)
- Optimizations: don't use 12 for every shift; multithreading