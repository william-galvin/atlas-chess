use crate::chess_move::ChessMove;
use crate::board::{Board, WHITE, BLACK, KING, QUEEN, ROOK, BISHOP, KNIGHT, PAWN};

/// Generates legal moves from a given position.
pub fn moves(board: &mut Board) -> Vec<ChessMove> {
    pseudo_moves(board)
    .into_iter()
    .filter(|&chess_move| {
        board.push_move(chess_move);
        let valid_move = !phantom_check(board);
        board.pop_move();
        valid_move
    })
    .collect()
}

/// Generates pseudo-legal moves from a given
/// position.
///
/// Pseudo-legal moves are like regular
/// legal moves, except they may result in the king
/// being in check, which is not legal.
pub fn pseudo_moves(board: &Board) -> Vec<ChessMove> {
    let mut moves = Vec::new();

    let to_move = board.to_move();

    let mut opp_mask = 0u64;
    let mut self_mask = 0u64;
    let (self_offset, opp_offset) = if to_move == WHITE { (0, 6) } else { (6, 0) };
    for i in 0..6 {
        opp_mask |= board.pieces[i + opp_offset];
        self_mask |= board.pieces[i + self_offset];
    }

    let mut k_moves = king_moves(board, self_offset, self_mask, opp_mask, to_move);
    let mut n_moves = knight_moves(board, self_offset, self_mask);
    let mut p_moves = pawn_moves(board, self_offset, self_mask, opp_mask, to_move);

    moves.append(&mut k_moves);
    moves.append(&mut n_moves);
    moves.append(&mut p_moves);

    moves
}

/// Finds pseudo-legal knight moves
fn knight_moves(board: &Board, self_offset: usize, self_mask: u64) -> Vec<ChessMove> {
    get_k_moves(
        board,
        board.pieces[self_offset + KNIGHT],
        self_mask,
        [-17, -15, -6, 10, 17, 15, 6, -10],
        2
    )
}

/// Finds pseudo-legal king moves
fn king_moves(board: &Board, self_offset: usize, self_mask: u64, opp_mask: u64, to_move: u64) -> Vec<ChessMove> {
    let mut moves = get_k_moves(
        board,
        board.pieces[self_offset + KING],
        self_mask,
        [-8, 8, 1, -1, 9, 7, -9, -7],
        1
    );

    if to_move == WHITE {
        if board.pieces[self_offset + KING] >> 4 & 1 == 1 && ! square_in_check(board, 4, board.to_move()) {
            if board.white_king_castle() && can_castle(board, vec![5, 6], self_mask, opp_mask) {
                moves.push(ChessMove::new(4, 6, 3));
            }
            if board.white_queen_castle() && can_castle(board, vec![2, 3], self_mask, opp_mask) {
                moves.push(ChessMove::new(4, 2, 3));
            }
        }
    } else if board.pieces[self_offset + KING] >> 60 & 1 == 1 && ! square_in_check(board, 60, board.to_move()) {
        if board.black_king_castle() && can_castle(board, vec![61, 62], self_mask, opp_mask) {
            moves.push(ChessMove::new(60, 62, 3));
        }
        if board.black_queen_castle() && can_castle(board, vec![58, 59], self_mask, opp_mask) {
            moves.push(ChessMove::new(60, 58, 3));
        }
    }

    moves
}

/// Helper for adding castle moves
fn can_castle(board: &Board, squares: Vec<u16>, self_mask: u64, opp_mask: u64) -> bool {
    for square in squares {
        if self_mask >> square & 1 == 1 || opp_mask >> square & 1 == 1 || square_in_check(board, square, board.to_move()) {
            return false;
        }
    }
    true
}

/// Helper for generating king/knight moves, since the procedure
/// is roughly the same for both
fn get_k_moves(board: &Board, bitboard: u64, self_mask: u64, directions: [i16; 8], threshold: i16) -> Vec<ChessMove> {
    let mut moves = Vec::new();
    for i in 0..64 {
        if bitboard >> i & 1 != 1 {
            continue;
        }
        for dir in directions {
            let to = match validate_move(i, dir, threshold) {
                Some(t) => t,
                None => continue
            };
            if self_mask >> to & 1 == 1 {
                continue;
            }
            moves.push(ChessMove::new(i, to, 0));
        }
    }
    moves
}

/// Returns true if the given square is being attacked by
/// the side who is **not** moving.
fn square_in_check(board: &Board, square: u16, to_move: u64) -> bool {
    let offset = if to_move == WHITE { 6 } else { 0 };

    if check_pawn_knight(
        board,
        board.pieces[offset + KNIGHT],
        square,
        vec![-17, -15, -6, 10, 17, 15, 6, -10],
        2
    ) || check_pawn_knight(
        board,
        board.pieces[offset + PAWN],
        square,
        if to_move == WHITE { vec![7, 9] } else { vec![-7, -9] },
        1
    ) {
        return true;
    }

    for update in [8, -8, 1, -1] {
        if check_sliding_direction(board, offset, square, update, vec![ROOK, QUEEN]) {
            return true
        }
    }
    for update in [7, -7, 9, -9] {
        if check_sliding_direction(board, offset, square, update, vec![BISHOP, QUEEN]) {
            return true
        }
    }

    false
}

/// Helper function to check if king is in check by a sliding piece, parameterized
/// to specify which direction
fn check_sliding_direction(board: &Board, offset: usize, square: u16, update: i16, pieces: Vec<usize>) -> bool {
    let mut target_square = square;
    loop {
        target_square = match validate_move(target_square, update, 1) {
            None => return false,
            Some(s) => s
        };
        for idx in 0..12 {
            if board.pieces[idx] >> target_square & 1 == 1 {
                for piece in pieces {
                    if idx == offset + piece {
                        return true;
                    }
                }
                return false;
            }
        }
    }
}

/// Helper for checking whether check is given by a pawn or
/// knight, since they are handled similarly
fn check_pawn_knight(board: &Board, bitboard: u64, square: u16, directions: Vec<i16>, threshold: i16) -> bool {
    for jump in directions {
        let to = match validate_move(square, jump, threshold) {
            Some(t) => t,
            None => continue
        };
        if bitboard >> to & 1 == 1 {
            return true
        }
    }
    false
}

/// Finds pseudo-legal pawn moves
fn pawn_moves(board: &Board, offset: usize, self_mask: u64, opp_mask: u64, to_move: u64) -> Vec<ChessMove> {
    let mut moves = Vec::new();

    let (start_rank, end_rank, direction, enpass_offset, enpass_rank) = if board.to_move() == WHITE {
        (1, 6, 1, 5, 4)
    } else {
        (6, 1, -1, 13, 3)
    };

    for i in 0..64 {
        if  board.pieces[offset + PAWN] >> i & 1 == 1 {
            for dir in [7, 9] {
                let to = match validate_move(i, dir * direction, 1) {
                    None => continue,
                    Some(t) => t
                };
                if opp_mask >> to & 1 == 1 {
                    if ChessMove::rank(i) == end_rank {
                        for promotion in 4..=7 {
                            moves.push(ChessMove::new(i, to, promotion));
                        }
                    } else {
                        moves.push(ChessMove::new(i, to, 0));
                    }
                }
                if ChessMove::rank(i) == enpass_rank && board.info >> (ChessMove::file(to) + enpass_offset) & 1 == 1 {
                    moves.push(ChessMove::new(i, to, 1));
                }
            }

            let to = match validate_move(i, 8 * direction, 0) {
                None => continue,
                Some(t) => t
            };
            if opp_mask >> to & 1 == self_mask >> to & 1 {
                if ChessMove::rank(i) == end_rank {
                    for promotion in 4..=7 {
                        moves.push(ChessMove::new(i, to, promotion));
                    }
                } else {
                    moves.push(ChessMove::new(i, to, 0));
                    if ChessMove::rank(i) == start_rank {
                        let to_jump = match validate_move(i, 16 * direction, 0) {
                            None => continue,
                            Some(t) => t
                        };
                        if opp_mask >> to_jump & 1 == self_mask >> to_jump & 1 {
                            moves.push(ChessMove::new(i, to_jump, 2));
                        }
                    }
                }
            }
        }
    }
    moves
}

/// Validates that a move + jump is legal
/// If so, returns Some(to).
fn validate_move(square: u16, jump: i16, threshold: i16) -> Option<u16> {
    if square as i16 + jump < 0 {
        return None
    }
    let to = (square as i16 + jump) as u16;
    if (ChessMove::file(to) as i16 - ChessMove::file(square) as i16).abs() > threshold || to >= 64 {
        return None;
    }
    Some(to)
}

/// Checks if king of side to move is in check
pub fn check(board: &Board) -> bool {
    let offset = if board.to_move() == WHITE { 0 } else { 6 };
    for i in 0..64 {
        if board.pieces[offset + KING] >> i & 1 == 1 {
            return square_in_check(board, i, board.to_move());
        }
    }
    panic!("No king found")
}

/// Checks if the side moving is giving check
/// Useful for legal move validation
pub fn phantom_check(board: &Board) -> bool {
    let offset = if board.to_move() == BLACK { 0 } else { 6 };
    for i in 0..64 {
        if board.pieces[offset + KING] >> i & 1 == 1 {
            return square_in_check(board, i, 1 - board.to_move());
        }
    }
    panic!("No king found")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{self, BufRead};
    use std::fs::{File, read_dir};
    use std::time::{Instant};

    #[test]
    fn check_test_knight() {
        let mut board = Board::from_fen("r1bqk2r/p1p1b1pp/3p1p2/1p2p3/4PP2/1PNB1nP1/1BPPQ2P/R3NRK1 w kq - 3 13");
        assert!(check(&board));

        let mut board = Board::from_fen("r1bqk2r/p1p1b1p1/3p1p1p/1p2p3/4PP2/1PNB1nP1/1BPPQ2P/R3NR1K w kq - 0 14");
        assert!(!check(&board));

        let mut board = Board::from_fen("r1bqk2r/p1N1b1p1/3p1p1p/1p2p3/4PP2/1P1B2P1/1BPPQ2P/R3nR1K b kq - 0 15");
        assert!(check(&board));

        let mut board = Board::from_fen("r1b1k2r/pRq1b1p1/3p1p1p/4p3/4PP2/1n4P1/1BPPQ2P/5R1K w kq - 0 19");
        assert!(!check(&board));

        let mut board = Board::from_fen("rnbq1bnr/pppp1ppp/4p3/3NP3/1k6/5N2/P1PP1PPP/R1BQKB1R b KQ - 1 6");
        assert!(check(&board));
    }

    #[test]
    fn check_test_pawn() {
        let mut board = Board::from_fen("rnbq1bnr/pppp1ppp/3kp3/4P3/8/2N2N2/PPPP1PPP/R1BQKB1R b KQ - 0 4");
        assert!(check(&board));

        let mut board = Board::from_fen("rnbq1bnr/pppp1ppp/4p3/2k1P3/8/2N2N2/PPPP1PPP/R1BQKB1R w KQ - 1 5");
        assert!(! check(&board));

        let mut board = Board::from_fen("rnbq1bnr/pppp1ppp/4p3/2k1P3/1P6/2N2N2/P1PP1PPP/R1BQKB1R b KQ - 0 5");
        assert!(check(&board));
    }

    #[test]
    fn test_check_rook_under() {
        let mut board = Board::from_fen("8/8/8/7p/1k5P/8/4K3/1R6 b - - 0 1");
        assert!(check(&board));

        let mut board = Board::from_fen("8/8/8/7p/k6P/8/4K3/1R6 b - - 0 1");
        assert!(!check(&board));

        let mut board = Board::from_fen("8/8/8/7p/k6P/8/N3K3/R7 b - - 0 1");
        assert!(!check(&board));

        let mut board = Board::from_fen("8/8/8/7p/k6P/8/4K3/Q1N5 b - - 0 1");
        assert!(check(&board));
    }

    #[test]
    fn test_validate_move() {
        assert_eq!(validate_move(16, -8, 0), Some(8u16));
    }

    #[test]
    fn check_bishop() {
        let mut board = Board::from_fen("8/3B4/8/7p/k6P/8/4K3/1R6 b - - 0 1");
        assert!(check(&board));

        let mut board = Board::from_fen("2BB4/4B3/8/7p/k6P/8/4K3/1RB5 b - - 0 1");
        assert!(!check(&board));
    }

    #[test]
    fn test_check_fuzz() -> io::Result<()> {
        let file = File::open("tests/checks.txt")?;
        let reader = io::BufReader::new(file);
        for line_result in reader.lines() {
            let mut board = Board::new();
            let game = line_result?;
            let mut moves = game.split_whitespace();
            loop {
                match moves.next() {
                    None => break,
                    Some(m) => {
                        board.push_move(ChessMove::from_str(m));
                        let c = moves.next().unwrap();
                        match c  {
                            "True" => {
                                assert!(check(&board));
                            },
                            "False" => {
                                assert!(!check(&board));
                            },
                            _ => {
                                panic!("Expected 'True' or 'False, got something else: {}", c);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Only see output when run with `cargo test -- --nocapture`
    #[test]
    fn bench_legal_vs_pseudo_generators() -> io::Result<()> {
        let start = Instant::now();
        let mut lens = 0;

        for iter in 0..30 {

            let entries = read_dir("tests/random_games")?;
            for entry in entries {
                let entry = entry?;
                let file = File::open(entry.path())?;
                let reader = io::BufReader::new(file);
                let mut lines = reader.lines();
                while let Some(line1) = lines.next() {
                    if let Some(line2) = lines.next() {

                        let _chess_move = line1?;
                        let fen = line2?;

                        let mut board = Board::from_fen(&fen);

                        // let v = moves(&mut board);
                        let v = pseudo_moves(&mut board);

                        lens += v.len();
                    }
                }
            }
        }


        println!("Time taken: {} seconds \t\t\t{}", start.elapsed().as_secs(), lens);
        Ok(())
    }

}