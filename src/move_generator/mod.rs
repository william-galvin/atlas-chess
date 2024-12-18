use crate::chess_move::ChessMove;
use crate::board::{Board, WHITE, BLACK, KING, QUEEN, ROOK, BISHOP, KNIGHT, PAWN};

use pyo3::prelude::*;

const ROOK_MAGICS: [usize; 64] = [
    0xa8002c000108020, 0x6c00049b0002001, 0x100200010090040, 0x2480041000800801, 0x280028004000800,
    0x900410008040022, 0x280020001001080, 0x2880002041000080, 0xa000800080400034, 0x4808020004000,
    0x2290802004801000, 0x411000d00100020, 0x402800800040080, 0xb000401004208, 0x2409000100040200,
    0x1002100004082, 0x22878001e24000, 0x1090810021004010, 0x801030040200012, 0x500808008001000,
    0xa08018014000880, 0x8000808004000200, 0x201008080010200, 0x801020000441091, 0x800080204005,
    0x1040200040100048, 0x120200402082, 0xd14880480100080, 0x12040280080080, 0x100040080020080,
    0x9020010080800200, 0x813241200148449, 0x491604001800080, 0x100401000402001, 0x4820010021001040,
    0x400402202000812, 0x209009005000802, 0x810800601800400, 0x4301083214000150, 0x204026458e001401,
    0x40204000808000, 0x8001008040010020, 0x8410820820420010, 0x1003001000090020, 0x804040008008080,
    0x12000810020004, 0x1000100200040208, 0x430000a044020001, 0x280009023410300, 0xe0100040002240,
    0x200100401700, 0x2244100408008080, 0x8000400801980, 0x2000810040200, 0x8010100228810400,
    0x2000009044210200, 0x4080008040102101, 0x40002080411d01, 0x2005524060000901, 0x502001008400422,
    0x489a000810200402, 0x1004400080a13, 0x4000011008020084, 0x26002114058042
];

const BISHOP_MAGICS: [usize; 64] = [
    0x89a1121896040240, 0x2004844802002010, 0x2068080051921000, 0x62880a0220200808, 0x4042004000000,
    0x100822020200011, 0xc00444222012000a, 0x28808801216001, 0x400492088408100, 0x201c401040c0084,
    0x840800910a0010, 0x82080240060, 0x2000840504006000, 0x30010c4108405004, 0x1008005410080802,
    0x8144042209100900, 0x208081020014400, 0x4800201208ca00, 0xf18140408012008, 0x1004002802102001,
    0x841000820080811, 0x40200200a42008, 0x800054042000, 0x88010400410c9000, 0x520040470104290,
    0x1004040051500081, 0x2002081833080021, 0x400c00c010142, 0x941408200c002000, 0x658810000806011,
    0x188071040440a00, 0x4800404002011c00, 0x104442040404200, 0x511080202091021, 0x4022401120400,
    0x80c0040400080120, 0x8040010040820802, 0x480810700020090, 0x102008e00040242, 0x809005202050100,
    0x8002024220104080, 0x431008804142000, 0x19001802081400, 0x200014208040080, 0x3308082008200100,
    0x41010500040c020, 0x4012020c04210308, 0x208220a202004080, 0x111040120082000, 0x6803040141280a00,
    0x2101004202410000, 0x8200000041108022, 0x21082088000, 0x2410204010040, 0x40100400809000,
    0x822088220820214, 0x40808090012004, 0x910224040218c9, 0x402814422015008, 0x90014004842410,
    0x1000042304105, 0x10008830412a00, 0x2520081090008908, 0x40102000a0a60140
];

const ROOK_SHIFTS: [u16; 64] = [
    52, 53, 53, 53, 53, 53, 53, 52,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    52, 53, 53, 53, 53, 53, 53, 52
];

const BISHOP_SHIFTS: [u16; 64] = [
    58, 59, 59, 59, 59, 59, 59, 58,
    59, 59, 59, 59, 59, 59, 59, 59,
    59, 59, 57, 57, 57, 57, 59, 59,
    59, 59, 57, 55, 55, 57, 59, 59,
    59, 59, 57, 55, 55, 57, 59, 59,
    59, 59, 57, 57, 57, 57, 59, 59,
    59, 59, 59, 59, 59, 59, 59, 59,
    58, 59, 59, 59, 59, 59, 59, 58
];

#[pyclass]
#[derive(Clone)]
pub struct MoveGenerator {
    sliding_moves: Vec<Vec<u64>>
}

#[pymethods]
impl MoveGenerator {
    #[new]
    pub fn py_new() -> PyResult<Self> {
        Ok(Self::new())
    }

    pub fn __call__(&self, board: &mut Board) -> PyResult<Vec<String>> {
        Ok(
            self.moves(board)
                .into_iter()
                .map(|c| c.to_string())
                .collect()
        )
    }
}

impl Default for MoveGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl MoveGenerator {
    pub fn new() -> Self {
        Self { sliding_moves: populate_sliding_moves() }
    }

    /// Generates legal moves from a given position.
    pub fn moves(&self, board: &mut Board) -> Vec<ChessMove> {
        self.pseudo_moves(board)
            .into_iter()
            .filter(|&chess_move| {
                board.push_move(chess_move);
                let valid_move = !self.phantom_check(board);
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
    pub fn pseudo_moves(&self, board: &Board) -> Vec<ChessMove> {
        let mut moves = Vec::with_capacity(50);

        let to_move = board.to_move();

        let mut opp_mask = 0u64;
        let mut self_mask = 0u64;
        let (self_offset, opp_offset) = if to_move == WHITE { (0, 6) } else { (6, 0) };
        for i in 0..6 {
            opp_mask |= board.pieces[i + opp_offset];
            self_mask |= board.pieces[i + self_offset];
        }

        let mut k_moves = self.king_moves(board, self_offset, self_mask, opp_mask, to_move);
        let mut n_moves = knight_moves(board, self_offset, self_mask);
        let mut p_moves = pawn_moves(board, self_offset, self_mask, opp_mask, to_move);
        let mut r_moves = self.magic_moves(board, self_offset, self_mask, opp_mask, ROOK);
        let mut b_moves = self.magic_moves(board, self_offset, self_mask, opp_mask, BISHOP);

        moves.append(&mut k_moves);
        moves.append(&mut n_moves);
        moves.append(&mut p_moves);
        moves.append(&mut r_moves);
        moves.append(&mut b_moves);

        moves
    }

    /// Finds pseudo-legal rook/bishop moves
    fn magic_moves(&self, board: &Board, offset: usize, self_mask: u64, opp_mask: u64, piece_type: usize) -> Vec<ChessMove> {
        let (magics, shifts, idx_offset) = if piece_type == ROOK {
            (ROOK_MAGICS, ROOK_SHIFTS, 0)
        } else {
            (BISHOP_MAGICS, BISHOP_SHIFTS, 1)
        };


        let mut moves = Vec::with_capacity(50);
        let piece_mask = board.pieces[offset + piece_type] | board.pieces[offset + QUEEN];
        for square in get_bits(piece_mask) {
            let blockers = (self_mask | opp_mask) & if piece_type == ROOK {
                get_rook_cross(square)
            } else {
                get_bishop_cross(square)
            };

            let key = (blockers as usize).wrapping_mul(magics[square]) >> shifts[square];
            let _moves = self.sliding_moves[square * 2 + idx_offset][key] & !(self_mask);
            for to in get_bits(_moves) {
                moves.push(ChessMove::new(square as u16, to as u16, 0))
            }
        }
        moves
    }

    /// Returns true if the given square is being attacked by
    /// the side who is **not** moving.
    fn square_in_check(&self, board: &Board, square: usize, to_move: u64) -> bool {
        let offset = if to_move == WHITE { 6 } else { 0 };

        if check_pawn_knight(
            board.pieces[offset + KNIGHT],
            square,
            vec![-17, -15, -6, 10, 17, 15, 6, -10],
            2
        ) || check_pawn_knight(
            board.pieces[offset + PAWN],
            square,
            if to_move == WHITE { vec![7, 9] } else { vec![-7, -9] },
            1
        ) {
            return true;
        }

        let mut blockers = 0u64;
        for i in 0..12 {
            blockers |= board.pieces[i];
        }

        let rook_bboard = board.pieces[ROOK + offset] | board.pieces[QUEEN + offset];
        let bishop_bboard = board.pieces[BISHOP + offset] | board.pieces[QUEEN + offset];
        let k_bboard = board.pieces[KING + offset];

        let r_blockers = get_rook_cross(square) & blockers;
        let b_blockers = get_bishop_cross(square) & blockers;
        
        let r_key = (r_blockers as usize).wrapping_mul(ROOK_MAGICS[square]) >> ROOK_SHIFTS[square];
        let b_key = (b_blockers as usize).wrapping_mul(BISHOP_MAGICS[square]) >> BISHOP_SHIFTS[square];

        let r_moves = self.sliding_moves[square * 2][r_key];
        let b_moves = self.sliding_moves[square * 2 + 1][b_key];
        let k_moves = get_king_circle(square);

        r_moves & rook_bboard != 0 || b_moves & bishop_bboard != 0 || k_moves & k_bboard != 0
    }

    /// Finds pseudo-legal king moves
    fn king_moves(&self, board: &Board, self_offset: usize, self_mask: u64, opp_mask: u64, to_move: u64) -> Vec<ChessMove> {
        let mut moves = get_k_moves(
            board.pieces[self_offset + KING],
            self_mask,
            [-8, 8, 1, -1, 9, 7, -9, -7],
            1
        );

        if to_move == WHITE {
            if board.pieces[self_offset + KING] >> 4 & 1 == 1 && ! self.square_in_check(board, 4, board.to_move()) {
                if board.white_king_castle() && self.can_castle(board, vec![5, 6], self_mask, opp_mask, None) {
                    moves.push(ChessMove::new(4, 6, 3));
                }
                if board.white_queen_castle() && self.can_castle(board, vec![2, 3], self_mask, opp_mask, Some(1)) {
                    moves.push(ChessMove::new(4, 2, 3));
                }
            }
        } else if board.pieces[self_offset + KING] >> 60 & 1 == 1 && ! self.square_in_check(board, 60, board.to_move()) {
            if board.black_king_castle() && self.can_castle(board, vec![61, 62], self_mask, opp_mask, None) {
                moves.push(ChessMove::new(60, 62, 3));
            }
            if board.black_queen_castle() && self.can_castle(board, vec![58, 59], self_mask, opp_mask, Some(57)) {
                moves.push(ChessMove::new(60, 58, 3));
            }
        }

        moves
    }

    /// Checks if king of side to move is in check
    pub fn check(&self, board: &Board) -> bool {
        let offset = if board.to_move() == WHITE { 0 } else { 6 };
        self.square_in_check(board, board.pieces[offset + KING].trailing_zeros() as usize, board.to_move())
    }

    /// Checks if the side moving is giving check
    /// Useful for legal move validation
    pub fn phantom_check(&self, board: &Board) -> bool {
        let offset = if board.to_move() == BLACK { 0 } else { 6 };
        self.square_in_check(board, board.pieces[offset + KING].trailing_zeros() as usize, 1 - board.to_move())
    }

    /// Helper for adding castle moves
    fn can_castle(&self, board: &Board, squares: Vec<usize>, self_mask: u64, opp_mask: u64, queen_side: Option<u16>) -> bool {
        let blockers = self_mask | opp_mask;
        for square in squares {
            if blockers >> square & 1 == 1 || self.square_in_check(board, square, board.to_move()) {
                return false;
            }
        }
        if let Some(queen_side) = queen_side {
            if blockers >> queen_side & 1 == 1 {
                return false
            }
        }
        true
    }
}

/// Finds pseudo-legal knight moves
fn knight_moves(board: &Board, self_offset: usize, self_mask: u64) -> Vec<ChessMove> {
    get_k_moves(
        board.pieces[self_offset + KNIGHT],
        self_mask,
        [-17, -15, -6, 10, 17, 15, 6, -10],
        2
    )
}

/// Helper for generating king/knight moves, since the procedure
/// is roughly the same for both
fn get_k_moves(bitboard: u64, self_mask: u64, directions: [i16; 8], threshold: u16) -> Vec<ChessMove> {
    let mut moves = Vec::new();
    for i in get_bits(bitboard) {
        for dir in directions {
            let to = match validate_move(i as u16, dir, threshold) {
                Some(t) => t,
                None => continue
            };
            if self_mask >> to & 1 == 1 {
                continue;
            }
            moves.push(ChessMove::new(i as u16, to, 0));
        }
    }
    moves
}

/// Helper function to check if king is in check by a sliding piece, parameterized
/// to specify which direction
#[allow(unused)]
fn check_sliding_direction(board: &Board, offset: usize, square: u16, update: i16, pieces: Vec<usize>, blockers: u64) -> bool {
    let mut target_square = square;
    let pieces_mask = pieces.iter().fold(0, |acc, &x| acc | board.pieces[x + offset]);
    loop {
        target_square = match validate_move(target_square, update, 1) {
            None => return false,
            Some(s) => s
        };
        if blockers >> target_square & 1 == 1 {
            if pieces_mask >> target_square & 1 == 1 {
                return true;
            }
            return false;
        }
    }
}

/// Helper for checking whether check is given by a pawn or
/// knight, since they are handled similarly
fn check_pawn_knight(bitboard: u64, square: usize, directions: Vec<i16>, threshold: u16) -> bool {
    for jump in directions {
        let to = match validate_move(square as u16, jump, threshold) {
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

    let (start_rank, end_rank, direction, enpass_offset, enpass_rank) = if to_move == WHITE {
        (1, 6, 1, 5, 4)
    } else {
        (6, 1, -1, 13, 3)
    };

    for i_size in get_bits(board.pieces[offset + PAWN]) {
        let i = i_size as u16;
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
    moves
}

/// Validates that a move + jump is legal
/// If so, returns Some(to).
fn validate_move(square: u16, jump: i16, threshold: u16) -> Option<u16> {
    if square as i16 + jump < 0 {
        return None
    }
    let to = (square as i16 + jump) as u16;
    if to >= 64 {
        return None
    }
    let to_file = ChessMove::file(to);
    let from_file = ChessMove::file(square);

    let diff = (to_file as i16 - from_file as i16).unsigned_abs();

    if diff <= threshold {
        Some(to)
    } else {
        None
    }
}

/// Precomputes the sliding piece moves.
fn populate_sliding_moves() -> Vec<Vec<u64>> {
    let mut array = vec![Vec::new(); 64 * 2];
    for square in 0..64 {
        array[square * 2] = rook_bishop_moves(square, get_rook_cross(square), ROOK).to_vec();
        array[square * 2 + 1] = rook_bishop_moves(square, get_bishop_cross(square), BISHOP).to_vec();
    }
    array
}

/// Precompute rook moves
fn rook_bishop_moves(square: usize, cross: u64, piece_type: usize) -> Box<[u64; 1 << 12]> {
    let mut array = Box::new([0u64; 1 << 12]);

    let (magics, shifts) = if piece_type == ROOK { (ROOK_MAGICS, ROOK_SHIFTS) } else { (BISHOP_MAGICS, BISHOP_SHIFTS) };

    let move_idxs = get_bits(cross);
    
    for pattern_idx in 0..(1 << move_idxs.len()) {
        let mut blocker_bitboard = 0u64;
        for bit_idx in 0..move_idxs.len() {
            let bit = (pattern_idx >> bit_idx) & 1;
            blocker_bitboard |= bit << move_idxs[bit_idx];
        }
        let moves = get_moves_from_blockers(square, blocker_bitboard, if piece_type == ROOK {
            [8, -8, 1, -1]
        } else {
            [9, -9, 7, -7]
        });

        let key = (blocker_bitboard as usize).wrapping_mul(magics[square]) >> shifts[square];
        array[key] = moves;
    }

    array
}

/// Returns mask of moves to which rook can move (including edges)
fn get_moves_from_blockers(square: usize, blocker_bitboard: u64, directions: [i16; 4]) -> u64 {
    let mut moves = 0u64;
    for direction in directions {
        let mut target_square = square as u16;
        loop {
            target_square = match validate_move(target_square, direction, 1) {
                None => break,
                Some(s) => s
            };
            moves |= 1 << target_square;
            if blocker_bitboard >> target_square & 1 == 1 {
                break
            }
        }
    }
    moves
}

/// Returns a mask of all squares surrounding a given square, 
/// as if accessible by a king
fn get_king_circle(square: usize) -> u64 {
    let king_pos: u64 = 1 << square; // Set the king's position as a single bit in a 64-bit integer

    // Calculate attacks
                    // Down-Left

    (king_pos << 8) | (king_pos >> 8) |                // Up and Down
                  ((king_pos & !0x8080808080808080) << 1) |               // Right (exclude leftmost column wrap)
                  ((king_pos & !0x0101010101010101) >> 1) |               // Left (exclude rightmost column wrap)
                  ((king_pos & !0x8080808080808080) << 9) |               // Up-Right
                  ((king_pos & !0x8080808080808080) >> 7) |               // Down-Right
                  ((king_pos & !0x0101010101010101) << 7) |               // Up-Left
                  ((king_pos & !0x0101010101010101) >> 9)
}

/// Returns a mask of all squares on rank + file as given
/// Square, excluding those on the edges
fn get_rook_cross(square: usize) -> u64 {
    let rank = 0b1111110;
    let file = 0b1000000010000000100000001000000010000000100000000;

    (file << ChessMove::file(square as u16) | rank << (8 * ChessMove::rank(square as u16))) & !(1 << square)
}

/// Returns mask of all squares on same diagonal
fn get_bishop_cross(square: usize) -> u64 {
    let mut cross = 0u64;
    for jump in [9, 7, -9, -7] {
        let mut target_square = square as u16;
        loop {
            target_square = match validate_move(target_square, jump, 1) {
                None => break,
                Some(s) => s
            };
            if ChessMove::rank(target_square) == 0 || ChessMove::rank(target_square) == 7 ||
                ChessMove::file(target_square) == 0 || ChessMove::file(target_square) == 7
            {
                break;
            }
            cross |= 1 << target_square;
        }
    }
    cross
}

/// returns a vector describing bits set to 1
fn get_bits(mut n: u64) -> Vec<usize> {
    let mut bits = Vec::new();
    while n != 0 {
        let idx = n.trailing_zeros() as usize;
        bits.push(idx);
        n &= !(1 << idx);
    }
    bits
}

/// https://www.chessprogramming.org/Perft
/// https://www.chessprogramming.org/Perft_Results
#[allow(unused)]
pub fn perft(depth: i32, board: &mut Board, move_gen: &MoveGenerator) -> usize {
    let moves = move_gen.moves(board);

    if depth == 1 {
        return moves.len();
    }

    let mut nodes = 0;
    for m in moves {
        board.push_move(m);
        nodes += perft(depth - 1, board, move_gen);
        board.pop_move();
    }
    nodes
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{self, BufRead};
    use std::fs::File;
    use std::time::{Instant};

    #[test]
    fn check_test_knight() {
        let mg = MoveGenerator::new();

        let board = Board::from_fen("r1bqk2r/p1p1b1pp/3p1p2/1p2p3/4PP2/1PNB1nP1/1BPPQ2P/R3NRK1 w kq - 3 13").unwrap();
        assert!(mg.check(&board));

        let board = Board::from_fen("r1bqk2r/p1p1b1p1/3p1p1p/1p2p3/4PP2/1PNB1nP1/1BPPQ2P/R3NR1K w kq - 0 14").unwrap();
        assert!(!mg.check(&board));

        let board = Board::from_fen("r1bqk2r/p1N1b1p1/3p1p1p/1p2p3/4PP2/1P1B2P1/1BPPQ2P/R3nR1K b kq - 0 15").unwrap();
        assert!(mg.check(&board));

        let board = Board::from_fen("r1b1k2r/pRq1b1p1/3p1p1p/4p3/4PP2/1n4P1/1BPPQ2P/5R1K w kq - 0 19").unwrap();
        assert!(!mg.check(&board));

        let board = Board::from_fen("rnbq1bnr/pppp1ppp/4p3/3NP3/1k6/5N2/P1PP1PPP/R1BQKB1R b KQ - 1 6").unwrap();
        assert!(mg.check(&board));
    }

    #[test]
    fn check_test_pawn() {
        let mg = MoveGenerator::new();

        let board = Board::from_fen("rnbq1bnr/pppp1ppp/3kp3/4P3/8/2N2N2/PPPP1PPP/R1BQKB1R b KQ - 0 4").unwrap();
        assert!(mg.check(&board));

        let board = Board::from_fen("rnbq1bnr/pppp1ppp/4p3/2k1P3/8/2N2N2/PPPP1PPP/R1BQKB1R w KQ - 1 5").unwrap();
        assert!(! mg.check(&board));

        let board = Board::from_fen("rnbq1bnr/pppp1ppp/4p3/2k1P3/1P6/2N2N2/P1PP1PPP/R1BQKB1R b KQ - 0 5").unwrap();
        assert!(mg.check(&board));
    }

    #[test]
    fn test_check_rook_under() {
        let mg = MoveGenerator::new();

        let board = Board::from_fen("8/8/8/7p/1k5P/8/4K3/1R6 b - - 0 1").unwrap();
        assert!(mg.check(&board));

        let board = Board::from_fen("8/8/8/7p/k6P/8/4K3/1R6 b - - 0 1").unwrap();
        assert!(!mg.check(&board));

        let board = Board::from_fen("8/8/8/7p/k6P/8/N3K3/R7 b - - 0 1").unwrap();
        assert!(!mg.check(&board));

        let board = Board::from_fen("8/8/8/7p/k6P/8/4K3/Q1N5 b - - 0 1").unwrap();
        assert!(mg.check(&board));
    }

    #[test]
    fn test_validate_move() {
        assert_eq!(validate_move(16, -8, 0), Some(8u16));
    }

    #[test]
    fn check_bishop() {
        let mg = MoveGenerator::new();

        let board = Board::from_fen("8/3B4/8/7p/k6P/8/4K3/1R6 b - - 0 1").unwrap();
        assert!(mg.check(&board));

        let board = Board::from_fen("2BB4/4B3/8/7p/k6P/8/4K3/1RB5 b - - 0 1").unwrap();
        assert!(!mg.check(&board));
    }

    #[test]
    fn check_get_king_circle() {
        let king = 0;
        let king_circle = get_king_circle(king);
        assert!(king_circle.count_ones() == 3);
        assert!((king_circle >> 1) & 0b1 == 1 && (king_circle >> 8) & 0b1 == 1 && (king_circle >> 9) & 0b1 == 1);
    }

    #[test]
    fn test_check_fuzz() -> io::Result<()> {
        let mg = MoveGenerator::new();

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
                        board.push_move(ChessMove::from_str(m).unwrap());
                        let c = moves.next().unwrap();
                        match c  {
                            "True" => {
                                assert!(mg.check(&board));
                            },
                            "False" => {
                                assert!(!mg.check(&board));
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

    #[test]
    fn test_perft() {
        let mg = MoveGenerator::new();

        let expected = [20, 400, 8902, 197281, 4865609];
        for i in 1..=5 {
            let mut board = Board::new();
            assert_eq!(perft(i, &mut board, &mg), expected[i as usize - 1]);
        }
    }

    #[test]
    fn test_perft2() {
        let mg = MoveGenerator::new();

        let expected = [48, 2039, 97862, 4085603];
        for i in 1..=4 {
            let mut board = Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - ").unwrap();
            assert_eq!(perft(i, &mut board, &mg), expected[i as usize - 1]);
        }

    }

    #[test]
    fn test_perft6() {
        let mg = MoveGenerator::new();

        let start = Instant::now();
        let expected = [46, 2079, 89890, 3894594];
        for i in 1..=4 {
            let mut board = Board::from_fen("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10 ").unwrap();
            assert_eq!(perft(i, &mut board, &mg), expected[i as usize - 1]);
        }
        let elapsed = start.elapsed();
        eprintln!("Elapsed: {:.4?}", elapsed);
    }
}
