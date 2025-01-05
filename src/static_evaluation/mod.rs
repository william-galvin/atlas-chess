use crate::board::{
    BISHOP,
    KING,
    KNIGHT,
    PAWN,
    QUEEN,
    ROOK
};
use crate::chess_move::ChessMove;
use crate::move_generator::{get_bits, get_king_circle, knight_moves, pawn_seen, MoveGenerator};

pub const QUEEN_VALUE: i16 = 1000;
pub const ROOK_VALUE: i16 = 525;
pub const BISHOP_VALUE: i16 = 350;
pub const KNIGHT_VALUE: i16 = 350;
pub const PAWN_VALUE: i16 = 100;

const BISHOP_PAIR_BONUS: i16 = 50;
const KNIGHT_PAWN_INTERACTION: i16 = 5;
const CONNECTED_PAWN_BONUS: i16 = 25;
const DOUBLED_PAWN_PENALTY: i16 = 25;
const ISOLATED_PAWN_PENALTY: i16 = 25;

const MINOR_ATTACK_UNITS: u32 = 2;
const ROOK_ATTACK_UNITS: u32 = 3;
const QUEEN_ATTACK_UNITS: u32 = 5;

const QUEEN_INVERSE: i16 = 4;
const ROOK_INVERSE: i16 = 10;
const BISHOP_INVERSE: i16 = 30;
const KNIGHT_INVERSE: i16 = 30;
const PAWN_INVERSE: i16 = 50;

// PIECE SQUARE TABLES
const INTERPOLANTS: [i16; 33] = [
    0, 0, 0, 0, 0, 1, 1, 2,
    2, 3, 3, 3, 4, 4, 4, 5,
    5, 5, 6, 6, 6, 7, 7, 7,
    8, 8, 9, 9, 10, 10, 10, 10, 10
];

const FLIP: [usize; 64] = [
    56,  57,  58,  59,  60,  61,  62,  63,
    48,  49,  50,  51,  52,  53,  54,  55,
    40,  41,  42,  43,  44,  45,  46,  47,
    32,  33,  34,  35,  36,  37,  38,  39,
    24,  25,  26,  27,  28,  29,  30,  31,
    16,  17,  18,  19,  20,  21,  22,  23,
     8,   9,  10,  11,  12,  13,  14,  15,
     0,   1,   2,   3,   4,   5,   6,   7
];

const MG_PAWN: [i16; 64] = [
    0,   0,   0,   0,   0,   0,  0,   0,
   98, 134,  61,  95,  68, 126, 34, -11,
   -6,   7,  26,  31,  65,  56, 25, -20,
  -14,  13,   6,  21,  23,  12, 17, -23,
  -27,  -2,  -5,  12,  17,   6, 10, -25,
  -26,  -4,  -4, -10,   3,   3, 33, -12,
  -35,  -1, -20, -23, -15,  24, 38, -22,
    0,   0,   0,   0,   0,   0,  0,   0,
];

const EG_PAWN: [i16; 64] = [
    0,   0,   0,   0,   0,   0,   0,   0,
  178, 173, 158, 134, 147, 132, 165, 187,
   94, 100,  85,  67,  56,  53,  82,  84,
   32,  24,  13,   5,  -2,   4,  17,  17,
   13,   9,  -3,  -7,  -7,  -8,   3,  -1,
    4,   7,  -6,   1,   0,  -5,  -1,  -8,
   13,   8,   8,  10,  13,   0,   2,  -7,
    0,   0,   0,   0,   0,   0,   0,   0,
];

const MG_KNIGHT: [i16; 64] = [
  -167, -89, -34, -49,  61, -97, -15, -107,
   -73, -41,  72,  36,  23,  62,   7,  -17,
   -47,  60,  37,  65,  84, 129,  73,   44,
    -9,  17,  19,  53,  37,  69,  18,   22,
   -13,   4,  16,  13,  28,  19,  21,   -8,
   -23,  -9,  12,  10,  19,  17,  25,  -16,
   -29, -53, -12,  -3,  -1,  18, -14,  -19,
  -105, -21, -58, -33, -17, -28, -19,  -23,
];

const EG_KNIGHT: [i16; 64] = [
  -58, -38, -13, -28, -31, -27, -63, -99,
  -25,  -8, -25,  -2,  -9, -25, -24, -52,
  -24, -20,  10,   9,  -1,  -9, -19, -41,
  -17,   3,  22,  22,  22,  11,   8, -18,
  -18,  -6,  16,  25,  16,  17,   4, -18,
  -23,  -3,  -1,  15,  10,  -3, -20, -22,
  -42, -20, -10,  -5,  -2, -20, -23, -44,
  -29, -51, -23, -15, -22, -18, -50, -64,
];

const MG_BISHOP: [i16; 64] = [
  -29,   4, -82, -37, -25, -42,   7,  -8,
  -26,  16, -18, -13,  30,  59,  18, -47,
  -16,  37,  43,  40,  35,  50,  37,  -2,
   -4,   5,  19,  50,  37,  37,   7,  -2,
   -6,  13,  13,  26,  34,  12,  10,   4,
    0,  15,  15,  15,  14,  27,  18,  10,
    4,  15,  16,   0,   7,  21,  33,   1,
  -33,  -3, -14, -21, -13, -12, -39, -21,
];

const EG_BISHOP: [i16; 64] = [
  -14, -21, -11,  -8, -7,  -9, -17, -24,
   -8,  -4,   7, -12, -3, -13,  -4, -14,
    2,  -8,   0,  -1, -2,   6,   0,   4,
   -3,   9,  12,   9, 14,  10,   3,   2,
   -6,   3,  13,  19,  7,  10,  -3,  -9,
  -12,  -3,   8,  10, 13,   3,  -7, -15,
  -14, -18,  -7,  -1,  4,  -9, -15, -27,
  -23,  -9, -23,  -5, -9, -16,  -5, -17,
];

const MG_ROOK: [i16; 64] = [
   32,  42,  32,  51, 63,  9,  31,  43,
   27,  32,  58,  62, 80, 67,  26,  44,
   -5,  19,  26,  36, 17, 45,  61,  16,
  -24, -11,   7,  26, 24, 35,  -8, -20,
  -36, -26, -12,  -1,  9, -7,   6, -23,
  -45, -25, -16, -17,  3,  0,  -5, -33,
  -44, -16, -20,  -9, -1, 11,  -6, -71,
  -19, -13,   1,  17, 16,  7, -37, -26,
];

const EG_ROOK: [i16; 64] = [
  13, 10, 18, 15, 12,  12,   8,   5,
  11, 13, 13, 11, -3,   3,   8,   3,
   7,  7,  7,  5,  4,  -3,  -5,  -3,
   4,  3, 13,  1,  2,   1,  -1,   2,
   3,  5,  8,  4, -5,  -6,  -8, -11,
  -4,  0, -5, -1, -7, -12,  -8, -16,
  -6, -6,  0,  2, -9,  -9, -11,  -3,
  -9,  2,  3, -1, -5, -13,   4, -20,
];

const MG_QUEEN: [i16; 64] = [
  -28,   0,  29,  12,  59,  44,  43,  45,
  -24, -39,  -5,   1, -16,  57,  28,  54,
  -13, -17,   7,   8,  29,  56,  47,  57,
  -27, -27, -16, -16,  -1,  17,  -2,   1,
   -9, -26,  -9, -10,  -2,  -4,   3,  -3,
  -14,   2, -11,  -2,  -5,   2,  14,   5,
  -35,  -8,  11,   2,   8,  15,  -3,   1,
   -1, -18,  -9,  10, -15, -25, -31, -50,
];

const EG_QUEEN: [i16; 64] = [
   -9,  22,  22,  27,  27,  19,  10,  20,
  -17,  20,  32,  41,  58,  25,  30,   0,
  -20,   6,   9,  49,  47,  35,  19,   9,
    3,  22,  24,  45,  57,  40,  57,  36,
  -18,  28,  19,  47,  31,  34,  39,  23,
  -16, -27,  15,   6,   9,  17,  10,   5,
  -22, -23, -30, -16, -16, -23, -36, -32,
  -33, -28, -22, -43,  -5, -32, -20, -41,
];

const MG_KING: [i16; 64] = [
  -65,  23,  16, -15, -56, -34,   2,  13,
   29,  -1, -20,  -7,  -8,  -4, -38, -29,
   -9,  24,   2, -16, -20,   6,  22, -22,
  -17, -20, -12, -27, -30, -25, -14, -36,
  -49,  -1, -27, -39, -46, -44, -33, -51,
  -14, -14, -22, -46, -44, -30, -15, -27,
    1,   7,  -8, -64, -43, -16,   9,   8,
  -15,  36,  12, -54,   8, -28,  24,  14,
];

const EG_KING: [i16; 64] = [
  -74, -35, -18, -18, -11,  15,   4, -17,
  -12,  17,  14,  17,  17,  38,  23,  11,
   10,  17,  23,  15,  20,  45,  44,  13,
   -8,  22,  24,  27,  26,  33,  26,   3,
  -18,  -4,  21,  24,  27,  23,   9, -11,
  -19,  -3,  11,  21,  23,  16,   7,  -9,
  -27, -11,   4,  13,  14,   4,  -5, -17,
  -53, -34, -21, -11, -28, -14, -24, -43
];

// ATTACK UNIT SAFETY TABLE
const SAFETY_TABLE: [i16; 100] = [
    0,  0,   1,   2,   3,   5,   7,   9,  12,  15,
  18,  22,  26,  30,  35,  39,  44,  50,  56,  62,
  68,  75,  82,  85,  89,  97, 105, 113, 122, 131,
 140, 150, 169, 180, 191, 202, 213, 225, 237, 248,
 260, 272, 283, 295, 307, 319, 330, 342, 354, 366,
 377, 389, 401, 412, 424, 436, 448, 459, 471, 483,
 494, 500, 500, 500, 500, 500, 500, 500, 500, 500,
 550, 550, 550, 550, 575, 575, 575, 625, 625, 625,
 650, 650, 650, 675, 675, 675, 700, 700, 700, 700,
 700, 700, 700, 700, 700, 700, 700, 700, 700, 700
];

/// Returns a large positive number if board is good for white,
/// large negative if good for black
pub fn static_evaluation(pieces: [u64; 12], move_generator: &MoveGenerator) -> i16 {
    let attacks = get_attacks(pieces, move_generator);

    material_eval(pieces) + 
    piece_square_tables(pieces) +
    pawn_structure(pieces) +
    king_safety(pieces, &attacks) * 2 + 
    square_control(&attacks) / 2 
}

/// Returns type: array A where A[i] gives an bitmask of attacking pieces for square i.
/// Eg A[34] := 0b000...01101 indicates one attacking white pawn and two white knights.
/// This only accounts for the first two instances of a piece
fn get_attacks(pieces: [u64; 12], move_generator: &MoveGenerator) -> [u32; 64] {
    let mut attacks = [0u32; 64];
    let mut moves = Vec::with_capacity(32);

    let mut occupied = 0u64;
    for p in pieces {
        occupied |= p;
    }

    pawn_seen(pieces, 0, 0, &mut moves);
    add_attackers(&mut attacks, &mut moves, 0);

    pawn_seen(pieces, 6, 1, &mut moves);
    add_attackers(&mut attacks, &mut moves, 6 * 2);
    
    knight_moves(pieces, 0, 0u64, [u64::MAX; 64], &mut moves);
    add_attackers(&mut attacks, &mut moves, KNIGHT * 2);

    knight_moves(pieces, 6, 0u64, [u64::MAX; 64], &mut moves);
    add_attackers(&mut attacks, &mut moves, (KNIGHT + 6) * 2);

    move_generator.magic_moves(pieces, 0, 0u64, occupied, ROOK, [u64::MAX; 64], &mut moves);
    add_magic_attackers(&mut attacks, &mut moves, [ROOK, QUEEN], pieces);

    move_generator.magic_moves(pieces, 6, 0u64, occupied, ROOK, [u64::MAX; 64], &mut moves);
    add_magic_attackers(&mut attacks, &mut moves, [ROOK + 6, QUEEN + 6], pieces);
    
    move_generator.magic_moves(pieces, 0, 0u64, occupied, BISHOP, [u64::MAX; 64], &mut moves);
    add_magic_attackers(&mut attacks, &mut moves, [BISHOP, QUEEN], pieces);

    move_generator.magic_moves(pieces, 6, 0u64, occupied, BISHOP, [u64::MAX; 64], &mut moves);
    add_magic_attackers(&mut attacks, &mut moves, [BISHOP + 6, QUEEN + 6], pieces);

    attacks
}

/// Updates bitboard with attackers or'd in 
fn add_attackers(attacks: &mut [u32; 64], moves: &mut Vec<ChessMove>, shift: usize) {
    let n = moves.len();
    for _ in 0..n {
        let to = moves.pop().unwrap().to() as usize;
        let prev = attacks[to];
        attacks[to] |= prev + (1 << shift);
    }
}

/// Updates bitboard with attackers or'd in, 
/// aware that attackers may be queen or bishop/rook
fn add_magic_attackers(attacks: &mut [u32; 64], moves: &mut Vec<ChessMove>, options: [usize; 2], pieces: [u64; 12]) {
    let n = moves.len();
    for _ in 0..n {
        let m = moves.pop().unwrap();

        let to = m.to() as usize;
        let from = m.from();
        
        let shift = if pieces[options[0]] >> from & 1 == 1 {
            options[0] * 2
        } else {
            options[1] * 2
        };

        let prev = attacks[to];
        attacks[to] |= prev + (1 << shift);
    }
}

/// Evaluation of attacks/defenses of squares, orthogonal to occupancy
fn square_control(attacks: &[u32; 64]) -> i16 {
    let mut sum = 0;

    for square in 0..64 {
        let attackers = attacks[square];
        for (piece, inverse) in [
            (PAWN, PAWN_INVERSE),
            (KNIGHT, KNIGHT_INVERSE),
            (BISHOP, BISHOP_INVERSE),
            (ROOK, ROOK_INVERSE),
            (QUEEN, QUEEN_INVERSE)
        ] {
            let white_shift = piece * 2;
            let black_shift = (piece + 6) * 2;

            sum += ((attackers >> white_shift) & 0b11).count_ones() as i16 * inverse;
            sum -= ((attackers >> black_shift) & 0b11).count_ones() as i16 * inverse;
        }
    }

    sum / 10
}

/// Evalutation based on king safety, as described
/// here: https://www.chessprogramming.org/King_Safety#Attack_Units
/// 
/// TODO: reuse calculations from move generator to find reachable squares
fn king_safety(pieces: [u64; 12], attacks: &[u32; 64]) -> i16 {
    let king_zone_white = get_king_zone(pieces[KING].trailing_zeros() as usize, true);
    let king_zone_black = get_king_zone(pieces[KING + 6].trailing_zeros() as usize, false);

    let mut units_arr = [0, 0];

    for (zone, opp_offset, idx) in [(king_zone_white, 6, 0), (king_zone_black, 0, 1)] {
        for square in get_bits(zone) {
            for (piece, units) in [
                    (KNIGHT, MINOR_ATTACK_UNITS), 
                    (BISHOP, MINOR_ATTACK_UNITS), 
                    (ROOK, ROOK_ATTACK_UNITS), 
                    (QUEEN, QUEEN_ATTACK_UNITS)
                ] {
                let shift = (piece + opp_offset) * 2;
                units_arr[idx] += (attacks[square] >> shift & 0b11).count_ones() * units;
            }
        }
    }
    
    SAFETY_TABLE[units_arr[1] as usize] - SAFETY_TABLE[units_arr[0] as usize]
}

/// Returns a mask of the "king zone", which includes the king's circle and
/// three additional squares from each square in the circle in the direction of the enemy.
pub fn get_king_zone(square: usize, white: bool) -> u64 {
    let mut king_zone = get_king_circle(square);
    for _ in 0..2 {
        king_zone |= if white { king_zone << 8 } else { king_zone >> 8};
    }
    king_zone
}

/// Evaluation based on patterns of pawns
fn pawn_structure(pieces: [u64; 12]) -> i16 {
    let mut sum = 0;
    for white_pawn in get_bits(pieces[PAWN]) {
        if ChessMove::rank(white_pawn as u16) == 1 {
            continue;
        }
        let adjacent_files = get_adjacent_files(white_pawn as u16);
        if (0b101u64 << (white_pawn + 7)) & adjacent_files & pieces[PAWN] != 0 {
            sum += CONNECTED_PAWN_BONUS;
        }
        if (get_file(white_pawn as u16) & pieces[PAWN]).count_ones() > 1 {
            sum -= DOUBLED_PAWN_PENALTY;
        } else if get_king_circle(white_pawn) & pieces[PAWN] == 0 {
            sum -= ISOLATED_PAWN_PENALTY;
        }
    }
    for black_pawn in get_bits(pieces[PAWN + 6]) {
        if ChessMove::rank(black_pawn as u16) == 6 {
            continue;
        }
        let adjacent_files = get_adjacent_files(black_pawn as u16);
        if (0b101u64 << (black_pawn - 9)) & adjacent_files & pieces[PAWN + 6] != 0 {
            sum += -CONNECTED_PAWN_BONUS;
        }
        if (get_file(black_pawn as u16) & pieces[PAWN + 6]).count_ones() > 1 {
            sum -= -DOUBLED_PAWN_PENALTY;
        } else if get_king_circle(black_pawn) & pieces[PAWN + 6] == 0 {
            sum -= -ISOLATED_PAWN_PENALTY;
        }
    }
    sum
}

/// Returns adjacent files without ranks 1 and 8 (pawns can't be on them)
fn get_adjacent_files(square: u16) -> u64 {
    let mut return_value = 0u64;
    let file = ChessMove::file(square);
    let file_template = 0b1000000010000000100000001000000010000000100000000;
    if file > 0 {
        return_value |= file_template << (file - 1)
    }
    if file < 7 {
        return_value |= file_template << (file + 1)
    }
    return_value
}

// return current file bitmask
fn get_file(square: u16) -> u64 {
    0b01000000010000000100000001000000010000000100000000u64 << ChessMove::file(square)
}

/// Evaluation based on position of pieces
/// Tables taken from PeSTO: https://www.chessprogramming.org/PeSTO%27s_Evaluation_Function
/// 
/// We interpolate between middle game (MG) and end game (EG)
/// tables, where >= 28 pieces is fully middle game and <= 4 is end game
fn piece_square_tables(pieces: [u64; 12]) -> i16 {
    let mut occupied = 0u64;
    for piece in pieces {
        occupied |= piece;
    }
    let n_pieces = occupied.count_ones();
    let mg_weight = INTERPOLANTS[n_pieces as usize];
    let eg_weight = 10 - mg_weight;

    let mut sum = 0;
    
    for (piece, mg_table, eg_table) in [
        (PAWN, MG_PAWN, EG_PAWN),
        (KNIGHT, MG_KNIGHT, EG_KNIGHT),
        (BISHOP, MG_BISHOP, EG_BISHOP),
        (ROOK, MG_ROOK, EG_ROOK), 
        (QUEEN, MG_QUEEN, EG_QUEEN), 
        (KING, MG_KING, EG_KING)
    ] {
        for square in get_bits(pieces[piece]) {
            let flip = FLIP[square];
            sum += mg_table[flip] * mg_weight + eg_table[flip] * eg_weight
        }
        for square in get_bits(pieces[piece + 6]) {
            sum -= mg_table[square] * mg_weight + eg_table[square] * eg_weight
        }
    }

    sum / 10
}

/// Evaluation based on the number and type of material
/// on the board, but not position of the material
fn material_eval(pieces: [u64; 12]) -> i16 { 
    count_pieces(pieces) + 
    bishop_pairs(pieces) + 
    knight_pawn_interacton(pieces)
}

/// Knights are more effective with pawns. For every pawn
/// over 8, each knight is worth an extra 5 centipawns.
/// For every pawn under 8, minus 5.
fn knight_pawn_interacton(pieces: [u64; 12]) -> i16 {
    let n_pawns = (pieces[PAWN] | pieces[PAWN + 6]).count_ones();

    (pieces[KNIGHT].count_ones() - pieces[KNIGHT + 6].count_ones()) as i16 * 
    (n_pawns as i16 - 8) * KNIGHT_PAWN_INTERACTION
}

/// A bishop pair is worth half a pawn
fn bishop_pairs(pieces: [u64; 12]) -> i16 {
    (pieces[BISHOP].count_ones() == 2) as i16 * BISHOP_PAIR_BONUS - 
    (pieces[BISHOP + 6].count_ones() == 2) as i16 * BISHOP_PAIR_BONUS
}

/// Naive piece counting
fn count_pieces(pieces: [u64; 12]) -> i16 {
    let mut sum = 0;
    let idxs_weights = vec![
        (PAWN, PAWN_VALUE),
        (KNIGHT, KNIGHT_VALUE),
        (BISHOP, BISHOP_VALUE),
        (ROOK, ROOK_VALUE),
        (QUEEN, QUEEN_VALUE)
    ];
    for &(idx, weight) in &idxs_weights {
        sum += weight * pieces[idx].count_ones() as i16;
        sum -= weight * pieces[idx + 6].count_ones() as i16;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    
    #[test]
    fn count_pieces_test() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(0, count_pieces(Board::new().pieces));
        assert_eq!(-600, count_pieces(Board::from_fen("2r3k1/pp1q4/1n2pr2/n2p3Q/2PP4/7P/2R2PP1/5RK1 w - - 0 32")?.pieces));
        Ok(())
    }

    #[test]
    fn bishop_pairs_respected() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(0, bishop_pairs(Board::new().pieces));
        assert_eq!(-PAWN_VALUE/2, bishop_pairs(Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK1NR w KQkq - 0 1")?.pieces));
        Ok(())
    }

    #[test]
    fn knight_pawn_interaction_respected() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(-8 * KNIGHT_PAWN_INTERACTION, knight_pawn_interacton(Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1")?.pieces));
        assert_eq!(0, knight_pawn_interacton(Board::from_fen("rnbqkbnr/4pppp/8/8/8/8/PPPP4/RNBQKB1R w KQkq - 0 1")?.pieces));
        assert_eq!(4 * KNIGHT_PAWN_INTERACTION, knight_pawn_interacton(Board::from_fen("rnbqkbnr/4pppp/8/8/8/8/8/RNBQKB1R w KQkq - 0 1")?.pieces));
        Ok(())
    }

    #[test]
    fn pawn_adjacent_files() {
        let square = 0;
        let adjacent: Vec<_> = get_bits(get_adjacent_files(square)).collect();
        assert_eq!(vec![9, 17, 25, 33, 41, 49], adjacent);

        let square = 1;
        let adjacent: Vec<_> = get_bits(get_adjacent_files(square)).collect();
        assert_eq!(vec![8, 10, 16, 18, 24, 26, 32, 34, 40, 42, 48, 50], adjacent);

        let square = 23;
        let adjacent: Vec<_> = get_bits(get_adjacent_files(square)).collect();
        assert_eq!(vec![14, 22, 30, 38, 46, 54], adjacent);
    }

    #[test]
    fn pawn_file() {
        let square = 0;
        let adjacent: Vec<_> = get_bits(get_file(square)).collect();
        assert_eq!(vec![8, 16, 24, 32, 40, 48], adjacent);

        let square = 1;
        let adjacent: Vec<_> = get_bits(get_file(square)).collect();
        assert_eq!(vec![9, 17, 25, 33, 41, 49], adjacent);

        let square = 23;
        let adjacent: Vec<_> = get_bits(get_file(square)).collect();
        assert_eq!(vec![15, 23, 31, 39, 47, 55], adjacent);
    }

    #[test]
    fn pawn_structure_connected() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(-CONNECTED_PAWN_BONUS * 2, pawn_structure(Board::from_fen("k7/8/2p5/3p4/4p3/8/8/K7 w - - 0 1")?.pieces));
        assert_eq!(-CONNECTED_PAWN_BONUS, pawn_structure(Board::from_fen("k7/8/6p1/7p/8/8/8/K7 w - - 0 1")?.pieces));
        assert_eq!(-CONNECTED_PAWN_BONUS, pawn_structure(Board::from_fen("k7/8/6p1/5p2/8/8/8/K7 w - - 0 1")?.pieces));
        assert_eq!(-CONNECTED_PAWN_BONUS * 2, pawn_structure(Board::from_fen("k7/8/p7/1p6/2p5/8/8/K7 w - - 0 1")?.pieces));
        assert_eq!(0, pawn_structure(Board::from_fen("k7/8/8/pp6/6pp/8/8/K7 w - - 0 1")?.pieces));

        assert_eq!(CONNECTED_PAWN_BONUS, pawn_structure(Board::from_fen("k7/8/4P3/3P4/8/8/8/K7 w - - 0 1")?.pieces));
        assert_eq!(CONNECTED_PAWN_BONUS, pawn_structure(Board::from_fen("k7/8/4P3/5P2/8/8/8/K7 w - - 0 1")?.pieces));
        assert_eq!(CONNECTED_PAWN_BONUS, pawn_structure(Board::from_fen("k7/7P/6P1/8/8/8/8/K7 w - - 0 1")?.pieces));
        assert_eq!(CONNECTED_PAWN_BONUS, pawn_structure(Board::from_fen("k7/8/8/8/P7/1P6/8/K7 w - - 0 1")?.pieces));
        assert_eq!(3 * CONNECTED_PAWN_BONUS, pawn_structure(Board::from_fen("k7/8/4P3/3P4/P1P5/1P6/8/K7 w - - 0 1")?.pieces));

        Ok(())
    }

    #[test]
    fn pawn_structure_doubled() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(DOUBLED_PAWN_PENALTY * 2, pawn_structure(Board::from_fen("k7/8/1p6/1p6/8/8/8/K7 w - - 0 1")?.pieces));
        assert_eq!(5 * DOUBLED_PAWN_PENALTY, pawn_structure(Board::from_fen("k7/4p3/1p2p3/1p2p3/4p3/8/8/K7 w - - 0 1")?.pieces));

        assert_eq!(-DOUBLED_PAWN_PENALTY,  pawn_structure(Board::from_fen("k7/8/8/8/8/6P1/6P1/K7 w - - 0 1")?.pieces));
        Ok(())
    }

    #[test]
    fn pawn_structure_isolated() -> Result<(), Box<dyn std::error::Error>> { 
        assert_eq!(-2 * ISOLATED_PAWN_PENALTY, pawn_structure(Board::from_fen("k7/8/8/8/4P3/6P1/8/K7 w - - 0 1")?.pieces));
        assert_eq!(2 * ISOLATED_PAWN_PENALTY, pawn_structure(Board::from_fen("k7/8/8/6p1/8/6p1/8/K7 w - - 0 1")?.pieces));
        Ok(())
    }

    #[test]
    fn attacks_knights_plausible() -> Result<(), Box<dyn std::error::Error>> {
        let move_generator = MoveGenerator::new();
        
        let board = Board::from_fen("8/8/8/8/3N4/5N2/8/8 w - - 0 1")?;
        let attacks = get_attacks(board.pieces, &move_generator);
        for square in [4, 6, 10, 12, 15, 17, 21, 27, 31, 33, 36, 37, 38, 42, 44] {
            assert_eq!(0b01, attacks[square] >> 2 & 0b11);
        }

        let board = Board::from_fen("8/8/8/8/3n4/5n2/8/8 w - - 0 1")?;
        let attacks = get_attacks(board.pieces, &move_generator);
        for square in [4, 6, 10, 12, 15, 17, 21, 27, 31, 33, 36, 37, 38, 42, 44] {
            assert_eq!(0b01, attacks[square] >> 14 & 0b11);
        }

        let board = Board::from_fen("8/8/2p1p3/1p2ppp1/7p/1p1N1N2/2p1p2p/4p1p1 w - - 0 1")?;
        let attacks = get_attacks(board.pieces, &move_generator);
        for square in [4, 36] {
            assert_eq!(0b11, attacks[square] >> 2 & 0b11);
        }

        Ok(())
    }

    #[test]
    fn attacks_sliding_plausible() -> Result<(), Box<dyn std::error::Error>> {
        let move_generator = MoveGenerator::new();
        let board = Board::from_fen("8/4B1Q1/8/6B1/8/2r1P3/2q5/8 w - - 0 1")?;

        let attacks = get_attacks(board.pieces, &move_generator);
        
        assert_eq!(0b01, attacks[20] >> ((ROOK + 6) * 2) & 0b11);
        assert_eq!(0b00, attacks[20] >> ((QUEEN + 6) * 2) & 0b11);

        assert_eq!(0b01, attacks[19] >> ((ROOK + 6) * 2) & 0b11);
        assert_eq!(0b01, attacks[19] >> ((QUEEN + 6) * 2) & 0b11);

        assert_eq!(0b11, attacks[45] >> (BISHOP * 2) & 0b11);
        assert_eq!(0b01, attacks[45] >> (QUEEN * 2) & 0b11);

        Ok(())
    }

    #[test]
    fn king_zone_correct() -> Result<(), Box<dyn std::error::Error>> {
        for square in 0..64 {
            let zone = get_king_zone(square, true);
            assert!(zone.count_ones() <= 15);

            let rank = ChessMove::rank(square as u16);
            let file = ChessMove::file(square as u16);

            if rank > 0 {
                assert_eq!(1, zone >> (square - 8) & 1);
                if file > 0 {
                    assert_eq!(1, zone >> (square - 9) & 1);
                }
                if file < 7 {
                    assert_eq!(1, zone >> (square - 7) & 1);
                }
            }
            if file > 0 {
                assert_eq!(1, zone >> (square - 1) & 1);
            }
            if file < 7 {
                assert_eq!(1, zone >> (square + 1) & 1);
            }
            for forward in 1..=2 {
                if rank <= 7 - forward as u16 {
                    assert_eq!(1, zone >> (square + 8 * forward) & 1);
                    if file > 0 {
                        assert_eq!(1, zone >> (square + (8 * forward) - 1) & 1);
                    }
                    if file < 7 {
                        assert_eq!(1, zone >> (square + (8 * forward) + 1) & 1);
                    }
                }
            }
        }

        Ok(())
    }
}