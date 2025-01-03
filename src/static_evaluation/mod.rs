use crate::board::{
    Board,
    PAWN,
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
};

pub const QUEEN_VALUE: i16 = 1000;
pub const ROOK_VALUE: i16 = 525;
pub const BISHOP_VALUE: i16 = 350;
pub const KNIGHT_VALUE: i16 = 350;
pub const PAWN_VALUE: i16 = 100;

const BISHOP_PAIR_BONUS: i16 = 50;
const KNIGHT_PAWN_INTERATION: i16 = 10;

/// Returns a large positive number if board is good for white,
/// large negative if good for black
pub fn static_evaluation(board: &Board) -> i16 {
    material_eval(board)
}

/// Evaluation based on the number and type of material
/// on the board, but not position of the material
fn material_eval(board: &Board) -> i16 { 
    count_pieces(board) + 
    bishop_pairs(board) + 
    knight_pawn_interacton(board)
}

/// Knights are more effective with pawns. For every pawn
/// over 8, each knight is worth an extra 10 centipawns.
/// For every pawn under 8, minus 10.
fn knight_pawn_interacton(board: &Board) -> i16 {
    let n_pawns = (board.pieces[PAWN] | board.pieces[PAWN + 6]).count_ones();

    (board.pieces[KNIGHT].count_ones() - board.pieces[KNIGHT + 6].count_ones()) as i16 * 
    (n_pawns as i16 - 8) * KNIGHT_PAWN_INTERATION
}

/// A bishop pair is worth half a pawn
fn bishop_pairs(board: &Board) -> i16 {
    (board.pieces[BISHOP].count_ones() == 2) as i16 * BISHOP_PAIR_BONUS - 
    (board.pieces[BISHOP + 6].count_ones() == 2) as i16 * BISHOP_PAIR_BONUS
}

/// Naive piece counting
fn count_pieces(board: &Board) -> i16 {
    let mut sum = 0;
    let idxs_weights = vec![
        (PAWN, PAWN_VALUE),
        (KNIGHT, KNIGHT_VALUE),
        (BISHOP, BISHOP_VALUE),
        (ROOK, ROOK_VALUE),
        (QUEEN, QUEEN_VALUE)
    ];
    for &(idx, weight) in &idxs_weights {
        sum += weight * board.pieces[idx].count_ones() as i16;
        sum -= weight * board.pieces[idx + 6].count_ones() as i16;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn count_pieces_test() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(0, count_pieces(&Board::new()));
        assert_eq!(-600, count_pieces(&Board::from_fen("2r3k1/pp1q4/1n2pr2/n2p3Q/2PP4/7P/2R2PP1/5RK1 w - - 0 32")?));
        Ok(())
    }

    #[test]
    fn bishop_pairs_respected() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(0, bishop_pairs(&Board::new()));
        assert_eq!(-PAWN_VALUE/2, bishop_pairs(&Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK1NR w KQkq - 0 1")?));
        Ok(())
    }

    #[test]
    fn knight_pawn_interaction_respected() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(-80, knight_pawn_interacton(&Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1")?));
        assert_eq!(0, knight_pawn_interacton(&Board::from_fen("rnbqkbnr/4pppp/8/8/8/8/PPPP4/RNBQKB1R w KQkq - 0 1")?));
        assert_eq!(40, knight_pawn_interacton(&Board::from_fen("rnbqkbnr/4pppp/8/8/8/8/8/RNBQKB1R w KQkq - 0 1")?));
        Ok(())
    }
}