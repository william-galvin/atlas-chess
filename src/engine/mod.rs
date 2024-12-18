use std::path::PathBuf;

use crate::board::{Board, WHITE};
use crate::constants::{DEEP_MOVE_ORDERING_DEPTH, NN_WEIGHTS, N_ONNX_THREADS};
use crate::zobrist::{ZobristHashTableEntry, ZobristHashTable};
use crate::move_generator::MoveGenerator;
use crate::chess_move::ChessMove;

use ort::session::{
    builder::GraphOptimizationLevel, 
    Session,
};
use ndarray::Array3;

pub struct Engine {
    move_generator: MoveGenerator,
    transposition_table: std::sync::Arc<ZobristHashTable>,
    nn: Session,
    depth: u8
}

impl Engine {
    pub fn new(move_generator: MoveGenerator, depth: u8, cache_size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let weights_path = get_weights();
        let nn = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(N_ONNX_THREADS)?
                .commit_from_file(weights_path.to_str().unwrap())?;

        let transposition_table = ZobristHashTable::new(cache_size);

        Ok(Self{ move_generator, transposition_table, nn, depth })
    }

    /// Returns a tuple of (best_move, eval) for a given board
    pub fn go(&mut self, board: &Board) -> Result<(Option<ChessMove>, i16), Box<dyn std::error::Error>> {
        let color = board.to_move() as i16 * 2 - 1;
        let mut board = board.copy()?;
        self.iddfs(&mut board, color)
    }

    /// See https://www.chessprogramming.org/Iterative_Deepening
    fn iddfs(&mut self, root: &mut Board, color: i16) -> Result<(Option<ChessMove>, i16), Box<dyn std::error::Error>> {
        let depth = self.depth;
        for d in 3..depth {
            let _ = negamax(
                root, d, d, i16::MIN + 1, i16::MAX - 1, color,
                self.transposition_table.clone(), &self.move_generator, &self.nn
            );
        }

        negamax(
            root, depth, depth, i16::MIN + 1, i16::MAX - 1, color,
            self.transposition_table.clone(), &self.move_generator, &self.nn
        )
    }
}

/// Minimax tree search with alpha-beta pruning + cache + deep move ordering
/// See https://en.wikipedia.org/wiki/Negamax
fn negamax(
    root: &mut Board,
    depth: u8,
    max_depth: u8,
    mut alpha: i16,
    beta: i16,
    color: i16,
    transposition_table: std::sync::Arc<ZobristHashTable>,
    move_generator: &MoveGenerator,
    nn: &Session,
) -> Result<(Option<ChessMove>, i16), Box<dyn std::error::Error>> {

    let z_key = root.zobrist.z_key;
    let z_sum = root.zobrist.z_sum;
    let mut principal_variation = None;
    let cached = transposition_table.get(z_key);
    if cached.z_sum == z_sum {
        if cached.depth >= depth {
            return Ok((Some(cached.chess_move), cached.eval));
        }
        if cached.chess_move.non_default() {
            principal_variation = Some(cached.chess_move);
        }
    }
    
    let mut child_nodes = vec![];
    let check = move_generator.check(root);
    if check {
        child_nodes = move_generator.moves(root);
        if child_nodes.is_empty() {
            let eval = -i16::MAX + depth as i16 + 1;
            transposition_table.put(z_key, ZobristHashTableEntry::new(z_sum, depth, eval, ChessMove::default()));
            return Ok((Some(ChessMove::default()), eval));
        }
    }

    if depth == 0 {
        let eval = static_evaluation(root) * color;
        transposition_table.put(z_key, ZobristHashTableEntry::new(z_sum, depth, eval, ChessMove::default()));
        return Ok((Some(ChessMove::default()), eval));
    } 
    
    if !check {
        child_nodes = move_generator.moves(root);
    }

    if child_nodes.is_empty() {
        transposition_table.put(z_key, ZobristHashTableEntry::new(z_sum, depth, 0, ChessMove::default()));
        return Ok((Some(ChessMove::default()), 0));
    }

    if depth >= max_depth - DEEP_MOVE_ORDERING_DEPTH {
        order_moves_nn(root, &mut child_nodes, nn)?;    
    } 

    if let Some(pv) = principal_variation {
        // why do we need this check? For some reason pv can be
        // other player's move??? hash collision or something worse?
        if child_nodes.contains(&pv) {
            child_nodes.insert(0, pv);
        }
    }

    let mut value = (None, i16::MIN);    
    for &child in &child_nodes {
        root.push_move(child);
        let (_child_move, child_value) = negamax(
            root, depth - 1, max_depth, -beta, -alpha, -color,
            transposition_table.clone(), move_generator, nn
        )?;
        root.pop_move();

        if -child_value > value.1 {
            value = (Some(child), -child_value);
        }
        if -child_value > alpha {
            alpha = -child_value;
        }
        if alpha >= beta {
            break;
        }
    }
    
    transposition_table.put(z_key, ZobristHashTableEntry::new(z_sum, depth, value.1, value.0.unwrap_or_default()));
    Ok((value.0, value.1))
}

/// Returns an Array3 in the shape that the NN expects
/// given the pieces bitboard from a board
fn bitboard_onnx_order_vec(pieces: [u64; 12]) -> Array3<f32> {
    let mut ret_val = Array3::<f32>::zeros((1, 64, 12));
    for i in 0..12 {
        for j in 0..64 {
            if (pieces[i] & (1 << j)) != 0 {
                ret_val[[0, j, i]] = 1.0;
            }
        }
    }
    ret_val
}

/// Returns index into NN prediction vector to retrieve move score
/// Reflects idx if black to play
fn get_move_idx(chess_move: &ChessMove, side: u64) -> [usize; 2] {
    if side == WHITE {
        [0, (chess_move.from() * 64 + chess_move.to()) as usize]
    } else {
        let reflected_move = chess_move.reflect();
        [0, (reflected_move.from() * 64 + reflected_move.to()) as usize]
    }
}

/// Given and a set of its legal moves, sorts the moves by what the NN thinks
/// is most likely to be played
fn order_moves_nn(root: &Board, child_nodes: &mut Vec<ChessMove>, nn: &Session) -> Result<(), Box<dyn std::error::Error>>{
    let pieces = if root.to_move() == WHITE { root.pieces } else { root.reflect_pieces() };
    let outputs = nn.run(ort::inputs![bitboard_onnx_order_vec(pieces)]?)?;
    let move_scores = outputs[2].try_extract_tensor::<f32>()?;
    child_nodes.sort_by(|m1, m2| {
        let f1 = move_scores[get_move_idx(m1, root.to_move())];
        let f2 = move_scores[get_move_idx(m2, root.to_move())];
        f2.partial_cmp(&f1).unwrap()
    });
    Ok(())
}

/// Returns a large positive number if board is good for white,
/// large negative if good for black
fn static_evaluation(board: &Board) -> i16 {
    count_pieces(board)
}

/// Static material evaluation
fn count_pieces(board: &Board) -> i16 {
    let mut sum = 0;
    let idxs_weights = vec![(0, 100), (1, 300), (2, 300), (3, 500), (4, 900)]; // TODO put in cofig
    for &(idx, weight) in &idxs_weights {
        sum += weight * board.pieces[idx].count_ones() as i16;
        sum -= weight * board.pieces[idx + 6].count_ones() as i16;
    }
    sum
}

fn get_weights() -> PathBuf {
    let exe_dir = std::env::current_exe()
        .expect("Failed to get the current executable path")
        .parent()
        .expect("Executable path has no parent")
        .to_path_buf();

    let mut weights = exe_dir.join(NN_WEIGHTS);
    if !weights.exists() {
        let exe_dir = std::env::current_exe()
        .expect("Failed to get the current executable path")
        .parent()
        .expect("Executable path has no parent")
        .parent()
        .expect("Executable path has no parent")
        .to_path_buf();

        weights = exe_dir.join(NN_WEIGHTS);
    }

    if !weights.exists() {
        panic!("weights don't exist");
    }

    weights
}


#[cfg(test)]
mod tests {
    use crate::constants;
    use super::*;
    
    use std::io::{self, BufRead};
    use std::fs::File;
    use std::time::Instant;

    #[test]
    fn engine_test() -> Result<(), ()> {
        let engine = Engine::new(
            MoveGenerator::new(), 
            4,
            constants::GIB / 2048
        ).unwrap();

        let board = Board::new();
        let res = engine.nn.run(ort::inputs![bitboard_onnx_order_vec(board.pieces)].unwrap()).unwrap();
        let _scores = &res[2].try_extract_tensor::<f32>().unwrap();

        Ok(())
    }

    #[test]
    fn count_pieces_test() -> Result<(), Box<dyn std::error::Error>> {
        assert!(count_pieces(&Board::new()) == 0);
        assert!(count_pieces(&Board::from_fen("2r3k1/pp1q4/1n2pr2/n2p3Q/2PP4/7P/2R2PP1/5RK1 w - - 0 32")?) == -500);
        Ok(())
    }

    #[test]
    fn can_find_mate_in_2() -> Result<(), Box<dyn std::error::Error>> {
        let mut engine = Engine::new(
            MoveGenerator::new(), 
            3,
            constants::GIB / 1024
        )?;

        let start = Instant::now();

        let file = File::open("tests/mate_in_2.csv")?;
        let reader = io::BufReader::new(file);
        let mut i = 0;
        for line_result in reader.lines() {
            if i > 20 {
                break;
            }
            i += 1;

            let line = line_result?;
            let fen = line.split(",").nth(1).unwrap();
            let moves = line.split(",").nth(2).unwrap();
            let mut board = Board::from_fen(fen)?;
            board.push_move(ChessMove::from_str(moves.split(" ").next().unwrap())?);
            let (_best_move, score) = engine.go(&board)?;
            assert!(score <= -i16::MAX + 3 || score >= i16::MAX - 3);
            eprint!(".");
        }
        println!("Time to check puzzles: {:?}", start.elapsed()); // Time to beat: 4.3 seconds (running in release, move ordering only on first ply + second, not generating moves unless needed)
        Ok(())
    }

    #[test]
    fn can_find_mate_in_3() -> Result<(), Box<dyn std::error::Error>> {
        let mut engine = Engine::new(
            MoveGenerator::new(), 
            5,
            constants::GIB / 8
        )?;

        let start = Instant::now();

        let file = File::open("tests/mate_in_3.csv")?;
        let reader = io::BufReader::new(file);
        let mut i = 0;
        for line_result in reader.lines() {
            if i > 20 {
                break;
            }
            i += 1;

            let line = line_result?;
            let fen = line.split(",").nth(1).unwrap();
            let moves = line.split(",").nth(2).unwrap();
            let mut board = Board::from_fen(fen)?;
            board.push_move(ChessMove::from_str(moves.split(" ").next().unwrap())?);
            let (_best_move, score) = engine.go(&board)?;
            let best_move = _best_move.unwrap();
            assert!(score <= -i16::MAX + 5 || score >= i16::MAX - 5, "didn't find mate in 3, score is {score} and move is {best_move}");
            eprint!(".");
        }
        println!("Time to check puzzles: {:?}", start.elapsed()); // Time to beat: 4.3 seconds (running in release, move ordering only on first ply + second, not generating moves unless needed)
        Ok(())
    }

    #[test]
    fn can_find_mate_in_four_simple() -> Result<(), Box<dyn std::error::Error>> {
        let mut engine = Engine::new(
            MoveGenerator::new(), 
            7,
            constants::GIB / 2048
        )?;

        let board = Board::from_fen("2r5/1k4q1/2pRp3/ppQ1Pp1p/8/2P1P3/PP3PrP/3R1K2 b - - 1 30")?;
        let (_best_move, score) = engine.go(&board)?;
        let best_move = _best_move.unwrap();
        assert!(score <= -i16::MAX + 7 || score >= i16::MAX - 7, "didn't find mate in 4, score is {score} and move is {best_move}");
        Ok(())
    }

    #[ignore = "long running"]
    #[test]
    fn can_find_mate_in_five_simple() -> Result<(), Box<dyn std::error::Error>> {
        let mut engine = Engine::new(
            MoveGenerator::new(), 
            9,
            constants::GIB / 2048
        )?;

        let start = Instant::now();
        let board = Board::from_fen("4r1k1/pp3p1p/q1p2np1/4P1Q1/8/1P1P3P/P3R1P1/7K w - - 0 29")?;
        let (_best_move, score) = engine.go(&board)?;
        let best_move = _best_move.unwrap();
        assert!(score <= -i16::MAX + 7 || score >= i16::MAX - 7, "didn't find mate in 4, score is {score} and move is {best_move}");
        println!("Time to check puzzles: {:?}", start.elapsed()); 
        Ok(())
    }

    #[ignore = "long running"]
    #[test]
    fn can_find_mate_in_4() -> Result<(), Box<dyn std::error::Error>> {
        let start = Instant::now();

        let mut engine = Engine::new(
            MoveGenerator::new(), 
            7,
            constants::GIB
        )?;

        let file = File::open("tests/mate_in_4.csv")?;
        let reader = io::BufReader::new(file);
        let mut i = 0;
        for line_result in reader.lines() {
            if i > 20 {
                break;
            }
            i += 1;

            let line = line_result?;
            let fen = line.split(",").nth(1).unwrap();
            let moves = line.split(",").nth(2).unwrap();
            let mut board = Board::from_fen(fen)?;
            board.push_move(ChessMove::from_str(moves.split(" ").next().unwrap())?);
            eprint!("Attempting {} ...", &board.to_fen());
            let start2 = Instant::now();
            let (_best_move, score) = engine.go(&board)?;
            let best_move = _best_move.unwrap();
            assert!(score <= -i16::MAX + 7 || score >= i16::MAX - 7, "didn't find mate in 4 for fen={fen}, instead saw best move as {best_move} with score {score}");
            eprintln!(" done in {:?}", start2.elapsed());
        }
        println!("Time to check puzzles: {:?}", start.elapsed()); 
        Ok(())
    }

}


