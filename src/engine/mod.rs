use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::board::{Board, WHITE};

use crate::constants::UCIConfig;
use crate::liches::tablebase_lookup;
use crate::lru_cache::LRUCache;
use crate::static_evaluation::{static_evaluation, QUEEN_VALUE};
use crate::zobrist::{ZobristHashTableEntry, ZobristHashTable};
use crate::move_generator::MoveGenerator;
use crate::chess_move::ChessMove;

use ort::session::{
    builder::GraphOptimizationLevel, 
    Session,
};
use ndarray::Array3;
use rand::seq::SliceRandom;
use rand::thread_rng;

pub struct Engine {
    move_generator: MoveGenerator,
    transposition_table: Arc<ZobristHashTable>,
    nn: Arc<Session>,
    ponder: Ponder,
    uci: UCIConfig,
    book: HashMap<String, (ChessMove, i16)>,
}

struct Ponder {
    cache: Arc<Mutex<LRUCache<String, (ChessMove, i16)>>>,
    deadline: Arc<RwLock<Instant>>
}

impl Engine {
    pub fn new(move_generator: MoveGenerator, uci: UCIConfig, ) -> Result<Self, Box<dyn std::error::Error>> {
        let weights_path = get_file("nn.quant.onnx");
        let nn = Arc::from(
            Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(uci.n_onnx_threads)?
                .commit_from_file(weights_path.to_str().unwrap())?
            );

        let transposition_table = ZobristHashTable::new(uci.tt_cache_size);
        let ponder = Ponder {
            cache: Arc::from(Mutex::new(LRUCache::new(uci.ponder_cache_size))), 
            deadline: Arc::from(RwLock::new(Instant::now()))
        };

        let book = get_book("opening_book.csv")?;

        Ok(Self{ move_generator, transposition_table, nn, ponder, uci, book })
    }

    /// Starts pondering a given board.
    /// The given board should be the state of the game directly after the move
    /// returned from `go` has been played.
    /// Pondering is done async - this is non-blocking. Call `ponder_stop`
    /// before next call to `go`.
    pub fn ponder_start(
        &self, 
        board: &Board, 
        depth: u8, 
    ) { 
        let mut deadline = self.ponder.deadline.write().unwrap();
        *deadline = Instant::now() + Duration::from_secs(3600);
        
        let thread_deadline = self.ponder.deadline.clone();
        let thread_cache = self.ponder.cache.clone();

        let mut board = board.clone();
        let transposition_table = self.transposition_table.clone();
        let move_generator = self.move_generator.clone();
        let nn = self.nn.clone();
        let uci = self.uci.clone();
        thread::spawn(move || {
            let mut moves = move_generator.moves(&mut board);
            order_moves_nn(&board, &mut moves, &nn).unwrap();

            let mut i = 0;
            let l = moves.len();
            for chess_move in moves {
                board.push_move(chess_move);
                let child_fen = board.to_fen();

                let color = board.to_move() as i16 * 2 - 1;
                let result = lazy_smp_iddfs(
                    &board, color, depth, transposition_table.clone(), 
                    &move_generator, nn.clone(), thread_deadline.clone(), uci.clone()
                );

                board.pop_move();

                if let Ok((_depth, best_move)) = result {
                    let mut cache = thread_cache.lock().unwrap();
                    cache.put(child_fen, (best_move.0.unwrap(), best_move.1));
                    i += 1;
                } else {
                    break;
                }
            }
            eprintln!("info ponder search: {}/{}", i, l);
        }); 
    }

    /// Signals thread to stop pondering
    pub fn ponder_stop(&self) {
        let mut deadline = self.ponder.deadline.write().unwrap();
        *deadline = Instant::now();
    }

    /// Returns a tuple of (best_move, eval) for a given board
    pub fn go(
        &mut self, 
        board: &Board, 
        depth: u8, 
        search_time: Duration
    ) -> Result<(Option<ChessMove>, i16), Box<dyn std::error::Error>> {

        if self.uci.own_book {
            if let Some((book_move, eval)) = self.book.get(&strip_en_pass(&board.to_fen())) {
                eprintln!("info book hit");
                return Ok((Some(*book_move), *eval));
            }
        }

        if self.uci.tablebase && board.count_pieces() <= 7 {
            if let Some(best_move) = tablebase_lookup(&board) {
                eprintln!("info tablebase hit");
                return Ok((Some(best_move), i16::MAX));
            }
        }

        if let Some(best_move) = self.ponder.cache.lock().unwrap().get(&board.to_fen()) {
            eprintln!("info ponder hit");
            return Ok((Some(best_move.0), best_move.1));
        }
        
        let deadline = Arc::from(RwLock::new(Instant::now() + search_time));
        let color = board.to_move() as i16 * 2 - 1;
        if let Ok((depth, result)) = lazy_smp_iddfs(
            &board, color, depth, self.transposition_table.clone(), 
            &self.move_generator, self.nn.clone(), deadline, self.uci.clone()
        ) {
            eprintln!("info search depth {}", depth);
            return Ok(result);
        } else {
            panic!("search didn't finish");
        }
    }
}

/// Spawn a worker to explore root at depth -- does not return value
fn async_negamax(
    root: &mut Board, 
    color: i16, 
    depth: u8, 
    transposition_table: Arc<ZobristHashTable>,
    move_generator: &MoveGenerator,
    nn: Arc<Session>,
    results: Arc<Mutex<Vec<(u8, (Option<ChessMove>, i16))>>>,
    deadline: Arc<RwLock<Instant>>, 
    uci: UCIConfig,
) -> JoinHandle<()> {
    let mut root = root.clone();
    let move_generator = move_generator.clone();
    let tt = transposition_table.clone();
    thread::spawn(move || {
        let result = negamax(
            &mut root, depth, depth, i16::MIN + 1, i16::MAX - 1, color,
            tt, &move_generator, &nn, deadline, uci
        );
        if let Ok(res) = result {
            let mut results = results.lock().unwrap();
            results.push((depth, res));
        }
    })
}

/// See https://www.chessprogramming.org/Iterative_Deepening
/// and https://www.chessprogramming.org/Lazy_SMP
fn lazy_smp_iddfs(
    root: &Board, 
    color: i16, 
    depth: u8,
    transposition_table: Arc<ZobristHashTable>,
    move_generator: &MoveGenerator,
    nn: Arc<Session>,
    deadline: Arc<RwLock<Instant>>, 
    uci: UCIConfig,
) -> Result<(u8, (Option<ChessMove>, i16)), Box<dyn std::error::Error>> {

    let root = &mut root.clone();

    let mut threads = vec![];
    let results = Arc::from(Mutex::new(vec![]));
    
    for d in 2..depth {
        threads.push(
            async_negamax(root, color, d, transposition_table.clone(),
            &move_generator.clone(), nn.clone(), results.clone(), deadline.clone(), uci.clone()
        ));
    }

    for _ in 0..uci.lazy_smp_parallel_root {
        threads.push(
            async_negamax(root, color, depth, transposition_table.clone(),
            &move_generator.clone(), nn.clone(), results.clone(), deadline.clone(), uci.clone()
        ));
    }

    let result = negamax(
        root, depth, depth, i16::MIN + 1, i16::MAX - 1, color,
        transposition_table.clone(), move_generator, &nn, deadline.clone(), uci.clone()
    );

    threads.into_iter().for_each(|t| { let _  = t.join(); } );

    let mut results = results.lock().unwrap();
    if let Ok(res) = result {
        results.push((depth, res));
    }
    if results.is_empty() {
        return Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, "search timed out")));
    }

    results.sort_by_key(|r| r.0);
    Ok(results.pop().unwrap())
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
    transposition_table: Arc<ZobristHashTable>,
    move_generator: &MoveGenerator,
    nn: &Session,
    deadline: Arc<RwLock<Instant>>, 
    uci: UCIConfig,
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
            let eval = -i16::MAX + (16 - depth) as i16 + 1;
            transposition_table.put(z_key, ZobristHashTableEntry::new(z_sum, depth, eval, ChessMove::default()));
            return Ok((Some(ChessMove::default()), eval));
        }
    }

    if depth == 0 {
        let eval = qsearch(root, alpha, beta, color, move_generator, uci.qsearch_depth);
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

    if depth >= max_depth - uci.deep_move_ordering_depth {
        order_moves_nn(root, &mut child_nodes, nn)?;    
    } 

    if let Some(pv) = principal_variation {
        // why do we need this check? For some reason pv can be
        // other player's move??? hash collision or something worse?
        if child_nodes.contains(&pv) {
            child_nodes.insert(0, pv);
        }
    }

    shuffle_tail(&mut child_nodes, uci.lazy_smp_shuffle_n);
    
    if Instant::now() > *deadline.read().unwrap() {
        return Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, "search timed out")));
    }

    let mut value = (None, i16::MIN);    
    for &child in &child_nodes {
        root.push_move(child);
        let (_child_move, child_value) = negamax(
            root, depth - 1, max_depth, -beta, -alpha, -color,
            transposition_table.clone(), move_generator, nn, deadline.clone(), uci.clone()
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

fn qsearch(
    root: &mut Board,
    mut alpha: i16,
    beta: i16,
    color: i16, 
    move_generator: &MoveGenerator, 
    depth: u8,
) -> i16 {

    let check = move_generator.check(root);
    let mut moves = vec![];
    if check {
        moves = move_generator.moves(root);
        if moves.len() == 0 {
            return  -i16::MAX + (16 - depth) as i16;
        }
    } 

    let mut best_value = static_evaluation(root.pieces) * color;

    if depth == 0 {
        return best_value;
    }

    // delta pruning
    let delta = QUEEN_VALUE;
    if !check && best_value + delta < alpha {
        return best_value;
    }

    if best_value >= beta {
        return best_value;
    }
    if alpha < best_value {
        alpha = best_value;
    }

    if moves.is_empty() {
        moves = move_generator.moves(root);
    }
    let occupied = root.occupied();
    for chess_move in moves {
        let capture = occupied >> chess_move.to() & 1 == 1;
        if !capture && !check {
            continue;
        }

        root.push_move(chess_move);
        let score = -qsearch(root, -beta, -alpha, -color, move_generator, depth - 1);
        root.pop_move();

        if score > best_value {
            best_value = score;
        }
        if best_value >= beta {
            return best_value;
        }
        if alpha < best_value {
            alpha = best_value;
        }

    }

    return best_value;
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
fn order_moves_nn(root: &Board, child_nodes: &mut [ChessMove], nn: &Session) -> Result<(), Box<dyn std::error::Error>>{
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

/// Shuffles the last LAZY_SMP_SHUFFLE_N elements
fn shuffle_tail(child_nodes: &mut [ChessMove], shuffle_n: usize) {
    if child_nodes.len() > shuffle_n {
        let (_first, rest) = child_nodes.split_at_mut(shuffle_n);
        rest.shuffle(&mut thread_rng())
    }
}

fn get_file(path: &str) -> PathBuf {
    let exe_dir = std::env::current_exe()
        .expect("Failed to get the current executable path")
        .parent()
        .expect("Executable path has no parent")
        .to_path_buf();

    let mut file = exe_dir.join(path);
    if !file.exists() {
        let exe_dir = std::env::current_exe()
        .expect("Failed to get the current executable path")
        .parent()
        .expect("Executable path has no parent")
        .parent()
        .expect("Executable path has no parent")
        .to_path_buf();

        file = exe_dir.join(path);
    }

    if !file.exists() {
        panic!("file don't exist");
    }

    file
}

fn get_book(path: &str) -> io::Result<HashMap<String, (ChessMove, i16)>> {
    let file = File::open(get_file(path))?;
    let reader = io::BufReader::new(file);

    let mut book = HashMap::new();

    for line in reader.lines() {
        let mut _line = line?;
        let mut line = _line.split(",");
        let fen = line.next().unwrap();

        let mut board = Board::from_fen(fen).unwrap();
        let color = board.to_move() as i16 * 2 - 1;

        let mut moves: Vec<(ChessMove, i16)> = line.next()
            .unwrap()
            .split(" ")
            .into_iter()
            .map(|m| ChessMove::from_str(m).unwrap())
            .map(|m| {
                board.push_move(m);
                let eval = static_evaluation(board.pieces);
                board.pop_move();
                (m, eval)
            })
            .collect();

        moves.sort_by_key(|k| k.1 * color);

        let key = strip_en_pass(fen);
        book.insert(key, moves.pop().unwrap());
    }

    Ok(book)
}

fn strip_en_pass(fen: &str) -> String {
    let mut components: Vec<&str> = fen.split_whitespace().collect();
    if !components.is_empty() {
        components.pop();
    }
    components.join(" ")
}

pub fn format_score_info(score: i16) -> String {
    if score < i16::MAX - 100 || score > i16::MIN + 100 {
        return format!("cp {}", score);
    } else {
        // hacky magic taken from negamax
        // eval = -i16::MAX + (16 - depth) as i16 + 1;
        //  => depth = -(eval + i16::MAX - 1 - 16)
        let ply = -(score + i16::MAX - 1 - 16);
        let moves = ply / 2;
        return format!("mate {}", moves);
    }
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
        let mut uci = UCIConfig::default();
        uci.tt_cache_size = constants::GIB / 2048;

        let engine = Engine::new(
            MoveGenerator::new(), 
            uci
        ).unwrap();

        let board = Board::new();
        let res = engine.nn.run(ort::inputs![bitboard_onnx_order_vec(board.pieces)].unwrap()).unwrap();
        let _scores = &res[2].try_extract_tensor::<f32>().unwrap();

        Ok(())
    }

    #[test]
    fn can_find_mate_in_2() -> Result<(), Box<dyn std::error::Error>> {
        let mut uci = UCIConfig::default();
        uci.tt_cache_size = constants::GIB / 1024;

        let mut engine = Engine::new(
            MoveGenerator::new(), 
            uci.clone()
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
            let (_best_move, score) = engine.go(&board, 3, uci.search_time)?;
            assert!(score <= -i16::MAX + 50 || score >= i16::MAX - 50);
            eprint!(".");
        }
        println!("Time to check mate in 2: {:?}", start.elapsed());
        Ok(())
    }

    #[test]
    fn can_find_mate_in_3() -> Result<(), Box<dyn std::error::Error>> {
        let mut uci = UCIConfig::default();
        uci.tt_cache_size = constants::GIB / 4;

        let mut engine = Engine::new(
            MoveGenerator::new(), 
            uci
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
            let (_best_move, score) = engine.go(&board, 5, Duration::from_secs(100))?;
            let best_move = _best_move.unwrap();
            assert!(score <= -i16::MAX + 50 || score >= i16::MAX - 50, "didn't find mate in 3, score is {score} and move is {best_move}");
            eprint!(".");
        }
        println!("Time to check mate in 3: {:?}", start.elapsed()); // Time to beat: 0.9 seconds 
        Ok(())
    }

    #[test]
    fn can_find_mate_in_four_simple() -> Result<(), Box<dyn std::error::Error>> {
        let mut uci = UCIConfig::default();
        uci.tt_cache_size = constants::GIB / 1024;

        let mut engine = Engine::new(
            MoveGenerator::new(),
            uci
        )?;

        let board = Board::from_fen("2r5/1k4q1/2pRp3/ppQ1Pp1p/8/2P1P3/PP3PrP/3R1K2 b - - 1 30")?;
        let (_best_move, score) = engine.go(&board, 7, Duration::from_secs(100))?;
        let best_move = _best_move.unwrap();
        assert!(score <= -i16::MAX + 50 || score >= i16::MAX - 50, "didn't find mate in 4, score is {score} and move is {best_move}");
        Ok(())
    }

    #[ignore = "long running"]
    #[test]
    fn can_find_mate_in_five_simple() -> Result<(), Box<dyn std::error::Error>> {
        let mut uci = UCIConfig::default();
        uci.tt_cache_size = constants::GIB / 2048;

        let mut engine = Engine::new(
            MoveGenerator::new(), 
            uci
        )?;

        let start = Instant::now();
        let board = Board::from_fen("4r1k1/pp3p1p/q1p2np1/4P1Q1/8/1P1P3P/P3R1P1/7K w - - 0 29")?;
        let (_best_move, score) = engine.go(&board, 9, Duration::from_secs(1000))?;
        let best_move = _best_move.unwrap();
        assert!(score <= -i16::MAX + 50 || score >= i16::MAX - 50, "didn't find mate in 4, score is {score} and move is {best_move}");
        println!("Time to check mate in 5 simple: {:?}", start.elapsed()); 
        Ok(())
    }

    #[ignore = "long running"]
    #[test]
    fn can_find_mate_in_4() -> Result<(), Box<dyn std::error::Error>> {
        let start = Instant::now();

        let mut uci = UCIConfig::default();
        uci.tt_cache_size = constants::GIB;

        let mut engine = Engine::new(
            MoveGenerator::new(),
            uci
        )?;

        let mut correct = 0;

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
            let (_best_move, score) = engine.go(&board, 7, Duration::from_secs(60))?;
            let __best_move = _best_move.unwrap();
            let right = score <= -i16::MAX + 50 || score >= i16::MAX - 50;
            if right {
                correct += 1;
            }
            eprintln!(" done in {:?}, correct: {}", start2.elapsed(), right); // time to beat: 10.31
        }
        println!("Time to check puzzles: {:?}, {}/20 correct", start.elapsed(), correct); 
        assert!(correct > 15);
        Ok(())
    }

}


