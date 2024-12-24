mod chess_move;
mod random;
mod board;
mod move_generator;
mod engine;
mod zobrist;
mod constants;

use std::io::{self, BufRead, Write};
use std::fs::File;

use constants::{PONDER_SEARCH_DEPTH, SEARCH_TIME};

use crate::board::Board;
use crate::engine::Engine;
use crate::move_generator::MoveGenerator;
use crate::chess_move::ChessMove;

struct GameManager {
    board: Board,
    engine: Engine,
}

impl GameManager {
    fn new() -> Self {
        Self {
            board: Board::new(),
            engine: Engine::new(
                MoveGenerator::new(), 
                constants::TT_CACHE_SIZE
            ).unwrap(),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut log = File::create("log.out")?;
    let mut game_manager = GameManager::new();

    for line in io::stdin().lock().lines() {
        let input= line?;
        writeln!(log, "{}", input)?;
        let msg: Vec<&str> = input.split_whitespace().collect();

        if msg.is_empty() {
            continue;
        }

        match msg[0] {
            "uci" => {
                println!("id name Atlas100");
                println!("id author William Galvin");
                println!("uciok");
            },
            "isready" => {
                println!("readyok");
            },
            "ucinewgame" => {},
            "position" => {
                let moves_offset;
                game_manager.board = match msg[1] {
                    "startpos" => {
                        moves_offset = 3;
                        Board::new()
                    },
                    "fen" => {
                        moves_offset = 9;
                        Board::from_fen(&msg[2..=7].join(" "))?
                    },
                    _ => panic!("Failed to parse position")
                };
                if msg.len() > moves_offset {
                    for s in &msg[moves_offset..] {
                        match ChessMove::from_str(s) {
                            Ok(valid_move) => game_manager.board.push_move(valid_move),
                            Err(e) => print!("Error: {}", e)
                        };
                    }
                }
            },
            "go" => {
                game_manager.engine.ponder_stop();
                let best_move = game_manager.engine.go(&game_manager.board, 100, SEARCH_TIME)?;
                println!("bestmove {:#}", best_move.0.unwrap());
                game_manager.board.push_move(best_move.0.unwrap());
                game_manager.engine.ponder_start(&game_manager.board, PONDER_SEARCH_DEPTH);
                game_manager.board.pop_move();
            }
            _ => {
                println!("Unknown command entered: {}", msg.join(" "));
            }
        }
    }
    Ok(())
}
