mod chess_move;
mod board;
mod move_generator;
mod engine;
mod zobrist;
mod constants;
mod lru_cache;

use std::io::{self, BufRead, Write};
use std::fs::File;

use constants::UCIConfig;

use crate::board::Board;
use crate::engine::Engine;
use crate::move_generator::MoveGenerator;
use crate::chess_move::ChessMove;

struct GameManager {
    board: Board,
    engine: Engine,
    uci: UCIConfig,
}

impl GameManager {
    fn new() -> Self {
        let uci = UCIConfig::default();

        Self {
            board: Board::new(),
            engine: Engine::new(
                MoveGenerator::new(), 
                uci.clone()
            ).unwrap(),
            uci: uci
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
                println!("id name Atlas");
                println!("id author William Galvin");
                println!();
                println!("{}", UCIConfig::help());
                println!("uciok");
            },
            "option" => {
                println!("{}", UCIConfig::help());
            },
            "setoption" => {
                let name = msg[2];
                let value = msg[4];
                if game_manager.uci.set(name, value)? {
                    game_manager.engine = Engine::new(
                        MoveGenerator::new(), 
                        game_manager.uci.clone()
                    ).unwrap()
                }
            }
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
                let best_move = game_manager.engine.go(&game_manager.board, u8::MAX, game_manager.uci.search_time)?;
                println!("bestmove {:#}", best_move.0.unwrap());
                game_manager.board.push_move(best_move.0.unwrap());
                game_manager.engine.ponder_start(&game_manager.board, game_manager.uci.ponder_search_depth);
                game_manager.board.pop_move();
            }
            _ => {
                println!("Unknown command entered: {}", msg.join(" "));
            }
        }
    }
    Ok(())
}
