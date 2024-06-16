mod chess_move;
mod random;
mod board;
mod move_generator;

use std::io::{self, BufRead, Write};
use std::fs::File;

use crate::board::Board;
use crate::move_generator::MoveGenerator;
use crate::chess_move::ChessMove;

struct GameManager {
    board: Board,
    move_generator: MoveGenerator
}

impl GameManager {
    fn new() -> Self {
        Self {
            board: Board::new(),
            move_generator: MoveGenerator::new()
        }
    }
}

fn main() -> io::Result<()> {

    let mut log = File::create("log.out")?;
    let mut game_manager = GameManager::new();

    for line in io::stdin().lock().lines() {
        let input= line?;
        writeln!(log, "{}", input)?;
        let msg: Vec<&str> = input.split_whitespace().collect();
        match msg[0] {
            "uci" => {
                println!("id name Atlas");
                println!("id author William Galvin");
                println!("uciok");
            },
            "isready" => {
                println!("readyok");
            },
            "ucinewgame" => {
                game_manager = GameManager::new()
            },
            "position" => {
                game_manager.board = match msg[1] {
                    "startpos" => {
                        Board::new()
                    },
                    "fen" => {
                        let mut board = Board::from_fen(&msg[2..=7].join(" ")).unwrap();
                        if msg.len() > 9 {
                            for s in &msg[9..] {
                                match ChessMove::from_str(s) {
                                    Ok(valid_move) => board.push_move(valid_move),
                                    Err(e) => print!("Error: {}", e)
                                };
                            }
                        }
                        board
                    },
                    _ => panic!("Failed to parse position")
                }
            },
            "go" => {
                let moves = game_manager.move_generator.moves(&mut game_manager.board);
                println!("bestmove {}", moves[0]);
            }
            _ => {
                println!("Unknown command entered: {}", msg.join(" "));
            }
        }
    }
    Ok(())
}
