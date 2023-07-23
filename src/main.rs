mod chess_move;
mod random;
mod board;
mod move_generator;

use chess_move::ChessMove;
use crate::board::Board;
use crate::move_generator::{MoveGenerator, perft};

use std::io::{self, BufRead, Write};
use std::fs::File;

fn main() -> std::io::Result<()> {
    //
    let mg = MoveGenerator::new();
    let mut board = Board::new();
    println!("{}", perft(6, &mut board, &mg));


    // let move_generator = MoveGenerator::new();
    //
    // let stdin = io::stdin();
    // let stdout = io::stdout();
    // let mut stdout_handle = stdout.lock();
    //
    // // Enter an infinite loop to read from stdin indefinitely
    // loop {
    //     let mut input_line = String::new();
    //
    //     match stdin.lock().read_line(&mut input_line) {
    //         Ok(bytes_read) => {
    //             if bytes_read == 0 {
    //                 // End of input (EOF), break the loop
    //                 break;
    //             }
    //
    //             let mut board = Board::from_fen(&input_line);
    //             let mut moves= vec![];
    //             for m in move_generator.moves(&mut board) {
    //                 moves.push(format!("{m}"))
    //             }
    //
    //             writeln!(stdout_handle, "{}", moves.join("::")).unwrap();
    //             stdout_handle.flush().unwrap();
    //         }
    //         Err(error) => {
    //             eprintln!("Error reading from stdin: {}", error);
    //             break;
    //         }
    //     }
    //
    // }
    Ok(())
}
