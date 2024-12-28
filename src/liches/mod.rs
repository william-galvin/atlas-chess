use std::io::{Read, Write};
use std::net::TcpStream;

use crate::{board::Board, chess_move::ChessMove};

/// Looks up endgame tablebase on lichess, returns winning move
/// if available
pub fn tablebase_lookup(board: &Board) -> Option<ChessMove> {
    let fen = board.to_fen().replace(" ", "_");
    if let Ok(json_result) = lichess_lookup(&fen) {
        if !is_win(&json_result) {
            return None;
        }
        if let Some(best_move) = extract_moves_uci(&json_result) {
            return Some(ChessMove::from_str(&best_move).unwrap());
        }
    } 
    return None;
}

fn lichess_lookup(fen: &str) -> Result<String, std::io::Error> {
    let host = "tablebase.lichess.ovh";
    let port = 80;
    let path = format!("/standard?fen={}", fen);

    let mut stream = TcpStream::connect((host, port))?;

    let request = format!(
        "GET {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
        path, host
    );
    stream.write_all(request.as_bytes())?;

    let mut response = String::new();
    stream.read_to_string(&mut response)?;

    if let Some(pos) = response.find("\r\n\r\n") {
        Ok(response[pos + 4..].to_string())
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Failed to parse HTTP response",
        ))
    }
}

fn is_win(input: &str) -> bool {
    if let Some(index) = input.find("\"category\":") {
        let after_category = &input[index + 11..];
        return after_category.trim_start().starts_with("\"win\"");
    }
    false
}

fn extract_moves_uci(input: &str) -> Option<String> {
    if let Some(start_index) = input.find("\"moves\":[{\"uci\":") {
        let after_uci = &input[start_index + 16..]; 
        
        if let Some(end_index) = after_uci.find(',') {
            return Some(after_uci[1..end_index - 1].to_string());
        }
    }
    None
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lichess_basic_lookup() {
        assert_eq!(ChessMove::from_str("h7h8q").unwrap(), tablebase_lookup(&Board::from_fen("4k3/6KP/8/8/8/8/7p/8 w - - 0 1").unwrap()).unwrap());
        assert_eq!(None, tablebase_lookup(&Board::from_fen("r1b1k1nr/ppp1pp1p/2n3pb/6N1/1P6/P1N5/1BPP1KPP/R4B1R w kq - 0 1").unwrap()));
    }

}