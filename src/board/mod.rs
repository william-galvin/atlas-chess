use std::fmt;

const START_WHITE_PAWN: u64 = 0xFF << 8;
const START_WHITE_KNIGHT: u64 = 1 << 1 | 1 << 6;
const START_WHITE_BISHOP: u64 = 1 << 2 | 1 << 5;
const START_WHITE_ROOK: u64 = 1 | 1 << 7;
const START_WHITE_QUEEN: u64 = 1 << 3;
const START_WHITE_KING: u64 = 1 << 4;

const START_BLACK_PAWN: u64 = 0xFF << 48;
const START_BLACK_KNIGHT: u64 = 1 << 57 | 1 << 62;
const START_BLACK_BISHOP: u64 = 1 << 58 | 1 << 61;
const START_BLACK_ROOK: u64 = 1 << 56 | 1 << 63;
const START_BLACK_QUEEN: u64 = 1 << 59;
const START_BLACK_KING: u64 = 1 << 60;

const START_INFO: u64 = 0b11111;

const MOVE_MASK: u64 = 1;

#[derive(PartialEq, Eq, Hash, Debug)]
struct Board {
    pieces: [u64; 12],
    info: u64
}

impl Board {

    /// Initializes a board in the
    /// default state
    pub fn new() -> Self {
        Self {
            pieces: [
                START_WHITE_PAWN,
                START_WHITE_KNIGHT,
                START_WHITE_BISHOP,
                START_WHITE_ROOK,
                START_WHITE_QUEEN,
                START_WHITE_KING,
                START_BLACK_PAWN,
                START_BLACK_KNIGHT,
                START_BLACK_BISHOP,
                START_BLACK_ROOK,
                START_BLACK_QUEEN,
                START_BLACK_KING
            ],
            info: START_INFO
        }
    }

    /// Parses a position from a given FEN
    pub fn from_fen(fen: &str) -> Self {
        let mut square = 0;
        let mut pieces = [0u64; 12];

        let mut components = fen.split_whitespace();

        for c in Self::reverse_fen(components.next().unwrap()).chars() {
            let idx = match c {
                '/' => continue,
                'p' | 'P' => 0,
                'n' | 'N' => 1,
                'b' | 'B' => 2,
                'r' | 'R' => 3,
                'q' | 'Q' => 4,
                'k' | 'K' => 5,
                _ => {
                    square += c.to_digit(10).unwrap();
                    continue
                }
            } + {
                if c.is_lowercase() {
                    6
                } else {
                    0
                }
            };
            pieces[idx] |= 1 << square;
            square += 1;
        }
        assert_eq!(square, 64);

        let mut info = 0u64;

        info |= match components.next().unwrap() {
            "w" => 1,
            "b" => 0,
            _ => panic!()
        };

        for c in components.next().unwrap().chars() {
            info |= 1 << match c {
                '-' => break,
                'K' => 1,
                'Q' => 2,
                'k' => 3,
                'q' => 4,
                _ => panic!()
            };
        }

        let en_pass = components.next().unwrap();
        if en_pass != "-" {
            let file = en_pass.chars().next().unwrap() as u8 - b'a';
            info |= 1 << (file + match info & MOVE_MASK {
                1 => 5,
                0 => 13,
                _ => panic!()
            });
        }

        Self {
            pieces,
            info
        }
    }

    /// Takes a FEN from standard notion, reverses the ranks
    /// so that rank 1 comes first
    #[inline]
    fn reverse_fen(fen: &str) -> String {
        fen.split('/').rev().collect::<Vec<&str>>().join("/")
    }

    pub fn to_fen(&self) -> String {
        let mut fen = String::new();

        let symbols = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k'];

        for rank in 0..8 {
            let mut empty_spaces = 0;
            for file in 0..8 {
                let square = 8 * rank + file;
                let mut found = false;
                for piece in 0..12 {
                    let bitboard = self.pieces[piece];
                    if bitboard >> square & 1 == 1 {
                        if empty_spaces != 0{
                            fen = format!("{}{}", fen, empty_spaces);
                        }
                        fen.push(symbols[piece]);
                        empty_spaces = 0;
                        found = true;
                        break
                    }
                }
                if !found {
                    empty_spaces += 1;
                }
            }
            if empty_spaces != 0 {
                fen = format!("{}{}", fen, empty_spaces);
            }
            fen.push('/');
        }
        fen = String::from(Self::reverse_fen(&fen));
        fen.remove(0);

        fen.push_str(match self.info & MOVE_MASK {
            1 => " w ",
            0 => " b ",
            _ => panic!()
        });

        let castling = ['K', 'Q', 'k', 'q'];
        let mut castle_rights = String::new();
        for idx in 1..=4 {
            if self.info >> idx & 1 == 1 {
                castle_rights.push(castling[idx - 1]);
            }
        }
        fen.push_str(if castle_rights.is_empty() {
            "-"
        } else {
          &castle_rights
        });
        fen.push(' ');

        let (offset, rank) = if self.info & MOVE_MASK == 1 {
            (5, 6)
        } else {
            (13, 3)
        };
        let mut enpass = String::new();
        for file in 0..8 {
            if self.info >> (file + offset) & 1 == 1 {
                enpass = format!("{}{}", (file as u8 + b'a') as char, rank);
                break;
            }
        }
        if enpass.is_empty() {
            enpass = String::from("-");
        }
        fen.push_str(&enpass);

        fen
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_fen())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{self, BufRead};
    use std::fs::File;

    #[test]
    fn test_reverse_fen() {
        let fen = "rnb1kbnr/pppp1ppp/8/4p3/1q1P1N2/8/PPP1PPPP/RNBQKB1R";
        let reversed = "RNBQKB1R/PPP1PPPP/8/1q1P1N2/4p3/8/pppp1ppp/rnb1kbnr";
        assert_eq!(reversed, Board::reverse_fen(fen));
    }

    #[test]
    fn test_from_fen_starting() {
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -";
        let mut b = Board::new();
        let f = Board::from_fen(fen);
        assert_eq!(b, f);
    }

    #[test]
    fn test_to_fen_starting() {
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -";
        let mut b = Board::new();
        let f = b.to_fen();
        assert_eq!(f, fen);
    }

    #[test]
    fn test_non_starting_fen() {
        let fen = "rnb1kbnr/pppp1ppp/8/4p3/1q1P1N2/8/PPP1PPPP/RNBQKB1R b K -";
        let b = Board::from_fen(fen);
        assert_eq!(b.to_fen(), fen);
    }

    #[test]
    fn test_to_from_fen_fuzzy() -> io::Result<()> {
        let file = File::open("src/board/test_fens.txt")?;
        let reader = io::BufReader::new(file);
        for line_result in reader.lines() {
            let fen = line_result?;
            let board = Board::from_fen(&fen);
            assert_eq!(board.to_fen(), fen);
            assert_eq!(fen, format!("{}", board));
        }
        Ok(())
    }
}