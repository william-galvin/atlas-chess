use std::fmt;
use crate::chess_move::ChessMove;
use crate::zobrist::ZobristBoardComponent;
use std::ops::Range;
use std::panic;

use pyo3::prelude::*;

const MOVE_MASK: u64 = 1;

pub const WHITE: u64 = 1;
pub const BLACK: u64 = 0;
pub const KING: usize = 5;
pub const QUEEN: usize = 4;
pub const ROOK: usize = 3;
pub const BISHOP: usize = 2;
pub const KNIGHT: usize = 1;
pub const PAWN: usize = 0;

#[pyclass]
#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub struct Board {
    pub pieces: [u64; 12],
    pub info: u64,
    pub zobrist: ZobristBoardComponent,
    pieces_stacks: [Vec<u64>; 12],
    move_stack: Vec<MoveRecord>,
}

#[derive(PartialEq, Eq, Hash, Debug, Clone)]
struct MoveRecord {
    info: u64,
    moved: u16
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

impl Board {
    /// Initializes a board in the
    /// default state
    pub fn new() -> Self {
        Self::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap()
    }

    /// Parses a position from a given FEN
    pub fn from_fen(fen: &str) -> Result<Self, String> {
        let pieces = Self::pieces_from_fen(fen);
        let mut components = fen.split_whitespace();
        components.next();
        let mut info = 0u64;

        info |= match components.next().unwrap() {
            "w" => 1,
            "b" => 0,
            _ => return Err("Invalid side-to-move".to_string())
        };

        for c in components.next().unwrap().chars() {
            info |= 1 << match c {
                '-' => break,
                'K' => 1,
                'Q' => 2,
                'k' => 3,
                'q' => 4,
                _ => return Err("Invalid castling specifier".to_string())
            };
        }

        let en_pass = components.next().unwrap();
        if en_pass != "-" {
            let file = en_pass.chars().next().unwrap() as u8 - b'a';
            info |= 1 << (file + match info & MOVE_MASK {
                WHITE => 5,
                BLACK => 13,
                _ => return Err("Failed to understand en passant".to_string())
            });
        }

        Ok(Self {
            pieces,
            info,
            pieces_stacks: Default::default(),
            move_stack: Vec::new(),
            zobrist: ZobristBoardComponent::new(&pieces, info)
        })
    }

    fn pieces_from_fen(fen: &str) -> [u64; 12] {
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
                    let digit = c.to_digit(10);
                    match digit {
                        Some(d) => {
                            square += d;
                            continue;
                        },
                        None => panic!("Invalid piece type")
                    }
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
        pieces
    }

    /// Takes a FEN from standard notion, reverses the ranks
    /// so that rank 1 comes first
    #[inline]
    fn reverse_fen(fen: &str) -> String {
        fen.split('/').rev().collect::<Vec<&str>>().join("/")
    }

    /// Represent the current position in the
    /// canonical FEN (without the move counts)
    pub fn to_fen(&self) -> String {
        let mut fen = String::new();

        let symbols = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k'];

        for rank in 0..8 {
            let mut empty_spaces = 0;
            for file in 0..8 {
                let square = 8 * rank + file;
                let mut found = false;
                for (piece, symbol) in symbols.iter().enumerate() {
                    let bitboard = self.pieces[piece];
                    if bitboard >> square & 1 == 1 {
                        if empty_spaces != 0{
                            fen = format!("{}{}", fen, empty_spaces);
                        }
                        fen.push(*symbol);
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
        fen = Self::reverse_fen(&fen);
        fen.remove(0);

        fen.push_str(match self.info & MOVE_MASK {
            WHITE => " w ",
            BLACK => " b ",
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

        let (offset, rank) = if self.info & MOVE_MASK == WHITE {
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

    /// Update the current state of the board
    /// to reflect the new move. Push the old state onto the stack.
    /// **ASSUMES THAT MOVE IS LEGAL**. Undefined behavior if not.
    pub fn push_move(&mut self, chess_move: ChessMove) {
        let from = chess_move.from();
        let to = chess_move.to();

        let mut blockers = 0u64;
        for i in 0..12 {
            blockers |= self.pieces[i];
        }

        let move_type = self.get_piece(from, 0..12, blockers).unwrap();

        let mut capture_type = usize::MAX;
        let mut promotion_type = usize::MAX;
        let mut castle_type = usize::MAX;

        let mut en_pass = u16::MAX;
        let mut castle=  0;

        match self.get_piece(to, if move_type < 6 {6..12} else {0..6}, blockers) {
            Some(p) => {
                capture_type = p;
            },
            None => {
                if (move_type == 0 || move_type == 6) && ChessMove::file(from) != ChessMove::file(to) {
                    en_pass = ChessMove::file(to);
                    capture_type = 6 - move_type;
                }
            }
        }

        if (move_type == 0 || move_type == 6) && (ChessMove::rank(to) == 0 || ChessMove::rank(to) == 7) {
            promotion_type = move_type + (chess_move.special() as usize) - 3;
        }

        if (move_type == 5 || move_type == 11) && ChessMove::file(from) == 4 {
            castle = ChessMove::file(to); // 6 if kingside, 2 if queenside
        }
        if castle == 6 || castle == 2 {
            castle_type = move_type - 2;
        }


        let mut move_record = MoveRecord{ info: self.info, moved: 0 };
        for p_type in [move_type, capture_type, castle_type, promotion_type] {
            if p_type != usize::MAX {
                self.pieces_stacks[p_type].push(self.pieces[p_type]);
                move_record.moved |= 1 << p_type;
            }
        }
        self.move_stack.push(move_record);


        self.pieces[move_type] = Self::move_bit(self.pieces[move_type], from, to);
        self.zobrist.xor(move_type, from);
        self.zobrist.xor(move_type, to);

        if en_pass != u16::MAX {
            let (capture_type, offset) = if self.info & MOVE_MASK == WHITE {
                (6, to - 8)
            } else {
                (0, to + 8)
            };
            self.pieces[capture_type] = Self::delete_bit(self.pieces[capture_type], offset);
            self.zobrist.xor(capture_type, offset);
        } else if capture_type != usize::MAX {
            self.pieces[capture_type] = Self::delete_bit(self.pieces[capture_type], to);
            self.zobrist.xor(capture_type, to);
        }

        if castle_type != usize::MAX {
            let offset =  if self.info & MOVE_MASK == WHITE {
                0
            } else {
                56
            };
            let (src_offset, dst_offset) = if ChessMove::file(to) == 6 {
                (offset + 7, offset + 5)
            } else {
                (offset, offset + 3)
            };
            self.pieces[castle_type] = Self::move_bit(self.pieces[castle_type], src_offset, dst_offset);
            self.zobrist.xor(castle_type, src_offset);
            self.zobrist.xor(castle_type, dst_offset);
        }

        if promotion_type != usize::MAX {
            self.pieces[move_type] = Self::delete_bit(self.pieces[move_type], to);
            self.pieces[promotion_type] |= 1 << to;
            self.zobrist.xor(move_type, to);
            self.zobrist.xor(promotion_type, to);
        }

        for i in 0..8 {
            let info_idx = i + if self.to_move() == WHITE { 5 } else { 13 };
            if (self.info >> info_idx & 1) == 1 {
                self.zobrist.xor_info(info_idx);
            }
        }
        let clear_enpass_mask = !(0b111111111111111100000);
        self.info &= clear_enpass_mask;
        if move_type == 0 && ChessMove::rank(from) + 2 == ChessMove::rank(to) {
            let info_idx = 13 + ChessMove::file(to);
            self.info |= 1 << info_idx;
            self.zobrist.xor_info(info_idx);
        }
        if move_type == 6 && ChessMove::rank(from) == ChessMove::rank(to) + 2 {
            let info_idx: u16 = 5 + ChessMove::file(to);
            self.info |= 1 << info_idx;
            self.zobrist.xor_info(info_idx);
        }

        if self.info & MOVE_MASK == WHITE {
            if self.info >> 1 & 1 == 1 && (move_type == 5 || (move_type == 3 && from == 7)) {
                self.info = Self::delete_bit(self.info, 1);
                self.zobrist.xor_info(1);
            }
            if self.info >> 2 & 1 == 1 && (move_type == 5 || (move_type == 3 && from == 0)) {
                self.info = Self::delete_bit(self.info, 2);
                self.zobrist.xor_info(2);
            }
        } else {
            if self.info >> 3 & 1 == 1 && (move_type == 11 || (move_type == 9 && from == 63)) {
                self.info = Self::delete_bit(self.info, 3);
                self.zobrist.xor_info(3);
            }
            if self.info >> 4 & 1 == 1 && (move_type == 11 || (move_type == 9 && from == 56)) {
                self.info = Self::delete_bit(self.info, 4);
                self.zobrist.xor_info(4);
            }
        }
        if self.info >> 1 & 1 == 1 && to == 7 && capture_type == ROOK {
            self.info = Self::delete_bit(self.info, 1);
            self.zobrist.xor_info(1);
        }
        if self.info >> 2 & 1 == 1 && to == 0 && capture_type == ROOK {
            self.info = Self::delete_bit(self.info, 2);
            self.zobrist.xor_info(2);
        }
        if self.info >> 3 & 1 == 1 && to == 63 && capture_type == ROOK + 6 {
            self.info = Self::delete_bit(self.info, 3);
            self.zobrist.xor_info(3);
        }
        if self.info >> 4 & 1 == 1 && to == 56 && capture_type == ROOK + 6 {
            self.info = Self::delete_bit(self.info, 4);
            self.zobrist.xor_info(4);
        }

        self.info ^= MOVE_MASK;
        self.zobrist.xor_info(0);
    }

    /// Returns board to state of previous move
    /// Assumes that there is at least one previous move,
    /// panics if not
    pub fn pop_move(&mut self) {
        let move_record = self.move_stack.pop().unwrap();

        let mut info_diff = self.info ^ move_record.info;
        while info_diff.count_ones() > 0 {
            self.zobrist.xor_info(info_diff.trailing_zeros() as u16);
            info_diff &= !(1 << info_diff.trailing_zeros());
        }
        self.info = move_record.info;

        for piece_type in 0..12 {
            if move_record.moved >> piece_type & 1 == 1 {
                let old = self.pieces_stacks[piece_type].pop().unwrap();
                let mut diff = self.pieces[piece_type] ^ old;
                
                while diff.count_ones() > 0 {
                    self.zobrist.xor(piece_type, diff.trailing_zeros() as u16);
                    diff &= !(1 << diff.trailing_zeros());
                }
                
                self.pieces[piece_type] = old;
            }
        }
    }

    /// Returns the piece type of the piece on the given square
    /// Returns None if no piece on that square
    pub fn get_piece(&self, square: u16, range: Range<usize>, blockers: u64) -> Option<usize> {
        if blockers >> square & 1 == 0 {
            return None;
        }
        for idx in range {
            if (self.pieces[idx] >> square) & 1 == 1 {
                return Some(idx)
            }
        }
        None
    }

    /// Swaps sides of the board
    pub fn reflect_pieces(&self) -> [u64; 12] {
        let mut reflected = [0u64; 12];
        for i in 0..=11 {
            reflected[i] = self.pieces[i];
        }

        for piece in 0..=11 {
            for rank in 0..=3 {
                let shift1 = rank * 8usize;
                let shift2 = (7 - rank) * 8usize; 

                let tmp = (reflected[piece] >> shift1) & 0b11111111;

                reflected[piece] &= !(0b11111111 << shift1);
                reflected[piece] |= ((reflected[piece] >> shift2) & 0b11111111) << shift1;

                reflected[piece] &= !(0b11111111 << shift2);
                reflected[piece] |= tmp << shift2;
            }
        }
        reflected
    }

    pub fn move_bit(value: u64, source_index: u16, destination_index: u16) -> u64 {
        let bit = (value >> source_index) & 1;
        let cleared_value = value & !(1 << source_index);
        cleared_value | (bit << destination_index)
    }

    pub fn delete_bit(value: u64, index: u16) -> u64 {
        let mask = !(1 << index);
        value & mask
    }

    // Compares if two fens are equal, ignoring move count.
    // If self.fen has an enpassant move but other doesn't still passes.
    // Doesn't go both ways though.
    #[cfg(test)]
    fn fen_eq(&self, other_fen: &str) -> bool {
        let this_fen = self.to_fen();
        let mut this = this_fen.split_whitespace();
        let mut other = other_fen.split_whitespace();
        for _ in 0..3 {
            if this.next().unwrap() != other.next().unwrap() {
                return false;
            }
        }

        let enpass_other = other.next().unwrap();
        enpass_other == "-" || enpass_other == this.next().unwrap()
    }

    /// Returns WHITE or BLACK (ints)
    /// corresponding to whose move it is
    pub fn to_move(&self) -> u64 {
        self.info & MOVE_MASK
    }

    /// True if white can castle king side, false otherwise
    pub fn white_king_castle(&self) -> bool {
        self.info >> 1 & 1 == 1
    }

    /// True if black can castle king side, false otherwise
    pub fn black_king_castle(&self) -> bool {
        self.info >> 3 & 1 == 1
    }

    /// True if white can castle queen side, false otherwise
    pub fn white_queen_castle(&self) -> bool {
        self.info >> 2 & 1 == 1
    }

    /// True if black can castle queen side, false otherwise
    pub fn black_queen_castle(&self) -> bool {
        self.info >> 4 & 1 == 1
    }
}

#[pymethods]
impl Board {
    #[new]
    #[pyo3(signature = (fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"))]
    pub fn py_new(fen: &str) -> PyResult<Self> {
        match Self::from_fen(fen) {
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("Invalid fen: {}", e))),
            Ok(b) => Ok(b)
        }
    }

    #[staticmethod]
    pub fn bitboard_from_fen(fen: &str) -> PyResult<[[bool; 64]; 12]> {
        let pieces = Self::pieces_from_fen(fen);
        let mut ret_val = [[false; 64]; 12];
        for i in 0..12 {
            for j in 0..64 {
                ret_val[i][j] = (pieces[i] & (1 << j)) != 0;
            }
        }
        Ok(ret_val)
    }

    pub fn fen(&self) -> PyResult<String> {
        Ok(self.to_fen())
    }

    pub fn push(&mut self, chess_move: &str) -> PyResult<()> {
        match ChessMove::from_str(chess_move) {
            Ok(valid_move) => {
                self.push_move(valid_move);
                Ok(())
            }
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("Invalid chess move: {}", e))), 
        }
    }

    pub fn pop(&mut self) {
        self.pop_move();
    }

    pub fn bitboard(&self) -> PyResult<[[bool; 64]; 12]> {
        let mut ret_val = [[false; 64]; 12];
        for i in 0..12 {
            for j in 0..64 {
                ret_val[i][j] = (self.pieces[i] & (1 << j)) != 0;
            }
        }
        Ok(ret_val)
    }

    pub fn turn(&self) -> PyResult<String> {
        Ok( String::from(
            match self.to_move() {
                WHITE => "white",
                BLACK => "black",
                _ => panic!("unexpected side to move")
            }
        ))
    }

    pub fn reflect(&mut self) -> PyResult<()> {
        for piece in 0..=11 {
            for rank in 0..=3 {
                let shift1 = rank * 8usize;
                let shift2 = (7 - rank) * 8usize; 

                let tmp = (self.pieces[piece] >> shift1) & 0b11111111;

                self.pieces[piece] &= !(0b11111111 << shift1);
                self.pieces[piece] |= ((self.pieces[piece] >> shift2) & 0b11111111) << shift1;

                self.pieces[piece] &= !(0b11111111 << shift2);
                self.pieces[piece] |= tmp << shift2;
            }
        }

        for i in 0..=5 {
            let i = i;
            self.pieces.swap(i, i + 6);
        }

        for i in 1..=2 {
            let tmp = (self.info >> i) & 1;
            
            self.info &= !(1 << i);
            self.info |= ((self.info >> (i + 2)) & 1) << i;
            
            self.info &= !(1 << (i + 2));
            self.info |= tmp << (i + 2);
        }

        for i in 5..=12 {
            let tmp = (self.info >> i) & 1;
            
            self.info &= !(1 << i);
            self.info |= ((self.info >> (i + 8)) & 1) << i;
            
            self.info &= !(1 << (i + 8));
            self.info |= tmp << (i + 8) ;
        }

        self.info ^= MOVE_MASK;

        self.pieces_stacks = Default::default();
        self.move_stack = Vec::new();

        Ok(())
    }

    pub fn copy(&self) -> PyResult<Self> {
        Ok(self.clone())
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
    use std::fs::{File, read_dir};

    #[test]
    fn test_reverse_fen() {
        let fen = "rnb1kbnr/pppp1ppp/8/4p3/1q1P1N2/8/PPP1PPPP/RNBQKB1R";
        let reversed = "RNBQKB1R/PPP1PPPP/8/1q1P1N2/4p3/8/pppp1ppp/rnb1kbnr";
        assert_eq!(reversed, Board::reverse_fen(fen));
    }

    #[test]
    fn test_from_fen_starting() {
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -";
        let b = Board::new();
        let f = Board::from_fen(fen).unwrap();
        assert_eq!(b, f);
    }

    #[test]
    fn test_to_fen_starting() {
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -";
        let b = Board::new();
        let f = b.to_fen();
        assert_eq!(f, fen);
    }

    #[test]
    fn test_non_starting_fen() {
        let fen = "rnb1kbnr/pppp1ppp/8/4p3/1q1P1N2/8/PPP1PPPP/RNBQKB1R b K -";
        let b = Board::from_fen(fen).unwrap();
        assert_eq!(b.to_fen(), fen);
    }

    #[test]
    fn test_to_from_fen_fuzzy() -> io::Result<()> {
        let file = File::open("tests/test_fens.txt")?;
        let reader = io::BufReader::new(file);
        for line_result in reader.lines() {
            let fen = line_result?;
            let board = Board::from_fen(&fen).unwrap();
            assert_eq!(board.to_fen(), fen);
            assert_eq!(fen, format!("{}", board));
        }
        Ok(())
    }

    #[test]
    fn test_move_bit() {
        assert_eq!(Board::move_bit(0b0010, 1, 2), 0b0100);
        assert_eq!(Board::move_bit(0b1001, 0, 2), 0b1100);
        assert_eq!(Board::move_bit(0b1011, 3, 1), 0b0011);
    }

    #[test]
    fn test_push_move_basic() {
        let mut board = Board::new();
        let cm = ChessMove::from_str("e2e4").unwrap();
        board.push_move(cm);
        assert!(board.fen_eq("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -"));

        board.push_move(ChessMove::from_str("e7e5").unwrap());
        assert!(board.fen_eq("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6"));

        board.push_move(ChessMove::from_str("f2f4").unwrap());
        assert_eq!("rnbqkbnr/pppp1ppp/8/4p3/4PP2/8/PPPP2PP/RNBQKBNR b KQkq f3", board.to_fen());

        board.push_move(ChessMove::from_str("d7d6").unwrap());
        assert_eq!("rnbqkbnr/ppp2ppp/3p4/4p3/4PP2/8/PPPP2PP/RNBQKBNR w KQkq -", board.to_fen());

        board.push_move(ChessMove::from_str("g1f3").unwrap());
        assert_eq!("rnbqkbnr/ppp2ppp/3p4/4p3/4PP2/5N2/PPPP2PP/RNBQKB1R b KQkq -", board.to_fen());
    }

    #[test]
    fn test_move_castle_king_white() {
        let mut board = Board::from_fen("r2qkbnr/ppp2ppp/2np4/4p3/2B1PPb1/5N2/PPPP2PP/RNBQK2R w KQkq - 4 5").unwrap();
        board.push_move(ChessMove::from_str("e1g1").unwrap());
        assert_eq!("r2qkbnr/ppp2ppp/2np4/4p3/2B1PPb1/5N2/PPPP2PP/RNBQ1RK1 b kq -", board.to_fen());
    }

    #[test]
    fn test_castle_queen_white() {
        let mut board = Board::from_fen("rnbqkbnr/pp2p2p/2p3p1/3p2p1/2PP4/2N5/PP1QPPPP/R3KBNR w KQkq - ").unwrap();
        board.push_move(ChessMove::from_str("e1c1").unwrap());
        assert_eq!("rnbqkbnr/pp2p2p/2p3p1/3p2p1/2PP4/2N5/PP1QPPPP/2KR1BNR b kq -", board.to_fen());
    }

    #[test]
    fn test_castle_king_black() {
        let mut board = Board::from_fen("rnbqk2r/pp5p/2p1pnp1/3p2p1/1bPP4/1PN1PN2/P2Q1PPP/2KR1B1R b kq -").unwrap();
        board.push_move(ChessMove::from_str("e8g8").unwrap());
        assert_eq!("rnbq1rk1/pp5p/2p1pnp1/3p2p1/1bPP4/1PN1PN2/P2Q1PPP/2KR1B1R w - -", board.to_fen());
    }

    #[test]
    fn test_castle_queen_black() {
        let mut board = Board::from_fen("r3kbnr/ppp2ppp/2np4/4p1q1/2B1PPb1/3P1N1P/PPP3P1/RNBQK2R b KQkq - ").unwrap();
        board.push_move(ChessMove::from_str("e8c8").unwrap());
        assert_eq!("2kr1bnr/ppp2ppp/2np4/4p1q1/2B1PPb1/3P1N1P/PPP3P1/RNBQK2R w KQ -", board.to_fen());
    }

    #[test]
    fn test_capture_black() {
        let mut board = Board::from_fen("2kr1bnr/ppp2ppp/2np4/4p1q1/2B1PPb1/3P1N1P/PPP3P1/RNBQK2R w KQ -").unwrap();
        board.push_move(ChessMove::from_str("f4g5").unwrap());
        assert_eq!("2kr1bnr/ppp2ppp/2np4/4p1P1/2B1P1b1/3P1N1P/PPP3P1/RNBQK2R b KQ -", board.to_fen());
    }

    #[test]
    fn test_capture_white() {
        let mut board = Board::from_fen("2kr1bnr/ppp2ppp/2np4/4p1P1/2B1P1b1/3P1N1P/PPP3P1/RNBQK2R b KQ -").unwrap();
        board.push_move(ChessMove::from_str("g4h3").unwrap());
        assert_eq!("2kr1bnr/ppp2ppp/2np4/4p1P1/2B1P3/3P1N1b/PPP3P1/RNBQK2R w KQ -", board.to_fen());
    }

    #[test]
    fn test_enpassant_black() {
        let mut board = Board::from_fen("rnbqkbnr/pppp2pp/5p2/3Pp3/8/8/PPP1PPPP/RNBQKBNR w KQkq e6").unwrap();
        board.push_move(ChessMove::from_str("d5e6").unwrap());
        assert_eq!("rnbqkbnr/pppp2pp/4Pp2/8/8/8/PPP1PPPP/RNBQKBNR b KQkq -", board.to_fen());
    }

    #[test]
    fn test_enpassant_white() {
        let mut board = Board::from_fen("rnbqkbnr/pp1p2pp/4Pp2/8/1Pp5/4P3/P1P2PPP/RNBQKBNR b KQkq b3").unwrap();
        board.push_move(ChessMove::from_str("c4b3").unwrap());
        assert_eq!("rnbqkbnr/pp1p2pp/4Pp2/8/8/1p2P3/P1P2PPP/RNBQKBNR w KQkq -", board.to_fen());
    }

    #[test]
    fn test_promote_white() {
        for p in ['n', 'b', 'r', 'q'] {
            let mut board = Board::from_fen("rnbq1bnr/pp1P1kpp/5p2/8/8/1p2P3/P1P2PPP/RNBQKBNR w KQ - ").unwrap();
            let move_ = format!("d7c8{}", p);
            board.push_move(ChessMove::from_str(&move_).unwrap());
            assert_eq!(format!("rn{}q1bnr/pp3kpp/5p2/8/8/1p2P3/P1P2PPP/RNBQKBNR b KQ -", p.to_uppercase()), board.to_fen());
        }
    }

    #[test]
    fn test_promote_black() {
        for p in ['n', 'b', 'r', 'q'] {
            let mut board = Board::from_fen("rnq2bnr/pp3kpp/5p2/8/P7/N3P3/1pP2PPP/R1BQKBNR b KQ - ").unwrap();
            let move_ = format!("b2b1{}", p);
            board.push_move(ChessMove::from_str(&move_).unwrap());
            assert_eq!(format!("rnq2bnr/pp3kpp/5p2/8/P7/N3P3/2P2PPP/R{}BQKBNR w KQ -", p), board.to_fen());
        }
    }

    #[test]
    fn test_push_pop_fuzzy() -> io::Result<()> {
        let entries = read_dir("tests/random_games")?;
        for entry in entries {
            let entry = entry?;
            let file = File::open(entry.path())?;
            let reader = io::BufReader::new(file);
            let mut board = Board::new();
            let mut lines = reader.lines();
            while let Some(line1) = lines.next() {
                if let Some(line2) = lines.next() {
                    let old_fen = board.to_fen();
                    let old_zobrist = board.zobrist.clone();

                    let chess_move = line1?;
                    let target_fen = line2?;
                    board.push_move(ChessMove::from_str(&chess_move).unwrap());
                    assert!(board.fen_eq(&target_fen));

                    board.pop_move();
                    assert!(board.fen_eq(&old_fen));
                    assert!(board.zobrist.eq(&old_zobrist), "played {chess_move} from {old_fen} and zobrist hashes didn't align on pop");

                    board.push_move(ChessMove::from_str(&chess_move).unwrap());
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_enpass_push_pop() {
        let fen = "r3k2r/p1ppqpb1/1n2pnp1/1b1PN3/Pp2P3/2N2Q1p/1PPBBPPP/1R2K2R b Kkq a3";
        let mut board = Board::from_fen(fen).unwrap();
        let old_zobrist = board.zobrist.clone();
        board.push_move(ChessMove::from_str("b4a3").unwrap());
        board.pop_move();
        assert!(board.fen_eq(fen));
        assert!(board.zobrist.eq(&old_zobrist));
    }

    #[test]
    fn test_capture_rook_castling() {
        let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/P1N2Q2/1PPBBPpP/1R2K2R b Kkq - 0 2";
        let mut board = Board::from_fen(fen).unwrap();
        board.push_move(ChessMove::from_str("g2h1b").unwrap());
        println!("fen is\n{}", board.to_fen());
        assert!(board.fen_eq("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/P1N2Q2/1PPBBP1P/1R2K2b w kq -"));
    }
}