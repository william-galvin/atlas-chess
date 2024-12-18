use std::fmt;

const FROM_MASK: u16 =    0b0000000000111111;
const TO_MASK: u16 =      0b0000111111000000;
const SPECIAL_MASK: u16 = 0b1111000000000000;

#[derive(PartialEq, Eq, Hash, Clone, Copy, Default)]
pub struct ChessMove {
    data: u16
}

impl ChessMove {

    /// Construct a new chess move
    /// args:
    ///     - from: u16 representing from square (0-63)
    ///     - to: u16 representing to square (0-63)
    ///     - special: u16 representing special moves characteristics (0-7)
    pub fn new(from: u16, to: u16, special: u16) -> Self {
        assert!(from <= 63, "Invalid FROM int");
        assert!(to <= 63, "Invalid TO int");
        assert!(special <= 15, "Invalid SPECIAL int");

        Self {
            data: from | to << 6 | special << 12
        }
    }

    /// Takes a string in UCI-compatible algabraic notation.
    /// For example, e2e4, e2e4, e1g1 (castling), e7e8q (promotion)
    /// Leaves the `SPECIAL` field blank, except for promotions
    pub fn from_str(s_move: &str) -> Result<Self, String> {
        let s_move = s_move.trim();
        let promotion = match s_move.len() {
            5 => {
                match s_move.chars().last().unwrap() {
                    'n' | 'N' => 4,
                    'b' | 'B' => 5,
                    'r' | 'R' => 6,
                    'q' | 'Q' => 7,
                    _ => return Err(format!("Unsupported promotion type for {}", s_move))
                }
            },
            4 => 0,
            _ => return Err(format!("Unsupported move notation. Must be 4 or 5 (promotion) characters. Found: {}", s_move))
        };

        let v_move: Vec<char> = s_move.chars().collect();

        let from_file = v_move[0usize] as u8 - b'a';
        let from_rank = v_move[1usize] as u8 - b'1';
        assert!(from_file <= 7);
        assert!(from_rank <= 7);

        let to_file = v_move[2usize] as u8 - b'a';
        let to_rank = v_move[3usize] as u8 - b'1';
        assert!(to_file <= 7);
        assert!(to_rank <= 7);

        let from = from_rank * 8 + from_file;
        let to = to_rank * 8 + to_file;

        Ok(Self::new(from as u16, to as u16, promotion))
    }

    /// Returns from square as u16
    pub fn from(&self) -> u16 {
        self.data & FROM_MASK
    }

    /// returns to square as u16
    pub fn to(&self) -> u16 {
        (self.data & TO_MASK) >> 6
    }

    /// returns special characteristics as u16
    pub fn special(&self) -> u16 {
        (self.data & SPECIAL_MASK) >> 12
    }

    /// Convert a int (0, 63) to the string representation of a square
    pub fn square(n: u16) -> String {
        let file = n % 8;
        let rank = n / 8;
        format!("{}{}", (file as u8 + b'a') as char, rank + 1)
    }

    /// Reflects the move as if it had been played from the other color
    pub fn reflect(&self) -> Self {
        let from_rank = 7 - Self::rank(self.from());
        let to_rank = 7 - Self::rank(self.to());
        
        let from_square = Self::file(self.from()) + 8 * from_rank;
        let to_square = Self::file(self.to()) + 8 * to_rank;

        Self { 
            data: from_square | to_square << 6 | self.special() << 12
        }
    }

    /// Returns true if the move is a real move
    pub fn non_default(&self) -> bool {
        self.data != 0
    }

    /// Returns rank (0-7)
    #[inline]
    pub fn rank(n: u16) -> u16 {
        n >> 3
    }

    /// Returns file (0-7)
    #[inline]
    pub fn file(n: u16) -> u16 {
        n & 7
    }
}

/// Display format is UCI-compatible SAN
/// {from}{to}[promotion]
impl fmt::Display for ChessMove {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{}{}",
           ChessMove::square(self.from()),
           ChessMove::square(self.to()),
           match self.special() {
               0u16..=3u16 => "",
               4 => "n",
               5 => "b",
               6 => "r",
               7 => "q",
               _ => panic!()
           }
        )
    }
}

impl fmt::Debug for ChessMove {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[to={}, from={}, special={}]",
               ChessMove::square(self.to()),
               ChessMove::square(self.from()),
               match self.special() {
                   0 => "None",
                   1 => "en passant",
                   2 => "pawn hop",
                   3 => "castle",
                   4 => "knight promotion",
                   5 => "bishop promotion",
                   6 => "rook promotion",
                   7 => "queen promotion",
                   _ => "error"
               }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{*, super::random::Rng};

    #[test]
    fn test_constructor() {
        let cm = ChessMove::new(0, 0, 0);
        assert_eq!(cm.data, 0);

        let cm1 = ChessMove::new(1, 0, 0);
        assert_eq!(cm1.data, 1);

        let cm2 = ChessMove::new(63, 63, 15);
        assert_eq!(cm2.data, u16::MAX);
    }

    #[test]
    fn test_from() {
        let mut rng = Rng::new();
        for i in 0..=63 {
            let cm = ChessMove::new(i,
                                    rng.bounded_random(0, 64) as u16,
                                    rng.bounded_random(0, 16) as u16);
            assert_eq!(cm.from(), i);
        }
    }

    #[test]
    fn test_to() {
        let mut rng = Rng::new();
        for i in 0..=63 {
            let cm = ChessMove::new(rng.bounded_random(0, 64) as u16,
                                    i,
                                    rng.bounded_random(0, 16) as u16);
            assert_eq!(cm.to(), i);
        }
    }

    #[test]
    fn test_special() {
        let mut rng = Rng::new();
        for i in 0..=15 {
            let cm = ChessMove::new(rng.bounded_random(0, 64) as u16,
                                    rng.bounded_random(0, 64) as u16,
                                    i);
            assert_eq!(cm.special(), i);
        }
    }

    #[test]
    fn test_from_str() {
        let cm = ChessMove::from_str("e2e4").unwrap();
        assert_eq!(cm.to(), 28);
        assert_eq!(cm.from(), 12);
        assert_eq!(cm.special(), 0);

        let cm0 = ChessMove::from_str("a7a8b").unwrap();
        assert_eq!(cm0.from(), 48);
        assert_eq!(cm0.to(), 56);
        assert_eq!(cm0.special(), 5);
    }

    #[test]
    fn test_debug() {
        let cm = ChessMove::from_str("a7a8b").unwrap();
        assert_eq!(format!("{:?}", cm), "[to=a8, from=a7, special=bishop promotion]");
    }

    #[test]
    fn test_display() {
        let cm = ChessMove::from_str("a7a8q").unwrap();
        assert_eq!(format!("{}", cm), "a7a8q");

        let cm1 = ChessMove::from_str("a7a8").unwrap();
        assert_eq!(format!("{}", cm1), "a7a8");
    }

    #[test]
    fn test_capital_promotion() {
        let cm = ChessMove::from_str("a7a8Q").unwrap();
        assert_eq!(format!("{}", cm), "a7a8q");
    }

    #[test]
    fn test_reflect_basic() {
        let cm = ChessMove::from_str("a7a8Q").unwrap();
        assert_eq!(format!("{}", cm.reflect()), "a2a1q");

        let cm1 = ChessMove::from_str("b3c6").unwrap();
        assert_eq!(format!("{}", cm1.reflect()), "b6c3");
    }
}