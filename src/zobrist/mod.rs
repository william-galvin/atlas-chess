use std::{fmt::Debug, mem::size_of, ops::Index, sync::Arc};

use crate::{
    chess_move::ChessMove, 
    random::Rng,
    constants::TT_CACHE_SIZE,
};

pub trait Uint: Sized {
    fn from_u64_truncate(value: u64) -> Self;
}

macro_rules! impl_uint {
    ($($ty:ty),*) => {
        $(
            impl Uint for $ty {
                fn from_u64_truncate(value: u64) -> Self {
                    value as $ty
                }
            }
        )*
    };
}

impl_uint!(u8, u16, u32, u64, usize, u128);

const N_FEATURES: usize = 64 * 12 + 22;
const ENTRY_SIZE: usize = size_of::<ZobristHashTableEntry>();
const N_ENTRIES: usize = TT_CACHE_SIZE / ENTRY_SIZE;


/// Table of N_FEATURES random bitstrings to 
/// incrementally update board hashes
#[derive(PartialEq, Eq, Hash, Debug, Clone)]
struct ZobristLookupTable<U: Uint> {
    features: [U; N_FEATURES],
}

/// Element belonging to a Board instance, 
/// with 64 bit hash (key), 16 bit hash (checksum), 
/// and references to lookup tables
#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub struct ZobristBoardComponent {
    pub z_key: u64,
    pub z_sum: u16,
    z64: Arc<ZobristLookupTable<u64>>,
    z16: Arc<ZobristLookupTable<u16>>,
}

/// Heap allocated hash table for use in engine cache
pub struct ZobristHashTable {
    data: Box<[ZobristHashTableEntry; N_ENTRIES]>
}

/// Entry into ZobristHashTable where z_sum is the 16 bit checksum, 
/// depth is the searched depth at the node, eval and chess_move are 
/// previously returned from `go`
#[derive(Default, Clone, Copy)]
pub struct ZobristHashTableEntry {
    pub z_sum: u16,
    pub depth: u8,
    pub eval: i16,
    pub chess_move: ChessMove
}

impl<U: Uint> ZobristLookupTable<U> {
    fn new(seed: u64) -> Self {
        let mut rng = Rng::new_deterministic(seed, size_of::<U>() as u64);
        let features = std::array::from_fn(|_| U::from_u64_truncate(rng.random()));
        Self { features }
    }
}

impl<U: Uint> Index<usize> for ZobristLookupTable<U> {
    type Output = U;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.features[index]
    }
}

impl ZobristBoardComponent {
    pub fn new(pieces: &[u64; 12], info: u64) -> Self {
        let mut z_key = 0;
        let mut z_sum = 0;

        let z64 = Arc::new(ZobristLookupTable::new(0));
        let z16 = Arc::new(ZobristLookupTable::new(404));

        for piece in 0..12 {
            for square in 0..64 {
                if (pieces[piece] >> square) & 1 == 1 {
                    z_key ^= z64[piece * 64 + square];
                    z_sum ^= z16[piece * 64 + square];
                }
            }
        }

        for i in 1..=21 {
            if (info >> 1 & 1) == 1 {
                z_key ^= z64[64 * 12 + i];
                z_sum ^= z16[64 * 12 + i];
            }
        }

        Self { z64, z16, z_key, z_sum }
    }

    // #[inline]
    pub fn xor(&mut self, piece: usize, square: u16) {
        self.z_key ^= self.z64[piece * 64 + square as usize];
        self.z_sum ^= self.z16[piece * 64 + square as usize];
    }

    // #[inline]
    pub fn xor_info(&mut self, info_idx: u16) {
        self.z_key ^= self.z64[64 * 12 + info_idx as usize];
        self.z_sum ^= self.z16[64 * 12 + info_idx as usize];
    }

    #[inline]
    pub fn xor_all_info(&mut self, info: u64) {
        for i in 0..=21 {
            if (info >> i & 1) == 1 {
                self.xor_info(i);
            }
        }
    }
}

impl ZobristHashTable {
    pub fn new(_cache_size: usize) -> Self {
        let data: Box<[ZobristHashTableEntry; N_ENTRIES]> = Box::new(
            std::array::from_fn(|_| ZobristHashTableEntry::default())
        );

        Self { data }
    }

    pub fn put(&mut self, key: u64, value: ZobristHashTableEntry) {
        self.data[key as usize % N_ENTRIES] = value;
    }

    pub fn get(&self, key: u64) -> ZobristHashTableEntry {
        self.data[key as usize % N_ENTRIES]
    }
}

impl ZobristHashTableEntry {
    pub fn new(z_sum: u16, depth: u8, eval: i16, chess_move: ChessMove) -> Self {
        Self { z_sum, depth, eval, chess_move }
    }
}

#[cfg(test)]
mod tests {    
    use crate::{board::Board, chess_move::ChessMove};

    #[test]
    fn zobrist_push_pop_pawn_push() -> Result<(), Box<dyn std::error::Error>> {
        let mut board = Board::from_fen("rnbqkbnr/pppppp1p/8/6p1/8/7N/PPPPPPPP/RNBQKB1R w KQkq g6")?;
        let old_z = board.zobrist.clone();

        board.push_move(ChessMove::from_str("f2f3")?);
        let med_z = board.zobrist.clone();

        board.pop_move();

        assert!(old_z != med_z && old_z == board.zobrist);
        Ok(())
    }

    #[test]
    fn zobrist_push_pop_disable_castling() -> Result<(), Box<dyn std::error::Error>> {
        let mut board = Board::from_fen("3r1b2/2Nk1qp1/pPpp1p2/6rp/4pP2/1P1PQ1nP/P3P1B1/1RBK2NR b - - 0 1")?;
        let old_z = board.zobrist.clone();

        board.push_move(ChessMove::from_str("g3h1")?);
        let med_z = board.zobrist.clone();

        board.pop_move();

        assert!(old_z != med_z && old_z == board.zobrist);
        Ok(())
    }

    #[test]
    fn zobrist_push_pop_enpass() -> Result<(), Box<dyn std::error::Error>> {
        let fen = "r3k2r/p1ppqpb1/1n2pnp1/1b1PN3/Pp2P3/2N2Q1p/1PPBBPPP/1R2K2R b Kkq a3";
        let mut board = Board::from_fen(fen).unwrap();
        let old_zobrist = board.zobrist.clone();
        board.push_move(ChessMove::from_str("b4a3").unwrap());
        board.pop_move();
        assert!(board.zobrist.eq(&old_zobrist));

        Ok(())
    }
}