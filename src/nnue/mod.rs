// Effieciently Updatable Neural Netowrk (nnue) as described in 
// https://github.com/asdfjkl/neural_network_chess/releases/download/v1.6/neural_networks_chess.pdf
// and https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md

use ndarray::{concatenate, Array1, Array2, Axis};
use numpy::{PyArray, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use std::{hash::{Hash, Hasher}, sync::Arc};

use rand::{thread_rng, Rng};

use crate::{
    board::{Board, KING, WHITE},
    move_generator::get_bits
};

const FLIP: [usize; 64] = [
    56,  57,  58,  59,  60,  61,  62,  63,
    48,  49,  50,  51,  52,  53,  54,  55,
    40,  41,  42,  43,  44,  45,  46,  47,
    32,  33,  34,  35,  36,  37,  38,  39,
    24,  25,  26,  27,  28,  29,  30,  31,
    16,  17,  18,  19,  20,  21,  22,  23,
     8,   9,  10,  11,  12,  13,  14,  15,
     0,   1,   2,   3,   4,   5,   6,   7
];

#[derive(Debug, Clone)]
#[pyclass]
pub struct Nnue {
    w256: Arc<Array2<f32>>,  // (256 x 40,960).T
    w64: Arc<Array2<f32>>,   // 64 x 256
    w8: Arc<Array2<f32>>,    // 8 x 64(2)
    w1: Arc<Array2<f32>>,    // 1 x 8
    hidden_white: Array1<f32>,  // 256
    hidden_black: Array1<f32>,  // 256
    dirty: bool,
}

impl PartialEq for Nnue {
    fn eq(&self, other: &Self) -> bool {
        self.hidden_white.iter().zip(other.hidden_white.iter()).all(|(a, b)| a.to_bits() == b.to_bits())
    }
}

impl Eq for Nnue {}

impl Hash for Nnue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for &item in self.hidden_white.iter() {
            item.to_bits().hash(state);
        }
    }
}

#[pymethods]
impl Nnue {
    pub fn forward(&self, white: bool) -> f32 {
        if self.dirty {
            panic!("forward pass on dirty hidden layer");
        }

        let (mut x_us, mut x_them) = if white {
            (self.hidden_white.clone(), self.hidden_black.clone())
        } else { 
            (self.hidden_black.clone(), self.hidden_white.clone()) 
        };

        layer_norm(&mut x_us);
        layer_norm(&mut x_them);

        leaky_relu(&mut x_us);
        leaky_relu(&mut x_them);
        
        x_us = self.w64.dot(&x_us);
        x_them = self.w64.dot(&x_them);

        layer_norm(&mut x_us);
        layer_norm(&mut x_them);

        let mut x = concatenate(Axis(0), &[x_us.view(), x_them.view()]).unwrap();

        leaky_relu(&mut x);
        x = self.w8.dot(&x);
        layer_norm(&mut x);
        leaky_relu(&mut x);
        x = self.w1.dot(&x);

        x[0]
    }

    #[new]
    pub fn zeros() -> Self {
        Self {
            w256: Arc::new(transposed(Array2::zeros((256, 40_960)))),
            w64: Arc::new(Array2::zeros((64, 256))),
            w8: Arc::new(Array2::zeros((8, 64 * 2))),
            w1: Arc::new(Array2::zeros((1, 8))),
            hidden_white: Array1::zeros(256),
            hidden_black: Array1::zeros(256),
            dirty: false,
        }
    }

    #[staticmethod]
    pub fn random() -> Self {
        let mut rng = thread_rng();
        
        Self {
            w256: Arc::new(transposed(Array2::from_shape_fn((256, 40_960), |_| rng.gen_range(-1.0..1.0)))),
            w64: Arc::new(Array2::from_shape_fn((64, 256), |_| rng.gen_range(-1.0..1.0))),
            w8: Arc::new(Array2::from_shape_fn((8, 64 * 2), |_| rng.gen_range(-1.0..1.0))),
            w1: Arc::new(Array2::from_shape_fn((1, 8), |_| rng.gen_range(-1.0..1.0))),
            hidden_white: Array1::from_shape_fn(256, |_| rng.gen_range(-1.0..1.0)),
            hidden_black: Array1::from_shape_fn(256, |_| rng.gen_range(-1.0..1.0)),
            dirty: false,
        }
    }

    pub fn set_weights(&mut self, w256: PyReadonlyArray2<f32>, w64: PyReadonlyArray2<f32>, w8: PyReadonlyArray2<f32>, w1: PyReadonlyArray2<f32>) {
        self.w256 = Arc::new(transposed(w256.as_array().to_owned()));
        self.w64 = Arc::new(w64.as_array().to_owned());
        self.w8 = Arc::new(w8.as_array().to_owned());
        self.w1 = Arc::new(w1.as_array().to_owned());
        self.dirty = true;
    }

    pub fn clear_hidden(&mut self) {
        self.hidden_white = Array1::zeros(256);
        self.hidden_black = Array1::zeros(256);
        self.dirty = false;
    }

    pub fn add(&mut self, white_king: usize, black_king: usize, piece: usize, piece_pos: usize) {
        self.add_white(white_king, piece, piece_pos);
        self.add_black(black_king, piece, piece_pos);
    }

    pub fn sub(&mut self, white_king: usize, black_king: usize, piece: usize, piece_pos: usize) {
        self.sub_white(white_king, piece, piece_pos);
        self.sub_black(black_king, piece, piece_pos);
    }

    pub fn add_white(&mut self, white_king: usize, piece: usize, piece_pos: usize) {
        let Some(idx) = get_idx_white(white_king, piece, piece_pos) else {
            return;
        };
        let col = self.w256.row(idx);
        self.hidden_white += &col;
    }

    pub fn sub_white(&mut self, white_king: usize, piece: usize, piece_pos: usize) {
        let Some(idx) = get_idx_white(white_king, piece, piece_pos) else {
            return;
        };
        let col = self.w256.row(idx);
        self.hidden_white -= &col;
    }

    pub fn add_black(&mut self, black_king: usize, piece: usize, piece_pos: usize) {
        let Some(idx) = get_idx_black(black_king, piece, piece_pos) else {
            return;
        };
        let col = self.w256.row(idx);
        self.hidden_black += &col;
    }

    pub fn sub_black(&mut self, black_king: usize, piece: usize, piece_pos: usize) {
        let Some(idx) = get_idx_black(black_king, piece, piece_pos) else {
            return;
        };
        let col = self.w256.row(idx);
        self.hidden_black -= &col;
    }

    #[staticmethod]
    pub fn features(board: &Board) -> Py<PyArray1<f32>> {
        let pieces = board.pieces;

        let mut result_black = Array1::zeros(40_960);
        let mut result_white = Array1::zeros(40_960);

        let wk = pieces[KING].trailing_zeros() as usize;
        let bk = pieces[KING + 6].trailing_zeros() as usize;

        for i in 0..12 {
            for square in get_bits(pieces[i]) {
                let Some(black_idx) = get_idx_black(bk, i, square) else {
                    continue;
                };
                let Some(white_idxs) = get_idx_white(wk, i, square) else {
                    continue;
                };

                result_black[black_idx] = 1.0;
                result_white[white_idxs] = 1.0;
            }
        }

        let combined = if board.to_move() == WHITE {
            concatenate(Axis(0), &[result_white.view(), result_black.view()]).unwrap()
        } else {
            concatenate(Axis(0), &[result_black.view(), result_white.view()]).unwrap()
        };

        let pyarr = Python::with_gil(|py| {
            PyArray::from_owned_array(py, combined).unbind()
        });

        pyarr
    }
}

impl Nnue {
    pub fn calculate_hidden(&mut self, pieces: &[u64; 12]) {
        self.clear_hidden();
        
        let wk = pieces[KING].trailing_zeros() as usize;
        let bk = pieces[KING + 6].trailing_zeros() as usize;

        for i in 0..12 {
            for square in get_bits(pieces[i]) {
                self.add_white(wk, i, square);
                self.add_black(bk, i, square);
            }
        }
    }
}

pub fn leaky_relu(x: &mut Array1<f32>) {
    x.mapv_inplace(|a| (0.05 * a).max(a));
}

pub fn layer_norm(x: &mut Array1<f32>) {
    let mu=  mean(x);
    let sigma = stdev(x, mu);
    x.mapv_inplace(|a| (a - mu) / sigma);
}

fn mean(x: &Array1<f32>) -> f32 {
    x.sum() / x.len() as f32
}

fn stdev(x: &Array1<f32>, mu: f32) -> f32 {
    (x.map(|i| (i - mu).powi(2)).sum() / (x.len() as f32)).powf(0.5)
}

pub fn get_idx_black(black_king: usize, mut piece: usize, piece_pos: usize) -> Option<usize> {
    if piece >=6 {
        piece -= 6;
    } else {
        piece += 6;
    }
    if piece == KING || piece == KING + 6 {
        return None;
    } 
    if piece > KING {
        piece -= 1;
    }
    Some(FLIP[black_king] * (64 * 10) + piece * 64 + FLIP[piece_pos])
}

pub fn get_idx_white(white_king: usize, mut piece: usize, piece_pos: usize) -> Option<usize> {
    if piece == KING || piece == KING + 6 {
        return None;
    } 
    if piece > KING {
        piece -= 1;
    }
    Some(white_king * (64 * 10) + piece * 64 + piece_pos)
}

fn transposed(arr: Array2<f32>) -> Array2<f32> {
    let a_t = arr.t();
    let mut a_t_owned = Array2::zeros(a_t.raw_dim());
    a_t_owned.assign(&a_t);
    a_t_owned
}


#[cfg(test)]
mod tests {
    use std::time::Instant;

    use crate::{board::Board, move_generator::MoveGenerator};

    use super::*;

    #[test]
    pub fn leaky_relu_works() {
        let mut arr = Array1::zeros(3);
        arr[0] = 3.0;
        arr[1] = 0.5;
        arr[2] = -1.0;

        leaky_relu(&mut arr);

        assert_eq!(3.0, arr[0]);
        assert_eq!(0.5, arr[1]);
        assert_eq!(-0.05, arr[2]);
    }

    #[test]
    pub fn network_invariant() {
        let mut n = Nnue::random();
        let board = Board::from_fen("2k5/2p5/3n4/8/8/3N4/2P5/2K5 w - - 0 1").unwrap();

        n.calculate_hidden(&board.pieces);
        assert!((n.forward(true) - n.forward(false)).abs() < 0.0001);

        let mut n = Nnue::random();
        let board = Board::new();

        n.calculate_hidden(&board.pieces);
        assert!((n.forward(true) - n.forward(false)).abs() < 0.0001);
    }

    #[test]
    pub fn board_hidden_updates() {
        let mut board = Board::from_fen("2k5/2p5/3n4/8/8/3N4/2P5/2K5 w - - 0 1").unwrap();
        let nnue = Nnue::random();
        board.enable_nnue(nnue);
        let mg = MoveGenerator::new();

        for m in mg.moves(&board) {
            let expected = board.forward().unwrap();

            board.push_move(m);

            // Can check that pushing actually does something...very flaky
            // because random weight init can be close to 0 or saturate in clipped relu

            // let diff = board.forward().unwrap();
            // assert!((expected - diff).abs() > 0.0001, "{expected} == {diff}");

            board.pop_move();
            let expected2 = board.forward().unwrap();

            assert!((expected - expected2).abs() < 0.0001, "{expected} != {expected2}");
        }
    }

    #[test]
    fn test_perft6_nnue() {
        let mg = MoveGenerator::new();
        let nnue = Nnue::random();

        let start = Instant::now();
        let expected = [46, 2079, 89890, 3894594, 164075551]; // Time to beat: 2.1 seconds ==> 3 or 4 seconds. Not horrible
        for i in 1..=expected.len() as i32 {
            let mut board = Board::from_fen("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10 ").unwrap();
            board.enable_nnue(nnue.clone());
            assert_eq!(crate::move_generator::perft(i, &mut board, &mg), expected[i as usize - 1]);
        }
        let elapsed = start.elapsed();
        eprintln!("perft6: {:.4?}", elapsed);
    }
}