mod chess_move;
mod board;
mod move_generator;
mod zobrist;
mod constants;
mod nnue;
mod mcts;

use crate::move_generator::MoveGenerator;
use crate::board::Board;
use crate::mcts::mcts_search;

use nnue::Nnue;
use pyo3::prelude::*;

#[pymodule]
fn atlas_chess(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Board>()?;
    m.add_class::<MoveGenerator>()?;
    m.add_class::<Nnue>()?;
    m.add_function(wrap_pyfunction!(mcts_search, m)?)?;
    Ok(())
}