mod chess_move;
mod random;
mod board;
mod move_generator;
mod zobrist;
mod constants;

use crate::move_generator::MoveGenerator;
use crate::board::Board;

use pyo3::prelude::*;

#[pymodule]
fn atlas_chess(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Board>()?;
    m.add_class::<MoveGenerator>()?;
    Ok(())
}