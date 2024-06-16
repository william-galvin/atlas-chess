mod chess_move;
mod random;
mod board;
mod move_generator;

use move_generator::MoveGenerator;
use pyo3::prelude::*;
use crate::board::Board;

#[pyfunction]
fn add(a: i32, b: i32) -> PyResult<i32> {
    println!{"We are in a rust function!"}
    Ok(a + b)
}

#[pymodule]
fn atlaschess(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_class::<Board>()?;
    m.add_class::<MoveGenerator>()?;
    Ok(())
}