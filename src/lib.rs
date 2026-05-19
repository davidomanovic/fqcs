use pyo3::prelude::*;

mod orbital_rotation;
mod pair_uccd;
mod correlators;

#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        correlators::apply_ucj_spin_restricted_in_place_num_rep,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        correlators::apply_ucj_spin_balanced_in_place_num_rep,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        correlators::apply_igcr2_spin_restricted_in_place_num_rep,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        correlators::apply_igcr3_spin_restricted_in_place_num_rep,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        correlators::apply_igcr4_spin_restricted_in_place_num_rep,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        pair_uccd::apply_pair_uccd_doci_unitary_in_place,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        orbital_rotation::apply_givens_rotation_in_place,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        orbital_rotation::apply_phase_shift_in_place,
        m
    )?)?;
    Ok(())
}
