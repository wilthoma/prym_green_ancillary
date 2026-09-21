//! Linear gluing constraints for homogeneous sections on the normalization.
//! Each row expresses multiplier[1]*s(P)-multiplier[0]*s(Q)=0.

use crate::{binary_form::BinaryForm, field::Field, points::PointPair};

pub fn constraint_matrix(
    point_pairs: &[PointPair],
    multipliers: &[[u64; 2]],
    degree: usize,
    field: Field,
) -> Vec<Vec<u64>> {
    point_pairs
        .iter()
        .zip(multipliers)
        .map(|(pair, multiplier)| {
            let eval_p = BinaryForm::monomial_evaluations(degree, &pair.p, field);
            let eval_q = BinaryForm::monomial_evaluations(degree, &pair.q, field);
            eval_p
                .iter()
                .zip(&eval_q)
                .map(|(&at_p, &at_q)| {
                    field.sub(
                        field.mul(multiplier[1], at_p),
                        field.mul(multiplier[0], at_q),
                    )
                })
                .collect()
        })
        .collect()
}
