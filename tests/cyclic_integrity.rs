//! Construction and verification remain independently exercised after removing
//! the old random geometry and CPU rank backends.

use prym_phi::cyclic::{
    CyclicGenerateOptions, CyclicInstanceFile, generate_cyclic_instance, verify_cyclic_instance,
};

fn small_instance() -> CyclicInstanceFile {
    generate_cyclic_instance(&CyclicGenerateOptions {
        genus: 6,
        order: 3,
        prime: 31,
        zeta: 5,
        orbit_reps: vec![[1, 2], [3, 4]],
    })
    .unwrap()
}

#[test]
fn serialized_small_cyclic_instance_retains_checksum_and_algebra() {
    let instance = small_instance();
    let text = serde_json::to_string(&instance).unwrap();
    let recovered = serde_json::from_str(&text).unwrap();
    assert_eq!(
        verify_cyclic_instance(&instance).unwrap(),
        verify_cyclic_instance(&recovered).unwrap()
    );
}

#[test]
fn verification_reconstructs_products_and_rejects_tampered_mu_without_checksum() {
    let mut instance = small_instance();
    instance.checksum_sha256 = None;
    instance.artinian.mu.values[0] = (instance.artinian.mu.values[0] + 1) % instance.modulus;
    assert!(
        verify_cyclic_instance(&instance)
            .unwrap_err()
            .contains("stored mu tensor does not match products")
    );
}

#[test]
fn cyclic_generator_rejects_colliding_orbits_and_wrong_order() {
    let mut options = CyclicGenerateOptions {
        genus: 6,
        order: 3,
        prime: 31,
        zeta: 5,
        orbit_reps: vec![[1, 2], [3, 4]],
    };
    options.orbit_reps[1] = [5, 4];
    assert!(
        generate_cyclic_instance(&options)
            .unwrap_err()
            .contains("repeats r-th power")
    );
    options.orbit_reps[1] = [3, 4];
    options.zeta = 1;
    assert!(
        generate_cyclic_instance(&options)
            .unwrap_err()
            .contains("not exact order")
    );
}
