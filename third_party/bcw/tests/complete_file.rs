//! Exercise the complete-file CLI boundary, including failure without polling.
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn run_case(name: &str, terms: usize) -> (std::process::Output, PathBuf) {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("prym-rank-{name}-{}-{nonce}", std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    let path = dir.join("case.wdm");
    // A=diag(1,2), V=(1,1): V^T A^k V = 1 + 2^k modulo 29.
    let mut power = 1u32;
    let sequence: Vec<String> = (0..terms)
        .map(|_| {
            let value = ((1 + power) % 29).to_string();
            power = (2 * power) % 29;
            value
        })
        .collect();
    let text = format!(
        "2 2 29 {terms} 1\n1 1\n1 1\n1 1\n1 1\n{}\n",
        sequence.join(" ")
    );
    std::fs::write(&path, text).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_prym-rank"))
        .arg(&path)
        .args(["-g", "--threads", "1"])
        .output()
        .unwrap();
    (output, dir)
}

#[test]
fn complete_sequence_yields_exact_lower_bound_and_generator() {
    let (output, dir) = run_case("complete", 50);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        std::fs::read_to_string(dir.join("case_result.txt")).unwrap(),
        "Matrix size: 2 x 2\nRank: 2\n"
    );
    let coefficients: Vec<u32> = std::fs::read_to_string(dir.join("case_generators.txt"))
        .unwrap()
        .split_whitespace()
        .map(|x| x.parse().unwrap())
        .collect();
    assert_eq!(coefficients.len(), 3);
    for eigenvalue in [1u32, 2] {
        let value = coefficients.iter().rev().fold(0, |value, coefficient| {
            (value * eigenvalue + coefficient) % 29
        });
        assert_eq!(value, 0, "generator must vanish at each eigenvalue");
    }
    assert!(!dir.join("case.wdm.stop").exists());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn insufficient_complete_sequence_fails_without_results() {
    let (output, dir) = run_case("short", 3);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("insufficient"));
    assert!(!dir.join("case_result.txt").exists());
    assert!(!dir.join("case_generators.txt").exists());
    assert!(!dir.join("case.wdm.stop").exists());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn initial_gram_term_alone_is_not_a_sequence() {
    let (output, dir) = run_case("initial", 1);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("no usable sequence"));
    assert!(!dir.join("case_result.txt").exists());
    std::fs::remove_dir_all(dir).unwrap();
}
