//! Exercise the production CPU stages against an independently assembled g12
//! certificate. No CUDA, Python interpreter, or sibling checkout is required.

use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
    time::{SystemTime, UNIX_EPOCH},
};

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "prym-deformation-test-{}-{stamp}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn read(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

fn command(stage: &str, args: &[(&str, &Path)]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_prym-phi"));
    cmd.env("RAYON_NUM_THREADS", "2").arg(stage);
    for (flag, path) in args {
        cmd.arg(flag).arg(path);
    }
    cmd.output().unwrap()
}

fn run(stage: &str, args: &[(&str, &Path)]) {
    let output = command(stage, args);
    assert!(
        output.status.success(),
        "{stage} failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn rows(value: &Value) -> Vec<Vec<u64>> {
    serde_json::from_value(value.clone()).unwrap()
}

fn write_vectors(path: &Path, block: &[Vec<u64>]) {
    // CUDA recovery writes one full vector per line, while JSON uses rows.
    let text = (0..block[0].len())
        .map(|column| {
            block
                .iter()
                .map(|row| row[column].to_string())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    fs::write(path, text).unwrap();
}

fn assert_dense(path: &Path, expected: &Value) {
    let output = read(path);
    let expected = rows(expected);
    assert_eq!(output["rows"], expected.len());
    assert_eq!(output["columns"], expected[0].len());
    assert_eq!(
        output["values"],
        json!(expected.into_iter().flatten().collect::<Vec<_>>())
    );
}

#[test]
fn all_deformation_stages_match_independent_certificate_and_reject_bad_kernel() {
    let scratch = Scratch::new();
    let instance = fixture("g12.json");
    let reference = read(&fixture("g12-reference.json"));
    let certificate = &reference["certificate"];
    let artifacts = scratch.0.join("artifacts");
    let candidates = scratch.0.join("base-vectors.txt");
    let kernel = rows(&certificate["K0"]);
    write_vectors(&candidates, &kernel);

    run(
        "deformation-extract-base-kernel",
        &[
            ("--instance", &instance),
            ("--kernel-vectors", &candidates),
            ("--out-dir", &artifacts),
        ],
    );
    let kernel_path = artifacts.join("K_sector0.json");
    assert_dense(&kernel_path, &certificate["K0"]);
    let kernel_report = read(&artifacts.join("base_kernel_report.json"));
    assert_eq!(kernel_report["rank"], 2);
    assert_eq!(kernel_report["F0_residual_nonzeros"], 0);
    assert_eq!(kernel_report["certificate_source"], "rust_matrix_free_F0");

    run(
        "deformation-derive-rhs",
        &[
            ("--instance", &instance),
            ("--kernel", &kernel_path),
            ("--out-dir", &artifacts),
            ("--expected-corrections", Path::new("1,5")),
        ],
    );
    let rhs_manifest = read(&artifacts.join("rhs_manifest.json"));
    assert_eq!(rhs_manifest["correction_sectors"], json!([1, 5]));

    let mut runs = Vec::new();
    for rhs in rhs_manifest["rhs"].as_array().unwrap() {
        let sector = rhs["sector"].as_u64().unwrap();
        let s = rows(&certificate["S_by_sector"][sector.to_string()]);
        let mut augmented = s;
        augmented.push(vec![1, 0]);
        augmented.push(vec![0, 1]);
        // Give normalization a nonidentity invertible bottom block; this
        // catches a regression that merely discards the two bottom rows.
        for row in &mut augmented {
            let (x, y) = (row[0], row[1]);
            row[0] = (2 * x + 5 * y) % 109;
            row[1] = (3 * x + 7 * y) % 109;
        }
        let wdm = scratch.0.join(format!("augmented-sector{sector}.wdm.zst"));
        write_vectors(
            &PathBuf::from(format!("{}_nullvectors_2.txt", wdm.display())),
            &augmented,
        );
        runs.push(json!({
            "run_id": format!("g12-augmented-{sector}"), "sector": sector,
            "rows": rhs["rows"], "columns": augmented.len(), "wdm_path": wdm,
            "args": ["--dense-file", rhs["path"]],
        }));
    }
    let manifest = scratch.0.join("augmented-manifest.json");
    fs::write(
        &manifest,
        serde_json::to_vec(&json!({"runs": runs})).unwrap(),
    )
    .unwrap();
    run(
        "deformation-normalize-augmented",
        &[
            ("--instance", &instance),
            ("--manifest", &manifest),
            ("--out-dir", &artifacts),
        ],
    );
    let solutions_path = artifacts.join("solutions_manifest.json");
    let solutions = read(&solutions_path);
    assert_eq!(solutions["first_order_equations_verified"], true);
    for solution in solutions["solutions"].as_array().unwrap() {
        let sector = solution["sector"].as_u64().unwrap().to_string();
        assert_dense(
            Path::new(solution["path"].as_str().unwrap()),
            &certificate["S_by_sector"][sector],
        );
        assert_eq!(solution["residual_nonzeros"], 0);
        assert_eq!(solution["certificate_source"], "rust_matrix_free_F0");
    }

    run(
        "deformation-derive-quadratic",
        &[
            ("--instance", &instance),
            ("--kernel", &kernel_path),
            ("--solutions-manifest", &solutions_path),
            ("--out-dir", &artifacts),
        ],
    );
    assert_dense(
        &artifacts.join("Z0_quadratic_obstruction.json"),
        &certificate["Z0"],
    );
    assert_eq!(
        read(&artifacts.join("replacement_drop_columns.json"))["indices"],
        certificate["J"]
    );

    let mut bad = kernel;
    bad[0][0] = (bad[0][0] + 1) % 109;
    write_vectors(&candidates, &bad);
    let rejected = command(
        "deformation-extract-base-kernel",
        &[
            ("--instance", &instance),
            ("--kernel-vectors", &candidates),
            ("--out-dir", &scratch.0.join("bad")),
        ],
    );
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("do not satisfy F0,0 K=0"));

    // An empty stage must not produce a vacuous "equations verified" report.
    fs::write(&manifest, br#"{"runs": []}"#).unwrap();
    let rejected = command(
        "deformation-normalize-augmented",
        &[
            ("--instance", &instance),
            ("--manifest", &manifest),
            ("--out-dir", &scratch.0.join("incomplete")),
        ],
    );
    assert!(!rejected.status.success());
    assert!(
        String::from_utf8_lossy(&rejected.stderr).contains("exactly the two augmented sectors")
    );
}
