//! Symmetric WDM text format, optionally wrapped in a zstd frame.
//! Vector columns and upper-triangular moment entries retain CUDA ordering.

use std::fmt::Display;
use std::fs::File;
use std::io::{self, BufRead, Read, Seek, SeekFrom, Write};
use std::str::FromStr;
// use std::process::Output;

// This module contains code to read and write .wdm files.
// These are text files used to store progress in the computation of the Wiedemann sequence, that is,
// the sequence u^T (A^TA)^k v for k=1,2,3....
// Structure of a .wdm file
// 1. First line: m n p N num_v -- with m,n matrix size, p the prime number, Nlen the length of the Wiedemann sequence num_u and num_v the number of columns in the U and V matrix
// 2. Second line: row_precond -- the row preconditioner
// 3. Third line: col_precond -- the column preconditioner
// 5. Next num_v lines: v's -- the v vectors, i.e., columns of the V matrix.
// 6. Next num_v lines: curv -- the current V matrix (A^TA)^{N} V
// 7. Next num_v x (num_v+1)/2 lines: seq -- the Wiedemann sequence M_k = V^T(A^TA)^{k} V for k=1,2,3...,N.

const ZSTD_MAGIC: [u8; 4] = [0x28, 0xb5, 0x2f, 0xfd];

fn open_wdm_reader(wdm_filename: &str) -> Result<Box<dyn BufRead>, Box<dyn std::error::Error>> {
    let mut file = File::open(wdm_filename)?;
    let mut magic = [0u8; 4];
    let bytes_read = file.read(&mut magic)?;
    file.seek(SeekFrom::Start(0))?;

    if bytes_read == ZSTD_MAGIC.len() && magic == ZSTD_MAGIC {
        let decoder = zstd::stream::read::Decoder::new(file)?;
        Ok(Box::new(io::BufReader::new(decoder)))
    } else {
        Ok(Box::new(io::BufReader::new(file)))
    }
}

fn should_compress_wdm_output(wdm_filename: &str) -> bool {
    wdm_filename.ends_with(".zst")
}

pub fn load_wdm_file_sym<T>(
    wdm_filename: &str,
    row_precond: &mut Vec<T>,
    col_precond: &mut Vec<T>,
    v_list: &mut Vec<Vec<T>>,
    curv_list: &mut Vec<Vec<T>>,
    seq_list: &mut Vec<Vec<T>>,
) -> Result<(u32, usize, usize, usize), Box<dyn std::error::Error>>
where
    T: Display
        + std::ops::Add<Output = T>
        + Copy
        + std::ops::Mul<Output = T>
        + std::ops::AddAssign
        + std::ops::Rem<Output = T>
        + std::str::FromStr,
    <T as FromStr>::Err: std::error::Error + 'static,
{
    let mut reader = open_wdm_reader(wdm_filename)?;

    // Read the first line: m n p Nlen num_u num_v
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let m: usize = parts.next().ok_or("Missing m")?.parse()?;
    let n: usize = parts.next().ok_or("Missing n")?.parse()?;
    let p: u32 = parts.next().ok_or("Missing p")?.parse()?;
    let nlen: usize = parts.next().ok_or("Missing Nlen")?.parse()?;
    let num_v: usize = parts.next().ok_or("Missing num_v")?.parse()?;

    //println!("First line ok m,n,p,Nlen,num_v: {} {} {} {} {}", m, n, p, nlen, num_v);

    // Read row_precond
    line.clear();
    reader.read_line(&mut line)?;
    *row_precond = line
        .split_whitespace()
        .map(|x| x.parse::<T>())
        .collect::<Result<Vec<_>, _>>()?;

    //println!("Row precond ok");

    // Read col_precond
    line.clear();
    reader.read_line(&mut line)?;
    *col_precond = line
        .split_whitespace()
        .map(|x| x.parse::<T>())
        .collect::<Result<Vec<_>, _>>()?;
    //println!("Col precond ok");
    // Read v_list
    v_list.clear();
    for _ in 0..num_v {
        line.clear();
        reader.read_line(&mut line)?;
        let v = line
            .split_whitespace()
            .map(|x| x.parse::<T>())
            .collect::<Result<Vec<_>, _>>()?;
        if v.len() != n {
            return Err("v vector length does not match matrix columns".into());
        }
        v_list.push(v);
    }
    //println!("V list ok");

    // Read curv_list
    curv_list.clear();
    for _ in 0..num_v {
        line.clear();
        reader.read_line(&mut line)?;
        let curv = line
            .split_whitespace()
            .map(|x| x.parse::<T>())
            .collect::<Result<Vec<_>, _>>()?;
        if curv.len() != n {
            return Err("curv vector length does not match matrix columns".into());
        }
        curv_list.push(curv);
    }
    //println!("Curv list ok");
    // Read seq_list
    seq_list.clear();
    for _ in 0..(num_v * (num_v + 1) / 2) {
        line.clear();
        reader.read_line(&mut line)?;
        let seq = line
            .split_whitespace()
            .take(nlen)
            .map(|x| x.parse::<T>())
            .collect::<Result<Vec<_>, _>>()?;
        if seq.len() != nlen {
            return Err("seq vector length does not match expected sequence length".into());
        }
        seq_list.push(seq);
    }
    //println!("Seq list ok");

    // Ensure all vectors are of the correct size
    if row_precond.len() != m {
        return Err("Row preconditioner length does not match matrix rows".into());
    }
    if col_precond.len() != n {
        return Err("Column preconditioner length does not match matrix columns".into());
    }

    Ok((p, m, n, num_v)) // Return the prime as the only return value
}

pub fn load_wdm_initial_state<T>(
    wdm_filename: &str,
    row_precond: &mut Vec<T>,
    col_precond: &mut Vec<T>,
    v_list: &mut Vec<Vec<T>>,
) -> Result<(u32, usize, usize, usize), Box<dyn std::error::Error>>
where
    T: std::str::FromStr,
    <T as FromStr>::Err: std::error::Error + 'static,
{
    let mut reader = open_wdm_reader(wdm_filename)?;

    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let m: usize = parts.next().ok_or("Missing m")?.parse()?;
    let n: usize = parts.next().ok_or("Missing n")?.parse()?;
    let p: u32 = parts.next().ok_or("Missing p")?.parse()?;
    let _nlen: usize = parts.next().ok_or("Missing Nlen")?.parse()?;
    let num_v: usize = parts.next().ok_or("Missing num_v")?.parse()?;

    line.clear();
    reader.read_line(&mut line)?;
    *row_precond = line
        .split_whitespace()
        .map(|x| x.parse::<T>())
        .collect::<Result<Vec<_>, _>>()?;

    line.clear();
    reader.read_line(&mut line)?;
    *col_precond = line
        .split_whitespace()
        .map(|x| x.parse::<T>())
        .collect::<Result<Vec<_>, _>>()?;

    v_list.clear();
    for _ in 0..num_v {
        line.clear();
        reader.read_line(&mut line)?;
        let v = line
            .split_whitespace()
            .map(|x| x.parse::<T>())
            .collect::<Result<Vec<_>, _>>()?;
        if v.len() != n {
            return Err("v vector length does not match matrix columns".into());
        }
        v_list.push(v);
    }

    if row_precond.len() != m {
        return Err("Row preconditioner length does not match matrix rows".into());
    }
    if col_precond.len() != n {
        return Err("Column preconditioner length does not match matrix columns".into());
    }

    Ok((p, m, n, num_v))
}

#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub fn save_wdm_file_sym<T>(
    wdm_filename: &str,
    n_rows: usize,
    n_cols: usize,
    theprime: T,
    row_precond: &[T],
    col_precond: &[T],
    v_list: &[Vec<T>],
    curv_list: &[Vec<T>],
    seq_list: &[Vec<T>],
) -> Result<(), Box<dyn std::error::Error>>
where
    T: Display + Copy,
{
    if seq_list.is_empty() {
        return Err("Cannot save WDM file with empty sequence list".into());
    }

    let tmp_filename = format!("{}.tmp", wdm_filename);
    let compress = should_compress_wdm_output(wdm_filename);
    {
        let file = File::create(&tmp_filename)?;
        let mut writer = io::BufWriter::new(file);

        if compress {
            let mut encoder = zstd::stream::write::Encoder::new(&mut writer, 3)?;
            write_wdm_text(
                &mut encoder,
                n_rows,
                n_cols,
                theprime,
                row_precond,
                col_precond,
                v_list,
                curv_list,
                seq_list,
            )?;
            let writer = encoder.finish()?;
            writer.flush()?;
        } else {
            write_wdm_text(
                &mut writer,
                n_rows,
                n_cols,
                theprime,
                row_precond,
                col_precond,
                v_list,
                curv_list,
                seq_list,
            )?;
            writer.flush()?;
        }
    }

    std::fs::rename(tmp_filename, wdm_filename)?;
    Ok(())
}

#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn write_wdm_text<T: Display + Copy>(
    writer: &mut impl Write,
    n_rows: usize,
    n_cols: usize,
    theprime: T,
    row_precond: &[T],
    col_precond: &[T],
    v_list: &[Vec<T>],
    curv_list: &[Vec<T>],
    seq_list: &[Vec<T>],
) -> Result<(), Box<dyn std::error::Error>> {
    writeln!(
        writer,
        "{} {} {} {} {}",
        n_rows,
        n_cols,
        theprime,
        seq_list[0].len(),
        v_list.len()
    )?;

    write_line(writer, row_precond)?;
    write_line(writer, col_precond)?;
    for v in v_list {
        write_line(writer, v)?;
    }
    for curv in curv_list {
        write_line(writer, curv)?;
    }
    for seq in seq_list {
        write_line(writer, seq)?;
    }
    Ok(())
}

#[allow(dead_code)]
fn write_line<T: Display + Copy>(
    writer: &mut impl Write,
    values: &[T],
) -> Result<(), Box<dyn std::error::Error>> {
    for (i, val) in values.iter().enumerate() {
        if i > 0 {
            write!(writer, " ")?;
        }
        write!(writer, "{}", val)?;
    }
    writeln!(writer)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_wdm_path(test_name: &str, compressed: bool) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let suffix = if compressed { ".wdm.zst" } else { ".wdm" };
        std::env::temp_dir().join(format!(
            "bcw_rank_{test_name}_{}_{}{suffix}",
            std::process::id(),
            nanos
        ))
    }

    fn assert_save_load_roundtrip(compressed: bool) -> Result<(), Box<dyn std::error::Error>> {
        let path = temp_wdm_path("roundtrip", compressed);
        let row_precond = vec![2u32, 3];
        let col_precond = vec![4u32, 5, 6];
        let v = vec![vec![1u32, 2, 3], vec![3, 2, 1]];
        let curv = vec![vec![7u32, 8, 9], vec![9, 8, 7]];
        let seq = vec![vec![1u32, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];

        save_wdm_file_sym(
            path.to_str().unwrap(),
            2,
            3,
            29u32,
            &row_precond,
            &col_precond,
            &v,
            &curv,
            &seq,
        )?;

        if compressed {
            let bytes = std::fs::read(&path)?;
            assert!(bytes.starts_with(&ZSTD_MAGIC));
        }

        let mut got_row_precond: Vec<u32> = Vec::new();
        let mut got_col_precond: Vec<u32> = Vec::new();
        let mut got_v: Vec<Vec<u32>> = Vec::new();
        let mut got_curv: Vec<Vec<u32>> = Vec::new();
        let mut got_seq: Vec<Vec<u32>> = Vec::new();
        let meta = load_wdm_file_sym(
            path.to_str().unwrap(),
            &mut got_row_precond,
            &mut got_col_precond,
            &mut got_v,
            &mut got_curv,
            &mut got_seq,
        )?;

        assert_eq!(meta, (29, 2, 3, 2));
        assert_eq!(got_row_precond, row_precond);
        assert_eq!(got_col_precond, col_precond);
        assert_eq!(got_v, v);
        assert_eq!(got_curv, curv);
        assert_eq!(got_seq, seq);

        let mut got_initial_row_precond: Vec<u32> = Vec::new();
        let mut got_initial_col_precond: Vec<u32> = Vec::new();
        let mut got_initial_v: Vec<Vec<u32>> = Vec::new();
        let initial_meta = load_wdm_initial_state(
            path.to_str().unwrap(),
            &mut got_initial_row_precond,
            &mut got_initial_col_precond,
            &mut got_initial_v,
        )?;

        assert_eq!(initial_meta, (29, 2, 3, 2));
        assert_eq!(got_initial_row_precond, row_precond);
        assert_eq!(got_initial_col_precond, col_precond);
        assert_eq!(got_initial_v, v);

        let _ = std::fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn save_load_roundtrip_preserves_wdm_layout() -> Result<(), Box<dyn std::error::Error>> {
        assert_save_load_roundtrip(false)
    }

    #[test]
    fn save_load_roundtrip_preserves_zstd_wdm_layout() -> Result<(), Box<dyn std::error::Error>> {
        assert_save_load_roundtrip(true)
    }
}
