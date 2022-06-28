//! This module contains "golden tests" for the cli: that is, we test that running the binary
//! with a particular input results in the expected output, typically as measured by the
//! output stdout/stderr.
//!
//! What is "expected" likely changes over time, but this means that changes to the output
//! get flagged by tests and has to be inspected. `pretty_assertions` is used to get a
//! manageable diff on failing tests.
//!
//! Generally, stderr is tested with `-vv`, meaning that we consider changes to the tracing
//! log output to be insignificant.
//!
//! To force overwrite the current golden test files, set the environmental variable
//! WINSFS_GOLDEN=overwrite, e.g. `WINSFS_GOLDEN=overwrite cargo test -- name_of_test`
//! in fish. Doing so will _not_ check whether the current output matches the expected,
//! but set the expected to the current.

use std::{
    env,
    error::Error,
    ffi::OsStr,
    fs::{read_to_string, remove_file, write, File},
    io,
    process::{Command, Output},
    str::from_utf8,
    thread,
};

use pretty_assertions::assert_eq;

const WINSFS: &'static str = env!("CARGO_BIN_EXE_winsfs");
const EXPECT_DIR: &'static str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/expected");
const TMP_DIR: &'static str = env!("CARGO_TARGET_TMPDIR");

// This has to be a macro in order to expand in calls to concat!,
// since concat! cannot take consts.
macro_rules! test_dir {
    () => {
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data")
    };
}

const SAF_A: &'static str = concat!(test_dir!(), "/A.saf.idx");
const SAF_B: &'static str = concat!(test_dir!(), "/B.saf.idx");
const SAF_C: &'static str = concat!(test_dir!(), "/C.saf.idx");

/// Get the path of the expected `.stdout` file in `EXPECT_DIR`.
fn get_expected_stdout(test_name: &str) -> String {
    format!("{EXPECT_DIR}/{test_name}.stdout")
}

/// Read the expected stdout from the `.stdout` file in `EXPECT_DIR`.
fn read_expected_stdout(test_name: &str) -> io::Result<String> {
    let path = get_expected_stdout(test_name);
    read_to_string(path)
}

/// Overwrite new stdout into the `.stdout` file in `EXPECT_DIR`.
fn overwrite_expected_stdout<C>(new: C, test_name: &str) -> io::Result<()>
where
    C: AsRef<[u8]>,
{
    let path = get_expected_stdout(test_name);
    write(path, new)
}

/// Get the path of the expected `.stderr` file in `EXPECT_DIR`.
fn get_expected_stderr(test_name: &str) -> String {
    format!("{EXPECT_DIR}/{test_name}.stderr")
}

/// Read the expected stderr from the `.stderr` file in `EXPECT_DIR`.
fn read_expected_stderr(test_name: &str) -> io::Result<String> {
    let path = get_expected_stderr(test_name);
    read_to_string(path)
}

/// Overwrite new stderr into the `.stderr` file in `EXPECT_DIR`.
fn overwrite_expected_stderr<C>(new: C, test_name: &str) -> io::Result<()>
where
    C: AsRef<[u8]>,
{
    let path = get_expected_stderr(test_name);
    write(path, new)
}

/// Removes any instance of the CARGO_MANIFEST_DIR and CARGO_TARGET_TMPDIR
/// (with trailing '/') from string.
///
/// Input file paths show up in log files with their full paths, which is unportable.
/// The expected files give the relative paths, so this is a hack to re-create the
/// relative path by simply removing the CARGO_MANIFEST_DIR and CARGO_TARGET_TMPDIR.
fn remove_dirs(s: &str) -> String {
    s.replace(concat!(env!("CARGO_MANIFEST_DIR"), "/"), "")
        .replace(concat!(env!("CARGO_TARGET_TMPDIR"), "/"), "")
}

/// Test if stdout and stderr from command matches the expected stdout and stderr.
///
/// The expected stdout and stderr are found in the EXPECT_DIR based on the test file name.
/// Note that getting the current thread name is a bit fragile: in particular, it may break when
/// setting the number of test threads to 1, see github.com/rust-lang/rust/issues/70492.
///
/// This also contains a helper for automatically overwriting the current expected, which will
/// happen exactly when the env variable WINSFS_GOLDEN is set to "overwrite".
fn test_output(output: Output) -> Result<(), Box<dyn Error>> {
    let test_name = thread::current()
        .name()
        .expect("failed to get test name")
        .to_string();

    let stdout = from_utf8(&output.stdout)?;
    let stderr = from_utf8(&output.stderr).map(remove_dirs)?;

    if let Ok(true) = env::var("WINSFS_GOLDEN").map(|s| s == "overwrite") {
        overwrite_expected_stdout(stdout, &test_name)?;
        overwrite_expected_stderr(stderr, &test_name)?;
        eprintln!("Overwrote expected test output; test will fail, please rerun!");
        assert!(false);
    } else {
        assert_eq!(stdout, read_expected_stdout(&test_name)?);
        assert_eq!(stderr, read_expected_stderr(&test_name)?);
    }

    Ok(())
}

/// Helper to run the produced winsfs binary with the provided arguments.
fn winsfs_cmd<I, S>(args: I) -> Command
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let mut cmd = Command::new(WINSFS);
    cmd.args(args);

    // If test fails, it's nice to see the commands involved
    eprintln!("Failing command: {}", format!("{cmd:?}").replace("\"", ""));

    cmd
}

#[test]
fn test_1d_estimate_default() -> Result<(), Box<dyn Error>> {
    winsfs_cmd(["-vv", "--seed", "1", SAF_A])
        .output()
        .map(test_output)?
}

#[test]
fn test_2d_estimate_default() -> Result<(), Box<dyn Error>> {
    winsfs_cmd(["-vv", "--seed", "1", SAF_A, SAF_B])
        .output()
        .map(test_output)?
}

#[test]
fn test_3d_estimate_3_epochs() -> Result<(), Box<dyn Error>> {
    winsfs_cmd([
        "-vv",
        "--seed",
        "1",
        "--max-epochs",
        "3",
        SAF_A,
        SAF_B,
        SAF_C,
    ])
    .output()
    .map(test_output)?
}

#[test]
fn test_1d_stream_estimate_default() -> Result<(), Box<dyn Error>> {
    let saf_shuf = format!("{TMP_DIR}/A.saf.shuf");

    winsfs_cmd(["shuffle", "--output", &saf_shuf, SAF_A])
        .spawn()?
        .wait()?;

    winsfs_cmd(["-vv", &saf_shuf]).output().map(test_output)??;

    remove_file(saf_shuf)?;

    Ok(())
}

#[test]
fn test_2d_stream_estimate_3_epochs() -> Result<(), Box<dyn Error>> {
    let saf_shuf = format!("{TMP_DIR}/A-B.saf.shuf");

    winsfs_cmd(["shuffle", "--output", &saf_shuf, SAF_A, SAF_B])
        .spawn()?
        .wait()?;

    winsfs_cmd(["-vv", "--max-epochs", "3", &saf_shuf])
        .output()
        .map(test_output)??;

    remove_file(saf_shuf)?;

    Ok(())
}

#[test]
fn test_3d_stream_estimate_1_epoch() -> Result<(), Box<dyn Error>> {
    let saf_shuf = format!("{TMP_DIR}/A-B-C.saf.shuf");

    winsfs_cmd(["shuffle", "--output", &saf_shuf, SAF_A, SAF_B, SAF_C])
        .spawn()?
        .wait()?;

    winsfs_cmd(["-vv", "--max-epochs", "1", &saf_shuf])
        .output()
        .map(test_output)??;

    remove_file(saf_shuf)?;

    Ok(())
}

#[test]
fn test_1d_log_likelihood() -> Result<(), Box<dyn Error>> {
    let sfs = format!("{TMP_DIR}/A.sfs");

    winsfs_cmd(["--seed", "1", SAF_A])
        .stdout(File::create(&sfs)?)
        .spawn()?
        .wait()?;

    winsfs_cmd(["log-likelihood", "-vv", "--sfs", &sfs, SAF_A])
        .output()
        .map(test_output)??;

    remove_file(sfs)?;

    Ok(())
}

#[test]
fn test_2d_log_likelihood() -> Result<(), Box<dyn Error>> {
    let sfs = format!("{TMP_DIR}/A-B.sfs");

    winsfs_cmd(["--seed", "1", SAF_A, SAF_B])
        .stdout(File::create(&sfs)?)
        .spawn()?
        .wait()?;

    winsfs_cmd(["log-likelihood", "-vv", "--sfs", &sfs, SAF_A, SAF_B])
        .output()
        .map(test_output)??;

    remove_file(sfs)?;

    Ok(())
}

#[test]
fn test_3d_log_likelihood() -> Result<(), Box<dyn Error>> {
    let sfs = format!("{TMP_DIR}/A-B-C.sfs");

    winsfs_cmd(["--seed", "1", "--max-epochs", "3", SAF_A, SAF_B, SAF_C])
        .stdout(File::create(&sfs)?)
        .spawn()?
        .wait()?;

    winsfs_cmd(["log-likelihood", "-vv", "--sfs", &sfs, SAF_A, SAF_B, SAF_C])
        .output()
        .map(test_output)??;

    remove_file(sfs)?;

    Ok(())
}
