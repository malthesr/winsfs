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

// clippy has some false positives when mapping to shorten covariant lifetimes in iterator,
// see github.com/rust-lang/rust-clippy/issues/9280
#![allow(clippy::map_identity)]

use std::{
    env,
    ffi::OsStr,
    fs::{read, read_to_string, remove_file, write},
    io::{self, Write},
    path::Path,
    process::{Command, Output, Stdio},
    str::from_utf8,
    thread,
};

use pretty_assertions::assert_eq;

const WINSFS: &str = env!("CARGO_BIN_EXE_winsfs");
const EXPECT_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/expected");
const TMP_DIR: &str = env!("CARGO_TARGET_TMPDIR");

type DynResult = Result<(), Box<dyn std::error::Error>>;

// This has to be a macro in order to expand in calls to concat!,
// since concat! cannot take consts.
macro_rules! test_dir {
    () => {
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data")
    };
}

const SAF_A: &str = concat!(test_dir!(), "/A.saf.idx");
const SAF_B: &str = concat!(test_dir!(), "/B.saf.idx");
const SAF_C: &str = concat!(test_dir!(), "/C.saf.idx");

const BANDED_SAF_D: &str = concat!(test_dir!(), "/D.banded.saf.idx");
const BANDED_SAF_E: &str = concat!(test_dir!(), "/E.banded.saf.idx");
const BANDED_SAF_F: &str = concat!(test_dir!(), "/F.banded.saf.idx");

const SFS_A: &str = concat!(test_dir!(), "/A.sfs");
const SFS_A_B: &str = concat!(test_dir!(), "/A-B.sfs");
const SFS_A_B_C: &str = concat!(test_dir!(), "/A-B-C.sfs");

const NPY_SFS_A: &str = concat!(test_dir!(), "/A.npy");
const NPY_SFS_A_B: &str = concat!(test_dir!(), "/A-B.npy");
const NPY_SFS_A_B_C: &str = concat!(test_dir!(), "/A-B-C.npy");

const SFS_D: &str = concat!(test_dir!(), "/D.sfs");
const SFS_D_E: &str = concat!(test_dir!(), "/D-E.sfs");
const SFS_D_E_F: &str = concat!(test_dir!(), "/D-E-F.sfs");

/// Get the path of the expected `.stdout` file in `EXPECT_DIR`.
fn get_expected_stdout(test_name: &str) -> String {
    format!("{EXPECT_DIR}/{test_name}.stdout")
}

/// Read the raw expected stdout from the `.stdout` file in `EXPECT_DIR`.
fn read_raw_expected_stdout(test_name: &str) -> io::Result<Vec<u8>> {
    let path = get_expected_stdout(test_name);
    read(path)
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

/// Note that getting the current thread name is a bit fragile: in particular, it may break when
/// setting the number of test threads to 1, see github.com/rust-lang/rust/issues/70492.
fn get_test_name() -> String {
    thread::current()
        .name()
        .expect("failed to get test name")
        .to_string()
}

/// Test if stdout and stderr from command matches the expected stdout and stderr.
///
/// The expected stdout and stderr are found in the EXPECT_DIR based on the test file name.
/// See [`get_test_name`] for some caveats on this.
///
/// This also contains a helper for automatically overwriting the current expected, which will
/// happen exactly when the env variable WINSFS_GOLDEN is set to "overwrite".
fn test_output(output: Output) -> DynResult {
    let test_name = get_test_name();

    let raw_stdout = &output.stdout;
    let stderr = from_utf8(&output.stderr).map(remove_dirs)?;

    if let Ok(true) = env::var("WINSFS_GOLDEN").map(|s| s == "overwrite") {
        overwrite_expected_stdout(raw_stdout, &test_name)?;
        overwrite_expected_stderr(stderr, &test_name)?;

        panic!("Overwrote expected test output; test will fail, please rerun!");
    } else {
        assert_eq!(raw_stdout, &read_raw_expected_stdout(&test_name)?);
        assert_eq!(stderr, read_expected_stderr(&test_name)?);
    }

    Ok(())
}

/// Runs the produced winsfs binary with the provided arguments.
fn winsfs_cmd<I, S>(args: I) -> Command
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let mut cmd = Command::new(WINSFS);
    cmd.args(args);

    // If test fails, it's nice to see the commands involved
    eprintln!("Failing command: {}", format!("{cmd:?}").replace('"', ""));

    cmd
}

/// Runs winsfs with the provided args and returns the output.
fn winsfs<I>(args: I) -> io::Result<Output>
where
    I: IntoIterator,
    I::Item: AsRef<OsStr>,
{
    winsfs_cmd(args).output()
}

/// Runs winsfs with the provided args and stdin and returns the output.
fn winsfs_with_stdin<I>(args: I, stdin: &[u8]) -> io::Result<Output>
where
    I: IntoIterator,
    I::Item: AsRef<OsStr>,
{
    let mut process = winsfs_cmd(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let process_stdin = process.stdin.as_mut().unwrap();
    process_stdin.write_all(stdin)?;

    process.wait_with_output()
}

/// Runs winsfs with the provided args and stdin from path and returns the output.
fn winsfs_with_stdin_path<I, P>(args: I, stdin_path: P) -> io::Result<Output>
where
    I: IntoIterator,
    I::Item: AsRef<OsStr>,
    P: AsRef<Path>,
{
    winsfs_with_stdin(args, &read(stdin_path)?)
}

#[test]
fn test_1d_view_fold_normalise() -> DynResult {
    winsfs(["view", "-vv", "--fold", "--normalise", SFS_A]).map(test_output)?
}

#[test]
fn test_1d_view_npy_from_stdin() -> DynResult {
    winsfs_with_stdin_path(["view", "-vv"], NPY_SFS_A).map(test_output)?
}

#[test]
fn test_1d_view_fold_from_stdin() -> DynResult {
    winsfs_with_stdin_path(["view", "-vv", "--fold"], SFS_A).map(test_output)?
}

#[test]
fn test_2d_view_normalise_from_stdin() -> DynResult {
    winsfs_with_stdin_path(["view", "-vv", "--normalise"], SFS_A_B).map(test_output)?
}

#[test]
fn test_2d_view_fold_npy() -> DynResult {
    winsfs(["view", "-vv", "--fold", NPY_SFS_A_B]).map(test_output)?
}

#[test]
fn test_2d_view_output_npy() -> DynResult {
    winsfs(["view", "-vv", "--output-format", "np", SFS_A_B]).map(test_output)?
}

#[test]
fn test_3d_view_normalise() -> DynResult {
    winsfs_with_stdin_path(["view", "-vv", "--normalise"], SFS_A_B_C).map(test_output)?
}

#[test]
fn test_3d_view_fold_npy_from_stdin() -> DynResult {
    winsfs_with_stdin_path(["view", "-vv", "--fold"], NPY_SFS_A_B_C).map(test_output)?
}

#[test]
fn test_3d_view_normalise_npy_output_npy() -> DynResult {
    winsfs([
        "view",
        "-vv",
        "--normalise",
        "--output-format",
        "np",
        NPY_SFS_A_B_C,
    ])
    .map(test_output)?
}

/// Run winsfs estimation with provided extra arguments from provided SAF files.
fn impl_test_estimate<'a, I1, I2>(args: I1, safs: I2) -> DynResult
where
    I1: IntoIterator<Item = &'a str>,
    I2: IntoIterator<Item = &'a str>,
{
    winsfs(args.into_iter().chain(["-vv", "--seed", "1"]).chain(safs)).map(test_output)?
}

#[test]
fn test_1d_estimate_default() -> DynResult {
    impl_test_estimate([], [SAF_A])
}

#[test]
fn test_1d_estimate_uneven_block_size() -> DynResult {
    impl_test_estimate(["--block-size", "439"], [SAF_A])
}

#[test]
fn test_1d_banded_estimate_default() -> DynResult {
    impl_test_estimate([], [BANDED_SAF_D])
}

#[test]
fn test_2d_estimate_default() -> DynResult {
    impl_test_estimate([], [SAF_A, SAF_B])
}

#[test]
fn test_2d_banded_estimate_10_epochs() -> DynResult {
    impl_test_estimate(["--max-epochs", "10"], [BANDED_SAF_D, BANDED_SAF_E])
}

#[test]
fn test_3d_estimate_3_epochs() -> DynResult {
    impl_test_estimate(["--max-epochs", "3"], [SAF_A, SAF_B, SAF_C])
}

#[test]
fn test_3d_banded_estimate_3_epochs() -> DynResult {
    impl_test_estimate(
        ["--max-epochs", "3"],
        [BANDED_SAF_D, BANDED_SAF_E, BANDED_SAF_F],
    )
}

/// Run stream winsfs estimation with provided extra arguments from provided SAF files.
///
/// The shuffled SAF file is automatically created in the CARGO_TARGET_TMPDIR. It is removed
/// at test tear-down.
fn impl_test_stream_estimate<'a, I1, I2>(args: I1, safs: I2) -> DynResult
where
    I1: IntoIterator<Item = &'a str>,
    I2: IntoIterator<Item = &'a str>,
{
    let saf_shuf = format!(
        "{TMP_DIR}/{test_name}.saf.shuf",
        test_name = get_test_name()
    );

    winsfs_cmd(
        ["shuffle", "--output", &saf_shuf]
            .into_iter()
            .chain(safs.into_iter().map(|x| x)),
    )
    .spawn()?
    .wait()?;

    winsfs_cmd(args.into_iter().map(|x| x).chain(["-vv", &saf_shuf]))
        .output()
        .map(test_output)??;

    remove_file(&saf_shuf)?;

    Ok(())
}

#[test]
fn test_1d_stream_estimate_default() -> DynResult {
    impl_test_stream_estimate([], [SAF_A])
}

#[test]
fn test_1d_stream_banded_estimate_default() -> DynResult {
    impl_test_stream_estimate([], [BANDED_SAF_D])
}

#[test]
fn test_2d_stream_estimate_3_epochs() -> DynResult {
    impl_test_stream_estimate(["--max-epochs", "3"], [SAF_A, SAF_B])
}

#[test]
fn test_2d_stream_banded_estimate_3_epochs() -> DynResult {
    impl_test_stream_estimate(["--max-epochs", "3"], [BANDED_SAF_D, BANDED_SAF_E])
}

#[test]
fn test_3d_stream_estimate_1_epoch() -> DynResult {
    impl_test_stream_estimate(["--max-epochs", "1"], [SAF_A, SAF_B, SAF_C])
}

#[test]
fn test_3d_stream_banded_estimate_1_epoch() -> DynResult {
    impl_test_stream_estimate(
        ["--max-epochs", "1"],
        [BANDED_SAF_D, BANDED_SAF_E, BANDED_SAF_F],
    )
}

/// Run winsfs log_likelihood with provided extra arguments from provided SFS and SAF files.
fn impl_test_log_likelihood<'a, I1, I2>(args: I1, sfs: &'a str, safs: I2) -> DynResult
where
    I1: IntoIterator<Item = &'a str>,
    I2: IntoIterator<Item = &'a str>,
{
    winsfs(
        ["log-likelihood", "-vv", "--sfs", sfs]
            .into_iter()
            .chain(args.into_iter().chain(safs)),
    )
    .map(test_output)?
}

#[test]
fn test_1d_log_likelihood() -> DynResult {
    impl_test_log_likelihood([], SFS_A, [SAF_A])
}

#[test]
fn test_1d_banded_log_likelihood() -> DynResult {
    impl_test_log_likelihood([], SFS_D, [BANDED_SAF_D])
}

#[test]
fn test_2d_log_likelihood() -> DynResult {
    impl_test_log_likelihood([], SFS_A_B, [SAF_A, SAF_B])
}

#[test]
fn test_2d_banded_log_likelihood() -> DynResult {
    impl_test_log_likelihood([], SFS_D_E, [BANDED_SAF_D, BANDED_SAF_E])
}

#[test]
fn test_3d_log_likelihood() -> DynResult {
    impl_test_log_likelihood([], SFS_A_B_C, [SAF_A, SAF_B, SAF_C])
}

#[test]
fn test_3d_banded_log_likelihood() -> DynResult {
    impl_test_log_likelihood([], SFS_D_E_F, [BANDED_SAF_D, BANDED_SAF_E, BANDED_SAF_F])
}
