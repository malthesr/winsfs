use std::{
    env, fs,
    io::{self, Write},
    path,
};

const DEFAULT_1D: usize = 101;
const DEFAULT_2D: usize = 61;

fn create_run_1d(f: &mut fs::File, dim: usize) -> io::Result<()> {
    writeln!(
        f,
        "{}",
        r#"pub fn run_1d<P>(path: P, args: &Cli) -> clap::Result<()>
where
    P: AsRef<Path>,
{
    let reader = saf::BgzfReader::from_bgzf_member_path(path)?;
    let n = reader.index().alleles() + 1;
    match n {"#
    )?;

    for i in (3..=dim).step_by(2) {
        writeln!(f, "\t\t\t{i} => run_1d_inner::<{i}, _>(reader, args),")?;
    }

    writeln!(
        f,
        "{}",
        r#"
        n => {
            Err(Cli::command().error(
            clap::ErrorKind::InvalidUtf8,
            format!("SAF allele dimension {n} not supported."),
        ))
        }
    }
}"#
    )?;

    Ok(())
}

fn create_run_2d(f: &mut fs::File, dim: usize) -> io::Result<()> {
    writeln!(
        f,
        "{}",
        r#"pub fn run_2d<P>(first_path: P, second_path: P, args: &Cli) -> clap::Result<()>
where
    P: AsRef<Path>,
{
    let first_reader = saf::BgzfReader::from_bgzf_member_path(first_path)?;
    let second_reader = saf::BgzfReader::from_bgzf_member_path(second_path)?;

    let n = first_reader.index().alleles() + 1;
    let m = second_reader.index().alleles() + 1;
    match (n, m) {"#
    )?;

    for i in (3..=dim).step_by(2) {
        for j in (3..=dim).step_by(2) {
            writeln!(
                f,
                "\t\t\t({i}, {j}) => run_2d_inner::<{i}, {j}>(first_reader, second_reader, args),"
            )?;
        }
    }

    writeln!(
        f,
        "{}",
        r#"
        (n, m) => {
            Err(Cli::command().error(
            clap::ErrorKind::InvalidUtf8,
            format!("SAF allele dimensions {n}/{m} not supported."),
        ))
        }
    }
}"#
    )?;

    Ok(())
}

fn main() -> io::Result<()> {
    let out_dir = env::var("OUT_DIR").expect("failed to get OUT_DIR in build.rs");
    let dest_path = path::Path::new(&out_dir).join("run.rs");
    let mut f = fs::File::create(&dest_path).unwrap();

    let max_1d_dim: usize = match env::var("WINSFS_1D").map(|v| v.parse()) {
        Ok(Ok(v)) => v,
        _ => DEFAULT_1D,
    };
    let max_2d_dim: usize = match env::var("WINSFS_2D").map(|v| v.parse()) {
        Ok(Ok(v)) => v,
        _ => DEFAULT_2D,
    };

    create_run_1d(&mut f, max_1d_dim)?;
    writeln!(f)?;
    create_run_2d(&mut f, max_2d_dim)?;

    Ok(())
}
