use std::path::{Path, PathBuf};

use angsd_io::saf;

use clap::Args;

use crate::Sfs;

use super::utils::validate_shape;

/// Calculate log-likelihood of site frequency spectrum.
///
/// The SAF files will be streamed, and therefore the calculation requires only constant memory
/// usage.
#[derive(Args, Debug)]
pub struct LogLikelihood {
    #[clap(from_global)]
    pub paths: Vec<PathBuf>,

    /// Input SFS to calculate log-likelihood from.
    #[clap(short = 'i', long)]
    pub sfs: PathBuf,
}

impl LogLikelihood {
    pub(crate) fn run(&self) -> Result<(), clap::Error> {
        match self.paths.as_slice() {
            [path] => run_1d(path, &self),
            [first_path, second_path] => run_2d(first_path, second_path, &self),
            _ => unreachable!(), // Checked by clap
        }
    }
}

fn run_1d<P>(saf_path: P, args: &LogLikelihood) -> clap::Result<()>
where
    P: AsRef<Path>,
{
    let mut sfs = Sfs::read_from_angsd(&args.sfs)?;
    sfs.normalise();

    let mut reader = saf::Reader::from_bgzf_member_path(saf_path)?;
    let sites = reader.index().total_sites();
    let cols = reader.index().alleles() + 1;

    validate_shape(sfs.shape(), [cols])?;

    log::info!(
        target: "log-likelihood",
        "Streamining {sites} sites in SAF file with {cols} cols."
    );

    let mut log_likelihood = 0.0;

    let value_reader = reader.value_reader_mut();
    let mut buf = vec![0.0; cols];
    let site = buf.as_mut_slice();

    while value_reader.read_values(site)?.is_not_done() {
        exp(site);

        log_likelihood += sfs.site_log_likelihood(&site);
    }

    println!("{log_likelihood}");

    Ok(())
}

pub fn run_2d<P>(first_path: P, second_path: P, args: &LogLikelihood) -> clap::Result<()>
where
    P: AsRef<Path>,
{
    let mut sfs = Sfs::read_from_angsd(&args.sfs)?;
    sfs.normalise();

    let first_reader = saf::Reader::from_bgzf_member_path(first_path)?;
    let first_sites = first_reader.index().total_sites();
    let first_cols = first_reader.index().alleles() + 1;

    let second_reader = saf::Reader::from_bgzf_member_path(second_path)?;
    let second_sites = second_reader.index().total_sites();
    let second_cols = second_reader.index().alleles() + 1;

    validate_shape(sfs.shape(), [first_cols, second_cols])?;

    log::info!(
        target: "log-likelihood",
        "Streamining intersecting sites among {first_sites}/{second_sites} total sites \
         in SAF files with {first_cols}/{second_cols} cols."
    );

    let mut reader = saf::reader::Intersect::new(first_reader, second_reader);

    let mut log_likelihood = 0.0;

    let (mut first_record, mut second_record) = reader.create_record_buf();

    while reader
        .read_record_pair(&mut first_record, &mut second_record)?
        .is_not_done()
    {
        exp(first_record.values_mut());
        exp(second_record.values_mut());

        log_likelihood += sfs.site_log_likelihood(first_record.values(), second_record.values());
    }

    println!("{log_likelihood}");

    Ok(())
}

fn exp(values: &mut [f32]) {
    values.iter_mut().for_each(|x| *x = x.exp());
}
