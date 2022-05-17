# winsfs

[![GitHub Actions status](https://github.com/malthesr/winsfs/workflows/CI/badge.svg)](https://github.com/malthesr/winsfs/actions)

**winsfs** is a tool for inference of the site frequency spectrum ("SFS"). 

## Quickstart

Assuming [SAF files][saf] have already been made, default (in-memory) estimation can be run in one or two dimensions, respectively:

```shell
> winsfs $saf1
> winsfs $saf1 $saf2
```

Here, `$saf1`/`$saf2` is the path to any SAF member file (i.e. some file with extension `.saf.idx`, `.saf.pos.gz`, or `.saf.gz`). In the multi-dimensional case, the SAF files are intersected automatically. The output SFS will be written to stdout.

It is also possible to run `winsfs` in streaming mode. To do so, an intermediate, pre-shuffled file must be prepared before running, using the `shuffle` subcommand:

```shell
> winsfs shuffle --output single.saf.shuf $saf1
> winsfs single.saf.shuf
> winsfs shuffle --output joint.saf.shuf $saf1 $saf2
> winsfs joint.saf.shuf
```

The shuffled file must be pre-allocated, and therefore cannot be written to stdout, but must be provided via the `--output` flag. Also note that in multiple dimensions, the shuffle must happen jointly, not per population. Estimation can be run exactly as in the in-memory case, streaming is detected based on the file type.

For more settings, see `winsfs -h` (short help) or `winsfs --help` (long help).

## Installation

### From source

To build from source, a recent Rust toolchain is required. Currently, this can be installed by running:

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

See [instructions][rust-installation] for more details.

Once the toolchain is installed, `winsfs` can be installed using `cargo`:

```shell
cargo install --git https://github.com/malthesr/winsfs
```

This will install to `$HOME/.cargo/bin` by default, which should be in the `$PATH` after installing `cargo`. Alternatively:

```shell
cargo install --git https://github.com/malthesr/winsfs --root $HOME
```

Will install to `$HOME/bin`.

[saf]: http://www.popgen.dk/angsd/index.php/Safv3
[rust-installation]: https://www.rust-lang.org/tools/install