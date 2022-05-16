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

To build from source, install the Rust toolchain (see [instructions][rust-installation]), and run:

```
cargo install --git https://github.com/malthesr/winsfs
```

This will install to `$HOME/.cargo/bin` by default.

Optionally, the code may be optimised for your specific CPU (for potentially better performance), using:

```
RUSTFLAGS="-C target-cpu=native" cargo install --git https://github.com/malthesr/winsfs
```

### Pre-compiled

Pre-compiled binaries are available from the [releases][releases] page ([linux][linux-binary], [mac][mac-binary], [windows][windows-binary]).

[saf]: http://www.popgen.dk/angsd/index.php/Safv3
[releases]: https://github.com/malthesr/winsfs/releases/latest/
[linux-binary]: https://github.com/malthesr/winsfs/releases/latest/download/winsfs-x86_64-unknown-linux-gnu.tar.gz
[mac-binary]: https://github.com/malthesr/winsfs/releases/latest/download/winsfs-x86_64-apple-darwin.tar.gz
[windows-binary]: https://github.com/malthesr/winsfs/releases/latest/download/winsfs-x86_64-pc-windows-msvc.zip
[rust-installation]: https://www.rust-lang.org/tools/install