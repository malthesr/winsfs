# winsfs

[![GitHub Actions status](https://github.com/malthesr/winsfs/workflows/CI/badge.svg)](https://github.com/malthesr/winsfs/actions)

**winsfs** is a tool for inference of the site frequency spectrum ("SFS"). 

## Quickstart

Assuming [SAF files][saf] have already been made, default estimation can be run in one or two dimensions, respectively:

```shell
winsfs $saf1
winsfs $saf1 $saf2
```

Here, `$saf1`/`$saf2` is the path to any SAF member file (i.e. some file with extension `.saf.idx`, `.saf.pos.gz`, or `.saf.gz`). In the two-dimensional case, the SAF files are intersected automatically.  Be aware that the SAF files are kept in RAM.

For more settings, see `winsfs -h` (short help) or `winsfs --help` (long help).## Installation

### Pre-compiled

Pre-compiled binaries are available from the [releases][releases] page ([linux][linux-binary], [mac][mac-binary], [windows][windows-binary]).

### From source

To build from source, install the Rust toolchain (see [instructions][rust-installation]), and run:

```
cargo install --git https://github.com/malthesr/winsfs
```

This will install to `$HOME/.cargo/bin` by default. Be aware, however, that the projects experiments with static dimensions in the SFS and SAFs, and as a result compilation is quite memory-intensive and may take a while.

[saf]: http://www.popgen.dk/angsd/index.php/Safv3
[releases]: https://github.com/malthesr/winsfs/releases/latest/
[linux-binary]: https://github.com/malthesr/winsfs/releases/latest/download/winsfs-x86_64-unknown-linux-gnu.tar.gz
[mac-binary]: https://github.com/malthesr/winsfs/releases/latest/download/winsfs-x86_64-apple-darwin.tar.gz
[windows-binary]: https://github.com/malthesr/winsfs/releases/latest/download/winsfs-x86_64-pc-windows-msvc.zip
[rust-installation]: https://www.rust-lang.org/tools/install