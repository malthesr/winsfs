# Changelog

## [Unreleased]

### Added 

- Added support for calculating common derived statistics using `winsfs stat`.

### Changed

- Updated `clap` to `4.0`, which causes some aesthetic changes to the CLI.

- When specifying blocks using `--blocks/-B` (or using the default `500`), the number of blocks requested will be exact. The behaviour of `--block-size/-b` is unchanged. This makes little difference in practice, but makes it a bit easier to reason about the blocking strategy.
