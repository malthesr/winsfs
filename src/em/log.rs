macro_rules! info_sfs {
    (target: $target:expr, $fmt_str:literal, $sfs:expr, $sites:expr) => {
        log_sfs!(target: $target, log::Level::Info, $fmt_str, $sfs, $sites)
    };
}

macro_rules! trace_sfs {
    (target: $target:expr, $fmt_str:literal, $sfs:expr, $sites:expr) => {
        log_sfs!(target: $target, log::Level::Trace, $fmt_str, $sfs, $sites)
    };
}

macro_rules! log_sfs {
    (target: $target:expr, $level:expr, $fmt_str:literal, $sfs:expr, $sites:expr) => {
        if log::log_enabled!(target: $target, $level) {
            let fmt_sfs = $sfs
                .iter()
                .map(|v| format!("{:.6}", v * $sites as f64))
                .collect::<Vec<_>>()
                .join(" ");

            log::log!(target: $target, $level, $fmt_str, fmt_sfs);
        }
    };
}
