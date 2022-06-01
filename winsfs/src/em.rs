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

mod core;
pub use self::core::Em;

mod stop;
pub use stop::{StoppingRule, DEFAULT_TOLERANCE};

mod window;
pub use window::Window;
