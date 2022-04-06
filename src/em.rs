#[macro_use]
mod log;

mod core;
pub use self::core::{Em, Input};

mod stop;
pub use stop::{StoppingRule, DEFAULT_TOLERANCE};

mod window;
pub use window::Window;
