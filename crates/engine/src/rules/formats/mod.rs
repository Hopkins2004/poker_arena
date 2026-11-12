//! One file per poker format. Each defines a zero-sized type implementing
//! [`FormatRules`](super::FormatRules) and a `format(...)` constructor for its standard
//! [`GameFormat`](crate::format::GameFormat).

pub mod flhe;
pub mod nlhe;
pub mod plo;
