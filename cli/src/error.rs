use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Python error: {0}")]
    Python(String),

    #[error("Manifest error: {0}")]
    Manifest(String),

    #[error("Training error: {0}")]
    Training(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Package error: {0}")]
    Package(String),

    #[error("Packaging error: {0}")]
    Packaging(String),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Installation error: {0}")]
    Installation(String),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Signing error: {0}")]
    Signing(String),

    #[error("Regex error: {0}")]
    Regex(#[from] regex::Error),

    #[error("{0}")]
    Other(String),
}

impl From<pyo3::PyErr> for Error {
    fn from(err: pyo3::PyErr) -> Self {
        Error::Python(err.to_string())
    }
}

impl<'a> From<pyo3::DowncastError<'a, '_>> for Error {
    fn from(err: pyo3::DowncastError) -> Self {
        Error::Python(format!("PyO3 downcast error: {}", err))
    }
}
