use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::io::Write;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CasConfig {
    pub distribute: bool,
    pub expand_binomials: bool,
    pub distribute_constants: bool,
    pub factor_difference_squares: bool,
}

impl Default for CasConfig {
    fn default() -> Self {
        Self {
            distribute: false, // Default: Conservative
            expand_binomials: true, // Needed for many simplifications
            distribute_constants: true, // Safe distribution of -1, etc.
            factor_difference_squares: false, // Can cause loops if not careful
        }
    }
}

impl CasConfig {
    pub fn load() -> Self {
        let path = Path::new("cas_config.toml");
        if path.exists() {
            match fs::read_to_string(path) {
                Ok(content) => match toml::from_str(&content) {
                    Ok(config) => return config,
                    Err(e) => println!("Error parsing config file: {}. Using defaults.", e),
                },
                Err(e) => println!("Error reading config file: {}. Using defaults.", e),
            }
        }
        Self::default()
    }

    pub fn save(&self) -> std::io::Result<()> {
        let content = toml::to_string_pretty(self).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let mut file = fs::File::create("cas_config.toml")?;
        file.write_all(content.as_bytes())?;
        Ok(())
    }

    pub fn restore() -> Self {
        let config = Self::default();
        let _ = config.save(); // Overwrite file with defaults
        config
    }
}
