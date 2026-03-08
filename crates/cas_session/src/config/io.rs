use std::fs;
use std::io::Write;
use std::path::Path;

use super::CasConfig;

impl CasConfig {
    pub fn load() -> Self {
        let path = Path::new("cas_config.toml");
        if path.exists() {
            match fs::read_to_string(path) {
                Ok(content) => match toml::from_str(&content) {
                    Ok(config) => return config,
                    Err(error) => println!("Error parsing config file: {}. Using defaults.", error),
                },
                Err(error) => println!("Error reading config file: {}. Using defaults.", error),
            }
        }
        Self::default()
    }

    pub fn save(&self) -> std::io::Result<()> {
        let content = toml::to_string_pretty(self).map_err(std::io::Error::other)?;
        let mut file = fs::File::create("cas_config.toml")?;
        file.write_all(content.as_bytes())?;
        Ok(())
    }

    pub fn restore() -> Self {
        let config = Self::default();
        let _ = config.save();
        config
    }
}
