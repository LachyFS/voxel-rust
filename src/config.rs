use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub rendering: RenderingConfig,
    pub terrain: TerrainConfig,
    pub camera: CameraConfig,
}

#[derive(Debug, Deserialize)]
pub struct RenderingConfig {
    /// Horizontal render distance in chunks
    pub render_distance: i32,
    /// Vertical render distance in chunks
    pub height_chunks: i32,
}

#[derive(Debug, Deserialize)]
pub struct TerrainConfig {
    /// World generation seed
    pub seed: u32,
    /// Sea level height in blocks
    pub sea_level: i32,
    /// Terrain noise scale
    pub terrain_scale: f64,
    /// Maximum terrain height variation
    pub terrain_height: f64,
    /// Cave generation scale
    pub cave_scale: f64,
    /// Cave threshold (higher = fewer caves)
    pub cave_threshold: f64,
}

#[derive(Debug, Deserialize)]
pub struct CameraConfig {
    /// Movement speed in blocks per second
    pub movement_speed: f32,
    /// Mouse sensitivity
    pub mouse_sensitivity: f32,
    /// Starting X position
    pub start_x: f32,
    /// Starting Y position
    pub start_y: f32,
    /// Starting Z position
    pub start_z: f32,
    /// Starting pitch in degrees
    pub start_pitch: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            rendering: RenderingConfig {
                render_distance: 4,
                height_chunks: 4,
            },
            terrain: TerrainConfig {
                seed: 42,
                sea_level: 32,
                terrain_scale: 0.02,
                terrain_height: 32.0,
                cave_scale: 0.08,
                cave_threshold: 0.55,
            },
            camera: CameraConfig {
                movement_speed: 20.0,
                mouse_sensitivity: 0.003,
                start_x: 32.0,
                start_y: 80.0,
                start_z: 32.0,
                start_pitch: -30.0,
            },
        }
    }
}

impl Config {
    /// Load configuration from a TOML file, falling back to defaults if not found
    pub fn load(path: &Path) -> Self {
        match fs::read_to_string(path) {
            Ok(contents) => match toml::from_str(&contents) {
                Ok(config) => {
                    log::info!("Loaded configuration from {:?}", path);
                    config
                }
                Err(e) => {
                    log::warn!("Failed to parse config file: {}. Using defaults.", e);
                    Self::default()
                }
            },
            Err(e) => {
                log::warn!("Failed to read config file: {}. Using defaults.", e);
                Self::default()
            }
        }
    }
}
