use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::voxel::BlockType;

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

// ============================================================================
// Biome Configuration
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct BiomesConfig {
    pub settings: BiomeSettings,
    pub biomes: HashMap<String, BiomeDefinition>,
}

#[derive(Debug, Deserialize)]
pub struct BiomeSettings {
    /// Scale for biome noise (smaller = larger biomes)
    pub biome_scale: f64,
}

#[derive(Debug, Deserialize)]
pub struct BiomeDefinition {
    /// Minimum temperature for this biome (-1.0 to 1.0)
    pub temperature_min: f64,
    /// Maximum temperature for this biome (-1.0 to 1.0)
    pub temperature_max: f64,
    /// Minimum moisture for this biome (-1.0 to 1.0)
    pub moisture_min: f64,
    /// Maximum moisture for this biome (-1.0 to 1.0)
    pub moisture_max: f64,
    /// Block type for the surface layer
    pub surface_block: String,
    /// Block type for subsurface layers (up to 4 blocks deep)
    pub subsurface_block: String,
    /// Height above sea level where stone replaces grass (optional)
    #[serde(default)]
    pub stone_altitude: Option<i32>,
}

/// Compiled biome data for efficient runtime lookup
#[derive(Clone, Debug)]
pub struct CompiledBiome {
    pub name: String,
    pub temperature_min: f64,
    pub temperature_max: f64,
    pub moisture_min: f64,
    pub moisture_max: f64,
    pub surface_block: BlockType,
    pub subsurface_block: BlockType,
    pub stone_altitude: Option<i32>,
}

/// Compiled biomes configuration ready for worldgen
#[derive(Clone, Debug)]
pub struct CompiledBiomesConfig {
    pub biome_scale: f64,
    pub biomes: Vec<CompiledBiome>,
}

impl Default for CompiledBiomesConfig {
    fn default() -> Self {
        Self {
            biome_scale: 0.005,
            biomes: vec![
                CompiledBiome {
                    name: "plains".to_string(),
                    temperature_min: -1.0,
                    temperature_max: 0.0,
                    moisture_min: 0.0,
                    moisture_max: 1.0,
                    surface_block: BlockType::Grass,
                    subsurface_block: BlockType::Dirt,
                    stone_altitude: None,
                },
                CompiledBiome {
                    name: "forest".to_string(),
                    temperature_min: 0.0,
                    temperature_max: 1.0,
                    moisture_min: 0.0,
                    moisture_max: 1.0,
                    surface_block: BlockType::Grass,
                    subsurface_block: BlockType::Dirt,
                    stone_altitude: None,
                },
                CompiledBiome {
                    name: "desert".to_string(),
                    temperature_min: 0.0,
                    temperature_max: 1.0,
                    moisture_min: -1.0,
                    moisture_max: 0.0,
                    surface_block: BlockType::Sand,
                    subsurface_block: BlockType::Sand,
                    stone_altitude: None,
                },
                CompiledBiome {
                    name: "mountains".to_string(),
                    temperature_min: -1.0,
                    temperature_max: 0.0,
                    moisture_min: -1.0,
                    moisture_max: 0.0,
                    surface_block: BlockType::Grass,
                    subsurface_block: BlockType::Dirt,
                    stone_altitude: Some(20),
                },
            ],
        }
    }
}

fn parse_block_type(name: &str) -> BlockType {
    match name.to_lowercase().as_str() {
        "air" => BlockType::Air,
        "grass" => BlockType::Grass,
        "dirt" => BlockType::Dirt,
        "stone" => BlockType::Stone,
        "cobblestone" => BlockType::Cobblestone,
        "sand" => BlockType::Sand,
        "wood" => BlockType::Wood,
        "leaves" => BlockType::Leaves,
        "brick" => BlockType::Brick,
        "water" => BlockType::Water,
        "snow" => BlockType::Snow,
        "ice" => BlockType::Ice,
        "gravel" => BlockType::Gravel,
        "clay" => BlockType::Clay,
        "cactus" => BlockType::Cactus,
        "dead_bush" => BlockType::DeadBush,
        "tall_grass" => BlockType::TallGrass,
        "podzol" => BlockType::Podzol,
        "spruce_wood" => BlockType::SpruceWood,
        "spruce_leaves" => BlockType::SpruceLeaves,
        _ => {
            log::warn!("Unknown block type '{}', defaulting to Stone", name);
            BlockType::Stone
        }
    }
}

impl CompiledBiomesConfig {
    /// Load biomes configuration from a TOML file
    pub fn load(path: &Path) -> Self {
        match fs::read_to_string(path) {
            Ok(contents) => match toml::from_str::<BiomesConfig>(&contents) {
                Ok(config) => {
                    let compiled = Self::compile(config);
                    log::info!("Loaded {} biomes from {:?}", compiled.biomes.len(), path);
                    compiled
                }
                Err(e) => {
                    log::warn!("Failed to parse biomes config: {}. Using defaults.", e);
                    Self::default()
                }
            },
            Err(e) => {
                log::warn!("Failed to read biomes config: {}. Using defaults.", e);
                Self::default()
            }
        }
    }

    /// Compile raw config into efficient runtime format
    fn compile(config: BiomesConfig) -> Self {
        let biomes = config
            .biomes
            .into_iter()
            .map(|(name, def)| CompiledBiome {
                name,
                temperature_min: def.temperature_min,
                temperature_max: def.temperature_max,
                moisture_min: def.moisture_min,
                moisture_max: def.moisture_max,
                surface_block: parse_block_type(&def.surface_block),
                subsurface_block: parse_block_type(&def.subsurface_block),
                stone_altitude: def.stone_altitude,
            })
            .collect();

        Self {
            biome_scale: config.settings.biome_scale,
            biomes,
        }
    }
}
