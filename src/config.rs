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
    /// Worm cave scale (smaller = longer tunnels)
    #[serde(default = "default_cave_worm_scale")]
    pub cave_worm_scale: f64,
    /// Worm cave threshold (smaller = more caves)
    #[serde(default = "default_cave_worm_threshold")]
    pub cave_worm_threshold: f64,
    /// Cheese cave scale (smaller = larger caverns)
    #[serde(default = "default_cave_cheese_scale")]
    pub cave_cheese_scale: f64,
    /// Cheese cave threshold (higher = fewer caves)
    #[serde(default = "default_cave_cheese_threshold")]
    pub cave_cheese_threshold: f64,
    /// Ravine scale (smaller = longer ravines)
    #[serde(default = "default_ravine_scale")]
    pub ravine_scale: f64,
    /// Ravine threshold (higher = fewer ravines)
    #[serde(default = "default_ravine_threshold")]
    pub ravine_threshold: f64,
    /// Ravine depth scale (how deep ravines cut)
    #[serde(default = "default_ravine_depth_scale")]
    pub ravine_depth_scale: f64,
}

fn default_cave_worm_scale() -> f64 { 0.04 }
fn default_cave_worm_threshold() -> f64 { 0.03 }
fn default_cave_cheese_scale() -> f64 { 0.06 }
fn default_cave_cheese_threshold() -> f64 { 0.7 }
fn default_ravine_scale() -> f64 { 0.015 }
fn default_ravine_threshold() -> f64 { 0.85 }
fn default_ravine_depth_scale() -> f64 { 40.0 }

#[derive(Debug, Deserialize)]
pub struct CameraConfig {
    /// Field of view in degrees
    #[serde(default = "default_fov")]
    pub fov: f32,
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

fn default_fov() -> f32 {
    70.0
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
                cave_worm_scale: 0.04,
                cave_worm_threshold: 0.03,
                cave_cheese_scale: 0.06,
                cave_cheese_threshold: 0.7,
                ravine_scale: 0.015,
                ravine_threshold: 0.85,
                ravine_depth_scale: 40.0,
            },
            camera: CameraConfig {
                fov: 90.0,
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
            Ok(contents) => Self::from_str(&contents, path),
            Err(e) => {
                log::warn!("Failed to read config file: {}. Using defaults.", e);
                Self::default()
            }
        }
    }

    /// Load configuration from embedded string data
    #[cfg(feature = "embed-assets")]
    pub fn load_embedded() -> Self {
        let contents = crate::embedded_assets::CONFIG_TOML;
        Self::from_str(contents, Path::new("(embedded)"))
    }

    fn from_str(contents: &str, source: &Path) -> Self {
        match toml::from_str(contents) {
            Ok(config) => {
                log::info!("Loaded configuration from {:?}", source);
                config
            }
            Err(e) => {
                log::warn!("Failed to parse config file: {}. Using defaults.", e);
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
    /// Blend radius for smooth biome transitions (default 0.15)
    #[serde(default = "default_blend_radius")]
    pub blend_radius: f64,
    /// Noise strength for irregular biome boundaries (default 0.08)
    #[serde(default = "default_blend_noise_strength")]
    pub blend_noise_strength: f64,
}

fn default_blend_radius() -> f64 {
    0.15
}

fn default_blend_noise_strength() -> f64 {
    0.08
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
    /// Terrain height multiplier (default 1.0, higher = taller terrain)
    #[serde(default = "default_height_scale")]
    pub height_scale: f64,
    /// Base height offset relative to sea level (default 0)
    #[serde(default)]
    pub base_height: i32,
    /// Terrain noise frequency multiplier (default 1.0, higher = more jagged)
    #[serde(default = "default_one")]
    pub noise_scale: f64,
    /// Detail noise strength (default 0.25, 0 = smooth, higher = rougher)
    #[serde(default = "default_detail_strength")]
    pub detail_strength: f64,
    /// Terrain flatness (0.0 = normal, 1.0 = completely flat)
    #[serde(default)]
    pub flatness: f64,
}

fn default_height_scale() -> f64 {
    1.0
}

fn default_one() -> f64 {
    1.0
}

fn default_detail_strength() -> f64 {
    0.25
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
    /// Terrain height multiplier (higher = taller terrain)
    pub height_scale: f64,
    /// Base height offset relative to sea level
    pub base_height: i32,
    /// Terrain noise frequency multiplier (higher = more jagged)
    pub noise_scale: f64,
    /// Detail noise strength (0 = smooth, higher = rougher)
    pub detail_strength: f64,
    /// Terrain flatness (0.0 = normal, 1.0 = completely flat)
    pub flatness: f64,
}

/// Compiled biomes configuration ready for worldgen
#[derive(Clone, Debug)]
pub struct CompiledBiomesConfig {
    pub biome_scale: f64,
    pub blend_radius: f64,
    /// Noise strength for irregular biome boundaries
    pub blend_noise_strength: f64,
    pub biomes: Vec<CompiledBiome>,
}

impl Default for CompiledBiomesConfig {
    fn default() -> Self {
        Self {
            biome_scale: 0.005,
            blend_radius: 0.15,
            blend_noise_strength: 0.08,
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
                    height_scale: 0.5,
                    base_height: 0,
                    noise_scale: 1.0,
                    detail_strength: 0.15,
                    flatness: 0.3,
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
                    height_scale: 0.7,
                    base_height: 5,
                    noise_scale: 1.0,
                    detail_strength: 0.25,
                    flatness: 0.0,
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
                    height_scale: 0.4,
                    base_height: -5,
                    noise_scale: 0.8,
                    detail_strength: 0.1,
                    flatness: 0.5,
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
                    height_scale: 2.0,
                    base_height: 20,
                    noise_scale: 1.5,
                    detail_strength: 0.4,
                    flatness: 0.0,
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
            Ok(contents) => Self::from_str(&contents, path),
            Err(e) => {
                log::warn!("Failed to read biomes config: {}. Using defaults.", e);
                Self::default()
            }
        }
    }

    /// Load biomes configuration from embedded string data
    #[cfg(feature = "embed-assets")]
    pub fn load_embedded() -> Self {
        let contents = crate::embedded_assets::BIOMES_TOML;
        Self::from_str(contents, Path::new("(embedded)"))
    }

    fn from_str(contents: &str, source: &Path) -> Self {
        match toml::from_str::<BiomesConfig>(contents) {
            Ok(config) => {
                let compiled = Self::compile(config);
                log::info!("Loaded {} biomes from {:?}", compiled.biomes.len(), source);
                compiled
            }
            Err(e) => {
                log::warn!("Failed to parse biomes config: {}. Using defaults.", e);
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
                height_scale: def.height_scale,
                base_height: def.base_height,
                noise_scale: def.noise_scale,
                detail_strength: def.detail_strength,
                flatness: def.flatness,
            })
            .collect();

        Self {
            biome_scale: config.settings.biome_scale,
            blend_radius: config.settings.blend_radius,
            blend_noise_strength: config.settings.blend_noise_strength,
            biomes,
        }
    }
}
