use noise::{Fbm, MultiFractal, NoiseFn, Perlin, Simplex};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

use crate::camera::Frustum;
use crate::config::{CompiledBiome, CompiledBiomesConfig, TerrainConfig};
use crate::texture::BlockTextureArray;
use crate::voxel::{BlockType, Chunk, ChunkNeighbors, NeighborBoundaries, Vertex, CHUNK_SIZE};
use glam::Vec3;

// ============================================================================
// Structure Generation
// ============================================================================

/// Maximum horizontal radius a structure can extend from its spawn point
const STRUCTURE_CHECK_RADIUS: i32 = 5;

/// Structure types that can spawn in the world
#[derive(Clone, Copy, Debug, PartialEq)]
enum StructureType {
    OakTree,
    Cactus,
    SpruceTree,
    DeadBush,
    TallGrass,
}

/// A block placement within a structure, relative to spawn point
#[derive(Clone, Copy)]
struct StructureBlock {
    dx: i32, // Offset from spawn X
    dy: i32, // Offset from spawn Y (spawn Y is surface + 1)
    dz: i32, // Offset from spawn Z
    block: BlockType,
}

/// Get the blocks that make up a structure
fn get_structure_blocks(structure_type: StructureType) -> &'static [StructureBlock] {
    match structure_type {
        StructureType::OakTree => &OAK_TREE_BLOCKS,
        StructureType::Cactus => &CACTUS_BLOCKS,
        StructureType::SpruceTree => &SPRUCE_TREE_BLOCKS,
        StructureType::DeadBush => &DEAD_BUSH_BLOCKS,
        StructureType::TallGrass => &TALL_GRASS_BLOCKS,
    }
}

/// Oak tree: 5 blocks tall trunk + 3x3x3 leaf canopy at top
static OAK_TREE_BLOCKS: [StructureBlock; 31] = [
    // Trunk (5 blocks, y=0 to y=4)
    StructureBlock { dx: 0, dy: 0, dz: 0, block: BlockType::Wood },
    StructureBlock { dx: 0, dy: 1, dz: 0, block: BlockType::Wood },
    StructureBlock { dx: 0, dy: 2, dz: 0, block: BlockType::Wood },
    StructureBlock { dx: 0, dy: 3, dz: 0, block: BlockType::Wood },
    StructureBlock { dx: 0, dy: 4, dz: 0, block: BlockType::Wood },
    // Leaves layer 1 (y=3, 3x3 minus corners)
    StructureBlock { dx: -1, dy: 3, dz: 0, block: BlockType::Leaves },
    StructureBlock { dx: 1, dy: 3, dz: 0, block: BlockType::Leaves },
    StructureBlock { dx: 0, dy: 3, dz: -1, block: BlockType::Leaves },
    StructureBlock { dx: 0, dy: 3, dz: 1, block: BlockType::Leaves },
    // Leaves layer 2 (y=4, 3x3)
    StructureBlock { dx: -1, dy: 4, dz: -1, block: BlockType::Leaves },
    StructureBlock { dx: -1, dy: 4, dz: 0, block: BlockType::Leaves },
    StructureBlock { dx: -1, dy: 4, dz: 1, block: BlockType::Leaves },
    StructureBlock { dx: 0, dy: 4, dz: -1, block: BlockType::Leaves },
    // trunk at (0, 4,0) - already added above
    StructureBlock { dx: 0, dy: 4, dz: 1, block: BlockType::Leaves },
    StructureBlock { dx: 1, dy: 4, dz: -1, block: BlockType::Leaves },
    StructureBlock { dx: 1, dy: 4, dz: 0, block: BlockType::Leaves },
    StructureBlock { dx: 1, dy: 4, dz: 1, block: BlockType::Leaves },
    // Leaves layer 3 (y=5, 3x3)
    StructureBlock { dx: -1, dy: 5, dz: -1, block: BlockType::Leaves },
    StructureBlock { dx: -1, dy: 5, dz: 0, block: BlockType::Leaves },
    StructureBlock { dx: -1, dy: 5, dz: 1, block: BlockType::Leaves },
    StructureBlock { dx: 0, dy: 5, dz: -1, block: BlockType::Leaves },
    StructureBlock { dx: 0, dy: 5, dz: 0, block: BlockType::Leaves },
    StructureBlock { dx: 0, dy: 5, dz: 1, block: BlockType::Leaves },
    StructureBlock { dx: 1, dy: 5, dz: -1, block: BlockType::Leaves },
    StructureBlock { dx: 1, dy: 5, dz: 0, block: BlockType::Leaves },
    StructureBlock { dx: 1, dy: 5, dz: 1, block: BlockType::Leaves },
    // Top leaves (y=6, cross pattern)
    StructureBlock { dx: 0, dy: 6, dz: 0, block: BlockType::Leaves },
    StructureBlock { dx: -1, dy: 6, dz: 0, block: BlockType::Leaves },
    StructureBlock { dx: 1, dy: 6, dz: 0, block: BlockType::Leaves },
    StructureBlock { dx: 0, dy: 6, dz: -1, block: BlockType::Leaves },
    StructureBlock { dx: 0, dy: 6, dz: 1, block: BlockType::Leaves },
];

/// Cactus: 3 blocks tall
static CACTUS_BLOCKS: [StructureBlock; 3] = [
    StructureBlock { dx: 0, dy: 0, dz: 0, block: BlockType::Cactus },
    StructureBlock { dx: 0, dy: 1, dz: 0, block: BlockType::Cactus },
    StructureBlock { dx: 0, dy: 2, dz: 0, block: BlockType::Cactus },
];

/// Spruce tree: 7 blocks tall trunk with triangular canopy
static SPRUCE_TREE_BLOCKS: [StructureBlock; 40] = [
    // Trunk (7 blocks, y=0 to y=6)
    StructureBlock { dx: 0, dy: 0, dz: 0, block: BlockType::SpruceWood },
    StructureBlock { dx: 0, dy: 1, dz: 0, block: BlockType::SpruceWood },
    StructureBlock { dx: 0, dy: 2, dz: 0, block: BlockType::SpruceWood },
    StructureBlock { dx: 0, dy: 3, dz: 0, block: BlockType::SpruceWood },
    StructureBlock { dx: 0, dy: 4, dz: 0, block: BlockType::SpruceWood },
    StructureBlock { dx: 0, dy: 5, dz: 0, block: BlockType::SpruceWood },
    StructureBlock { dx: 0, dy: 6, dz: 0, block: BlockType::SpruceWood },
    // Bottom layer (y=2, 3x3 cross)
    StructureBlock { dx: -1, dy: 2, dz: 0, block: BlockType::SpruceLeaves },
    StructureBlock { dx: 1, dy: 2, dz: 0, block: BlockType::SpruceLeaves },
    StructureBlock { dx: 0, dy: 2, dz: -1, block: BlockType::SpruceLeaves },
    StructureBlock { dx: 0, dy: 2, dz: 1, block: BlockType::SpruceLeaves },
    StructureBlock { dx: -2, dy: 2, dz: 0, block: BlockType::SpruceLeaves },
    StructureBlock { dx: 2, dy: 2, dz: 0, block: BlockType::SpruceLeaves },
    StructureBlock { dx: 0, dy: 2, dz: -2, block: BlockType::SpruceLeaves },
    StructureBlock { dx: 0, dy: 2, dz: 2, block: BlockType::SpruceLeaves },
    // Layer 2 (y=3, smaller)
    StructureBlock { dx: -1, dy: 3, dz: 0, block: BlockType::SpruceLeaves },
    StructureBlock { dx: 1, dy: 3, dz: 0, block: BlockType::SpruceLeaves },
    StructureBlock { dx: 0, dy: 3, dz: -1, block: BlockType::SpruceLeaves },
    StructureBlock { dx: 0, dy: 3, dz: 1, block: BlockType::SpruceLeaves },
    // Layer 3 (y=4, cross)
    StructureBlock { dx: -1, dy: 4, dz: 0, block: BlockType::SpruceLeaves },
    StructureBlock { dx: 1, dy: 4, dz: 0, block: BlockType::SpruceLeaves },
    StructureBlock { dx: 0, dy: 4, dz: -1, block: BlockType::SpruceLeaves },
    StructureBlock { dx: 0, dy: 4, dz: 1, block: BlockType::SpruceLeaves },
    StructureBlock { dx: -1, dy: 4, dz: -1, block: BlockType::SpruceLeaves },
    StructureBlock { dx: -1, dy: 4, dz: 1, block: BlockType::SpruceLeaves },
    StructureBlock { dx: 1, dy: 4, dz: -1, block: BlockType::SpruceLeaves },
    StructureBlock { dx: 1, dy: 4, dz: 1, block: BlockType::SpruceLeaves },
    // Layer 4 (y=5, smaller)
    StructureBlock { dx: -1, dy: 5, dz: 0, block: BlockType::SpruceLeaves },
    StructureBlock { dx: 1, dy: 5, dz: 0, block: BlockType::SpruceLeaves },
    StructureBlock { dx: 0, dy: 5, dz: -1, block: BlockType::SpruceLeaves },
    StructureBlock { dx: 0, dy: 5, dz: 1, block: BlockType::SpruceLeaves },
    // Layer 5 (y=6, cross around trunk)
    StructureBlock { dx: -1, dy: 6, dz: 0, block: BlockType::SpruceLeaves },
    StructureBlock { dx: 1, dy: 6, dz: 0, block: BlockType::SpruceLeaves },
    StructureBlock { dx: 0, dy: 6, dz: -1, block: BlockType::SpruceLeaves },
    StructureBlock { dx: 0, dy: 6, dz: 1, block: BlockType::SpruceLeaves },
    // Top (y=7, single block)
    StructureBlock { dx: 0, dy: 7, dz: 0, block: BlockType::SpruceLeaves },
    // Extra corner leaves at y=6
    StructureBlock { dx: -1, dy: 6, dz: -1, block: BlockType::SpruceLeaves },
    StructureBlock { dx: -1, dy: 6, dz: 1, block: BlockType::SpruceLeaves },
    StructureBlock { dx: 1, dy: 6, dz: -1, block: BlockType::SpruceLeaves },
    StructureBlock { dx: 1, dy: 6, dz: 1, block: BlockType::SpruceLeaves },
];

/// Dead bush: single block
static DEAD_BUSH_BLOCKS: [StructureBlock; 1] = [
    StructureBlock { dx: 0, dy: 0, dz: 0, block: BlockType::DeadBush },
];

/// Tall grass: single block
static TALL_GRASS_BLOCKS: [StructureBlock; 1] = [
    StructureBlock { dx: 0, dy: 0, dz: 0, block: BlockType::TallGrass },
];

/// Deterministic hash for structure spawn decisions
/// Returns a value 0-255 based on position and seed
#[inline]
fn structure_hash(x: i32, z: i32, seed: u32) -> u8 {
    let mut h = (x as u32).wrapping_mul(73856093)
        ^ (z as u32).wrapping_mul(19349663)
        ^ seed.wrapping_mul(83492791);
    h ^= h >> 16;
    h = h.wrapping_mul(0x85ebca6b);
    h ^= h >> 13;
    (h & 0xFF) as u8
}

/// Chunk coordinate key for HashMap
pub type ChunkPos = (i32, i32, i32);

// ============================================================================
// Biome Blending
// ============================================================================

/// Blended terrain parameters for smooth biome transitions
#[derive(Clone, Debug)]
struct BlendedTerrainParams {
    height_scale: f64,
    base_height: f64,
    noise_scale: f64,
    detail_strength: f64,
    flatness: f64,
}

/// Calculate distance from a value to a range (0 if inside range)
#[inline]
fn distance_to_range(value: f64, min: f64, max: f64) -> f64 {
    if value < min {
        min - value
    } else if value > max {
        value - max
    } else {
        0.0
    }
}

/// World generation parameters
pub struct WorldGenConfig {
    pub seed: u32,
    pub sea_level: i32,
    pub terrain_scale: f64,
    pub terrain_height: f64,
    pub cave_scale: f64,
    pub cave_threshold: f64,
}

impl Default for WorldGenConfig {
    fn default() -> Self {
        Self {
            seed: 12345,
            sea_level: 32,
            terrain_scale: 0.02,
            terrain_height: 32.0,
            cave_scale: 0.08,
            cave_threshold: 0.55,
        }
    }
}

/// Procedural world generator
pub struct WorldGenerator {
    config: WorldGenConfig,
    biomes_config: CompiledBiomesConfig,
    terrain_noise: Fbm<Perlin>,
    detail_noise: Perlin,
    cave_noise: Simplex,
    cave_noise_2: Simplex,
    ore_noise: Perlin,
    biome_noise: Perlin,
}

impl WorldGenerator {
    pub fn new(config: WorldGenConfig, biomes_config: CompiledBiomesConfig) -> Self {
        let seed = config.seed;

        let terrain_noise = Fbm::<Perlin>::new(seed)
            .set_octaves(6)
            .set_frequency(1.0)
            .set_persistence(0.5)
            .set_lacunarity(2.0);

        Self {
            terrain_noise,
            detail_noise: Perlin::new(seed.wrapping_add(1)),
            cave_noise: Simplex::new(seed.wrapping_add(2)),
            cave_noise_2: Simplex::new(seed.wrapping_add(3)),
            ore_noise: Perlin::new(seed.wrapping_add(4)),
            biome_noise: Perlin::new(seed.wrapping_add(5)),
            config,
            biomes_config,
        }
    }

    pub fn generate_chunk(&self, chunk_x: i32, chunk_y: i32, chunk_z: i32) -> Chunk {
        let mut chunk = Chunk::new([chunk_x, chunk_y, chunk_z]);

        let world_x = chunk_x * CHUNK_SIZE as i32;
        let world_y = chunk_y * CHUNK_SIZE as i32;
        let world_z = chunk_z * CHUNK_SIZE as i32;

        // Pre-compute structure blocks for this chunk region (O(n) once instead of O(n²) per block)
        // We need to check spawn points in a radius around the chunk that could place blocks inside
        let structure_blocks = self.precompute_structure_blocks(
            world_x,
            world_y,
            world_z,
            CHUNK_SIZE as i32,
        );

        for local_x in 0..CHUNK_SIZE {
            for local_z in 0..CHUNK_SIZE {
                let wx = world_x + local_x as i32;
                let wz = world_z + local_z as i32;

                let height = self.get_terrain_height(wx, wz);
                let biome = self.get_biome(wx, wz);

                for local_y in 0..CHUNK_SIZE {
                    let wy = world_y + local_y as i32;
                    let block = self.get_block_at_with_structures(
                        wx, wy, wz, height, &biome, &structure_blocks
                    );
                    chunk.set_block(local_x, local_y, local_z, block);
                }
            }
        }

        chunk
    }

    fn get_terrain_height(&self, x: i32, z: i32) -> i32 {
        // Use blended parameters for smooth biome transitions
        let params = self.get_blended_terrain_params(x, z);
        self.get_terrain_height_with_params(x, z, &params)
    }

    fn get_terrain_height_with_params(&self, x: i32, z: i32, params: &BlendedTerrainParams) -> i32 {
        let scale = self.config.terrain_scale * params.noise_scale;

        // Get base terrain noise
        let base = self.terrain_noise.get([x as f64 * scale, z as f64 * scale]);

        // Apply flatness - interpolate between noise and 0 (flat)
        let flattened_base = base * (1.0 - params.flatness);

        // Get detail noise with biome-specific strength
        let detail = self.detail_noise.get([x as f64 * scale * 4.0, z as f64 * scale * 4.0])
            * params.detail_strength;

        // Combine and normalize to 0-1 range
        let combined = flattened_base + detail;
        let normalized = (combined + 1.0) / 2.0;

        // Apply blended height scaling and base height offset
        let height_variation = (normalized * self.config.terrain_height * params.height_scale) as i32;

        self.config.sea_level + params.base_height as i32 + height_variation
    }

    #[allow(dead_code)]
    fn get_terrain_height_for_biome(&self, x: i32, z: i32, biome: &CompiledBiome) -> i32 {
        let scale = self.config.terrain_scale * biome.noise_scale;

        // Get base terrain noise
        let base = self.terrain_noise.get([x as f64 * scale, z as f64 * scale]);

        // Apply flatness - interpolate between noise and 0 (flat)
        let flattened_base = base * (1.0 - biome.flatness);

        // Get detail noise with biome-specific strength
        let detail = self.detail_noise.get([x as f64 * scale * 4.0, z as f64 * scale * 4.0])
            * biome.detail_strength;

        // Combine and normalize to 0-1 range
        let combined = flattened_base + detail;
        let normalized = (combined + 1.0) / 2.0;

        // Apply biome-specific height scaling and base height offset
        let height_variation = (normalized * self.config.terrain_height * biome.height_scale) as i32;

        self.config.sea_level + biome.base_height + height_variation
    }

    fn get_biome(&self, x: i32, z: i32) -> &CompiledBiome {
        let scale = self.biomes_config.biome_scale;
        let temperature = self.biome_noise.get([x as f64 * scale, z as f64 * scale]);
        let moisture = self
            .biome_noise
            .get([x as f64 * scale + 1000.0, z as f64 * scale + 1000.0]);

        // Find the first biome that matches the temperature/moisture ranges
        for biome in &self.biomes_config.biomes {
            if temperature >= biome.temperature_min
                && temperature < biome.temperature_max
                && moisture >= biome.moisture_min
                && moisture < biome.moisture_max
            {
                return biome;
            }
        }

        // Fallback to first biome if none match (shouldn't happen with proper config)
        &self.biomes_config.biomes[0]
    }

    /// Calculate blend weights for all biomes based on temperature/moisture distance
    /// Returns a vector of (biome_index, weight) pairs for biomes with non-zero weights
    fn get_biome_weights(&self, x: i32, z: i32) -> Vec<(usize, f64)> {
        let scale = self.biomes_config.biome_scale;
        let temperature = self.biome_noise.get([x as f64 * scale, z as f64 * scale]);
        let moisture = self
            .biome_noise
            .get([x as f64 * scale + 1000.0, z as f64 * scale + 1000.0]);

        // Blend radius in temperature/moisture space (how far to blend at biome edges)
        let blend_radius = self.biomes_config.blend_radius;

        let mut weights: Vec<(usize, f64)> = Vec::new();
        let mut total_weight = 0.0;

        for (i, biome) in self.biomes_config.biomes.iter().enumerate() {
            // Calculate distance to biome's temperature/moisture range
            let temp_dist = distance_to_range(temperature, biome.temperature_min, biome.temperature_max);
            let moist_dist = distance_to_range(moisture, biome.moisture_min, biome.moisture_max);

            // Combined distance (Euclidean in temperature/moisture space)
            let distance = (temp_dist * temp_dist + moist_dist * moist_dist).sqrt();

            // Only consider biomes within blend radius
            if distance < blend_radius {
                // Smooth falloff: 1.0 at distance 0, 0.0 at blend_radius
                // Using smoothstep for nicer interpolation
                let t = distance / blend_radius;
                let weight = 1.0 - (t * t * (3.0 - 2.0 * t)); // smoothstep

                weights.push((i, weight));
                total_weight += weight;
            }
        }

        // Normalize weights so they sum to 1.0
        if total_weight > 0.0 {
            for (_, weight) in &mut weights {
                *weight /= total_weight;
            }
        } else {
            // Fallback: if no biomes in range, use the closest one with weight 1.0
            let mut min_dist = f64::MAX;
            let mut closest_idx = 0;
            for (i, biome) in self.biomes_config.biomes.iter().enumerate() {
                let temp_dist = distance_to_range(temperature, biome.temperature_min, biome.temperature_max);
                let moist_dist = distance_to_range(moisture, biome.moisture_min, biome.moisture_max);
                let distance = (temp_dist * temp_dist + moist_dist * moist_dist).sqrt();
                if distance < min_dist {
                    min_dist = distance;
                    closest_idx = i;
                }
            }
            weights.push((closest_idx, 1.0));
        }

        weights
    }

    /// Get blended terrain parameters by interpolating across nearby biomes
    fn get_blended_terrain_params(&self, x: i32, z: i32) -> BlendedTerrainParams {
        let weights = self.get_biome_weights(x, z);

        let mut height_scale = 0.0;
        let mut base_height = 0.0;
        let mut noise_scale = 0.0;
        let mut detail_strength = 0.0;
        let mut flatness = 0.0;

        for (idx, weight) in weights {
            let biome = &self.biomes_config.biomes[idx];
            height_scale += biome.height_scale * weight;
            base_height += biome.base_height as f64 * weight;
            noise_scale += biome.noise_scale * weight;
            detail_strength += biome.detail_strength * weight;
            flatness += biome.flatness * weight;
        }

        BlendedTerrainParams {
            height_scale,
            base_height,
            noise_scale,
            detail_strength,
            flatness,
        }
    }

    /// Get block at position (legacy method without structure support)
    #[allow(dead_code)]
    fn get_block_at(&self, x: i32, y: i32, z: i32, surface_height: i32, biome: &CompiledBiome) -> BlockType {
        if y > surface_height {
            if y <= self.config.sea_level {
                return BlockType::Water;
            }
            return BlockType::Air;
        }

        if self.is_cave(x, y, z) && y < surface_height - 1 {
            return BlockType::Air;
        }

        // Surface block
        if y == surface_height {
            // Check if we're above the stone altitude threshold
            if let Some(stone_alt) = biome.stone_altitude {
                if y > self.config.sea_level + stone_alt {
                    return BlockType::Stone;
                }
            }
            return biome.surface_block;
        }

        // Subsurface (up to 4 blocks deep)
        if y > surface_height - 4 {
            return biome.subsurface_block;
        }

        if let Some(ore) = self.get_ore(x, y, z) {
            return ore;
        }

        if y < 10 {
            let bedrock_noise =
                self.ore_noise
                    .get([x as f64 * 0.5, y as f64 * 0.5, z as f64 * 0.5]);
            if bedrock_noise > 0.3 + (y as f64 * 0.05) {
                return BlockType::Cobblestone;
            }
        }

        BlockType::Stone
    }

    fn is_cave(&self, x: i32, y: i32, z: i32) -> bool {
        if y < 5 || y > self.config.sea_level + 30 {
            return false;
        }

        let scale = self.config.cave_scale;
        let noise1 = self
            .cave_noise
            .get([x as f64 * scale, y as f64 * scale, z as f64 * scale]);
        let noise2 = self.cave_noise_2.get([
            x as f64 * scale * 2.0,
            y as f64 * scale * 2.0,
            z as f64 * scale * 2.0,
        ]);

        let combined = (noise1.abs() + noise2.abs()) / 2.0;
        combined > self.config.cave_threshold
    }

    fn get_ore(&self, x: i32, y: i32, z: i32) -> Option<BlockType> {
        if y < 80 && y > 5 {
            let coal =
                self.ore_noise
                    .get([x as f64 * 0.15, y as f64 * 0.15, z as f64 * 0.15]);
            if coal > 0.75 {
                return Some(BlockType::Cobblestone);
            }
        }

        if y < 40 && y > 5 {
            let iron = self.ore_noise.get([
                x as f64 * 0.12 + 100.0,
                y as f64 * 0.12,
                z as f64 * 0.12,
            ]);
            if iron > 0.8 {
                return Some(BlockType::Brick);
            }
        }

        None
    }

    /// Check if a structure spawns at the given world position
    /// Returns the structure type if one spawns there
    fn get_structure_at(&self, x: i32, z: i32, biome: &CompiledBiome, surface_height: i32) -> Option<StructureType> {
        // Don't spawn structures underwater
        if surface_height <= self.config.sea_level {
            return None;
        }

        let hash = structure_hash(x, z, self.config.seed);

        // Check biome-specific structures
        match biome.name.as_str() {
            "forest" => {
                // Higher tree density in forests (hash < 8 = ~3% chance)
                if hash < 8 {
                    return Some(StructureType::OakTree);
                }
                // Tall grass in forests
                if hash >= 8 && hash < 25 {
                    return Some(StructureType::TallGrass);
                }
            }
            "plains" => {
                // Lower tree density in plains (hash < 2 = ~0.8% chance)
                if hash < 2 {
                    return Some(StructureType::OakTree);
                }
                // Tall grass scattered in plains
                if hash >= 2 && hash < 15 {
                    return Some(StructureType::TallGrass);
                }
            }
            "desert" => {
                // Cacti in deserts (hash < 4 = ~1.5% chance)
                if hash < 4 {
                    return Some(StructureType::Cactus);
                }
                // Dead bushes in deserts
                if hash >= 4 && hash < 12 {
                    return Some(StructureType::DeadBush);
                }
            }
            "taiga" => {
                // Spruce trees in taiga (hash < 10 = ~4% chance)
                if hash < 10 {
                    return Some(StructureType::SpruceTree);
                }
            }
            "swamp" => {
                // Oak trees in swamp (lower density)
                if hash < 5 {
                    return Some(StructureType::OakTree);
                }
                // Tall grass in swamps
                if hash >= 5 && hash < 20 {
                    return Some(StructureType::TallGrass);
                }
            }
            "savanna" => {
                // Sparse oak trees in savanna
                if hash < 2 {
                    return Some(StructureType::OakTree);
                }
                // Tall grass in savanna
                if hash >= 2 && hash < 18 {
                    return Some(StructureType::TallGrass);
                }
                // Dead bushes scattered
                if hash >= 18 && hash < 22 {
                    return Some(StructureType::DeadBush);
                }
            }
            "tundra" | "ice_plains" => {
                // Very sparse vegetation in tundra
                if hash < 3 {
                    return Some(StructureType::DeadBush);
                }
            }
            "mountains" => {
                // Sparse trees at lower altitudes
                if hash < 3 && surface_height < self.config.sea_level + 25 {
                    return Some(StructureType::OakTree);
                }
            }
            _ => {}
        }

        None
    }

    /// Pre-compute all structure blocks that fall within a chunk region
    /// This is called once per chunk instead of per-block for O(1) lookup
    fn precompute_structure_blocks(
        &self,
        chunk_world_x: i32,
        chunk_world_y: i32,
        chunk_world_z: i32,
        chunk_size: i32,
    ) -> HashMap<(i32, i32, i32), BlockType> {
        let mut structure_blocks = HashMap::new();

        // Maximum structure height (spruce tree is tallest at 8 blocks)
        const MAX_STRUCTURE_HEIGHT: i32 = 8;

        // Check spawn points in a region that could place blocks inside this chunk
        // We need to check spawn points outside the chunk that could have structures
        // reaching into our chunk
        let min_x = chunk_world_x - STRUCTURE_CHECK_RADIUS;
        let max_x = chunk_world_x + chunk_size + STRUCTURE_CHECK_RADIUS;
        let min_z = chunk_world_z - STRUCTURE_CHECK_RADIUS;
        let max_z = chunk_world_z + chunk_size + STRUCTURE_CHECK_RADIUS;

        for sx in min_x..max_x {
            for sz in min_z..max_z {
                // Get biome and height for this potential spawn point
                let biome = self.get_biome(sx, sz);
                let surface_height = self.get_terrain_height(sx, sz);

                // Check if a structure spawns here
                if let Some(structure_type) = self.get_structure_at(sx, sz, biome, surface_height) {
                    let spawn_y = surface_height + 1;

                    // Only process if structure could reach into our chunk's Y range
                    if spawn_y + MAX_STRUCTURE_HEIGHT < chunk_world_y
                        || spawn_y >= chunk_world_y + chunk_size
                    {
                        continue;
                    }

                    // Add all blocks from this structure that fall within our chunk
                    let blocks = get_structure_blocks(structure_type);
                    for block in blocks {
                        let bx = sx + block.dx;
                        let by = spawn_y + block.dy;
                        let bz = sz + block.dz;

                        // Check if this block is within our chunk bounds
                        if bx >= chunk_world_x
                            && bx < chunk_world_x + chunk_size
                            && by >= chunk_world_y
                            && by < chunk_world_y + chunk_size
                            && bz >= chunk_world_z
                            && bz < chunk_world_z + chunk_size
                        {
                            structure_blocks.insert((bx, by, bz), block.block);
                        }
                    }
                }
            }
        }

        structure_blocks
    }

    /// Get block at position using pre-computed structure blocks (O(1) lookup)
    fn get_block_at_with_structures(
        &self,
        x: i32,
        y: i32,
        z: i32,
        surface_height: i32,
        biome: &CompiledBiome,
        structure_blocks: &HashMap<(i32, i32, i32), BlockType>,
    ) -> BlockType {
        // Check for structures first (they can be above ground)
        if y > surface_height {
            // O(1) lookup instead of O(n²) search
            if let Some(&block) = structure_blocks.get(&(x, y, z)) {
                return block;
            }

            if y <= self.config.sea_level {
                return BlockType::Water;
            }
            return BlockType::Air;
        }

        if self.is_cave(x, y, z) && y < surface_height - 1 {
            return BlockType::Air;
        }

        // Surface block
        if y == surface_height {
            // Check if we're above the stone altitude threshold
            if let Some(stone_alt) = biome.stone_altitude {
                if y > self.config.sea_level + stone_alt {
                    return BlockType::Stone;
                }
            }
            return biome.surface_block;
        }

        // Subsurface (up to 4 blocks deep)
        if y > surface_height - 4 {
            return biome.subsurface_block;
        }

        if let Some(ore) = self.get_ore(x, y, z) {
            return ore;
        }

        if y < 10 {
            let bedrock_noise =
                self.ore_noise
                    .get([x as f64 * 0.5, y as f64 * 0.5, z as f64 * 0.5]);
            if bedrock_noise > 0.3 + (y as f64 * 0.05) {
                return BlockType::Cobblestone;
            }
        }

        BlockType::Stone
    }
}


/// Manages multiple chunks with HashMap storage for O(1) neighbor lookup
pub struct World {
    /// Chunks stored by position for fast neighbor lookup
    pub chunks: HashMap<ChunkPos, Chunk>,
    generator: WorldGenerator,
    /// Horizontal render distance in chunks
    pub render_distance: i32,
    /// Vertical render distance (height) in chunks
    pub height_chunks: i32,
    /// Tracks which chunks need remeshing (boundary changed)
    chunks_needing_remesh: Vec<ChunkPos>,
    /// Last player chunk position for detecting movement
    last_player_chunk: Option<(i32, i32)>,
    /// Pre-allocated mesh buffers to avoid allocation during updates
    mesh_vertices: Vec<Vertex>,
    mesh_indices: Vec<u32>,
    /// Whether the combined mesh needs rebuilding
    mesh_dirty: bool,
    /// Pre-allocated set for desired chunk positions (reused each frame)
    desired_chunks: HashSet<ChunkPos>,
    /// Biomes configuration
    biomes_config: CompiledBiomesConfig,
}

impl World {
    pub fn new(seed: u32) -> Self {
        Self::with_render_distance(seed, 4, 4)
    }

    pub fn with_render_distance(seed: u32, render_distance: i32, height_chunks: i32) -> Self {
        let config = WorldGenConfig {
            seed,
            ..Default::default()
        };
        let biomes_config = CompiledBiomesConfig::default();

        Self::with_world_gen_config(config, biomes_config, render_distance, height_chunks)
    }

    /// Create a new world with terrain configuration from config file
    pub fn with_config(
        terrain_config: &TerrainConfig,
        biomes_config: CompiledBiomesConfig,
        render_distance: i32,
        height_chunks: i32,
    ) -> Self {
        let config = WorldGenConfig {
            seed: terrain_config.seed,
            sea_level: terrain_config.sea_level,
            terrain_scale: terrain_config.terrain_scale,
            terrain_height: terrain_config.terrain_height,
            cave_scale: terrain_config.cave_scale,
            cave_threshold: terrain_config.cave_threshold,
        };

        Self::with_world_gen_config(config, biomes_config, render_distance, height_chunks)
    }

    fn with_world_gen_config(
        config: WorldGenConfig,
        biomes_config: CompiledBiomesConfig,
        render_distance: i32,
        height_chunks: i32,
    ) -> Self {
        // Pre-allocate mesh buffers based on expected size
        // Rough estimate: ~500k vertices, ~750k indices for a typical view
        let estimated_vertices = 500_000;
        let estimated_indices = 750_000;

        // Pre-allocate desired_chunks HashSet based on render distance
        // (2*rd+1)^2 * height = max chunks
        let max_chunks = ((2 * render_distance + 1) * (2 * render_distance + 1) * height_chunks) as usize;

        Self {
            chunks: HashMap::with_capacity(max_chunks),
            generator: WorldGenerator::new(config, biomes_config.clone()),
            render_distance,
            height_chunks,
            chunks_needing_remesh: Vec::new(),
            last_player_chunk: None,
            mesh_vertices: Vec::with_capacity(estimated_vertices),
            mesh_indices: Vec::with_capacity(estimated_indices),
            mesh_dirty: true,
            desired_chunks: HashSet::with_capacity(max_chunks),
            biomes_config,
        }
    }

    /// Convert world position to chunk position
    pub fn world_to_chunk(x: f32, z: f32) -> (i32, i32) {
        (
            (x / CHUNK_SIZE as f32).floor() as i32,
            (z / CHUNK_SIZE as f32).floor() as i32,
        )
    }

    /// Update chunks based on player position. Returns true if chunks changed.
    pub fn update_for_position(&mut self, player_x: f32, player_z: f32) -> bool {
        let (player_cx, player_cz) = Self::world_to_chunk(player_x, player_z);

        // Check if player moved to a new chunk
        if self.last_player_chunk == Some((player_cx, player_cz)) {
            return false;
        }

        self.last_player_chunk = Some((player_cx, player_cz));

        let mut changed = false;

        // Determine which chunks should exist (reuse pre-allocated HashSet)
        self.desired_chunks.clear();

        for cx in (player_cx - self.render_distance)..=(player_cx + self.render_distance) {
            for cz in (player_cz - self.render_distance)..=(player_cz + self.render_distance) {
                for cy in 0..self.height_chunks {
                    self.desired_chunks.insert((cx, cy, cz));
                }
            }
        }

        // Unload chunks that are too far
        let chunks_to_remove: Vec<ChunkPos> = self
            .chunks
            .keys()
            .filter(|pos| !self.desired_chunks.contains(pos))
            .copied()
            .collect();

        for pos in chunks_to_remove {
            self.chunks.remove(&pos);
            changed = true;
            // Mark neighbors for remesh since boundary changed
            self.mark_neighbors_for_remesh(pos);
        }

        // Find chunks that need to be loaded and sort by distance to player
        let mut chunks_to_load: Vec<ChunkPos> = self.desired_chunks
            .iter()
            .filter(|pos| !self.chunks.contains_key(pos))
            .copied()
            .collect();

        if !chunks_to_load.is_empty() {
            // Sort by squared distance to player chunk (closest first)
            // We use squared distance to avoid sqrt, and weight Y less since vertical distance matters less
            chunks_to_load.sort_by_key(|&(cx, cy, cz)| {
                let dx = cx - player_cx;
                let dz = cz - player_cz;
                // Horizontal distance matters most, vertical (Y) is weighted less
                // since chunks directly below/above are less important than those ahead
                let dy = cy - (self.height_chunks / 2); // Distance from middle height
                dx * dx + dz * dz + (dy * dy) / 4
            });

            // Limit chunks per frame to spread load (max 8 chunks per update)
            let max_chunks_per_frame = 8;
            let chunks_this_frame: Vec<ChunkPos> = chunks_to_load
                .into_iter()
                .take(max_chunks_per_frame)
                .collect();

            // Generate chunks in parallel
            let new_chunks: Vec<(ChunkPos, Chunk)> = chunks_this_frame
                .par_iter()
                .map(|&pos| {
                    let chunk = self.generator.generate_chunk(pos.0, pos.1, pos.2);
                    (pos, chunk)
                })
                .collect();

            // Insert all new chunks (sequential, but fast)
            for (pos, chunk) in new_chunks {
                self.chunks.insert(pos, chunk);
                self.mark_neighbors_for_remesh(pos);
            }
            changed = true;

            // Force re-check next frame if we didn't load all chunks
            if chunks_this_frame.len() == max_chunks_per_frame {
                self.last_player_chunk = None;
            }
        }

        if changed {
            log::debug!(
                "Chunks updated: {} loaded, render distance {}",
                self.chunks.len(),
                self.render_distance
            );
        }

        changed
    }

    /// Mark all 6 neighbors of a chunk position for remeshing
    fn mark_neighbors_for_remesh(&mut self, pos: ChunkPos) {
        let (cx, cy, cz) = pos;
        let neighbors = [
            (cx - 1, cy, cz),
            (cx + 1, cy, cz),
            (cx, cy - 1, cz),
            (cx, cy + 1, cz),
            (cx, cy, cz - 1),
            (cx, cy, cz + 1),
        ];

        for neighbor_pos in neighbors {
            if let Some(chunk) = self.chunks.get_mut(&neighbor_pos) {
                chunk.mark_dirty();
            }
        }
    }

    /// Check if any chunks need remeshing and clear the list
    pub fn take_chunks_needing_remesh(&mut self) -> Vec<ChunkPos> {
        std::mem::take(&mut self.chunks_needing_remesh)
    }

    /// Check if mesh needs full rebuild
    pub fn needs_mesh_update(&self) -> bool {
        !self.chunks_needing_remesh.is_empty()
    }

    /// Generate initial chunks around a position (uses player position)
    /// Call update_for_position for ongoing updates
    pub fn generate_initial(&mut self, player_x: f32, player_z: f32) {
        self.chunks.clear();
        self.last_player_chunk = None;

        // Use update_for_position to generate initial area
        self.update_for_position(player_x, player_z);

        log::info!(
            "Initial generation: {} chunks, render distance {}",
            self.chunks.len(),
            self.render_distance
        );
    }

    /// Generate chunks in a radius around origin (legacy API)
    /// Two-phase: generate all chunk data first, then mesh with neighbor awareness
    pub fn generate_area(&mut self, center_x: i32, center_z: i32, radius: i32, height_chunks: i32) {
        self.chunks.clear();
        self.render_distance = radius;
        self.height_chunks = height_chunks;

        // Phase 1: Generate all chunk block data (can be parallelized)
        log::info!("Phase 1: Generating chunk data...");
        for cx in (center_x - radius)..=(center_x + radius) {
            for cz in (center_z - radius)..=(center_z + radius) {
                for cy in 0..height_chunks {
                    let chunk = self.generator.generate_chunk(cx, cy, cz);
                    self.chunks.insert((cx, cy, cz), chunk);
                }
            }
        }

        log::info!(
            "Generated {} chunks ({}x{}x{})",
            self.chunks.len(),
            radius * 2 + 1,
            height_chunks,
            radius * 2 + 1
        );
    }

    /// Get a chunk by position
    pub fn get_chunk(&self, pos: ChunkPos) -> Option<&Chunk> {
        self.chunks.get(&pos)
    }

    /// Build neighbors struct for a chunk position
    pub fn get_neighbors(&self, pos: ChunkPos) -> ChunkNeighbors {
        let (cx, cy, cz) = pos;
        ChunkNeighbors {
            neg_x: self.chunks.get(&(cx - 1, cy, cz)),
            pos_x: self.chunks.get(&(cx + 1, cy, cz)),
            neg_y: self.chunks.get(&(cx, cy - 1, cz)),
            pos_y: self.chunks.get(&(cx, cy + 1, cz)),
            neg_z: self.chunks.get(&(cx, cy, cz - 1)),
            pos_z: self.chunks.get(&(cx, cy, cz + 1)),
        }
    }

    /// Rebuild meshes for only dirty chunks using parallel processing
    pub fn update_dirty_meshes(&mut self, texture_indices: &BlockTextureArray) {
        // Collect positions of dirty chunks first (to avoid borrow issues)
        let dirty_positions: Vec<ChunkPos> = self
            .chunks
            .iter()
            .filter(|(_, chunk)| chunk.is_dirty())
            .map(|(&pos, _)| pos)
            .collect();

        if dirty_positions.is_empty() {
            return;
        }

        log::debug!("Rebuilding {} dirty chunk meshes", dirty_positions.len());
        self.mesh_dirty = true;

        // For parallel mesh generation, we need to extract chunks and their neighbor data
        // We can't hold mutable borrows across threads, so we extract what we need first

        // IMPORTANT: Extract ALL boundary data BEFORE removing any chunks from the HashMap
        // Otherwise, if two adjacent chunks are both dirty, removing the first one
        // will cause the second one to not find its neighbor for boundary extraction
        let boundary_data: Vec<_> = dirty_positions
            .iter()
            .map(|&pos| (pos, self.extract_neighbor_boundaries(pos)))
            .collect();

        // Now remove chunks and pair with their pre-extracted boundary data
        let chunks_with_neighbors: Vec<_> = boundary_data
            .into_iter()
            .filter_map(|(pos, neighbor_data)| {
                self.chunks.remove(&pos).map(|chunk| (pos, chunk, neighbor_data))
            })
            .collect();

        // Generate meshes in parallel
        let processed_chunks: Vec<_> = chunks_with_neighbors
            .into_par_iter()
            .map(|(pos, mut chunk, neighbor_data)| {
                // Create a temporary ChunkNeighbors from extracted boundary data
                let neighbors = neighbor_data.to_chunk_neighbors();
                chunk.generate_mesh_with_boundaries(texture_indices, &neighbors);
                (pos, chunk)
            })
            .collect();

        // Re-insert all processed chunks
        for (pos, chunk) in processed_chunks {
            self.chunks.insert(pos, chunk);
        }
    }

    /// Extract boundary block data from neighbors for a chunk position
    /// This allows parallel mesh generation without holding references to neighbors
    fn extract_neighbor_boundaries(&self, pos: ChunkPos) -> NeighborBoundaries {
        let (cx, cy, cz) = pos;

        NeighborBoundaries {
            neg_x: self.chunks.get(&(cx - 1, cy, cz)).map(|c| c.extract_pos_x_boundary()),
            pos_x: self.chunks.get(&(cx + 1, cy, cz)).map(|c| c.extract_neg_x_boundary()),
            neg_y: self.chunks.get(&(cx, cy - 1, cz)).map(|c| c.extract_pos_y_boundary()),
            pos_y: self.chunks.get(&(cx, cy + 1, cz)).map(|c| c.extract_neg_y_boundary()),
            neg_z: self.chunks.get(&(cx, cy, cz - 1)).map(|c| c.extract_pos_z_boundary()),
            pos_z: self.chunks.get(&(cx, cy, cz + 1)).map(|c| c.extract_neg_z_boundary()),
        }
    }

    /// Combine all cached chunk meshes into pre-allocated buffers with frustum culling
    /// Only includes chunks that are visible in the camera frustum
    /// Returns slices to avoid allocation
    pub fn collect_world_mesh(&mut self, frustum: &Frustum) -> (&[Vertex], &[u32]) {
        self.mesh_vertices.clear();
        self.mesh_indices.clear();

        let chunk_size = CHUNK_SIZE as f32;
        let mut visible_chunks = 0;
        let total_chunks = self.chunks.len();

        for (pos, chunk) in self.chunks.iter() {
            // Calculate chunk AABB in world space
            let min = Vec3::new(
                pos.0 as f32 * chunk_size,
                pos.1 as f32 * chunk_size,
                pos.2 as f32 * chunk_size,
            );
            let max = min + Vec3::splat(chunk_size);

            // Frustum cull: skip chunks that are completely outside the view frustum
            if !frustum.intersects_aabb(min, max) {
                continue;
            }

            visible_chunks += 1;
            let base_vertex = self.mesh_vertices.len() as u32;
            self.mesh_vertices.extend_from_slice(&chunk.mesh.vertices);
            self.mesh_indices.extend(chunk.mesh.indices.iter().map(|i| i + base_vertex));
        }

        if visible_chunks < total_chunks {
            log::trace!(
                "Frustum culling: rendering {}/{} chunks",
                visible_chunks,
                total_chunks
            );
        }

        self.mesh_dirty = false;
        (&self.mesh_vertices, &self.mesh_indices)
    }

    /// Generate mesh for entire world with neighbor-aware culling and frustum culling
    /// Returns slices into pre-allocated buffers (no allocation)
    pub fn generate_world_mesh(
        &mut self,
        texture_indices: &BlockTextureArray,
        frustum: &Frustum,
    ) -> (&[Vertex], &[u32]) {
        self.update_dirty_meshes(texture_indices);
        self.collect_world_mesh(frustum)
    }
}
