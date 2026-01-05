use noise::{Fbm, MultiFractal, NoiseFn, Perlin, Simplex};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

use crate::camera::Frustum;
use crate::config::{CompiledBiome, CompiledBiomesConfig, TerrainConfig};
use crate::fluid::MAX_WATER_LEVEL;
use crate::texture::BlockTextureArray;
use crate::voxel::{add_smooth_water_top, add_water_face, add_water_side_face, add_waterfall, BlockType, Chunk, ChunkNeighbors, Face, NeighborBoundaries, Vertex, WaterNeighborLevels, CHUNK_SIZE};
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
    // Cave system parameters
    pub cave_worm_scale: f64,       // Scale for worm-like cave tunnels
    pub cave_worm_threshold: f64,   // Threshold for worm caves (lower = more caves)
    pub cave_cheese_scale: f64,     // Scale for cheese-like cave pockets
    pub cave_cheese_threshold: f64, // Threshold for cheese caves
    // Ravine parameters
    pub ravine_scale: f64,          // Scale for ravine noise
    pub ravine_threshold: f64,      // Threshold for ravines (higher = fewer)
    pub ravine_depth_scale: f64,    // How deep ravines cut
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
            // Spaghetti/worm caves - thin winding tunnels
            cave_worm_scale: 0.04,
            cave_worm_threshold: 0.03,
            // Cheese caves - larger pockets
            cave_cheese_scale: 0.06,
            cave_cheese_threshold: 0.7,
            // Ravines - deep vertical cuts
            ravine_scale: 0.015,
            ravine_threshold: 0.85,
            ravine_depth_scale: 40.0,
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
    // Additional noise for improved caves and ravines
    cave_worm_noise_x: Simplex,   // 3D noise for worm cave X displacement
    cave_worm_noise_y: Simplex,   // 3D noise for worm cave Y displacement
    cave_worm_noise_z: Simplex,   // 3D noise for worm cave Z displacement
    cave_cheese_noise: Simplex,   // 3D noise for cheese caves (large pockets)
    ravine_noise: Perlin,         // 2D noise for ravine placement
    ravine_depth_noise: Perlin,   // Noise for ravine depth variation
    // Biome blend noise for irregular transitions
    biome_blend_noise: Simplex,   // Adds irregularity to biome boundaries
    // Pond and lake noise
    pond_noise: Perlin,           // 2D noise for pond/lake placement
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
            // Additional noise for improved caves and ravines
            cave_worm_noise_x: Simplex::new(seed.wrapping_add(10)),
            cave_worm_noise_y: Simplex::new(seed.wrapping_add(11)),
            cave_worm_noise_z: Simplex::new(seed.wrapping_add(12)),
            cave_cheese_noise: Simplex::new(seed.wrapping_add(13)),
            ravine_noise: Perlin::new(seed.wrapping_add(20)),
            ravine_depth_noise: Perlin::new(seed.wrapping_add(21)),
            // Biome blend noise
            biome_blend_noise: Simplex::new(seed.wrapping_add(30)),
            // Pond and lake noise
            pond_noise: Perlin::new(seed.wrapping_add(40)),
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
                // Get blended flatness for smooth biome transitions
                let params = self.get_blended_terrain_params(wx, wz);
                let flatness = params.flatness;

                for local_y in 0..CHUNK_SIZE {
                    let wy = world_y + local_y as i32;
                    let block = self.get_block_at_with_structures(
                        wx, wy, wz, height, flatness, &biome, &structure_blocks
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
        // Also reduce detail noise based on flatness - flat terrain shouldn't have detail bumps
        let detail = self.detail_noise.get([x as f64 * scale * 4.0, z as f64 * scale * 4.0])
            * params.detail_strength
            * (1.0 - params.flatness * 0.8);

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
        // Also reduce detail noise based on flatness - flat terrain shouldn't have detail bumps
        let detail = self.detail_noise.get([x as f64 * scale * 4.0, z as f64 * scale * 4.0])
            * biome.detail_strength
            * (1.0 - biome.flatness * 0.8);

        // Combine and normalize to 0-1 range
        let combined = flattened_base + detail;
        let normalized = (combined + 1.0) / 2.0;

        // Apply biome-specific height scaling and base height offset
        let height_variation = (normalized * self.config.terrain_height * biome.height_scale) as i32;

        self.config.sea_level + biome.base_height + height_variation
    }

    /// Get the biome at a world position
    pub fn get_biome(&self, x: i32, z: i32) -> &CompiledBiome {
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
        let base_temperature = self.biome_noise.get([x as f64 * scale, z as f64 * scale]);
        let base_moisture = self
            .biome_noise
            .get([x as f64 * scale + 1000.0, z as f64 * scale + 1000.0]);

        // Add noise to create irregular biome boundaries
        // Use a higher frequency noise to create jagged edges
        let blend_noise_scale = self.biomes_config.biome_scale * 8.0; // Higher frequency for detail
        let blend_noise_strength = self.biomes_config.blend_noise_strength;

        // Sample noise for temperature and moisture perturbation
        let temp_noise = self.biome_blend_noise.get([
            x as f64 * blend_noise_scale,
            z as f64 * blend_noise_scale,
        ]) * blend_noise_strength;

        let moisture_noise = self.biome_blend_noise.get([
            x as f64 * blend_noise_scale + 500.0,
            z as f64 * blend_noise_scale + 500.0,
        ]) * blend_noise_strength;

        // Apply noise perturbation
        let temperature = base_temperature + temp_noise;
        let moisture = base_moisture + moisture_noise;

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
        // Get blended flatness for smooth biome transitions
        let params = self.get_blended_terrain_params(x, z);
        let flatness = params.flatness;

        // Use 3D noise for terrain density - enables overhangs and cliffs
        let is_solid = self.is_terrain_solid(x, y, z, surface_height, flatness);

        if !is_solid {
            // Carve caves in air space too (for cave ceilings)
            if y <= self.config.sea_level {
                return BlockType::Water;
            }
            return BlockType::Air;
        }

        // Carve caves - can now reach the surface for cave entrances
        if self.is_cave(x, y, z) {
            return BlockType::Air;
        }

        // Determine if this is a surface block by checking if block above is air
        let above_solid = self.is_terrain_solid(x, y + 1, z, surface_height, flatness);
        let is_surface = !above_solid || y == surface_height;

        // Surface block
        if is_surface {
            // Check if we're above the stone altitude threshold
            if let Some(stone_alt) = biome.stone_altitude {
                if y > self.config.sea_level + stone_alt {
                    return BlockType::Stone;
                }
            }
            return biome.surface_block;
        }

        // Subsurface (check a few blocks to see if near surface)
        let depth_to_surface = if y < surface_height {
            // Check upward for air
            let mut depth = 0;
            for dy in 1..=4 {
                if !self.is_terrain_solid(x, y + dy, z, surface_height, flatness) {
                    depth = dy;
                    break;
                }
            }
            depth
        } else {
            0
        };

        if depth_to_surface > 0 && depth_to_surface <= 4 {
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

    /// Determines if a block position should be solid terrain
    /// Starts with height-based terrain, then adds 3D noise for overhangs/cliffs
    fn is_terrain_solid(&self, x: i32, y: i32, z: i32, surface_height: i32, _flatness: f64) -> bool {
        let height_diff = y - surface_height;

        // Well above surface - always air
        if height_diff > 10 {
            return false;
        }
        // Well below surface - always solid
        if height_diff < -20 {
            return true;
        }

        // Add 3D noise for local variation (overhangs, cliffs)
        // This only affects blocks near the surface (within ~10 blocks)
        let density_scale = 0.035;
        let noise = self.cave_cheese_noise.get([
            x as f64 * density_scale,
            y as f64 * density_scale * 0.5,
            z as f64 * density_scale,
        ]);

        // The noise can push the surface up or down by a few blocks
        // noise range is roughly -1 to 1, we map to -3 to +3 blocks offset
        let noise_offset = (noise * 3.0) as i32;

        y <= surface_height + noise_offset
    }

    /// Check if a position is within a pond
    /// Returns Some((water_level, carve_depth)) if in a pond
    fn get_pond_at(&self, x: i32, z: i32, biome: &CompiledBiome) -> Option<(i32, i32)> {
        // No ponds in desert or ice biomes
        match biome.name.as_str() {
            "desert" | "ice_plains" | "ocean" | "beach" | "mountains" => return None,
            _ => {}
        }

        // Grid-based pond placement - check nearby grid cells for pond centers
        let grid_size = 64;
        let cell_x = x.div_euclid(grid_size);
        let cell_z = z.div_euclid(grid_size);

        // Check this cell and neighbors
        for dx in -1..=1 {
            for dz in -1..=1 {
                let cx = cell_x + dx;
                let cz = cell_z + dz;

                // Deterministic check if this cell has a pond
                let cell_noise = self.pond_noise.get([cx as f64 * 0.7, cz as f64 * 0.7]);
                if cell_noise < 0.3 {
                    continue; // No pond in this cell
                }

                // Pond center position within the cell (offset from cell corner)
                let offset_noise_x = self.pond_noise.get([cx as f64 * 1.3 + 100.0, cz as f64 * 1.3]);
                let offset_noise_z = self.pond_noise.get([cx as f64 * 1.3, cz as f64 * 1.3 + 100.0]);
                let pond_x = cx * grid_size + (grid_size / 4) + ((offset_noise_x + 1.0) * (grid_size as f64 / 4.0)) as i32;
                let pond_z = cz * grid_size + (grid_size / 4) + ((offset_noise_z + 1.0) * (grid_size as f64 / 4.0)) as i32;

                // Pond size (radius 4-10 blocks)
                let size_noise = self.pond_noise.get([cx as f64 * 2.1 + 200.0, cz as f64 * 2.1]);
                let radius = 4.0 + (size_noise + 1.0) * 3.0; // 4-10 blocks
                let radius_sq = radius * radius;

                // Check if we're within this pond's radius
                let dist_x = (x - pond_x) as f64;
                let dist_z = (z - pond_z) as f64;
                let dist_sq = dist_x * dist_x + dist_z * dist_z;

                if dist_sq > radius_sq {
                    continue; // Outside this pond
                }

                // Get terrain height at pond center for water level
                let center_height = self.get_terrain_height(pond_x, pond_z);

                // Check biome at pond center
                let center_biome = self.get_biome(pond_x, pond_z);
                match center_biome.name.as_str() {
                    "desert" | "ice_plains" | "ocean" | "beach" | "mountains" => continue,
                    _ => {}
                }

                // Water level is 1 below center terrain height
                let water_level = center_height - 1;

                // Depth based on distance from center (deeper in middle)
                let dist_factor = 1.0 - (dist_sq / radius_sq).sqrt();
                let max_depth = 2 + ((size_noise + 1.0) * 1.5) as i32; // 2-5 blocks deep
                let carve_depth = 1 + (dist_factor * max_depth as f64) as i32;

                return Some((water_level, carve_depth));
            }
        }

        None
    }

    /// Check if a position should be carved out as a cave or ravine
    fn is_cave(&self, x: i32, y: i32, z: i32) -> bool {
        // Don't carve below y=5 (bedrock layer)
        if y < 5 {
            return false;
        }

        // Check for ravines first (they can go higher than caves)
        if y <= self.config.sea_level + 50 && self.is_ravine(x, y, z) {
            return true;
        }

        // Get surface height for this column
        let surface_height = self.get_terrain_height(x, z);

        // Don't generate caves above the surface
        if y > surface_height {
            return false;
        }

        let depth_below_terrain = surface_height - y;

        // === Large Cave Entrances (Minecraft-style) ===
        // These create big openings in hillsides that lead into cave systems
        if self.is_cave_entrance(x, y, z, surface_height) {
            return true;
        }

        // === Surface Protection with Entrance Regions ===
        // Use 2D noise to define regions where caves CAN reach the surface
        let entrance_noise = self.biome_blend_noise.get([
            x as f64 * 0.006,
            z as f64 * 0.006,
        ]);

        // ~40% allows surface entrances, ~30% shallow protection, ~30% deep protection
        let min_depth = if entrance_noise > 0.2 {
            0  // Cave entrance region - caves can reach surface
        } else if entrance_noise > -0.3 {
            2  // Transition - caves stop just below surface
        } else {
            4  // Protected region - caves stay underground
        };

        if depth_below_terrain < min_depth {
            return false;
        }

        // === Spaghetti/Worm Caves ===
        // Creates connected tunnel networks by finding where two 3D noise "surfaces" intersect
        let worm_scale = self.config.cave_worm_scale;

        let worm_a = self.cave_worm_noise_x.get([
            x as f64 * worm_scale,
            y as f64 * worm_scale * 0.5,
            z as f64 * worm_scale,
        ]);
        let worm_b = self.cave_worm_noise_y.get([
            x as f64 * worm_scale + 1000.0,
            y as f64 * worm_scale * 0.5,
            z as f64 * worm_scale + 1000.0,
        ]);

        // Cave exists where both noise values are close to zero
        let worm_dist = (worm_a * worm_a + worm_b * worm_b).sqrt();
        if worm_dist < self.config.cave_worm_threshold {
            return true;
        }

        // === Cheese Caves (Large Caverns) ===
        // These need more depth to avoid huge surface holes
        if self.config.cave_cheese_threshold < 1.0 && depth_below_terrain >= 8 {
            let cheese_scale = self.config.cave_cheese_scale;
            let cheese_noise = self.cave_cheese_noise.get([
                x as f64 * cheese_scale,
                y as f64 * cheese_scale * 0.5,
                z as f64 * cheese_scale,
            ]);

            if cheese_noise > self.config.cave_cheese_threshold {
                return true;
            }
        }

        // === Additional tunnel layer ===
        if self.config.cave_threshold < 1.0 {
            let scale = self.config.cave_scale;
            let noise1 = self.cave_noise.get([
                x as f64 * scale,
                y as f64 * scale * 0.5,
                z as f64 * scale,
            ]);
            let noise2 = self.cave_noise_2.get([
                x as f64 * scale * 1.5 + 500.0,
                y as f64 * scale * 0.5,
                z as f64 * scale * 1.5 + 500.0,
            ]);

            let tunnel_dist = (noise1 * noise1 + noise2 * noise2).sqrt();
            let tunnel_threshold = (1.0 - self.config.cave_threshold) * 0.12;
            if tunnel_dist < tunnel_threshold {
                return true;
            }
        }

        false
    }

    /// Check if a position should be carved as a ravine
    /// Ravines are deep, narrow canyons that cut through the terrain
    fn is_ravine(&self, x: i32, y: i32, z: i32) -> bool {
        let ravine_scale = self.config.ravine_scale;

        // Use 2D noise to determine ravine placement (they go vertically)
        let ravine_value = self.ravine_noise.get([
            x as f64 * ravine_scale,
            z as f64 * ravine_scale,
        ]);

        // Ravines only form where noise is very high
        if ravine_value < self.config.ravine_threshold {
            return false;
        }

        // Calculate ravine depth at this position
        // Ravines are deeper where the noise is higher
        let ravine_strength = (ravine_value - self.config.ravine_threshold) / (1.0 - self.config.ravine_threshold);

        // Add some variation to the ravine depth using another noise
        let depth_variation = self.ravine_depth_noise.get([
            x as f64 * ravine_scale * 3.0,
            z as f64 * ravine_scale * 3.0,
        ]);

        // Calculate the bottom of the ravine at this XZ position
        let base_depth = self.config.ravine_depth_scale * ravine_strength;
        let varied_depth = base_depth * (0.7 + 0.3 * (depth_variation + 1.0) / 2.0);

        // Get surface height to know where ravine starts
        let surface_height = self.get_terrain_height(x, z);
        let ravine_bottom = (surface_height as f64 - varied_depth) as i32;
        let ravine_bottom = ravine_bottom.max(8); // Don't go below y=8

        // Check if we're within the ravine's vertical extent
        if y < ravine_bottom || y > surface_height - 1 {
            return false;
        }

        // Make ravines narrower at the bottom (V-shaped profile)
        let depth_into_ravine = (surface_height - y) as f64;
        let max_depth = (surface_height - ravine_bottom) as f64;
        let depth_ratio = depth_into_ravine / max_depth.max(1.0);

        // Width decreases with depth (V-shape)
        // At surface: full width, at bottom: narrow
        let width_at_depth = 1.0 - depth_ratio * 0.7;

        // Use noise to vary the width along the ravine
        let width_noise = self.ravine_depth_noise.get([
            x as f64 * ravine_scale * 5.0 + 500.0,
            z as f64 * ravine_scale * 5.0 + 500.0,
        ]);
        let width_threshold = self.config.ravine_threshold + (1.0 - self.config.ravine_threshold) * (1.0 - width_at_depth) * (0.8 + 0.2 * width_noise);

        ravine_value > width_threshold
    }

    /// Find the actual surface Y position at a given XZ, accounting for 3D terrain and caves
    /// Returns None if the surface is unsuitable for structure placement (e.g., over a cave)
    fn find_actual_surface_for_structure(&self, x: i32, z: i32) -> Option<i32> {
        let base_height = self.get_terrain_height(x, z);
        let biome = self.get_biome(x, z);

        // Don't place structures in ponds/lakes
        if self.get_pond_at(x, z, biome).is_some() {
            return None;
        }

        // Get blended flatness for smooth biome transitions
        let params = self.get_blended_terrain_params(x, z);
        let flatness = params.flatness;

        // Start scanning from above the expected surface
        let scan_start = base_height + 10;
        let scan_end = base_height - 20;

        // Find the first solid block from above (that's the actual surface)
        for y in (scan_end..=scan_start).rev() {
            if self.is_terrain_solid(x, y, z, base_height, flatness) {
                // Make sure there's air above (not inside an overhang)
                if !self.is_terrain_solid(x, y + 1, z, base_height, flatness) {
                    // Check if this block would be carved by a cave
                    if self.is_cave(x, y, z) {
                        continue; // Keep looking lower
                    }

                    // Check that there's solid ground below (not placing over a cave)
                    // We need at least 2 blocks of solid ground
                    let mut solid_below = true;
                    for dy in 1..=2 {
                        let check_y = y - dy;
                        if self.is_cave(x, check_y, z) || !self.is_terrain_solid(x, check_y, z, base_height, flatness) {
                            solid_below = false;
                            break;
                        }
                    }

                    if solid_below {
                        return Some(y);
                    }
                }
            }
        }

        // No valid surface found - don't place structure here
        None
    }

    /// Check if a position should be carved as a large cave entrance
    /// These are big, dramatic openings in hillsides like Minecraft has
    fn is_cave_entrance(&self, x: i32, y: i32, z: i32, surface_height: i32) -> bool {
        // Cave entrances only form in terrain with some height variation
        // They need to be on a slope/hillside
        let depth_below_surface = surface_height - y;

        // Only carve entrances within 20 blocks of surface
        if depth_below_surface > 20 || depth_below_surface < 0 {
            return false;
        }

        // Use 2D noise to place cave entrance locations (rare, large features)
        let entrance_scale = 0.008;
        let entrance_noise = self.cave_worm_noise_x.get([
            x as f64 * entrance_scale,
            z as f64 * entrance_scale,
        ]);

        // Only ~10% of locations can have entrances
        if entrance_noise < 0.75 {
            return false;
        }

        // Check if we're on a slope by sampling nearby terrain heights
        let height_nx = self.get_terrain_height(x - 4, z);
        let height_px = self.get_terrain_height(x + 4, z);
        let height_nz = self.get_terrain_height(x, z - 4);
        let height_pz = self.get_terrain_height(x, z + 4);

        let slope_x = (height_px - height_nx).abs();
        let slope_z = (height_pz - height_nz).abs();
        let max_slope = slope_x.max(slope_z);

        // Need some slope for a hillside entrance (at least 3 blocks over 8)
        if max_slope < 3 {
            return false;
        }

        // Use 3D noise for the entrance shape - creates arch-like openings
        let shape_scale = 0.06;
        let shape_noise = self.cave_cheese_noise.get([
            x as f64 * shape_scale,
            y as f64 * shape_scale * 0.7,  // Stretch vertically for taller entrances
            z as f64 * shape_scale,
        ]);

        // Entrance carves where noise is high, with vertical falloff
        // More likely to carve near bottom of entrance zone
        let vertical_factor = 1.0 - (depth_below_surface as f64 / 20.0);
        let carve_threshold = 0.3 + vertical_factor * 0.4; // 0.3 at bottom, 0.7 at surface

        shape_noise > carve_threshold
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
                // Get biome and find valid surface (accounting for 3D terrain and caves)
                let biome = self.get_biome(sx, sz);
                let Some(surface_height) = self.find_actual_surface_for_structure(sx, sz) else {
                    continue; // No valid surface - skip this position
                };

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
        flatness: f64,
        biome: &CompiledBiome,
        structure_blocks: &HashMap<(i32, i32, i32), BlockType>,
    ) -> BlockType {
        // Check for ponds/lakes - carve a bowl and fill with water
        if let Some((water_level, carve_depth)) = self.get_pond_at(x, z, biome) {
            let pond_bottom = water_level - carve_depth;

            // Fill with water from bottom to water level
            if y > pond_bottom && y <= water_level {
                return BlockType::Water;
            }
            // Carve out blocks above water level up to terrain surface
            if y > water_level && y <= surface_height {
                return BlockType::Air;
            }
        }

        // Use 3D noise for terrain density - enables overhangs and cliffs
        let is_solid = self.is_terrain_solid(x, y, z, surface_height, flatness);

        // Check for structures first (they can be above ground)
        if !is_solid {
            // O(1) lookup instead of O(n²) search
            if let Some(&block) = structure_blocks.get(&(x, y, z)) {
                return block;
            }

            if y <= self.config.sea_level {
                return BlockType::Water;
            }
            return BlockType::Air;
        }

        // Carve caves - can now reach the surface for cave entrances
        if self.is_cave(x, y, z) {
            return BlockType::Air;
        }

        // Determine if this is a surface block by checking if block above is air
        let above_solid = self.is_terrain_solid(x, y + 1, z, surface_height, flatness);
        let is_surface = !above_solid;

        // Surface block
        if is_surface {
            // Check if we're above the stone altitude threshold
            if let Some(stone_alt) = biome.stone_altitude {
                if y > self.config.sea_level + stone_alt {
                    return BlockType::Stone;
                }
            }
            return biome.surface_block;
        }

        // Subsurface - check if near any air pocket above
        let mut near_surface = false;
        for dy in 1..=4 {
            if !self.is_terrain_solid(x, y + dy, z, surface_height, flatness) {
                near_surface = true;
                break;
            }
        }
        if near_surface {
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
    /// Pre-allocated water mesh buffers (separate for transparent rendering)
    water_vertices: Vec<Vertex>,
    water_indices: Vec<u32>,
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
            cave_worm_scale: terrain_config.cave_worm_scale,
            cave_worm_threshold: terrain_config.cave_worm_threshold,
            cave_cheese_scale: terrain_config.cave_cheese_scale,
            cave_cheese_threshold: terrain_config.cave_cheese_threshold,
            ravine_scale: terrain_config.ravine_scale,
            ravine_threshold: terrain_config.ravine_threshold,
            ravine_depth_scale: terrain_config.ravine_depth_scale,
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
            water_vertices: Vec::with_capacity(estimated_vertices / 8), // Water is less common
            water_indices: Vec::with_capacity(estimated_indices / 8),
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

    /// Generate mesh for world including separate water mesh for transparent rendering
    /// Returns (terrain_vertices, terrain_indices, water_vertices, water_indices)
    pub fn generate_world_mesh_with_water(
        &mut self,
        texture_indices: &BlockTextureArray,
        frustum: &Frustum,
    ) -> (&[Vertex], &[u32], &[Vertex], &[u32]) {
        self.update_dirty_meshes(texture_indices);
        self.collect_world_mesh_with_water(texture_indices, frustum, None)
    }

    /// Generate mesh for world including separate water mesh with fluid level support
    /// Returns (terrain_vertices, terrain_indices, water_vertices, water_indices)
    pub fn generate_world_mesh_with_fluids(
        &mut self,
        texture_indices: &BlockTextureArray,
        frustum: &Frustum,
        fluid_sim: &crate::fluid::FluidSimulator,
    ) -> (&[Vertex], &[u32], &[Vertex], &[u32]) {
        self.update_dirty_meshes(texture_indices);
        self.collect_world_mesh_with_water(texture_indices, frustum, Some(fluid_sim))
    }

    /// Collect meshes from all chunks, separating water from terrain
    /// Water gets special subdivided meshes for smooth wave displacement
    fn collect_world_mesh_with_water(
        &mut self,
        texture_indices: &BlockTextureArray,
        frustum: &Frustum,
        fluid_sim: Option<&crate::fluid::FluidSimulator>,
    ) -> (&[Vertex], &[u32], &[Vertex], &[u32]) {
        // Clear pre-allocated buffers
        self.mesh_vertices.clear();
        self.mesh_indices.clear();
        self.water_vertices.clear();
        self.water_indices.clear();

        let water_tex_layer = texture_indices[BlockType::Water.as_index()].top;

        // Helper to get block at world coordinates (using chunks HashMap directly)
        let get_block_at = |chunks: &HashMap<(i32, i32, i32), Chunk>, wx: i32, wy: i32, wz: i32| -> Option<BlockType> {
            let chunk_size = CHUNK_SIZE as i32;
            let chunk_pos = (
                wx.div_euclid(chunk_size),
                wy.div_euclid(chunk_size),
                wz.div_euclid(chunk_size),
            );
            let chunk = chunks.get(&chunk_pos)?;
            let local_x = wx.rem_euclid(chunk_size) as usize;
            let local_y = wy.rem_euclid(chunk_size) as usize;
            let local_z = wz.rem_euclid(chunk_size) as usize;
            Some(chunk.get_block(local_x, local_y, local_z))
        };

        // Helper to get neighbor water level - returns 0 if neighbor is not water
        // This is used for smooth water interpolation
        let get_neighbor_water_level = |chunks: &HashMap<(i32, i32, i32), Chunk>, fluid_sim: Option<&crate::fluid::FluidSimulator>, wx: i32, wy: i32, wz: i32| -> u8 {
            // First check if neighbor is actually water
            match get_block_at(chunks, wx, wy, wz) {
                Some(BlockType::Water) => {
                    // It's water - get the fluid level
                    match fluid_sim {
                        Some(sim) => {
                            let level = sim.get_water_level(wx, wy, wz);
                            // If level is 0, the water block isn't registered yet - treat as full
                            if level == 0 { MAX_WATER_LEVEL } else { level }
                        }
                        None => MAX_WATER_LEVEL,
                    }
                }
                _ => 0, // Not water - return 0 to exclude from averaging
            }
        };

        // Collect meshes from visible chunks
        for chunk in self.chunks.values() {
            // Frustum culling - skip chunks outside view
            let chunk_min = glam::Vec3::new(
                chunk.position[0] as f32 * CHUNK_SIZE as f32,
                chunk.position[1] as f32 * CHUNK_SIZE as f32,
                chunk.position[2] as f32 * CHUNK_SIZE as f32,
            );
            let chunk_max = chunk_min + glam::Vec3::splat(CHUNK_SIZE as f32);

            if !frustum.intersects_aabb(chunk_min, chunk_max) {
                continue;
            }

            // Separate terrain from water - terrain uses existing mesh
            let mut terrain_vertices = Vec::new();
            let mut terrain_indices = Vec::new();

            let vertices = &chunk.mesh.vertices;
            let indices = &chunk.mesh.indices;

            // Copy non-water triangles to terrain
            let mut i = 0;
            while i < indices.len() {
                if i + 2 < indices.len() {
                    let idx0 = indices[i] as usize;
                    let idx1 = indices[i + 1] as usize;
                    let idx2 = indices[i + 2] as usize;

                    if idx0 < vertices.len() && idx1 < vertices.len() && idx2 < vertices.len() {
                        let is_water = vertices[idx0].tex_layer == water_tex_layer;

                        if !is_water {
                            let base = terrain_vertices.len() as u32;
                            terrain_vertices.push(vertices[idx0]);
                            terrain_vertices.push(vertices[idx1]);
                            terrain_vertices.push(vertices[idx2]);
                            terrain_indices.push(base);
                            terrain_indices.push(base + 1);
                            terrain_indices.push(base + 2);
                        }
                    }
                }
                i += 3;
            }

            // Append terrain mesh
            if !terrain_vertices.is_empty() {
                let base_vertex = self.mesh_vertices.len() as u32;
                self.mesh_vertices.extend_from_slice(&terrain_vertices);
                for idx in terrain_indices {
                    self.mesh_indices.push(base_vertex + idx);
                }
            }

            // Generate water mesh by scanning chunk for water blocks
            let chunk_offset = glam::Vec3::new(
                chunk.position[0] as f32 * CHUNK_SIZE as f32,
                chunk.position[1] as f32 * CHUNK_SIZE as f32,
                chunk.position[2] as f32 * CHUNK_SIZE as f32,
            );

            let chunk_pos = chunk.position;

            for x in 0..CHUNK_SIZE {
                for y in 0..CHUNK_SIZE {
                    for z in 0..CHUNK_SIZE {
                        let block = chunk.get_block(x, y, z);
                        if block != BlockType::Water {
                            continue;
                        }

                        let world_x = chunk_pos[0] * CHUNK_SIZE as i32 + x as i32;
                        let world_y = chunk_pos[1] * CHUNK_SIZE as i32 + y as i32;
                        let world_z = chunk_pos[2] * CHUNK_SIZE as i32 + z as i32;
                        let world_pos = chunk_offset + glam::Vec3::new(x as f32, y as f32, z as f32);

                        // Get water level from fluid simulator (default to MAX_WATER_LEVEL if not tracked)
                        // If fluid_sim returns 0, the water isn't registered yet - treat as full
                        let water_level = match fluid_sim {
                            Some(sim) => {
                                let level = sim.get_water_level(world_x, world_y, world_z);
                                if level == 0 { MAX_WATER_LEVEL } else { level }
                            }
                            None => MAX_WATER_LEVEL,
                        };

                        // Top face (+Y) - render if above is air, OR if water level < full and above is solid
                        // (when water is partial and has a solid block above, there's a gap that needs rendering)
                        let above = if y < CHUNK_SIZE - 1 {
                            chunk.get_block(x, y + 1, z)
                        } else {
                            self.chunks.get(&(chunk_pos[0], chunk_pos[1] + 1, chunk_pos[2]))
                                .map(|c| c.get_block(x, 0, z))
                                .unwrap_or(BlockType::Air)
                        };
                        let should_render_top = above == BlockType::Air
                            || (above != BlockType::Water && water_level < MAX_WATER_LEVEL);
                        if should_render_top {
                            // Gather neighbor water levels for smooth interpolation
                            let neighbors = WaterNeighborLevels {
                                neg_x: get_neighbor_water_level(&self.chunks, fluid_sim, world_x - 1, world_y, world_z),
                                pos_x: get_neighbor_water_level(&self.chunks, fluid_sim, world_x + 1, world_y, world_z),
                                neg_z: get_neighbor_water_level(&self.chunks, fluid_sim, world_x, world_y, world_z - 1),
                                pos_z: get_neighbor_water_level(&self.chunks, fluid_sim, world_x, world_y, world_z + 1),
                                neg_x_neg_z: get_neighbor_water_level(&self.chunks, fluid_sim, world_x - 1, world_y, world_z - 1),
                                pos_x_neg_z: get_neighbor_water_level(&self.chunks, fluid_sim, world_x + 1, world_y, world_z - 1),
                                neg_x_pos_z: get_neighbor_water_level(&self.chunks, fluid_sim, world_x - 1, world_y, world_z + 1),
                                pos_x_pos_z: get_neighbor_water_level(&self.chunks, fluid_sim, world_x + 1, world_y, world_z + 1),
                            };
                            add_smooth_water_top(
                                &mut self.water_vertices,
                                &mut self.water_indices,
                                world_pos,
                                water_tex_layer,
                                water_level,
                                &neighbors,
                            );
                        }

                        // Bottom face (-Y) - render if below is air
                        let below = if y > 0 {
                            chunk.get_block(x, y - 1, z)
                        } else {
                            self.chunks.get(&(chunk_pos[0], chunk_pos[1] - 1, chunk_pos[2]))
                                .map(|c| c.get_block(x, CHUNK_SIZE - 1, z))
                                .unwrap_or(BlockType::Air)
                        };
                        if below == BlockType::Air {
                            add_water_face(
                                &mut self.water_vertices,
                                &mut self.water_indices,
                                world_pos,
                                water_tex_layer,
                                Face::Bottom,
                                water_level,
                            );
                        }

                        // Front face (+Z)
                        let front = if z < CHUNK_SIZE - 1 {
                            chunk.get_block(x, y, z + 1)
                        } else {
                            self.chunks.get(&(chunk_pos[0], chunk_pos[1], chunk_pos[2] + 1))
                                .map(|c| c.get_block(x, y, 0))
                                .unwrap_or(BlockType::Air)
                        };
                        if front == BlockType::Air {
                            add_water_face(
                                &mut self.water_vertices,
                                &mut self.water_indices,
                                world_pos,
                                water_tex_layer,
                                Face::Front,
                                water_level,
                            );
                            // Check for waterfall: is there a drop below the front neighbor?
                            let drop = self.count_air_drop(world_x, world_y - 1, world_z + 1);
                            if drop > 0 {
                                add_waterfall(
                                    &mut self.water_vertices,
                                    &mut self.water_indices,
                                    world_pos,
                                    water_tex_layer,
                                    Face::Front,
                                    water_level,
                                    drop,
                                );
                            }
                        } else if front == BlockType::Water {
                            // Check if neighbor water has lower level - render exposed strip
                            let neighbor_level = get_neighbor_water_level(&self.chunks, fluid_sim, world_x, world_y, world_z + 1);
                            if neighbor_level < water_level {
                                add_water_side_face(
                                    &mut self.water_vertices,
                                    &mut self.water_indices,
                                    world_pos,
                                    water_tex_layer,
                                    Face::Front,
                                    water_level,
                                    neighbor_level,
                                );
                            }
                        }

                        // Back face (-Z)
                        let back = if z > 0 {
                            chunk.get_block(x, y, z - 1)
                        } else {
                            self.chunks.get(&(chunk_pos[0], chunk_pos[1], chunk_pos[2] - 1))
                                .map(|c| c.get_block(x, y, CHUNK_SIZE - 1))
                                .unwrap_or(BlockType::Air)
                        };
                        if back == BlockType::Air {
                            add_water_face(
                                &mut self.water_vertices,
                                &mut self.water_indices,
                                world_pos,
                                water_tex_layer,
                                Face::Back,
                                water_level,
                            );
                            // Check for waterfall
                            let drop = self.count_air_drop(world_x, world_y - 1, world_z - 1);
                            if drop > 0 {
                                add_waterfall(
                                    &mut self.water_vertices,
                                    &mut self.water_indices,
                                    world_pos,
                                    water_tex_layer,
                                    Face::Back,
                                    water_level,
                                    drop,
                                );
                            }
                        } else if back == BlockType::Water {
                            let neighbor_level = get_neighbor_water_level(&self.chunks, fluid_sim, world_x, world_y, world_z - 1);
                            if neighbor_level < water_level {
                                add_water_side_face(
                                    &mut self.water_vertices,
                                    &mut self.water_indices,
                                    world_pos,
                                    water_tex_layer,
                                    Face::Back,
                                    water_level,
                                    neighbor_level,
                                );
                            }
                        }

                        // Right face (+X)
                        let right = if x < CHUNK_SIZE - 1 {
                            chunk.get_block(x + 1, y, z)
                        } else {
                            self.chunks.get(&(chunk_pos[0] + 1, chunk_pos[1], chunk_pos[2]))
                                .map(|c| c.get_block(0, y, z))
                                .unwrap_or(BlockType::Air)
                        };
                        if right == BlockType::Air {
                            add_water_face(
                                &mut self.water_vertices,
                                &mut self.water_indices,
                                world_pos,
                                water_tex_layer,
                                Face::Right,
                                water_level,
                            );
                            // Check for waterfall
                            let drop = self.count_air_drop(world_x + 1, world_y - 1, world_z);
                            if drop > 0 {
                                add_waterfall(
                                    &mut self.water_vertices,
                                    &mut self.water_indices,
                                    world_pos,
                                    water_tex_layer,
                                    Face::Right,
                                    water_level,
                                    drop,
                                );
                            }
                        } else if right == BlockType::Water {
                            let neighbor_level = get_neighbor_water_level(&self.chunks, fluid_sim, world_x + 1, world_y, world_z);
                            if neighbor_level < water_level {
                                add_water_side_face(
                                    &mut self.water_vertices,
                                    &mut self.water_indices,
                                    world_pos,
                                    water_tex_layer,
                                    Face::Right,
                                    water_level,
                                    neighbor_level,
                                );
                            }
                        }

                        // Left face (-X)
                        let left = if x > 0 {
                            chunk.get_block(x - 1, y, z)
                        } else {
                            self.chunks.get(&(chunk_pos[0] - 1, chunk_pos[1], chunk_pos[2]))
                                .map(|c| c.get_block(CHUNK_SIZE - 1, y, z))
                                .unwrap_or(BlockType::Air)
                        };
                        if left == BlockType::Air {
                            add_water_face(
                                &mut self.water_vertices,
                                &mut self.water_indices,
                                world_pos,
                                water_tex_layer,
                                Face::Left,
                                water_level,
                            );
                            // Check for waterfall
                            let drop = self.count_air_drop(world_x - 1, world_y - 1, world_z);
                            if drop > 0 {
                                add_waterfall(
                                    &mut self.water_vertices,
                                    &mut self.water_indices,
                                    world_pos,
                                    water_tex_layer,
                                    Face::Left,
                                    water_level,
                                    drop,
                                );
                            }
                        } else if left == BlockType::Water {
                            let neighbor_level = get_neighbor_water_level(&self.chunks, fluid_sim, world_x - 1, world_y, world_z);
                            if neighbor_level < water_level {
                                add_water_side_face(
                                    &mut self.water_vertices,
                                    &mut self.water_indices,
                                    world_pos,
                                    water_tex_layer,
                                    Face::Left,
                                    water_level,
                                    neighbor_level,
                                );
                            }
                        }
                    }
                }
            }
        }

        self.mesh_dirty = false;
        (
            &self.mesh_vertices,
            &self.mesh_indices,
            &self.water_vertices,
            &self.water_indices,
        )
    }

    /// Set a block at world coordinates and mark affected chunks for remeshing
    /// Returns true if the block was successfully set
    pub fn set_block(&mut self, world_x: i32, world_y: i32, world_z: i32, block: BlockType) -> bool {
        let chunk_size = CHUNK_SIZE as i32;

        // Convert world position to chunk position
        let chunk_pos = (
            world_x.div_euclid(chunk_size),
            world_y.div_euclid(chunk_size),
            world_z.div_euclid(chunk_size),
        );

        // Get the chunk (must exist)
        let Some(chunk) = self.chunks.get_mut(&chunk_pos) else {
            return false;
        };

        // Convert to local coordinates within the chunk
        let local_x = world_x.rem_euclid(chunk_size) as usize;
        let local_y = world_y.rem_euclid(chunk_size) as usize;
        let local_z = world_z.rem_euclid(chunk_size) as usize;

        // Set the block
        chunk.set_block(local_x, local_y, local_z, block);
        chunk.mark_dirty();

        // Mark neighbor chunks for remesh if block is at chunk boundary
        if local_x == 0 {
            if let Some(neighbor) = self.chunks.get_mut(&(chunk_pos.0 - 1, chunk_pos.1, chunk_pos.2)) {
                neighbor.mark_dirty();
            }
        }
        if local_x == CHUNK_SIZE - 1 {
            if let Some(neighbor) = self.chunks.get_mut(&(chunk_pos.0 + 1, chunk_pos.1, chunk_pos.2)) {
                neighbor.mark_dirty();
            }
        }
        if local_y == 0 {
            if let Some(neighbor) = self.chunks.get_mut(&(chunk_pos.0, chunk_pos.1 - 1, chunk_pos.2)) {
                neighbor.mark_dirty();
            }
        }
        if local_y == CHUNK_SIZE - 1 {
            if let Some(neighbor) = self.chunks.get_mut(&(chunk_pos.0, chunk_pos.1 + 1, chunk_pos.2)) {
                neighbor.mark_dirty();
            }
        }
        if local_z == 0 {
            if let Some(neighbor) = self.chunks.get_mut(&(chunk_pos.0, chunk_pos.1, chunk_pos.2 - 1)) {
                neighbor.mark_dirty();
            }
        }
        if local_z == CHUNK_SIZE - 1 {
            if let Some(neighbor) = self.chunks.get_mut(&(chunk_pos.0, chunk_pos.1, chunk_pos.2 + 1)) {
                neighbor.mark_dirty();
            }
        }

        true
    }

    /// Get a block at world coordinates
    /// Returns None if the chunk doesn't exist
    pub fn get_block(&self, world_x: i32, world_y: i32, world_z: i32) -> Option<BlockType> {
        let chunk_size = CHUNK_SIZE as i32;

        // Convert world position to chunk position
        let chunk_pos = (
            world_x.div_euclid(chunk_size),
            world_y.div_euclid(chunk_size),
            world_z.div_euclid(chunk_size),
        );

        // Get the chunk
        let chunk = self.chunks.get(&chunk_pos)?;

        // Convert to local coordinates within the chunk
        let local_x = world_x.rem_euclid(chunk_size) as usize;
        let local_y = world_y.rem_euclid(chunk_size) as usize;
        let local_z = world_z.rem_euclid(chunk_size) as usize;

        Some(chunk.get_block(local_x, local_y, local_z))
    }

    /// Count how many air blocks are below a given position (for waterfall detection)
    /// Returns 0-16, capped for performance
    pub fn count_air_drop(&self, world_x: i32, start_y: i32, world_z: i32) -> u8 {
        let mut drop = 0u8;
        for dy in 0..16i32 {
            let check_y = start_y - dy;
            match self.get_block(world_x, check_y, world_z) {
                Some(BlockType::Air) => drop += 1,
                _ => break,
            }
        }
        drop
    }

    /// Destroy a block at world coordinates (set to Air)
    /// Returns the block type that was destroyed, or None if failed
    pub fn destroy_block(&mut self, world_x: i32, world_y: i32, world_z: i32) -> Option<BlockType> {
        let old_block = self.get_block(world_x, world_y, world_z)?;
        if old_block == BlockType::Air {
            return None;
        }
        if self.set_block(world_x, world_y, world_z, BlockType::Air) {
            Some(old_block)
        } else {
            None
        }
    }

    /// Place a block at world coordinates
    /// Returns true if successfully placed
    pub fn place_block(&mut self, world_x: i32, world_y: i32, world_z: i32, block: BlockType) -> bool {
        // Check if the position is valid (chunk exists and position is air)
        if let Some(existing) = self.get_block(world_x, world_y, world_z) {
            if existing != BlockType::Air {
                return false; // Can't place in non-air block
            }
        } else {
            return false; // Chunk doesn't exist
        }

        self.set_block(world_x, world_y, world_z, block)
    }

    /// Get the biome name at a world position
    pub fn get_biome_name(&self, world_x: f32, world_z: f32) -> &str {
        let biome = self.generator.get_biome(world_x as i32, world_z as i32);
        &biome.name
    }
}
