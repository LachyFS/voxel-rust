use noise::{Fbm, MultiFractal, NoiseFn, Perlin, Simplex};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

use crate::camera::Frustum;
use crate::config::TerrainConfig;
use crate::texture::BlockTextureArray;
use crate::voxel::{BlockType, Chunk, ChunkNeighbors, NeighborBoundaries, Vertex, CHUNK_SIZE};
use glam::Vec3;

/// Chunk coordinate key for HashMap
pub type ChunkPos = (i32, i32, i32);

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
    terrain_noise: Fbm<Perlin>,
    detail_noise: Perlin,
    cave_noise: Simplex,
    cave_noise_2: Simplex,
    ore_noise: Perlin,
    biome_noise: Perlin,
}

impl WorldGenerator {
    pub fn new(config: WorldGenConfig) -> Self {
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
        }
    }

    pub fn generate_chunk(&self, chunk_x: i32, chunk_y: i32, chunk_z: i32) -> Chunk {
        let mut chunk = Chunk::new([chunk_x, chunk_y, chunk_z]);

        let world_x = chunk_x * CHUNK_SIZE as i32;
        let world_y = chunk_y * CHUNK_SIZE as i32;
        let world_z = chunk_z * CHUNK_SIZE as i32;

        for local_x in 0..CHUNK_SIZE {
            for local_z in 0..CHUNK_SIZE {
                let wx = world_x + local_x as i32;
                let wz = world_z + local_z as i32;

                let height = self.get_terrain_height(wx, wz);
                let biome = self.get_biome(wx, wz);

                for local_y in 0..CHUNK_SIZE {
                    let wy = world_y + local_y as i32;
                    let block = self.get_block_at(wx, wy, wz, height, &biome);
                    chunk.set_block(local_x, local_y, local_z, block);
                }
            }
        }

        chunk
    }

    fn get_terrain_height(&self, x: i32, z: i32) -> i32 {
        let scale = self.config.terrain_scale;
        let base = self.terrain_noise.get([x as f64 * scale, z as f64 * scale]);
        let detail = self.detail_noise.get([x as f64 * scale * 4.0, z as f64 * scale * 4.0]) * 0.25;
        let normalized = (base + detail + 1.0) / 2.0;
        self.config.sea_level + (normalized * self.config.terrain_height) as i32
    }

    fn get_biome(&self, x: i32, z: i32) -> Biome {
        let temperature = self.biome_noise.get([x as f64 * 0.005, z as f64 * 0.005]);
        let moisture =
            self.biome_noise
                .get([x as f64 * 0.005 + 1000.0, z as f64 * 0.005 + 1000.0]);

        match (temperature > 0.0, moisture > 0.0) {
            (true, true) => Biome::Forest,
            (true, false) => Biome::Desert,
            (false, true) => Biome::Plains,
            (false, false) => Biome::Mountains,
        }
    }

    fn get_block_at(&self, x: i32, y: i32, z: i32, surface_height: i32, biome: &Biome) -> BlockType {
        if y > surface_height {
            if y <= self.config.sea_level {
                return BlockType::Water;
            }
            return BlockType::Air;
        }

        if self.is_cave(x, y, z) && y < surface_height - 1 {
            return BlockType::Air;
        }

        if y == surface_height {
            return match biome {
                Biome::Desert => BlockType::Sand,
                Biome::Forest | Biome::Plains => BlockType::Grass,
                Biome::Mountains => {
                    if y > self.config.sea_level + 20 {
                        BlockType::Stone
                    } else {
                        BlockType::Grass
                    }
                }
            };
        }

        if y > surface_height - 4 {
            return match biome {
                Biome::Desert => BlockType::Sand,
                _ => BlockType::Dirt,
            };
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
}

#[derive(Clone, Copy, Debug)]
enum Biome {
    Plains,
    Forest,
    Desert,
    Mountains,
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

        Self::with_world_gen_config(config, render_distance, height_chunks)
    }

    /// Create a new world with terrain configuration from config file
    pub fn with_config(terrain_config: &TerrainConfig, render_distance: i32, height_chunks: i32) -> Self {
        let config = WorldGenConfig {
            seed: terrain_config.seed,
            sea_level: terrain_config.sea_level,
            terrain_scale: terrain_config.terrain_scale,
            terrain_height: terrain_config.terrain_height,
            cave_scale: terrain_config.cave_scale,
            cave_threshold: terrain_config.cave_threshold,
        };

        Self::with_world_gen_config(config, render_distance, height_chunks)
    }

    fn with_world_gen_config(config: WorldGenConfig, render_distance: i32, height_chunks: i32) -> Self {
        // Pre-allocate mesh buffers based on expected size
        // Rough estimate: ~500k vertices, ~750k indices for a typical view
        let estimated_vertices = 500_000;
        let estimated_indices = 750_000;

        // Pre-allocate desired_chunks HashSet based on render distance
        // (2*rd+1)^2 * height = max chunks
        let max_chunks = ((2 * render_distance + 1) * (2 * render_distance + 1) * height_chunks) as usize;

        Self {
            chunks: HashMap::with_capacity(max_chunks),
            generator: WorldGenerator::new(config),
            render_distance,
            height_chunks,
            chunks_needing_remesh: Vec::new(),
            last_player_chunk: None,
            mesh_vertices: Vec::with_capacity(estimated_vertices),
            mesh_indices: Vec::with_capacity(estimated_indices),
            mesh_dirty: true,
            desired_chunks: HashSet::with_capacity(max_chunks),
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

        log::debug!("Player moved to chunk ({}, {})", player_cx, player_cz);
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

        // Find chunks that need to be loaded
        let chunks_to_load: Vec<ChunkPos> = self.desired_chunks
            .iter()
            .filter(|pos| !self.chunks.contains_key(pos))
            .copied()
            .collect();

        if !chunks_to_load.is_empty() {
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
