//! Fluid simulation system for 3D Terraria-style water physics
//!
//! Water exists as discrete levels (0-16) that flow downward with gravity
//! and spread horizontally to equalize pressure. 16 levels provides smooth
//! gradations while keeping integer arithmetic for efficiency.
//!
//! Performance optimizations:
//! - FxHashMap/FxHashSet for faster hashing (non-cryptographic)
//! - Bitwise operations for coordinate conversion
//! - SmallVec for stack-allocated neighbor lists
//! - Reusable buffers to avoid per-tick allocations
//! - Adaptive dense/sparse storage based on water density
//! - Batched activations to reduce push overhead

use std::collections::VecDeque;

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::voxel::BlockType;
use crate::worldgen::{ChunkPos, World};

/// Maximum water level (full block)
pub const MAX_WATER_LEVEL: u8 = 16;

/// Volume threshold above which a water body is treated as infinite (ocean)
/// Water blocks in bodies larger than this won't lose water when spreading
pub const OCEAN_THRESHOLD: usize = 500;

/// Threshold for switching from sparse to dense storage (number of water blocks)
const DENSE_THRESHOLD: usize = 256;

/// Fluid simulation configuration
#[derive(Clone)]
pub struct FluidConfig {
    /// Ticks per second for fluid simulation
    pub tick_rate: f32,
    /// Maximum blocks to process per tick (performance limit)
    pub max_blocks_per_tick: usize,
    /// Whether fluids are enabled
    pub enabled: bool,
}

impl Default for FluidConfig {
    fn default() -> Self {
        Self {
            tick_rate: 4.0, // 4 ticks per second
            max_blocks_per_tick: 10000,
            enabled: true,
        }
    }
}

/// Adaptive water storage - switches between sparse HashMap and dense array
/// based on water block count to optimize for both sparse and dense scenarios
#[derive(Clone)]
pub enum WaterStorage {
    /// Sparse storage for chunks with few water blocks (< DENSE_THRESHOLD)
    Sparse(FxHashMap<u16, u8>),
    /// Dense storage for chunks with many water blocks (>= DENSE_THRESHOLD)
    /// Uses a fixed 4096-byte array (16³) with 0 meaning no water
    Dense(Box<[u8; 4096]>),
}

impl Default for WaterStorage {
    fn default() -> Self {
        WaterStorage::Sparse(FxHashMap::default())
    }
}

impl WaterStorage {
    #[inline]
    pub fn get(&self, packed: u16) -> u8 {
        match self {
            WaterStorage::Sparse(map) => map.get(&packed).copied().unwrap_or(0),
            WaterStorage::Dense(arr) => arr[packed as usize],
        }
    }

    #[inline]
    pub fn set(&mut self, packed: u16, level: u8) {
        match self {
            WaterStorage::Sparse(map) => {
                if level == 0 {
                    map.remove(&packed);
                } else {
                    map.insert(packed, level);
                }
            }
            WaterStorage::Dense(arr) => {
                arr[packed as usize] = level;
            }
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match self {
            WaterStorage::Sparse(map) => map.len(),
            WaterStorage::Dense(arr) => arr.iter().filter(|&&l| l > 0).count(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        match self {
            WaterStorage::Sparse(map) => map.is_empty(),
            WaterStorage::Dense(arr) => !arr.iter().any(|&l| l > 0),
        }
    }

    /// Convert to dense storage if we exceed the threshold
    pub fn maybe_convert_to_dense(&mut self) {
        if let WaterStorage::Sparse(map) = self {
            if map.len() >= DENSE_THRESHOLD {
                let mut arr = Box::new([0u8; 4096]);
                for (&packed, &level) in map.iter() {
                    arr[packed as usize] = level;
                }
                *self = WaterStorage::Dense(arr);
            }
        }
    }

    /// Convert back to sparse if we drop below threshold (with hysteresis)
    pub fn maybe_convert_to_sparse(&mut self) {
        if let WaterStorage::Dense(arr) = self {
            let count = arr.iter().filter(|&&l| l > 0).count();
            // Use hysteresis: only convert back at half the threshold
            if count < DENSE_THRESHOLD / 2 {
                let mut map = FxHashMap::default();
                for (i, &level) in arr.iter().enumerate() {
                    if level > 0 {
                        map.insert(i as u16, level);
                    }
                }
                *self = WaterStorage::Sparse(map);
            }
        }
    }

    /// Iterate over all water blocks
    pub fn iter(&self) -> impl Iterator<Item = (u16, u8)> + '_ {
        let sparse_iter = if let WaterStorage::Sparse(map) = self {
            Some(map.iter().map(|(&k, &v)| (k, v)))
        } else {
            None
        };

        let dense_iter = if let WaterStorage::Dense(arr) = self {
            Some(
                arr.iter()
                    .enumerate()
                    .filter(|(_, &l)| l > 0)
                    .map(|(i, &l)| (i as u16, l)),
            )
        } else {
            None
        };

        sparse_iter
            .into_iter()
            .flatten()
            .chain(dense_iter.into_iter().flatten())
    }
}

/// Fluid state for a single chunk
#[derive(Default, Clone)]
pub struct ChunkFluidState {
    /// Water levels - adaptive sparse/dense storage
    levels: WaterStorage,

    /// Blocks that need processing next tick (packed local coordinates)
    active: FxHashSet<u16>,

    /// Reusable buffer for draining active blocks (avoids allocation each tick)
    active_buffer: Vec<u16>,

    /// Whether this chunk's fluid state changed this tick
    dirty: bool,
}

impl ChunkFluidState {
    /// Pack local chunk coordinates into a single u16
    /// Uses bit packing: x + y*16 + z*256 (12 bits total for 16³ chunk)
    #[inline]
    pub fn pack_coords(x: u8, y: u8, z: u8) -> u16 {
        debug_assert!(x < 16 && y < 16 && z < 16);
        x as u16 | ((y as u16) << 4) | ((z as u16) << 8)
    }

    /// Unpack a u16 back into local chunk coordinates using bitwise ops
    #[inline]
    pub fn unpack_coords(packed: u16) -> (u8, u8, u8) {
        let x = (packed & 0xF) as u8;
        let y = ((packed >> 4) & 0xF) as u8;
        let z = ((packed >> 8) & 0xF) as u8;
        (x, y, z)
    }

    /// Get water level at local coordinates (0 if no water)
    #[inline]
    pub fn get_level(&self, x: u8, y: u8, z: u8) -> u8 {
        self.levels.get(Self::pack_coords(x, y, z))
    }

    /// Set water level at local coordinates
    /// Level 0 removes the water entry
    pub fn set_level(&mut self, x: u8, y: u8, z: u8, level: u8) {
        let key = Self::pack_coords(x, y, z);
        self.levels.set(key, level.min(MAX_WATER_LEVEL));
        self.dirty = true;

        // Check if we need to convert storage type
        if level > 0 {
            self.levels.maybe_convert_to_dense();
        } else {
            self.levels.maybe_convert_to_sparse();
        }
    }

    /// Mark a block as needing simulation next tick
    #[inline]
    pub fn mark_active(&mut self, x: u8, y: u8, z: u8) {
        self.active.insert(Self::pack_coords(x, y, z));
    }

    /// Check if a block is marked active
    #[inline]
    pub fn is_active(&self, x: u8, y: u8, z: u8) -> bool {
        self.active.contains(&Self::pack_coords(x, y, z))
    }

    /// Take all active blocks into reusable buffer (avoids allocation)
    pub fn take_active(&mut self) -> &[u16] {
        self.active_buffer.clear();
        self.active_buffer.extend(self.active.drain());
        &self.active_buffer
    }

    /// Check if this chunk has any water
    #[inline]
    pub fn has_water(&self) -> bool {
        !self.levels.is_empty()
    }

    /// Check if this chunk has any active water to simulate
    #[inline]
    pub fn has_active(&self) -> bool {
        !self.active.is_empty()
    }

    /// Get number of active blocks
    #[inline]
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Check if fluid state changed this tick
    #[inline]
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Clear the dirty flag
    #[inline]
    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }

    /// Get number of water blocks in this chunk
    #[inline]
    pub fn water_count(&self) -> usize {
        self.levels.len()
    }

    /// Iterate over all water blocks in this chunk
    pub fn iter_water(&self) -> impl Iterator<Item = (u8, u8, u8, u8)> + '_ {
        self.levels.iter().map(|(packed, level)| {
            let (x, y, z) = Self::unpack_coords(packed);
            (x, y, z, level)
        })
    }
}

/// Pending water level update (batched for efficiency)
#[derive(Clone, Copy)]
struct FluidUpdate {
    world_x: i32,
    world_y: i32,
    world_z: i32,
    new_level: u8,
}

/// Neighbor info for horizontal flow calculations
/// Using a struct instead of tuple for clarity
struct NeighborInfo {
    pos: (i32, i32, i32),
    level: u8,
    has_hole_below: bool,
    #[allow(dead_code)]
    can_flow: bool,
}

/// Global fluid simulation coordinator
pub struct FluidSimulator {
    /// Per-chunk fluid states
    pub chunk_states: FxHashMap<ChunkPos, ChunkFluidState>,

    /// Configuration
    config: FluidConfig,

    /// Tick accumulator for fixed timestep
    tick_accumulator: f32,

    /// Pending block updates (applied at end of tick)
    pending_updates: Vec<FluidUpdate>,

    /// Chunks that need remeshing after this tick
    dirty_chunks: FxHashSet<ChunkPos>,

    /// Positions to mark active after updates applied
    pending_activations: Vec<(i32, i32, i32)>,

    /// Cache of positions known to be part of large water bodies (oceans)
    /// These blocks act as infinite sources
    ocean_blocks: FxHashSet<(i32, i32, i32)>,

    /// Reusable BFS visited set (avoids allocation each search)
    bfs_visited: FxHashSet<(i32, i32, i32)>,

    /// Reusable BFS queue (avoids allocation each search)
    bfs_queue: VecDeque<(i32, i32, i32)>,
}

impl FluidSimulator {
    pub fn new() -> Self {
        Self::with_config(FluidConfig::default())
    }

    pub fn with_config(config: FluidConfig) -> Self {
        Self {
            chunk_states: FxHashMap::default(),
            config,
            tick_accumulator: 0.0,
            pending_updates: Vec::with_capacity(1000),
            dirty_chunks: FxHashSet::default(),
            pending_activations: Vec::with_capacity(1000),
            ocean_blocks: FxHashSet::default(),
            bfs_visited: FxHashSet::default(),
            bfs_queue: VecDeque::with_capacity(1000),
        }
    }

    /// Initialize fluid state for a chunk that was just generated
    /// Call this when a chunk is created to register existing water blocks
    pub fn init_chunk(
        &mut self,
        chunk_pos: ChunkPos,
        water_blocks: impl Iterator<Item = (u8, u8, u8)>,
    ) {
        let state = self.chunk_states.entry(chunk_pos).or_default();
        for (x, y, z) in water_blocks {
            // Worldgen water starts at full level, NOT active (static until disturbed)
            state.set_level(x, y, z, MAX_WATER_LEVEL);
        }
        state.clear_dirty(); // Don't mark dirty just from initialization
    }

    /// Scan a chunk from the World and register all water blocks
    /// Call this when loading/generating a chunk
    pub fn scan_chunk_for_water(
        &mut self,
        chunk_pos: ChunkPos,
        blocks: &[[[BlockType; 16]; 16]; 16],
    ) {
        let state = self.chunk_states.entry(chunk_pos).or_default();
        for x in 0..16 {
            for y in 0..16 {
                for z in 0..16 {
                    if blocks[x][y][z] == BlockType::Water {
                        state.set_level(x as u8, y as u8, z as u8, MAX_WATER_LEVEL);
                    }
                }
            }
        }
        state.clear_dirty();
    }

    /// Remove fluid state for a chunk that was unloaded
    /// Also invalidates ocean cache entries for this chunk
    pub fn unload_chunk(&mut self, chunk_pos: ChunkPos) {
        self.chunk_states.remove(&chunk_pos);

        // Invalidate ocean cache for blocks in this chunk
        self.ocean_blocks.retain(|&(x, y, z)| {
            let (cp, _) = Self::world_to_chunk_local(x, y, z);
            cp != chunk_pos
        });
    }

    /// Called when a block is destroyed - activates adjacent water for simulation
    /// Also registers water blocks that weren't yet in the fluid system
    /// Propagates activation through the entire connected water body
    pub fn on_block_destroyed_with_world(
        &mut self,
        world_x: i32,
        world_y: i32,
        world_z: i32,
        world: &World,
    ) {
        #[cfg(debug_assertions)]
        log::debug!(
            "on_block_destroyed_with_world called at ({}, {}, {})",
            world_x,
            world_y,
            world_z
        );

        // Also check if we destroyed a water block - remove it from fluid system
        let destroyed_level = self.get_water_level(world_x, world_y, world_z);
        if destroyed_level > 0 {
            #[cfg(debug_assertions)]
            log::debug!(
                "Destroyed water block at ({}, {}, {}) with level {}",
                world_x,
                world_y,
                world_z,
                destroyed_level
            );
            let (chunk_pos, local) = Self::world_to_chunk_local(world_x, world_y, world_z);
            if let Some(state) = self.chunk_states.get_mut(&chunk_pos) {
                state.set_level(local.0, local.1, local.2, 0);
            }
        }

        // Use reusable BFS buffers
        self.bfs_visited.clear();
        self.bfs_queue.clear();

        // Start with immediate neighbors of destroyed block
        const NEIGHBORS: [(i32, i32, i32); 6] = [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ];

        for (dx, dy, dz) in NEIGHBORS {
            let (nx, ny, nz) = (world_x + dx, world_y + dy, world_z + dz);
            if let Some(BlockType::Water) = world.get_block(nx, ny, nz) {
                self.bfs_queue.push_back((nx, ny, nz));
                self.bfs_visited.insert((nx, ny, nz));
            }
        }

        // Limit propagation to prevent massive floods from taking too long
        const MAX_PROPAGATION: usize = 10000;
        let mut water_activated = 0;

        while let Some((x, y, z)) = self.bfs_queue.pop_front() {
            if self.bfs_visited.len() >= MAX_PROPAGATION {
                #[cfg(debug_assertions)]
                log::debug!("Hit propagation limit at {} blocks", MAX_PROPAGATION);
                break;
            }

            // Register and activate this water block
            let current_level = self.get_water_level(x, y, z);
            let (chunk_pos, local) = Self::world_to_chunk_local(x, y, z);
            let state = self.chunk_states.entry(chunk_pos).or_default();

            if current_level == 0 {
                // Worldgen water not yet registered - register as full
                #[cfg(debug_assertions)]
                log::debug!(
                    "Registering worldgen water at ({}, {}, {}) with level {}",
                    x,
                    y,
                    z,
                    MAX_WATER_LEVEL
                );
                state.set_level(local.0, local.1, local.2, MAX_WATER_LEVEL);
            }
            state.mark_active(local.0, local.1, local.2);
            water_activated += 1;

            // Check neighbors and add unvisited water to queue
            for (dx, dy, dz) in NEIGHBORS {
                let (nx, ny, nz) = (x + dx, y + dy, z + dz);
                if self.bfs_visited.contains(&(nx, ny, nz)) {
                    continue;
                }

                if let Some(BlockType::Water) = world.get_block(nx, ny, nz) {
                    self.bfs_visited.insert((nx, ny, nz));
                    self.bfs_queue.push_back((nx, ny, nz));
                }
            }
        }

        if water_activated > 0 {
            #[cfg(debug_assertions)]
            log::debug!(
                "Block destroyed at ({}, {}, {}): activated {} connected water blocks",
                world_x,
                world_y,
                world_z,
                water_activated
            );
        }
    }

    /// Called when a block is destroyed - activates adjacent water for simulation
    /// (Legacy version without world reference - only works for already-registered water)
    pub fn on_block_destroyed(&mut self, world_x: i32, world_y: i32, world_z: i32) {
        // Check all 6 neighbors for water using batched iteration
        const NEIGHBORS: [(i32, i32, i32); 6] = [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ];

        for (dx, dy, dz) in NEIGHBORS {
            let (nx, ny, nz) = (world_x + dx, world_y + dy, world_z + dz);
            if self.get_water_level(nx, ny, nz) > 0 {
                self.mark_active_world(nx, ny, nz);
            }
        }
    }

    /// Get water level at world coordinates
    #[inline]
    pub fn get_water_level(&self, world_x: i32, world_y: i32, world_z: i32) -> u8 {
        let (chunk_pos, local) = Self::world_to_chunk_local(world_x, world_y, world_z);
        self.chunk_states
            .get(&chunk_pos)
            .map(|s| s.get_level(local.0, local.1, local.2))
            .unwrap_or(0)
    }

    /// Set water level at world coordinates (queues update)
    #[inline]
    fn queue_water_update(&mut self, world_x: i32, world_y: i32, world_z: i32, level: u8) {
        self.pending_updates.push(FluidUpdate {
            world_x,
            world_y,
            world_z,
            new_level: level,
        });
    }

    /// Mark a world position as active for simulation
    #[inline]
    fn mark_active_world(&mut self, world_x: i32, world_y: i32, world_z: i32) {
        let (chunk_pos, local) = Self::world_to_chunk_local(world_x, world_y, world_z);
        if let Some(state) = self.chunk_states.get_mut(&chunk_pos) {
            state.mark_active(local.0, local.1, local.2);
        }
    }

    /// Queue a position to be marked active after updates are applied
    #[inline]
    fn queue_activation(&mut self, world_x: i32, world_y: i32, world_z: i32) {
        self.pending_activations.push((world_x, world_y, world_z));
    }

    /// Queue activation of a block AND all its cardinal neighbors
    /// This helps propagate pressure changes further through water bodies
    #[inline]
    fn queue_activation_with_neighbors(&mut self, world_x: i32, world_y: i32, world_z: i32) {
        self.pending_activations.extend([
            (world_x, world_y, world_z),
            (world_x + 1, world_y, world_z),
            (world_x - 1, world_y, world_z),
            (world_x, world_y, world_z + 1),
            (world_x, world_y, world_z - 1),
        ]);
    }

    /// Check if a position is part of a large water body (ocean)
    /// Uses cached results when available, otherwise does a limited flood-fill
    fn is_ocean_block(&mut self, x: i32, y: i32, z: i32, world: &World) -> bool {
        // Check cache first
        if self.ocean_blocks.contains(&(x, y, z)) {
            return true;
        }

        // Do a limited flood-fill to count connected full water blocks
        let count = self.count_connected_water(x, y, z, world, OCEAN_THRESHOLD + 1);

        if count > OCEAN_THRESHOLD {
            // This is an ocean - cache this block
            self.ocean_blocks.insert((x, y, z));
            #[cfg(debug_assertions)]
            log::debug!(
                "Detected ocean at ({},{},{}): {} connected blocks",
                x,
                y,
                z,
                count
            );
            true
        } else {
            false
        }
    }

    /// Count connected full water blocks using BFS, stopping early if we exceed limit
    /// Uses reusable buffers to avoid allocation
    fn count_connected_water(
        &mut self,
        start_x: i32,
        start_y: i32,
        start_z: i32,
        world: &World,
        limit: usize,
    ) -> usize {
        self.bfs_visited.clear();
        self.bfs_queue.clear();

        self.bfs_queue.push_back((start_x, start_y, start_z));
        self.bfs_visited.insert((start_x, start_y, start_z));

        const NEIGHBORS: [(i32, i32, i32); 6] = [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ];

        while let Some((x, y, z)) = self.bfs_queue.pop_front() {
            if self.bfs_visited.len() >= limit {
                return self.bfs_visited.len(); // Early exit - definitely an ocean
            }

            for (dx, dy, dz) in NEIGHBORS {
                let (nx, ny, nz) = (x + dx, y + dy, z + dz);
                if self.bfs_visited.contains(&(nx, ny, nz)) {
                    continue;
                }

                // Only count full water blocks (level == MAX)
                let level = self.get_water_level(nx, ny, nz);
                if level == MAX_WATER_LEVEL {
                    // Also verify it's actually water in the world
                    if let Some(BlockType::Water) = world.get_block(nx, ny, nz) {
                        self.bfs_visited.insert((nx, ny, nz));
                        self.bfs_queue.push_back((nx, ny, nz));
                    }
                }
            }
        }

        self.bfs_visited.len()
    }

    /// Convert world coordinates to chunk position and local coordinates
    /// Uses bitwise operations for speed (assumes CHUNK_SIZE = 16 = 2^4)
    /// Rust's arithmetic right shift (>>) already floors toward negative infinity,
    /// and bitwise AND (&) gives correct modulo for powers of 2.
    #[inline]
    fn world_to_chunk_local(world_x: i32, world_y: i32, world_z: i32) -> (ChunkPos, (u8, u8, u8)) {
        // Chunk position: arithmetic right shift by 4 (equivalent to div_euclid(16))
        let chunk_x = world_x >> 4;
        let chunk_y = world_y >> 4;
        let chunk_z = world_z >> 4;

        // Local coords: bitwise AND with 15 (equivalent to rem_euclid(16))
        let local_x = (world_x & 0xF) as u8;
        let local_y = (world_y & 0xF) as u8;
        let local_z = (world_z & 0xF) as u8;

        ((chunk_x, chunk_y, chunk_z), (local_x, local_y, local_z))
    }

    /// Main update function - call each frame with delta time
    /// Returns set of chunks that need remeshing
    pub fn update<F, G>(&mut self, dt: f32, get_block: F, mut set_block: G) -> &FxHashSet<ChunkPos>
    where
        F: Fn(i32, i32, i32) -> Option<BlockType>,
        G: FnMut(i32, i32, i32, BlockType),
    {
        self.dirty_chunks.clear();

        if !self.config.enabled {
            return &self.dirty_chunks;
        }

        self.tick_accumulator += dt;
        let tick_interval = 1.0 / self.config.tick_rate;

        while self.tick_accumulator >= tick_interval {
            self.tick_accumulator -= tick_interval;
            self.simulate_tick(&get_block, &mut set_block);
        }

        &self.dirty_chunks
    }

    /// Convenience method to update with a World reference directly
    /// This avoids borrow checker issues with closures
    pub fn update_with_world(&mut self, dt: f32, world: &mut World) {
        self.dirty_chunks.clear();

        if !self.config.enabled {
            return;
        }

        self.tick_accumulator += dt;
        let tick_interval = 1.0 / self.config.tick_rate;

        while self.tick_accumulator >= tick_interval {
            self.tick_accumulator -= tick_interval;
            self.simulate_tick_with_world(world);
        }
    }

    /// Run one simulation tick using World directly
    fn simulate_tick_with_world(&mut self, world: &mut World) {
        let mut blocks_processed = 0;

        // Collect all active blocks from all chunks
        let chunk_positions: Vec<ChunkPos> = self
            .chunk_states
            .iter()
            .filter(|(_, s)| s.has_active())
            .map(|(&pos, _)| pos)
            .collect();

        #[cfg(debug_assertions)]
        {
            let total_active: usize = self.chunk_states.values().map(|s| s.active_count()).sum();
            if total_active > 0 {
                log::info!(
                    "simulate_tick_with_world: {} active chunks, {} total active blocks",
                    chunk_positions.len(),
                    total_active
                );
            }
        }

        for chunk_pos in chunk_positions {
            if blocks_processed >= self.config.max_blocks_per_tick {
                break;
            }

            // Take active blocks from this chunk (uses reusable buffer)
            let active_blocks: Vec<u16> =
                if let Some(state) = self.chunk_states.get_mut(&chunk_pos) {
                    state.take_active().to_vec()
                } else {
                    continue;
                };

            for packed in active_blocks {
                if blocks_processed >= self.config.max_blocks_per_tick {
                    // Re-add unprocessed blocks for next tick
                    if let Some(state) = self.chunk_states.get_mut(&chunk_pos) {
                        let (x, y, z) = ChunkFluidState::unpack_coords(packed);
                        state.mark_active(x, y, z);
                    }
                    continue;
                }

                let (lx, ly, lz) = ChunkFluidState::unpack_coords(packed);
                let world_x = chunk_pos.0 * 16 + lx as i32;
                let world_y = chunk_pos.1 * 16 + ly as i32;
                let world_z = chunk_pos.2 * 16 + lz as i32;

                self.simulate_block_with_world(world_x, world_y, world_z, world);
                blocks_processed += 1;
            }
        }

        // Apply all pending updates
        self.apply_pending_updates_with_world(world);
    }

    /// Simulate a single water block using World directly
    /// Uses Terraria-style flow: gravity first, then spread horizontally seeking holes
    fn simulate_block_with_world(&mut self, x: i32, y: i32, z: i32, world: &World) {
        let current_level = self.get_water_level(x, y, z);
        if current_level == 0 {
            return;
        }

        // Check if this block is part of an ocean (large water body)
        // Ocean blocks act as infinite sources - they give water without losing any
        let is_ocean = if current_level == MAX_WATER_LEVEL {
            self.is_ocean_block(x, y, z, world)
        } else {
            false
        };

        // Phase 1: Try to flow DOWN (gravity priority) - all water falls as fast as possible
        let below_block = world.get_block(x, y - 1, z);
        let below_level = self.get_water_level(x, y - 1, z);

        #[cfg(debug_assertions)]
        log::trace!(
            "Simulating ({},{},{}): level={}, below={:?}, below_level={}, ocean={}",
            x,
            y,
            z,
            current_level,
            below_block,
            below_level,
            is_ocean
        );

        let can_flow_down = match below_block {
            Some(BlockType::Air) => true,
            Some(BlockType::Water) => below_level < MAX_WATER_LEVEL,
            Some(block) if block.is_foliage() => true, // Water destroys foliage
            _ => false,
        };

        if can_flow_down {
            let space_below = MAX_WATER_LEVEL - below_level;
            let flow_amount = current_level.min(space_below);

            if flow_amount > 0 {
                #[cfg(debug_assertions)]
                log::debug!(
                    "Water DOWN ({},{},{}): {} -> {} (below_level={}, space={}, ocean={})",
                    x,
                    y,
                    z,
                    current_level,
                    current_level - flow_amount,
                    below_level,
                    space_below,
                    is_ocean
                );

                // Ocean blocks don't lose water - they act as infinite sources
                if !is_ocean {
                    self.queue_water_update(x, y, z, current_level - flow_amount);
                }
                self.queue_water_update(x, y - 1, z, below_level + flow_amount);

                // Activate self, below, and horizontal neighbors in one batch
                self.pending_activations.extend([
                    (x, y, z),
                    (x, y - 1, z),
                    (x + 1, y, z),
                    (x - 1, y, z),
                    (x, y, z + 1),
                    (x, y, z - 1),
                ]);
                return; // Prioritize downward flow
            }
        } else {
            #[cfg(debug_assertions)]
            log::trace!(
                "Cannot flow down from ({},{},{}): below={:?}",
                x,
                y,
                z,
                below_block
            );
        }

        // Phase 2: Horizontal flow - Terraria style
        // Water spreads to neighbors that have lower level OR whose below is open
        // This makes water seek out holes to fall into
        // Include diagonal neighbors for more natural spreading

        const HORIZONTAL_NEIGHBORS: [(i32, i32); 8] = [
            // Cardinal directions (priority)
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            // Diagonal directions
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ];

        // Collect info about each neighbor using SmallVec (stack allocated for <= 8 elements)
        let mut neighbors: SmallVec<[NeighborInfo; 8]> = SmallVec::new();

        for (dx, dz) in HORIZONTAL_NEIGHBORS {
            let (nx, ny, nz) = (x + dx, y, z + dz);
            let block = world.get_block(nx, ny, nz);
            let can_flow = match block {
                Some(BlockType::Air) => true,
                Some(BlockType::Water) => true,
                Some(b) if b.is_foliage() => true, // Water destroys foliage
                _ => false,
            };

            if can_flow {
                let neighbor_level = self.get_water_level(nx, ny, nz);

                // Check if neighbor has a hole below it (air, non-full water, or foliage)
                let below_neighbor = world.get_block(nx, ny - 1, nz);
                let below_neighbor_level = self.get_water_level(nx, ny - 1, nz);
                let has_hole_below = match below_neighbor {
                    Some(BlockType::Air) => true,
                    Some(BlockType::Water) => below_neighbor_level < MAX_WATER_LEVEL,
                    Some(b) if b.is_foliage() => true, // Foliage can be destroyed
                    _ => false,
                };

                neighbors.push(NeighborInfo {
                    pos: (nx, ny, nz),
                    level: neighbor_level,
                    has_hole_below,
                    can_flow: true,
                });
            }
        }

        // Priority 1: Flow toward neighbors with holes below them (seeking lowest point)
        let neighbors_with_holes: SmallVec<[&NeighborInfo; 8]> = neighbors
            .iter()
            .filter(|n| n.has_hole_below && n.level < current_level)
            .collect();

        if !neighbors_with_holes.is_empty() && current_level > 1 {
            // Give water to neighbors with holes - they'll carry it down
            // Ocean blocks give water without losing any
            if is_ocean {
                // Ocean: just fill neighbors without tracking remaining
                for neighbor in &neighbors_with_holes {
                    let new_neighbor_level = (neighbor.level + 1).min(MAX_WATER_LEVEL);
                    if new_neighbor_level != neighbor.level {
                        self.queue_water_update(
                            neighbor.pos.0,
                            neighbor.pos.1,
                            neighbor.pos.2,
                            new_neighbor_level,
                        );
                        self.queue_activation(neighbor.pos.0, neighbor.pos.1, neighbor.pos.2);
                        #[cfg(debug_assertions)]
                        log::debug!(
                            "Water OCEAN SEEK HOLE ({},{},{})->({},{},{}): gave 1 (infinite)",
                            x,
                            y,
                            z,
                            neighbor.pos.0,
                            neighbor.pos.1,
                            neighbor.pos.2
                        );
                    }
                }
                self.queue_activation(x, y, z); // Keep ocean block active
                return;
            }

            let flow_per_neighbor = (current_level - 1) / neighbors_with_holes.len() as u8;
            let flow_per_neighbor = flow_per_neighbor.max(1).min(current_level - 1);

            let mut remaining = current_level;
            for neighbor in &neighbors_with_holes {
                if remaining <= 1 {
                    break;
                }
                let give = flow_per_neighbor.min(remaining - 1);
                let new_neighbor_level = (neighbor.level + give).min(MAX_WATER_LEVEL);
                let actual_give = new_neighbor_level - neighbor.level;

                if actual_give > 0 {
                    self.queue_water_update(
                        neighbor.pos.0,
                        neighbor.pos.1,
                        neighbor.pos.2,
                        new_neighbor_level,
                    );
                    self.queue_activation(neighbor.pos.0, neighbor.pos.1, neighbor.pos.2);
                    remaining -= actual_give;
                    #[cfg(debug_assertions)]
                    log::debug!(
                        "Water SEEK HOLE ({},{},{})->({},{},{}): gave {}",
                        x,
                        y,
                        z,
                        neighbor.pos.0,
                        neighbor.pos.1,
                        neighbor.pos.2,
                        actual_give
                    );
                }
            }

            if remaining != current_level {
                self.queue_water_update(x, y, z, remaining);
                self.queue_activation(x, y, z);
            }
            return;
        }

        // Priority 2: Equalize with ALL cardinal neighbors (true pressure equalization)
        // This creates proper water leveling across entire bodies
        if current_level > 0 {
            // Only use cardinal neighbors (first 4) for equalization
            let cardinal_neighbors: SmallVec<[&NeighborInfo; 4]> =
                neighbors.iter().take(4).filter(|n| n.can_flow).collect();

            if !cardinal_neighbors.is_empty() {
                // Find the minimum level among all connected blocks (including self)
                let min_neighbor = cardinal_neighbors
                    .iter()
                    .map(|n| n.level)
                    .min()
                    .unwrap_or(current_level);
                let min_level = current_level.min(min_neighbor);
                let max_level = current_level.max(
                    cardinal_neighbors
                        .iter()
                        .map(|n| n.level)
                        .max()
                        .unwrap_or(current_level),
                );

                // Ocean blocks don't lose water when spreading horizontally
                if is_ocean {
                    // Just fill neighbors up to our level
                    for neighbor in &cardinal_neighbors {
                        if neighbor.level < current_level {
                            let target = current_level.min(MAX_WATER_LEVEL);
                            if target != neighbor.level {
                                self.queue_water_update(
                                    neighbor.pos.0,
                                    neighbor.pos.1,
                                    neighbor.pos.2,
                                    target,
                                );
                                self.queue_activation_with_neighbors(
                                    neighbor.pos.0,
                                    neighbor.pos.1,
                                    neighbor.pos.2,
                                );
                                #[cfg(debug_assertions)]
                                log::debug!(
                                    "Water OCEAN SPREAD ({},{},{})->({},{},{}): {} -> {} (infinite)",
                                    x,
                                    y,
                                    z,
                                    neighbor.pos.0,
                                    neighbor.pos.1,
                                    neighbor.pos.2,
                                    neighbor.level,
                                    target
                                );
                            }
                        }
                    }
                    self.queue_activation(x, y, z); // Keep ocean active
                    return;
                }

                // If all blocks are at exactly the same level, we're done
                if max_level == min_level {
                    return;
                }

                // Calculate equalized levels
                let total_water: u16 = current_level as u16
                    + cardinal_neighbors
                        .iter()
                        .map(|n| n.level as u16)
                        .sum::<u16>();
                let num_blocks = 1 + cardinal_neighbors.len() as u16;
                let avg_level = (total_water / num_blocks) as u8;
                let remainder = (total_water % num_blocks) as u8;

                #[cfg(debug_assertions)]
                log::debug!(
                    "Water EQUALIZE ({},{},{}): total={}, blocks={}, avg={}, remainder={}, range=[{},{}]",
                    x,
                    y,
                    z,
                    total_water,
                    num_blocks,
                    avg_level,
                    remainder,
                    min_level,
                    max_level
                );

                // Build list of all blocks with their current levels
                let mut all_blocks: SmallVec<[((i32, i32, i32), u8); 5]> =
                    SmallVec::with_capacity(5);
                all_blocks.push(((x, y, z), current_level));
                for n in &cardinal_neighbors {
                    all_blocks.push((n.pos, n.level));
                }
                // Sort by level descending so higher blocks get remainder
                all_blocks.sort_by_key(|(_, level)| std::cmp::Reverse(*level));

                // Distribute water and propagate activation to neighbors
                let mut any_changed = false;
                for (i, (pos, old_level)) in all_blocks.iter().enumerate() {
                    let target_level = if (i as u8) < remainder {
                        avg_level + 1
                    } else {
                        avg_level
                    };
                    if target_level != *old_level {
                        any_changed = true;
                        self.queue_water_update(pos.0, pos.1, pos.2, target_level);
                        // Use activation with neighbors to propagate pressure through the water body
                        self.queue_activation_with_neighbors(pos.0, pos.1, pos.2);
                        #[cfg(debug_assertions)]
                        {
                            if *pos == (x, y, z) {
                                log::debug!(
                                    "Water SELF ({},{},{}): {} -> {}",
                                    x,
                                    y,
                                    z,
                                    old_level,
                                    target_level
                                );
                            } else {
                                log::debug!(
                                    "Water SPREAD ({},{},{})->({},{},{}): {} -> {}",
                                    x,
                                    y,
                                    z,
                                    pos.0,
                                    pos.1,
                                    pos.2,
                                    old_level,
                                    target_level
                                );
                            }
                        }
                    }
                }

                // If levels changed, also activate neighbors of unchanged blocks
                // This helps propagate the pressure wave further
                if any_changed {
                    for n in &cardinal_neighbors {
                        self.queue_activation_with_neighbors(n.pos.0, n.pos.1, n.pos.2);
                    }
                }
            }
        }
    }

    /// Apply all pending water level updates using World directly
    fn apply_pending_updates_with_world(&mut self, world: &mut World) {
        for update in self.pending_updates.drain(..) {
            let (chunk_pos, local) =
                Self::world_to_chunk_local(update.world_x, update.world_y, update.world_z);

            // Get or create chunk fluid state
            let state = self.chunk_states.entry(chunk_pos).or_default();
            let old_level = state.get_level(local.0, local.1, local.2);

            // Only update if level actually changed
            if old_level != update.new_level {
                state.set_level(local.0, local.1, local.2, update.new_level);
                self.dirty_chunks.insert(chunk_pos);

                // Update the block type in the world
                if update.new_level == 0 && old_level > 0 {
                    // Water removed - set to air
                    #[cfg(debug_assertions)]
                    log::debug!(
                        "Removing water block at ({}, {}, {})",
                        update.world_x,
                        update.world_y,
                        update.world_z
                    );
                    world.set_block(update.world_x, update.world_y, update.world_z, BlockType::Air);
                } else if update.new_level > 0 && old_level == 0 {
                    // Water added - set to water block
                    #[cfg(debug_assertions)]
                    log::debug!(
                        "Adding water block at ({}, {}, {}) with level {}",
                        update.world_x,
                        update.world_y,
                        update.world_z,
                        update.new_level
                    );
                    world.set_block(
                        update.world_x,
                        update.world_y,
                        update.world_z,
                        BlockType::Water,
                    );
                }
            }
        }

        // Apply pending activations - collect first to avoid borrow conflict
        let activations: Vec<_> = self.pending_activations.drain(..).collect();
        for (x, y, z) in activations {
            self.mark_active_world(x, y, z);
        }
    }

    /// Run one simulation tick
    fn simulate_tick<F, G>(&mut self, get_block: &F, set_block: &mut G)
    where
        F: Fn(i32, i32, i32) -> Option<BlockType>,
        G: FnMut(i32, i32, i32, BlockType),
    {
        let mut blocks_processed = 0;

        // Collect all active blocks from all chunks
        let chunk_positions: Vec<ChunkPos> = self
            .chunk_states
            .iter()
            .filter(|(_, s)| s.has_active())
            .map(|(&pos, _)| pos)
            .collect();

        for chunk_pos in chunk_positions {
            if blocks_processed >= self.config.max_blocks_per_tick {
                break;
            }

            // Take active blocks from this chunk (uses reusable buffer)
            let active_blocks: Vec<u16> =
                if let Some(state) = self.chunk_states.get_mut(&chunk_pos) {
                    state.take_active().to_vec()
                } else {
                    continue;
                };

            for packed in active_blocks {
                if blocks_processed >= self.config.max_blocks_per_tick {
                    // Re-add unprocessed blocks for next tick
                    if let Some(state) = self.chunk_states.get_mut(&chunk_pos) {
                        let (x, y, z) = ChunkFluidState::unpack_coords(packed);
                        state.mark_active(x, y, z);
                    }
                    continue;
                }

                let (lx, ly, lz) = ChunkFluidState::unpack_coords(packed);
                let world_x = chunk_pos.0 * 16 + lx as i32;
                let world_y = chunk_pos.1 * 16 + ly as i32;
                let world_z = chunk_pos.2 * 16 + lz as i32;

                self.simulate_block(world_x, world_y, world_z, get_block);
                blocks_processed += 1;
            }
        }

        // Apply all pending updates
        self.apply_pending_updates(set_block);
    }

    /// Simulate a single water block
    fn simulate_block<F>(&mut self, x: i32, y: i32, z: i32, get_block: &F)
    where
        F: Fn(i32, i32, i32) -> Option<BlockType>,
    {
        let current_level = self.get_water_level(x, y, z);
        if current_level == 0 {
            return;
        }

        // Phase 1: Try to flow DOWN (gravity priority)
        let below_block = get_block(x, y - 1, z);
        let below_level = self.get_water_level(x, y - 1, z);

        let can_flow_down = match below_block {
            Some(BlockType::Air) => true,
            Some(BlockType::Water) => below_level < MAX_WATER_LEVEL,
            Some(block) if block.is_foliage() => true, // Water destroys foliage
            _ => false,
        };

        if can_flow_down {
            let space_below = MAX_WATER_LEVEL - below_level;
            let flow_amount = current_level.min(space_below);

            if flow_amount > 0 {
                self.queue_water_update(x, y, z, current_level - flow_amount);
                self.queue_water_update(x, y - 1, z, below_level + flow_amount);

                // Batch all activations in one extend
                self.pending_activations.extend([
                    (x, y, z),
                    (x, y - 1, z),
                    (x + 1, y, z),
                    (x - 1, y, z),
                    (x, y, z + 1),
                    (x, y, z - 1),
                ]);
                return; // Prioritize downward flow
            }
        }

        // Phase 2: Horizontal spreading (pressure equalization)
        const HORIZONTAL_NEIGHBORS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

        // Find neighbors we can flow into (lower level or air)
        let mut flowable: SmallVec<[(i32, i32, i32, u8); 4]> = SmallVec::new();

        for (dx, dz) in HORIZONTAL_NEIGHBORS {
            let (nx, ny, nz) = (x + dx, y, z + dz);
            let block = get_block(nx, ny, nz);
            let can_flow = match block {
                Some(BlockType::Air) => true,
                Some(BlockType::Water) => true,
                Some(b) if b.is_foliage() => true, // Water destroys foliage
                _ => false,
            };

            if can_flow {
                let neighbor_level = self.get_water_level(nx, ny, nz);
                // Only flow to neighbors with significantly lower level
                if neighbor_level < current_level.saturating_sub(1) {
                    flowable.push((nx, ny, nz, neighbor_level));
                }
            }
        }

        if !flowable.is_empty() {
            // Calculate total water for equalization
            let total_water: u16 =
                current_level as u16 + flowable.iter().map(|(_, _, _, l)| *l as u16).sum::<u16>();
            let num_blocks = 1 + flowable.len();
            let base_level = (total_water / num_blocks as u16) as u8;
            let remainder = (total_water % num_blocks as u16) as usize;

            // Distribute water - this block gets base + 1 if there's remainder
            let my_new_level = if remainder > 0 {
                base_level + 1
            } else {
                base_level
            };
            self.queue_water_update(x, y, z, my_new_level);
            self.queue_activation(x, y, z);

            // Distribute to neighbors
            for (i, (nx, ny, nz, _)) in flowable.iter().enumerate() {
                let level = if i + 1 < remainder {
                    base_level + 1
                } else {
                    base_level
                };
                self.queue_water_update(*nx, *ny, *nz, level);
                self.queue_activation(*nx, *ny, *nz);
            }
        }
    }

    /// Apply all pending water level updates
    fn apply_pending_updates<G>(&mut self, set_block: &mut G)
    where
        G: FnMut(i32, i32, i32, BlockType),
    {
        for update in self.pending_updates.drain(..) {
            let (chunk_pos, local) =
                Self::world_to_chunk_local(update.world_x, update.world_y, update.world_z);

            // Get or create chunk fluid state
            let state = self.chunk_states.entry(chunk_pos).or_default();
            let old_level = state.get_level(local.0, local.1, local.2);

            // Only update if level actually changed
            if old_level != update.new_level {
                state.set_level(local.0, local.1, local.2, update.new_level);
                self.dirty_chunks.insert(chunk_pos);

                // Update the block type in the world
                if update.new_level == 0 && old_level > 0 {
                    // Water removed - set to air
                    set_block(update.world_x, update.world_y, update.world_z, BlockType::Air);
                } else if update.new_level > 0 && old_level == 0 {
                    // Water added - set to water block
                    set_block(
                        update.world_x,
                        update.world_y,
                        update.world_z,
                        BlockType::Water,
                    );
                }
            }
        }

        // Apply pending activations - collect first to avoid borrow conflict
        let activations: Vec<_> = self.pending_activations.drain(..).collect();
        for (x, y, z) in activations {
            self.mark_active_world(x, y, z);
        }
    }

    /// Add water at a position (for bucket placement, rain, etc.)
    pub fn add_water<F>(&mut self, world_x: i32, world_y: i32, world_z: i32, amount: u8, get_block: F)
    where
        F: Fn(i32, i32, i32) -> Option<BlockType>,
    {
        let block = get_block(world_x, world_y, world_z);
        let can_place = matches!(block, Some(BlockType::Air) | Some(BlockType::Water));

        if !can_place {
            return;
        }

        let current = self.get_water_level(world_x, world_y, world_z);
        let new_level = (current + amount).min(MAX_WATER_LEVEL);

        let (chunk_pos, local) = Self::world_to_chunk_local(world_x, world_y, world_z);
        let state = self.chunk_states.entry(chunk_pos).or_default();
        state.set_level(local.0, local.1, local.2, new_level);
        state.mark_active(local.0, local.1, local.2);
        self.dirty_chunks.insert(chunk_pos);
    }

    /// Remove water at a position (for bucket pickup)
    /// Returns the amount of water removed
    pub fn remove_water(&mut self, world_x: i32, world_y: i32, world_z: i32) -> u8 {
        let level = self.get_water_level(world_x, world_y, world_z);
        if level > 0 {
            let (chunk_pos, local) = Self::world_to_chunk_local(world_x, world_y, world_z);
            if let Some(state) = self.chunk_states.get_mut(&chunk_pos) {
                state.set_level(local.0, local.1, local.2, 0);
                self.dirty_chunks.insert(chunk_pos);

                // Mark neighbors active using const array
                const NEIGHBORS: [(i32, i32, i32); 6] = [
                    (1, 0, 0),
                    (-1, 0, 0),
                    (0, 1, 0),
                    (0, -1, 0),
                    (0, 0, 1),
                    (0, 0, -1),
                ];
                for (dx, dy, dz) in NEIGHBORS {
                    self.mark_active_world(world_x + dx, world_y + dy, world_z + dz);
                }
            }
        }
        level
    }

    /// Check if simulation is enabled
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Enable/disable fluid simulation
    #[inline]
    pub fn set_enabled(&mut self, enabled: bool) {
        self.config.enabled = enabled;
    }
}

impl Default for FluidSimulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_coords() {
        for x in 0..16u8 {
            for y in 0..16u8 {
                for z in 0..16u8 {
                    let packed = ChunkFluidState::pack_coords(x, y, z);
                    let (ux, uy, uz) = ChunkFluidState::unpack_coords(packed);
                    assert_eq!((x, y, z), (ux, uy, uz));
                }
            }
        }
    }

    #[test]
    fn test_chunk_fluid_state_basic() {
        let mut state = ChunkFluidState::default();

        // Initially empty
        assert_eq!(state.get_level(0, 0, 0), 0);
        assert!(!state.has_water());

        // Set water
        state.set_level(5, 10, 3, 8);
        assert_eq!(state.get_level(5, 10, 3), 8);
        assert!(state.has_water());

        // Remove water
        state.set_level(5, 10, 3, 0);
        assert_eq!(state.get_level(5, 10, 3), 0);
        assert!(!state.has_water());
    }

    #[test]
    fn test_active_tracking() {
        let mut state = ChunkFluidState::default();

        state.mark_active(1, 2, 3);
        assert!(state.is_active(1, 2, 3));
        assert!(!state.is_active(0, 0, 0));
        assert!(state.has_active());

        let active = state.take_active();
        assert_eq!(active.len(), 1);
        assert!(!state.has_active());
    }

    #[test]
    fn test_world_to_chunk_local_positive() {
        // Test positive coordinates
        let (chunk, local) = FluidSimulator::world_to_chunk_local(17, 33, 5);
        assert_eq!(chunk, (1, 2, 0));
        assert_eq!(local, (1, 1, 5));
    }

    #[test]
    fn test_world_to_chunk_local_negative() {
        // Test negative coordinates
        // -1 >> 4 = -1, -1 & 15 = 15
        // -17 >> 4 = -2, -17 & 15 = 15
        // -32 >> 4 = -2, -32 & 15 = 0
        let (chunk, local) = FluidSimulator::world_to_chunk_local(-1, -17, -32);
        assert_eq!(chunk, (-1, -2, -2));
        assert_eq!(local, (15, 15, 0));

        // Additional edge case tests
        // -16 >> 4 = -1, -16 & 15 = 0
        let (chunk2, local2) = FluidSimulator::world_to_chunk_local(-16, -16, -16);
        assert_eq!(chunk2, (-1, -1, -1));
        assert_eq!(local2, (0, 0, 0));

        // -17 >> 4 = -2, -17 & 15 = 15
        let (chunk3, local3) = FluidSimulator::world_to_chunk_local(-17, -17, -17);
        assert_eq!(chunk3, (-2, -2, -2));
        assert_eq!(local3, (15, 15, 15));
    }

    #[test]
    fn test_water_storage_sparse_to_dense() {
        let mut storage = WaterStorage::default();

        // Fill with water blocks up to threshold
        for i in 0..DENSE_THRESHOLD {
            storage.set(i as u16, 16);
            // Conversion is triggered by ChunkFluidState::set_level, so call it manually
            storage.maybe_convert_to_dense();
        }

        // Should have converted to dense
        assert!(matches!(storage, WaterStorage::Dense(_)));
        assert_eq!(storage.len(), DENSE_THRESHOLD);

        // Values should still be accessible
        for i in 0..DENSE_THRESHOLD {
            assert_eq!(storage.get(i as u16), 16);
        }
    }

    #[test]
    fn test_water_storage_dense_to_sparse() {
        let mut storage = WaterStorage::Dense(Box::new([0u8; 4096]));

        // Set a few blocks
        storage.set(0, 16);
        storage.set(100, 8);

        // Should convert back to sparse (below hysteresis threshold)
        storage.maybe_convert_to_sparse();
        assert!(matches!(storage, WaterStorage::Sparse(_)));

        // Values should still be accessible
        assert_eq!(storage.get(0), 16);
        assert_eq!(storage.get(100), 8);
        assert_eq!(storage.get(50), 0);
    }
}
