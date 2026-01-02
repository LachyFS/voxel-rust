use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use crate::texture::BlockTextureArray;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
    pub normal: [f32; 3],
    pub tex_layer: u32,
    pub ao: f32, // Ambient occlusion value (0.0 = fully occluded, 1.0 = no occlusion)
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
        0 => Float32x3,  // position
        1 => Float32x2,  // uv
        2 => Float32x3,  // normal
        3 => Uint32,     // tex_layer
        4 => Float32,    // ao
    ];

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

/// Block types with texture references
/// Uses repr(u8) to guarantee minimal memory footprint (1 byte per block)
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlockType {
    Air = 0,
    Grass = 1,
    Dirt = 2,
    Stone = 3,
    Cobblestone = 4,
    Sand = 5,
    Wood = 6,
    Leaves = 7,
    Brick = 8,
    Water = 9,
    Snow = 10,
    Ice = 11,
    Gravel = 12,
    Clay = 13,
    Cactus = 14,
    DeadBush = 15,
    TallGrass = 16,
    Podzol = 17,
    SpruceWood = 18,
    SpruceLeaves = 19,
}

impl BlockType {
    /// Total number of block types (for array indexing)
    pub const COUNT: usize = 20;

    /// Convert to array index for direct texture lookup
    #[inline]
    pub fn as_index(self) -> usize {
        self as usize
    }
}

impl BlockType {
    pub fn is_solid(&self) -> bool {
        !matches!(self, BlockType::Air)
    }

    pub fn is_transparent(&self) -> bool {
        matches!(
            self,
            BlockType::Air
                | BlockType::Water
                | BlockType::Leaves
                | BlockType::SpruceLeaves
                | BlockType::Ice
                | BlockType::DeadBush
                | BlockType::TallGrass
        )
    }

    /// Determine if a face should be rendered between current block and adjacent block
    /// For opaque blocks: render if adjacent is not solid OR adjacent is transparent (visible through it)
    /// For transparent blocks: render if adjacent is different type (prevents self-culling but shows all other faces)
    #[inline]
    pub fn should_render_face_against(&self, adjacent: BlockType) -> bool {
        // Air doesn't render faces
        if *self == BlockType::Air {
            return false;
        }

        // Always render against air
        if adjacent == BlockType::Air {
            return true;
        }

        // For cutout blocks (leaves with holes), always render faces even against same type
        // This ensures you can see through the transparent parts to blocks behind
        if self.is_cutout() {
            return true;
        }

        // For other transparent blocks (water, ice), render unless adjacent is the same type
        // This prevents z-fighting between identical transparent blocks
        if self.is_transparent() {
            return *self != adjacent;
        }

        // For opaque blocks, render if:
        // 1. Adjacent is not solid (air) - already handled above
        // 2. Adjacent is transparent (so we can see this block through the transparent one)
        adjacent.is_transparent()
    }

    /// Returns true for blocks with cutout transparency (holes in texture)
    /// These need all faces rendered even between identical blocks
    pub fn is_cutout(&self) -> bool {
        matches!(
            self,
            BlockType::Leaves | BlockType::SpruceLeaves | BlockType::DeadBush | BlockType::TallGrass
        )
    }

    /// Returns true for foliage blocks that should be rendered as crossed quads
    /// instead of cube faces (e.g., tall grass, dead bush)
    pub fn is_foliage(&self) -> bool {
        matches!(self, BlockType::DeadBush | BlockType::TallGrass)
    }

    pub fn name(&self) -> &'static str {
        match self {
            BlockType::Air => "air",
            BlockType::Grass => "grass",
            BlockType::Dirt => "dirt",
            BlockType::Stone => "stone",
            BlockType::Cobblestone => "cobblestone",
            BlockType::Sand => "sand",
            BlockType::Wood => "wood",
            BlockType::Leaves => "leaves",
            BlockType::Brick => "brick",
            BlockType::Water => "water",
            BlockType::Snow => "snow",
            BlockType::Ice => "ice",
            BlockType::Gravel => "gravel",
            BlockType::Clay => "clay",
            BlockType::Cactus => "cactus",
            BlockType::DeadBush => "dead_bush",
            BlockType::TallGrass => "tall_grass",
            BlockType::Podzol => "podzol",
            BlockType::SpruceWood => "spruce_wood",
            BlockType::SpruceLeaves => "spruce_leaves",
        }
    }
}

impl Default for BlockType {
    fn default() -> Self {
        BlockType::Air
    }
}

pub const CHUNK_SIZE: usize = 16;

/// A 2D slice of block type data for a chunk boundary face
/// Used for parallel mesh generation where we can't hold chunk references
/// Stores actual BlockType to support transparent block rendering decisions
pub type BoundarySlice = [[BlockType; CHUNK_SIZE]; CHUNK_SIZE];

/// Extracted boundary data from neighboring chunks for parallel mesh generation
#[derive(Clone)]
pub struct NeighborBoundaries {
    pub neg_x: Option<BoundarySlice>, // Boundary facing +X (from -X neighbor)
    pub pos_x: Option<BoundarySlice>, // Boundary facing -X (from +X neighbor)
    pub neg_y: Option<BoundarySlice>, // Boundary facing +Y (from -Y neighbor)
    pub pos_y: Option<BoundarySlice>, // Boundary facing -Y (from +Y neighbor)
    pub neg_z: Option<BoundarySlice>, // Boundary facing +Z (from -Z neighbor)
    pub pos_z: Option<BoundarySlice>, // Boundary facing -Z (from +Z neighbor)
}

impl NeighborBoundaries {
    /// Convert to a BoundaryNeighbors struct for mesh generation
    pub fn to_chunk_neighbors(&self) -> BoundaryNeighbors {
        BoundaryNeighbors {
            neg_x: self.neg_x.as_ref(),
            pos_x: self.pos_x.as_ref(),
            neg_y: self.neg_y.as_ref(),
            pos_y: self.pos_y.as_ref(),
            neg_z: self.neg_z.as_ref(),
            pos_z: self.pos_z.as_ref(),
        }
    }
}

/// References to boundary slices for mesh generation (used in parallel processing)
pub struct BoundaryNeighbors<'a> {
    pub neg_x: Option<&'a BoundarySlice>,
    pub pos_x: Option<&'a BoundarySlice>,
    pub neg_y: Option<&'a BoundarySlice>,
    pub pos_y: Option<&'a BoundarySlice>,
    pub neg_z: Option<&'a BoundarySlice>,
    pub pos_z: Option<&'a BoundarySlice>,
}

impl<'a> BoundaryNeighbors<'a> {
    /// Get the block type at neighbor boundary position
    /// Returns None if neighbor chunk doesn't exist (treat as solid for culling)
    fn get_block_at(&self, dir: Face, local_x: usize, local_y: usize, local_z: usize) -> Option<BlockType> {
        match dir {
            Face::Right => self.pos_x.map(|b| b[local_y][local_z]),
            Face::Left => self.neg_x.map(|b| b[local_y][local_z]),
            Face::Top => self.pos_y.map(|b| b[local_x][local_z]),
            Face::Bottom => self.neg_y.map(|b| b[local_x][local_z]),
            Face::Front => self.pos_z.map(|b| b[local_x][local_y]),
            Face::Back => self.neg_z.map(|b| b[local_x][local_y]),
        }
    }

    /// Check if face should render against neighbor boundary
    /// current_block: the block we're rendering
    fn should_render_face(&self, current_block: BlockType, dir: Face, local_x: usize, local_y: usize, local_z: usize) -> bool {
        match self.get_block_at(dir, local_x, local_y, local_z) {
            Some(adjacent) => current_block.should_render_face_against(adjacent),
            // No neighbor chunk loaded - don't render to avoid seams
            None => false,
        }
    }
}

/// References to neighboring chunks for boundary-aware meshing
#[derive(Default)]
pub struct ChunkNeighbors<'a> {
    pub neg_x: Option<&'a Chunk>, // -X neighbor
    pub pos_x: Option<&'a Chunk>, // +X neighbor
    pub neg_y: Option<&'a Chunk>, // -Y neighbor (below)
    pub pos_y: Option<&'a Chunk>, // +Y neighbor (above)
    pub neg_z: Option<&'a Chunk>, // -Z neighbor
    pub pos_z: Option<&'a Chunk>, // +Z neighbor
}

impl<'a> ChunkNeighbors<'a> {
    /// Get block type at neighbor position
    /// Returns None if neighbor doesn't exist
    fn get_block_at(&self, dir: Face, local_x: usize, local_y: usize, local_z: usize) -> Option<BlockType> {
        match dir {
            Face::Right => {
                // +X: check neg_x face of pos_x neighbor
                self.pos_x.map(|c| c.blocks[0][local_y][local_z])
            }
            Face::Left => {
                // -X: check pos_x face of neg_x neighbor
                self.neg_x.map(|c| c.blocks[CHUNK_SIZE - 1][local_y][local_z])
            }
            Face::Top => {
                // +Y: check neg_y face of pos_y neighbor
                self.pos_y.map(|c| c.blocks[local_x][0][local_z])
            }
            Face::Bottom => {
                // -Y: check pos_y face of neg_y neighbor
                self.neg_y.map(|c| c.blocks[local_x][CHUNK_SIZE - 1][local_z])
            }
            Face::Front => {
                // +Z: check neg_z face of pos_z neighbor
                self.pos_z.map(|c| c.blocks[local_x][local_y][0])
            }
            Face::Back => {
                // -Z: check pos_z face of neg_z neighbor
                self.neg_z.map(|c| c.blocks[local_x][local_y][CHUNK_SIZE - 1])
            }
        }
    }

    /// Check if face should render against neighbor chunk
    fn should_render_face(&self, current_block: BlockType, dir: Face, local_x: usize, local_y: usize, local_z: usize) -> bool {
        match self.get_block_at(dir, local_x, local_y, local_z) {
            Some(adjacent) => current_block.should_render_face_against(adjacent),
            // No neighbor chunk loaded - don't render to avoid seams
            None => false,
        }
    }
}

/// Cached mesh data for a chunk
#[derive(Clone)]
pub struct ChunkMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub dirty: bool,
}

impl Default for ChunkMesh {
    fn default() -> Self {
        // Pre-allocate based on typical chunk density
        // A fully exposed chunk surface has ~1536 faces (16*16*6), each with 4 vertices and 6 indices
        // Most chunks are partially solid, so we estimate ~1000 vertices and ~1500 indices
        Self {
            vertices: Vec::with_capacity(1024),
            indices: Vec::with_capacity(1536),
            dirty: true,
        }
    }
}

pub struct Chunk {
    pub blocks: [[[BlockType; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE],
    pub position: [i32; 3],
    pub mesh: ChunkMesh,
}

impl Chunk {
    pub fn new(position: [i32; 3]) -> Self {
        Self {
            blocks: [[[BlockType::Air; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE],
            position,
            mesh: ChunkMesh::default(), // Pre-allocated mesh buffers
        }
    }

    /// Mark this chunk as needing remeshing
    pub fn mark_dirty(&mut self) {
        self.mesh.dirty = true;
    }

    /// Check if chunk needs remeshing
    pub fn is_dirty(&self) -> bool {
        self.mesh.dirty
    }

    pub fn set_block(&mut self, x: usize, y: usize, z: usize, block: BlockType) {
        if x < CHUNK_SIZE && y < CHUNK_SIZE && z < CHUNK_SIZE {
            self.blocks[x][y][z] = block;
        }
    }

    pub fn get_block(&self, x: usize, y: usize, z: usize) -> BlockType {
        if x < CHUNK_SIZE && y < CHUNK_SIZE && z < CHUNK_SIZE {
            self.blocks[x][y][z]
        } else {
            BlockType::Air
        }
    }

    /// Extract the +X boundary (x = CHUNK_SIZE-1) as a 2D slice of block types
    /// Used by -X neighbor to check if faces should render
    pub fn extract_pos_x_boundary(&self) -> BoundarySlice {
        let mut slice = [[BlockType::Air; CHUNK_SIZE]; CHUNK_SIZE];
        for y in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                slice[y][z] = self.blocks[CHUNK_SIZE - 1][y][z];
            }
        }
        slice
    }

    /// Extract the -X boundary (x = 0) as a 2D slice of block types
    /// Used by +X neighbor to check if faces should render
    pub fn extract_neg_x_boundary(&self) -> BoundarySlice {
        let mut slice = [[BlockType::Air; CHUNK_SIZE]; CHUNK_SIZE];
        for y in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                slice[y][z] = self.blocks[0][y][z];
            }
        }
        slice
    }

    /// Extract the +Y boundary (y = CHUNK_SIZE-1) as a 2D slice of block types
    pub fn extract_pos_y_boundary(&self) -> BoundarySlice {
        let mut slice = [[BlockType::Air; CHUNK_SIZE]; CHUNK_SIZE];
        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                slice[x][z] = self.blocks[x][CHUNK_SIZE - 1][z];
            }
        }
        slice
    }

    /// Extract the -Y boundary (y = 0) as a 2D slice of block types
    pub fn extract_neg_y_boundary(&self) -> BoundarySlice {
        let mut slice = [[BlockType::Air; CHUNK_SIZE]; CHUNK_SIZE];
        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                slice[x][z] = self.blocks[x][0][z];
            }
        }
        slice
    }

    /// Extract the +Z boundary (z = CHUNK_SIZE-1) as a 2D slice of block types
    pub fn extract_pos_z_boundary(&self) -> BoundarySlice {
        let mut slice = [[BlockType::Air; CHUNK_SIZE]; CHUNK_SIZE];
        for x in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                slice[x][y] = self.blocks[x][y][CHUNK_SIZE - 1];
            }
        }
        slice
    }

    /// Extract the -Z boundary (z = 0) as a 2D slice of block types
    pub fn extract_neg_z_boundary(&self) -> BoundarySlice {
        let mut slice = [[BlockType::Air; CHUNK_SIZE]; CHUNK_SIZE];
        for x in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                slice[x][y] = self.blocks[x][y][0];
            }
        }
        slice
    }

    /// Generate mesh using greedy meshing algorithm with extracted boundary data
    /// Merges adjacent faces with the same texture into larger quads for efficiency
    pub fn generate_mesh_with_boundaries(
        &mut self,
        texture_indices: &BlockTextureArray,
        neighbors: &BoundaryNeighbors,
    ) {
        // Skip if not dirty
        if !self.mesh.dirty {
            return;
        }

        self.mesh.vertices.clear();
        self.mesh.indices.clear();

        let chunk_offset = Vec3::new(
            self.position[0] as f32 * CHUNK_SIZE as f32,
            self.position[1] as f32 * CHUNK_SIZE as f32,
            self.position[2] as f32 * CHUNK_SIZE as f32,
        );

        // Process each axis direction with greedy meshing
        // For each slice perpendicular to the axis, we build a 2D mask and merge faces

        // +X faces (Right) - iterate over x slices, each slice is y-z plane
        self.greedy_mesh_axis_with_boundaries(
            texture_indices,
            neighbors,
            chunk_offset,
            Face::Right,
            |x, y, z| (x, y, z),  // coord mapper: axis, row, col -> x, y, z
            |blocks, x, y, z| {
                if x == CHUNK_SIZE - 1 {
                    None // boundary case handled separately
                } else {
                    Some(blocks[x + 1][y][z])
                }
            },
            |tex| tex.sides,
        );

        // -X faces (Left)
        self.greedy_mesh_axis_with_boundaries(
            texture_indices,
            neighbors,
            chunk_offset,
            Face::Left,
            |x, y, z| (x, y, z),
            |blocks, x, y, z| {
                if x == 0 {
                    None
                } else {
                    Some(blocks[x - 1][y][z])
                }
            },
            |tex| tex.sides,
        );

        // +Y faces (Top)
        self.greedy_mesh_axis_with_boundaries(
            texture_indices,
            neighbors,
            chunk_offset,
            Face::Top,
            |y: usize, x: usize, z: usize| (x, y, z),  // axis=y, row=x, col=z
            |blocks: &[[[BlockType; 16]; 16]; 16], x, y, z| {
                if y == CHUNK_SIZE - 1 {
                    None
                } else {
                    Some(blocks[x][y + 1][z])
                }
            },
            |tex| tex.top,
        );

        // -Y faces (Bottom)
        self.greedy_mesh_axis_with_boundaries(
            texture_indices,
            neighbors,
            chunk_offset,
            Face::Bottom,
            |y, x, z| (x, y, z),
            |blocks, x, y, z| {
                if y == 0 {
                    None
                } else {
                    Some(blocks[x][y - 1][z])
                }
            },
            |tex| tex.bottom,
        );

        // +Z faces (Front)
        self.greedy_mesh_axis_with_boundaries(
            texture_indices,
            neighbors,
            chunk_offset,
            Face::Front,
            |z, x, y| (x, y, z),  // axis=z, row=x, col=y
            |blocks, x, y, z| {
                if z == CHUNK_SIZE - 1 {
                    None
                } else {
                    Some(blocks[x][y][z + 1])
                }
            },
            |tex| tex.sides,
        );

        // -Z faces (Back)
        self.greedy_mesh_axis_with_boundaries(
            texture_indices,
            neighbors,
            chunk_offset,
            Face::Back,
            |z, x, y| (x, y, z),
            |blocks, x, y, z| {
                if z == 0 {
                    None
                } else {
                    Some(blocks[x][y][z - 1])
                }
            },
            |tex| tex.sides,
        );

        // Generate crossed quads for foliage blocks (tall grass, dead bush, etc.)
        self.generate_foliage_mesh(texture_indices, chunk_offset);

        self.mesh.dirty = false;
    }

    /// Generate crossed quad meshes for foliage blocks
    fn generate_foliage_mesh(&mut self, texture_indices: &BlockTextureArray, chunk_offset: Vec3) {
        for x in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for z in 0..CHUNK_SIZE {
                    let block = self.blocks[x][y][z];
                    if block.is_foliage() {
                        let pos = Vec3::new(x as f32, y as f32, z as f32) + chunk_offset;
                        let tex = &texture_indices[block.as_index()];
                        // Use the sides texture for foliage (which is the "all" texture)
                        add_crossed_quads(&mut self.mesh.vertices, &mut self.mesh.indices, pos, tex.sides);
                    }
                }
            }
        }
    }

    /// Greedy mesh one face direction across all slices
    /// Now includes AO-aware merging: only merges faces with identical AO patterns
    fn greedy_mesh_axis_with_boundaries<F, G, T>(
        &mut self,
        texture_indices: &BlockTextureArray,
        neighbors: &BoundaryNeighbors,
        chunk_offset: Vec3,
        face: Face,
        coord_mapper: F,
        get_adjacent_internal: G,
        get_texture: T,
    ) where
        F: Fn(usize, usize, usize) -> (usize, usize, usize),
        // Returns None for boundary case, Some(adjacent_block) for internal
        G: Fn(&[[[BlockType; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE], usize, usize, usize) -> Option<BlockType>,
        T: Fn(&crate::texture::BlockTextureIndices) -> u32,
    {
        // Create AO sampler for this chunk
        let ao_sampler = AoSampler::new(&self.blocks, neighbors);

        // Mask stores (texture layer, ao_key) for each cell
        // ao_key is a compact representation of the 4 AO values for merging decisions
        const NO_FACE: u32 = u32::MAX;
        let mut mask = [[(NO_FACE, 0u32); CHUNK_SIZE]; CHUNK_SIZE];

        for axis in 0..CHUNK_SIZE {
            // Build mask for this slice
            for row in 0..CHUNK_SIZE {
                for col in 0..CHUNK_SIZE {
                    let (x, y, z) = coord_mapper(axis, row, col);
                    let block = self.blocks[x][y][z];

                    // Skip air and foliage blocks (foliage uses crossed quads, not cube faces)
                    if !block.is_solid() || block.is_foliage() {
                        mask[row][col] = (NO_FACE, 0);
                        continue;
                    }

                    // Check if face should render using proper transparent block handling
                    let should_render = match get_adjacent_internal(&self.blocks, x, y, z) {
                        Some(adjacent) => block.should_render_face_against(adjacent),
                        None => neighbors.should_render_face(block, face, x, y, z), // boundary case
                    };

                    if should_render {
                        let tex = &texture_indices[block.as_index()];
                        let tex_layer = get_texture(tex);

                        // Calculate AO for this face and create a key for merging
                        let ao = ao_sampler.calculate_face_ao(x as i32, y as i32, z as i32, face);
                        // Pack AO values into a u32 key (4 values * 8 bits each)
                        // This ensures faces only merge if they have identical AO
                        let ao_key = ao_to_key(&ao);

                        mask[row][col] = (tex_layer, ao_key);
                    } else {
                        mask[row][col] = (NO_FACE, 0);
                    }
                }
            }

            // Greedy merge the mask into quads
            let mut processed = [[false; CHUNK_SIZE]; CHUNK_SIZE];

            for row in 0..CHUNK_SIZE {
                for col in 0..CHUNK_SIZE {
                    if processed[row][col] || mask[row][col].0 == NO_FACE {
                        continue;
                    }

                    let (tex_layer, ao_key) = mask[row][col];

                    // Find width (extend along col) - must match texture AND AO
                    let mut width = 1;
                    while col + width < CHUNK_SIZE
                        && !processed[row][col + width]
                        && mask[row][col + width] == (tex_layer, ao_key)
                    {
                        width += 1;
                    }

                    // Find height (extend along row) - must match texture AND AO
                    let mut height = 1;
                    'height: while row + height < CHUNK_SIZE {
                        for c in col..col + width {
                            if processed[row + height][c] || mask[row + height][c] != (tex_layer, ao_key) {
                                break 'height;
                            }
                        }
                        height += 1;
                    }

                    // Mark cells as processed
                    for r in row..row + height {
                        for c in col..col + width {
                            processed[r][c] = true;
                        }
                    }

                    // Emit the merged quad
                    let (x, y, z) = coord_mapper(axis, row, col);
                    let pos = Vec3::new(x as f32, y as f32, z as f32) + chunk_offset;

                    // Check if this block type has random rotation enabled
                    let block = self.blocks[x][y][z];
                    let random_rotation = texture_indices[block.as_index()].random_rotation;

                    // Get the AO values from the key (all merged faces have same AO)
                    let ao_values = key_to_ao(ao_key);

                    add_greedy_face(
                        &mut self.mesh.vertices,
                        &mut self.mesh.indices,
                        pos,
                        tex_layer,
                        face,
                        width,
                        height,
                        random_rotation,
                        ao_values,
                    );
                }
            }
        }
    }

    /// Generate mesh with neighbor-aware culling and cache it
    /// Uses direct array indexing for O(1) texture lookup
    /// Note: This legacy path doesn't support full AO (uses default AO of 1.0)
    pub fn generate_mesh(
        &mut self,
        texture_indices: &BlockTextureArray,
        neighbors: &ChunkNeighbors,
    ) {
        // Skip if not dirty
        if !self.mesh.dirty {
            return;
        }

        self.mesh.vertices.clear();
        self.mesh.indices.clear();

        let chunk_offset = Vec3::new(
            self.position[0] as f32 * CHUNK_SIZE as f32,
            self.position[1] as f32 * CHUNK_SIZE as f32,
            self.position[2] as f32 * CHUNK_SIZE as f32,
        );

        // Create a simple AO calculator for this chunk (local only, no boundary sampling)
        let ao_calc = SimpleAoCalculator::new(&self.blocks);

        for x in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for z in 0..CHUNK_SIZE {
                    let block = self.blocks[x][y][z];
                    if !block.is_solid() {
                        continue;
                    }

                    let pos = Vec3::new(x as f32, y as f32, z as f32) + chunk_offset;
                    // Direct array indexing - O(1) instead of HashMap lookup
                    let tex = &texture_indices[block.as_index()];

                    // Foliage blocks use crossed quads instead of cube faces
                    if block.is_foliage() {
                        add_crossed_quads(&mut self.mesh.vertices, &mut self.mesh.indices, pos, tex.sides);
                        continue;
                    }

                    // +X face (Right)
                    let should_render_pos_x = if x == CHUNK_SIZE - 1 {
                        neighbors.should_render_face(block, Face::Right, x, y, z)
                    } else {
                        block.should_render_face_against(self.blocks[x + 1][y][z])
                    };
                    if should_render_pos_x {
                        let ao = ao_calc.calculate_face_ao(x as i32, y as i32, z as i32, Face::Right);
                        add_face(&mut self.mesh.vertices, &mut self.mesh.indices, pos, tex.sides, Face::Right, ao);
                    }

                    // -X face (Left)
                    let should_render_neg_x = if x == 0 {
                        neighbors.should_render_face(block, Face::Left, x, y, z)
                    } else {
                        block.should_render_face_against(self.blocks[x - 1][y][z])
                    };
                    if should_render_neg_x {
                        let ao = ao_calc.calculate_face_ao(x as i32, y as i32, z as i32, Face::Left);
                        add_face(&mut self.mesh.vertices, &mut self.mesh.indices, pos, tex.sides, Face::Left, ao);
                    }

                    // +Y face (Top)
                    let should_render_pos_y = if y == CHUNK_SIZE - 1 {
                        neighbors.should_render_face(block, Face::Top, x, y, z)
                    } else {
                        block.should_render_face_against(self.blocks[x][y + 1][z])
                    };
                    if should_render_pos_y {
                        let ao = ao_calc.calculate_face_ao(x as i32, y as i32, z as i32, Face::Top);
                        add_face(&mut self.mesh.vertices, &mut self.mesh.indices, pos, tex.top, Face::Top, ao);
                    }

                    // -Y face (Bottom)
                    let should_render_neg_y = if y == 0 {
                        neighbors.should_render_face(block, Face::Bottom, x, y, z)
                    } else {
                        block.should_render_face_against(self.blocks[x][y - 1][z])
                    };
                    if should_render_neg_y {
                        let ao = ao_calc.calculate_face_ao(x as i32, y as i32, z as i32, Face::Bottom);
                        add_face(&mut self.mesh.vertices, &mut self.mesh.indices, pos, tex.bottom, Face::Bottom, ao);
                    }

                    // +Z face (Front)
                    let should_render_pos_z = if z == CHUNK_SIZE - 1 {
                        neighbors.should_render_face(block, Face::Front, x, y, z)
                    } else {
                        block.should_render_face_against(self.blocks[x][y][z + 1])
                    };
                    if should_render_pos_z {
                        let ao = ao_calc.calculate_face_ao(x as i32, y as i32, z as i32, Face::Front);
                        add_face(&mut self.mesh.vertices, &mut self.mesh.indices, pos, tex.sides, Face::Front, ao);
                    }

                    // -Z face (Back)
                    let should_render_neg_z = if z == 0 {
                        neighbors.should_render_face(block, Face::Back, x, y, z)
                    } else {
                        block.should_render_face_against(self.blocks[x][y][z - 1])
                    };
                    if should_render_neg_z {
                        let ao = ao_calc.calculate_face_ao(x as i32, y as i32, z as i32, Face::Back);
                        add_face(&mut self.mesh.vertices, &mut self.mesh.indices, pos, tex.sides, Face::Back, ao);
                    }
                }
            }
        }

        self.mesh.dirty = false;
    }
}

/// Simple AO calculator that only uses local chunk data (no boundary sampling)
/// Used by the legacy non-greedy mesh generation path
struct SimpleAoCalculator<'a> {
    blocks: &'a [[[BlockType; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE],
}

impl<'a> SimpleAoCalculator<'a> {
    fn new(blocks: &'a [[[BlockType; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE]) -> Self {
        Self { blocks }
    }

    #[inline]
    fn is_solid(&self, x: i32, y: i32, z: i32) -> bool {
        if x >= 0 && x < CHUNK_SIZE as i32 && y >= 0 && y < CHUNK_SIZE as i32 && z >= 0 && z < CHUNK_SIZE as i32 {
            !self.blocks[x as usize][y as usize][z as usize].is_transparent()
        } else {
            false // Treat out-of-bounds as air
        }
    }

    fn calculate_face_ao(&self, x: i32, y: i32, z: i32, face: Face) -> [f32; 4] {
        match face {
            Face::Top => {
                let y = y + 1;
                let n_xn = self.is_solid(x - 1, y, z);
                let n_xp = self.is_solid(x + 1, y, z);
                let n_zn = self.is_solid(x, y, z - 1);
                let n_zp = self.is_solid(x, y, z + 1);
                let n_xn_zn = self.is_solid(x - 1, y, z - 1);
                let n_xn_zp = self.is_solid(x - 1, y, z + 1);
                let n_xp_zn = self.is_solid(x + 1, y, z - 1);
                let n_xp_zp = self.is_solid(x + 1, y, z + 1);
                [
                    calculate_ao(n_xn, n_zp, n_xn_zp),
                    calculate_ao(n_xp, n_zp, n_xp_zp),
                    calculate_ao(n_xp, n_zn, n_xp_zn),
                    calculate_ao(n_xn, n_zn, n_xn_zn),
                ]
            }
            Face::Bottom => {
                let y = y - 1;
                let n_xn = self.is_solid(x - 1, y, z);
                let n_xp = self.is_solid(x + 1, y, z);
                let n_zn = self.is_solid(x, y, z - 1);
                let n_zp = self.is_solid(x, y, z + 1);
                let n_xn_zn = self.is_solid(x - 1, y, z - 1);
                let n_xn_zp = self.is_solid(x - 1, y, z + 1);
                let n_xp_zn = self.is_solid(x + 1, y, z - 1);
                let n_xp_zp = self.is_solid(x + 1, y, z + 1);
                [
                    calculate_ao(n_xn, n_zn, n_xn_zn),
                    calculate_ao(n_xp, n_zn, n_xp_zn),
                    calculate_ao(n_xp, n_zp, n_xp_zp),
                    calculate_ao(n_xn, n_zp, n_xn_zp),
                ]
            }
            Face::Left => {
                let x = x - 1;
                let n_yn = self.is_solid(x, y - 1, z);
                let n_yp = self.is_solid(x, y + 1, z);
                let n_zn = self.is_solid(x, y, z - 1);
                let n_zp = self.is_solid(x, y, z + 1);
                let n_yn_zn = self.is_solid(x, y - 1, z - 1);
                let n_yn_zp = self.is_solid(x, y - 1, z + 1);
                let n_yp_zn = self.is_solid(x, y + 1, z - 1);
                let n_yp_zp = self.is_solid(x, y + 1, z + 1);
                [
                    calculate_ao(n_yn, n_zn, n_yn_zn),
                    calculate_ao(n_yn, n_zp, n_yn_zp),
                    calculate_ao(n_yp, n_zp, n_yp_zp),
                    calculate_ao(n_yp, n_zn, n_yp_zn),
                ]
            }
            Face::Right => {
                let x = x + 1;
                let n_yn = self.is_solid(x, y - 1, z);
                let n_yp = self.is_solid(x, y + 1, z);
                let n_zn = self.is_solid(x, y, z - 1);
                let n_zp = self.is_solid(x, y, z + 1);
                let n_yn_zn = self.is_solid(x, y - 1, z - 1);
                let n_yn_zp = self.is_solid(x, y - 1, z + 1);
                let n_yp_zn = self.is_solid(x, y + 1, z - 1);
                let n_yp_zp = self.is_solid(x, y + 1, z + 1);
                [
                    calculate_ao(n_yn, n_zp, n_yn_zp),
                    calculate_ao(n_yn, n_zn, n_yn_zn),
                    calculate_ao(n_yp, n_zn, n_yp_zn),
                    calculate_ao(n_yp, n_zp, n_yp_zp),
                ]
            }
            Face::Front => {
                let z = z + 1;
                let n_xn = self.is_solid(x - 1, y, z);
                let n_xp = self.is_solid(x + 1, y, z);
                let n_yn = self.is_solid(x, y - 1, z);
                let n_yp = self.is_solid(x, y + 1, z);
                let n_xn_yn = self.is_solid(x - 1, y - 1, z);
                let n_xn_yp = self.is_solid(x - 1, y + 1, z);
                let n_xp_yn = self.is_solid(x + 1, y - 1, z);
                let n_xp_yp = self.is_solid(x + 1, y + 1, z);
                [
                    calculate_ao(n_xn, n_yn, n_xn_yn),
                    calculate_ao(n_xp, n_yn, n_xp_yn),
                    calculate_ao(n_xp, n_yp, n_xp_yp),
                    calculate_ao(n_xn, n_yp, n_xn_yp),
                ]
            }
            Face::Back => {
                let z = z - 1;
                let n_xn = self.is_solid(x - 1, y, z);
                let n_xp = self.is_solid(x + 1, y, z);
                let n_yn = self.is_solid(x, y - 1, z);
                let n_yp = self.is_solid(x, y + 1, z);
                let n_xn_yn = self.is_solid(x - 1, y - 1, z);
                let n_xn_yp = self.is_solid(x - 1, y + 1, z);
                let n_xp_yn = self.is_solid(x + 1, y - 1, z);
                let n_xp_yp = self.is_solid(x + 1, y + 1, z);
                [
                    calculate_ao(n_xp, n_yn, n_xp_yn),
                    calculate_ao(n_xn, n_yn, n_xn_yn),
                    calculate_ao(n_xn, n_yp, n_xn_yp),
                    calculate_ao(n_xp, n_yp, n_xp_yp),
                ]
            }
        }
    }
}

#[derive(Clone, Copy)]
pub enum Face {
    Top,
    Bottom,
    Left,
    Right,
    Front,
    Back,
}

/// Calculate AO values for a merged quad's 4 corners
/// For greedy-merged faces, we need to sample AO at the corner positions of the merged quad
fn calculate_merged_quad_ao(
    blocks: &[[[BlockType; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE],
    neighbors: &BoundaryNeighbors,
    x: i32,
    y: i32,
    z: i32,
    face: Face,
    width: usize,
    height: usize,
) -> [f32; 4] {
    // Create a temporary AO sampler
    let ao_sampler = AoSampler::new(blocks, neighbors);

    // For a 1x1 quad, just use standard AO calculation
    if width == 1 && height == 1 {
        return ao_sampler.calculate_face_ao(x, y, z, face);
    }

    // For merged quads, we calculate AO at each of the 4 corner block positions
    // The corners depend on the face orientation and how greedy meshing extends the quad
    //
    // Greedy meshing extends:
    // - width along the "col" dimension
    // - height along the "row" dimension

    // For merged quads, we need to calculate AO at each corner block position
    // and pick the specific vertex AO that corresponds to that corner of the merged quad.
    //
    // The vertex order in add_greedy_face for each face type:
    // - Top:    [0]=(-X,+Z), [1]=(+X,+Z), [2]=(+X,-Z), [3]=(-X,-Z) relative to block
    // - Bottom: [0]=(-X,-Z), [1]=(+X,-Z), [2]=(+X,+Z), [3]=(-X,+Z)
    // - Left:   [0]=(-Y,-Z), [1]=(-Y,+Z), [2]=(+Y,+Z), [3]=(+Y,-Z)
    // - Right:  [0]=(-Y,+Z), [1]=(-Y,-Z), [2]=(+Y,-Z), [3]=(+Y,+Z)
    // - Front:  [0]=(-X,-Y), [1]=(+X,-Y), [2]=(+X,+Y), [3]=(-X,+Y)
    // - Back:   [0]=(+X,-Y), [1]=(-X,-Y), [2]=(-X,+Y), [3]=(+X,+Y)
    //
    // For merged quads, the corners are at different block positions, but we need
    // to pick the AO value that corresponds to the corner of the merged quad.

    match face {
        Face::Top => {
            // Vertex positions in add_greedy_face:
            // [0] = (x, y+1, z+w)     -> block at (x, y, z+w-1), need its -X,+Z corner = ao[0]
            // [1] = (x+h, y+1, z+w)   -> block at (x+h-1, y, z+w-1), need its +X,+Z corner = ao[1]
            // [2] = (x+h, y+1, z)     -> block at (x+h-1, y, z), need its +X,-Z corner = ao[2]
            // [3] = (x, y+1, z)       -> block at (x, y, z), need its -X,-Z corner = ao[3]
            let ao0 = ao_sampler.calculate_face_ao(x, y, z + width as i32 - 1, face)[0];
            let ao1 = ao_sampler.calculate_face_ao(x + height as i32 - 1, y, z + width as i32 - 1, face)[1];
            let ao2 = ao_sampler.calculate_face_ao(x + height as i32 - 1, y, z, face)[2];
            let ao3 = ao_sampler.calculate_face_ao(x, y, z, face)[3];
            [ao0, ao1, ao2, ao3]
        }
        Face::Bottom => {
            // Vertex positions:
            // [0] = (x, y, z)         -> block at (x, y, z), need its -X,-Z corner = ao[0]
            // [1] = (x+h, y, z)       -> block at (x+h-1, y, z), need its +X,-Z corner = ao[1]
            // [2] = (x+h, y, z+w)     -> block at (x+h-1, y, z+w-1), need its +X,+Z corner = ao[2]
            // [3] = (x, y, z+w)       -> block at (x, y, z+w-1), need its -X,+Z corner = ao[3]
            let ao0 = ao_sampler.calculate_face_ao(x, y, z, face)[0];
            let ao1 = ao_sampler.calculate_face_ao(x + height as i32 - 1, y, z, face)[1];
            let ao2 = ao_sampler.calculate_face_ao(x + height as i32 - 1, y, z + width as i32 - 1, face)[2];
            let ao3 = ao_sampler.calculate_face_ao(x, y, z + width as i32 - 1, face)[3];
            [ao0, ao1, ao2, ao3]
        }
        Face::Left => {
            // Vertex positions:
            // [0] = (x, y, z)         -> block at (x, y, z), need its -Y,-Z corner = ao[0]
            // [1] = (x, y, z+w)       -> block at (x, y, z+w-1), need its -Y,+Z corner = ao[1]
            // [2] = (x, y+h, z+w)     -> block at (x, y+h-1, z+w-1), need its +Y,+Z corner = ao[2]
            // [3] = (x, y+h, z)       -> block at (x, y+h-1, z), need its +Y,-Z corner = ao[3]
            let ao0 = ao_sampler.calculate_face_ao(x, y, z, face)[0];
            let ao1 = ao_sampler.calculate_face_ao(x, y, z + width as i32 - 1, face)[1];
            let ao2 = ao_sampler.calculate_face_ao(x, y + height as i32 - 1, z + width as i32 - 1, face)[2];
            let ao3 = ao_sampler.calculate_face_ao(x, y + height as i32 - 1, z, face)[3];
            [ao0, ao1, ao2, ao3]
        }
        Face::Right => {
            // Vertex positions:
            // [0] = (x+1, y, z+w)     -> block at (x, y, z+w-1), need its -Y,+Z corner = ao[0]
            // [1] = (x+1, y, z)       -> block at (x, y, z), need its -Y,-Z corner = ao[1]
            // [2] = (x+1, y+h, z)     -> block at (x, y+h-1, z), need its +Y,-Z corner = ao[2]
            // [3] = (x+1, y+h, z+w)   -> block at (x, y+h-1, z+w-1), need its +Y,+Z corner = ao[3]
            let ao0 = ao_sampler.calculate_face_ao(x, y, z + width as i32 - 1, face)[0];
            let ao1 = ao_sampler.calculate_face_ao(x, y, z, face)[1];
            let ao2 = ao_sampler.calculate_face_ao(x, y + height as i32 - 1, z, face)[2];
            let ao3 = ao_sampler.calculate_face_ao(x, y + height as i32 - 1, z + width as i32 - 1, face)[3];
            [ao0, ao1, ao2, ao3]
        }
        Face::Front => {
            // Vertex positions:
            // [0] = (x, y, z+1)       -> block at (x, y, z), need its -X,-Y corner = ao[0]
            // [1] = (x+h, y, z+1)     -> block at (x+h-1, y, z), need its +X,-Y corner = ao[1]
            // [2] = (x+h, y+w, z+1)   -> block at (x+h-1, y+w-1, z), need its +X,+Y corner = ao[2]
            // [3] = (x, y+w, z+1)     -> block at (x, y+w-1, z), need its -X,+Y corner = ao[3]
            let ao0 = ao_sampler.calculate_face_ao(x, y, z, face)[0];
            let ao1 = ao_sampler.calculate_face_ao(x + height as i32 - 1, y, z, face)[1];
            let ao2 = ao_sampler.calculate_face_ao(x + height as i32 - 1, y + width as i32 - 1, z, face)[2];
            let ao3 = ao_sampler.calculate_face_ao(x, y + width as i32 - 1, z, face)[3];
            [ao0, ao1, ao2, ao3]
        }
        Face::Back => {
            // Vertex positions:
            // [0] = (x+h, y, z)       -> block at (x+h-1, y, z), need its +X,-Y corner = ao[0]
            // [1] = (x, y, z)         -> block at (x, y, z), need its -X,-Y corner = ao[1]
            // [2] = (x, y+w, z)       -> block at (x, y+w-1, z), need its -X,+Y corner = ao[2]
            // [3] = (x+h, y+w, z)     -> block at (x+h-1, y+w-1, z), need its +X,+Y corner = ao[3]
            let ao0 = ao_sampler.calculate_face_ao(x + height as i32 - 1, y, z, face)[0];
            let ao1 = ao_sampler.calculate_face_ao(x, y, z, face)[1];
            let ao2 = ao_sampler.calculate_face_ao(x, y + width as i32 - 1, z, face)[2];
            let ao3 = ao_sampler.calculate_face_ao(x + height as i32 - 1, y + width as i32 - 1, z, face)[3];
            [ao0, ao1, ao2, ao3]
        }
    }
}

/// Add a single 1x1 face (used by non-greedy meshing)
fn add_face(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    pos: Vec3,
    tex_layer: u32,
    face: Face,
    ao_values: [f32; 4],
) {
    add_greedy_face(vertices, indices, pos, tex_layer, face, 1, 1, false, ao_values);
}

/// Simple hash function for deterministic random rotation based on position
#[inline]
fn position_hash(x: i32, y: i32, z: i32, face: u8) -> u8 {
    // Simple hash combining position and face for consistent rotation
    let mut h = x.wrapping_mul(73856093) ^ y.wrapping_mul(19349663) ^ z.wrapping_mul(83492791);
    h ^= (face as i32).wrapping_mul(48611);
    (h & 3) as u8 // Returns 0, 1, 2, or 3 for 0°, 90°, 180°, 270° rotation
}

/// Rotate UV coordinates by 90 degrees * rotation_count
#[inline]
fn rotate_uv(uv: [f32; 2], rotation: u8, w: f32, h: f32) -> [f32; 2] {
    match rotation & 3 {
        0 => uv,                           // 0°
        1 => [h - uv[1], uv[0]],          // 90° CCW: (u,v) -> (h-v, u)
        2 => [w - uv[0], h - uv[1]],      // 180°: (u,v) -> (w-u, h-v)
        3 => [uv[1], w - uv[0]],          // 270° CCW: (u,v) -> (v, w-u)
        _ => unreachable!(),
    }
}

/// Pack 4 AO values into a u32 key for greedy mesh comparison
/// Each AO value is quantized to 8 bits
#[inline]
fn ao_to_key(ao: &[f32; 4]) -> u32 {
    let a0 = (ao[0] * 255.0) as u32;
    let a1 = (ao[1] * 255.0) as u32;
    let a2 = (ao[2] * 255.0) as u32;
    let a3 = (ao[3] * 255.0) as u32;
    (a0 << 24) | (a1 << 16) | (a2 << 8) | a3
}

/// Unpack a u32 key back into 4 AO values
#[inline]
fn key_to_ao(key: u32) -> [f32; 4] {
    [
        ((key >> 24) & 0xFF) as f32 / 255.0,
        ((key >> 16) & 0xFF) as f32 / 255.0,
        ((key >> 8) & 0xFF) as f32 / 255.0,
        (key & 0xFF) as f32 / 255.0,
    ]
}

/// Calculate ambient occlusion value for a vertex based on neighboring blocks
/// Uses the standard voxel AO algorithm: count solid blocks in the 3 neighbors
/// side1, side2 are the two edge neighbors, corner is the diagonal neighbor
/// Returns AO value from 0.0 (fully occluded) to 1.0 (no occlusion)
#[inline]
fn calculate_ao(side1: bool, side2: bool, corner: bool) -> f32 {
    // Standard voxel AO formula
    // If both sides are solid, corner doesn't matter (fully occluded)
    // Otherwise count solid neighbors
    let ao_level = if side1 && side2 {
        0
    } else {
        3 - (side1 as u8 + side2 as u8 + corner as u8)
    };
    // Map 0-3 to brightness values with nice curve
    match ao_level {
        0 => 0.2,  // Fully occluded (dark corner)
        1 => 0.5,
        2 => 0.75,
        3 => 1.0,  // No occlusion
        _ => 1.0,
    }
}

/// AO lookup data for a chunk - caches which positions are solid for AO sampling
/// This allows sampling outside chunk boundaries
pub struct AoSampler<'a> {
    blocks: &'a [[[BlockType; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE],
    neighbors: &'a BoundaryNeighbors<'a>,
}

impl<'a> AoSampler<'a> {
    pub fn new(blocks: &'a [[[BlockType; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE], neighbors: &'a BoundaryNeighbors<'a>) -> Self {
        Self { blocks, neighbors }
    }

    /// Check if a block position is solid (for AO calculation)
    /// Handles positions outside chunk boundaries using neighbor data
    #[inline]
    fn is_solid(&self, x: i32, y: i32, z: i32) -> bool {
        // Inside chunk bounds
        if x >= 0 && x < CHUNK_SIZE as i32 && y >= 0 && y < CHUNK_SIZE as i32 && z >= 0 && z < CHUNK_SIZE as i32 {
            return !self.blocks[x as usize][y as usize][z as usize].is_transparent();
        }

        // Outside chunk - check neighbor boundaries
        // For AO, we treat missing neighbors as air (not solid)
        if x < 0 {
            if let Some(boundary) = self.neighbors.neg_x {
                if y >= 0 && y < CHUNK_SIZE as i32 && z >= 0 && z < CHUNK_SIZE as i32 {
                    return !boundary[y as usize][z as usize].is_transparent();
                }
            }
            return false;
        }
        if x >= CHUNK_SIZE as i32 {
            if let Some(boundary) = self.neighbors.pos_x {
                if y >= 0 && y < CHUNK_SIZE as i32 && z >= 0 && z < CHUNK_SIZE as i32 {
                    return !boundary[y as usize][z as usize].is_transparent();
                }
            }
            return false;
        }
        if y < 0 {
            if let Some(boundary) = self.neighbors.neg_y {
                if x >= 0 && x < CHUNK_SIZE as i32 && z >= 0 && z < CHUNK_SIZE as i32 {
                    return !boundary[x as usize][z as usize].is_transparent();
                }
            }
            return false;
        }
        if y >= CHUNK_SIZE as i32 {
            if let Some(boundary) = self.neighbors.pos_y {
                if x >= 0 && x < CHUNK_SIZE as i32 && z >= 0 && z < CHUNK_SIZE as i32 {
                    return !boundary[x as usize][z as usize].is_transparent();
                }
            }
            return false;
        }
        if z < 0 {
            if let Some(boundary) = self.neighbors.neg_z {
                if x >= 0 && x < CHUNK_SIZE as i32 && y >= 0 && y < CHUNK_SIZE as i32 {
                    return !boundary[x as usize][y as usize].is_transparent();
                }
            }
            return false;
        }
        if z >= CHUNK_SIZE as i32 {
            if let Some(boundary) = self.neighbors.pos_z {
                if x >= 0 && x < CHUNK_SIZE as i32 && y >= 0 && y < CHUNK_SIZE as i32 {
                    return !boundary[x as usize][y as usize].is_transparent();
                }
            }
            return false;
        }

        false // Default to air for corners outside all boundaries
    }

    /// Calculate AO values for the 4 vertices of a face
    /// Returns [ao0, ao1, ao2, ao3] for the quad vertices
    pub fn calculate_face_ao(&self, x: i32, y: i32, z: i32, face: Face) -> [f32; 4] {
        // For each face, we need to sample the 8 neighbors in the plane of the face
        // Each vertex uses 3 of those neighbors: 2 edge neighbors and 1 corner
        match face {
            Face::Top => {
                // Face at y+1, sample neighbors at y+1 level
                let y = y + 1;
                // Neighbors in the XZ plane at y+1
                let n_xn = self.is_solid(x - 1, y, z);     // -X
                let n_xp = self.is_solid(x + 1, y, z);     // +X
                let n_zn = self.is_solid(x, y, z - 1);     // -Z
                let n_zp = self.is_solid(x, y, z + 1);     // +Z
                let n_xn_zn = self.is_solid(x - 1, y, z - 1); // -X -Z
                let n_xn_zp = self.is_solid(x - 1, y, z + 1); // -X +Z
                let n_xp_zn = self.is_solid(x + 1, y, z - 1); // +X -Z
                let n_xp_zp = self.is_solid(x + 1, y, z + 1); // +X +Z

                [
                    calculate_ao(n_xn, n_zp, n_xn_zp), // vertex at (-X, +Z)
                    calculate_ao(n_xp, n_zp, n_xp_zp), // vertex at (+X, +Z)
                    calculate_ao(n_xp, n_zn, n_xp_zn), // vertex at (+X, -Z)
                    calculate_ao(n_xn, n_zn, n_xn_zn), // vertex at (-X, -Z)
                ]
            }
            Face::Bottom => {
                // Face at y, sample neighbors at y-1 level
                let y = y - 1;
                let n_xn = self.is_solid(x - 1, y, z);
                let n_xp = self.is_solid(x + 1, y, z);
                let n_zn = self.is_solid(x, y, z - 1);
                let n_zp = self.is_solid(x, y, z + 1);
                let n_xn_zn = self.is_solid(x - 1, y, z - 1);
                let n_xn_zp = self.is_solid(x - 1, y, z + 1);
                let n_xp_zn = self.is_solid(x + 1, y, z - 1);
                let n_xp_zp = self.is_solid(x + 1, y, z + 1);

                [
                    calculate_ao(n_xn, n_zn, n_xn_zn), // vertex at (-X, -Z)
                    calculate_ao(n_xp, n_zn, n_xp_zn), // vertex at (+X, -Z)
                    calculate_ao(n_xp, n_zp, n_xp_zp), // vertex at (+X, +Z)
                    calculate_ao(n_xn, n_zp, n_xn_zp), // vertex at (-X, +Z)
                ]
            }
            Face::Left => {
                // Face at x, sample neighbors at x-1 level
                let x = x - 1;
                let n_yn = self.is_solid(x, y - 1, z);
                let n_yp = self.is_solid(x, y + 1, z);
                let n_zn = self.is_solid(x, y, z - 1);
                let n_zp = self.is_solid(x, y, z + 1);
                let n_yn_zn = self.is_solid(x, y - 1, z - 1);
                let n_yn_zp = self.is_solid(x, y - 1, z + 1);
                let n_yp_zn = self.is_solid(x, y + 1, z - 1);
                let n_yp_zp = self.is_solid(x, y + 1, z + 1);

                [
                    calculate_ao(n_yn, n_zn, n_yn_zn), // vertex at (-Y, -Z)
                    calculate_ao(n_yn, n_zp, n_yn_zp), // vertex at (-Y, +Z)
                    calculate_ao(n_yp, n_zp, n_yp_zp), // vertex at (+Y, +Z)
                    calculate_ao(n_yp, n_zn, n_yp_zn), // vertex at (+Y, -Z)
                ]
            }
            Face::Right => {
                // Face at x+1, sample neighbors at x+1 level
                let x = x + 1;
                let n_yn = self.is_solid(x, y - 1, z);
                let n_yp = self.is_solid(x, y + 1, z);
                let n_zn = self.is_solid(x, y, z - 1);
                let n_zp = self.is_solid(x, y, z + 1);
                let n_yn_zn = self.is_solid(x, y - 1, z - 1);
                let n_yn_zp = self.is_solid(x, y - 1, z + 1);
                let n_yp_zn = self.is_solid(x, y + 1, z - 1);
                let n_yp_zp = self.is_solid(x, y + 1, z + 1);

                [
                    calculate_ao(n_yn, n_zp, n_yn_zp), // vertex at (-Y, +Z)
                    calculate_ao(n_yn, n_zn, n_yn_zn), // vertex at (-Y, -Z)
                    calculate_ao(n_yp, n_zn, n_yp_zn), // vertex at (+Y, -Z)
                    calculate_ao(n_yp, n_zp, n_yp_zp), // vertex at (+Y, +Z)
                ]
            }
            Face::Front => {
                // Face at z+1, sample neighbors at z+1 level
                let z = z + 1;
                let n_xn = self.is_solid(x - 1, y, z);
                let n_xp = self.is_solid(x + 1, y, z);
                let n_yn = self.is_solid(x, y - 1, z);
                let n_yp = self.is_solid(x, y + 1, z);
                let n_xn_yn = self.is_solid(x - 1, y - 1, z);
                let n_xn_yp = self.is_solid(x - 1, y + 1, z);
                let n_xp_yn = self.is_solid(x + 1, y - 1, z);
                let n_xp_yp = self.is_solid(x + 1, y + 1, z);

                [
                    calculate_ao(n_xn, n_yn, n_xn_yn), // vertex at (-X, -Y)
                    calculate_ao(n_xp, n_yn, n_xp_yn), // vertex at (+X, -Y)
                    calculate_ao(n_xp, n_yp, n_xp_yp), // vertex at (+X, +Y)
                    calculate_ao(n_xn, n_yp, n_xn_yp), // vertex at (-X, +Y)
                ]
            }
            Face::Back => {
                // Face at z, sample neighbors at z-1 level
                let z = z - 1;
                let n_xn = self.is_solid(x - 1, y, z);
                let n_xp = self.is_solid(x + 1, y, z);
                let n_yn = self.is_solid(x, y - 1, z);
                let n_yp = self.is_solid(x, y + 1, z);
                let n_xn_yn = self.is_solid(x - 1, y - 1, z);
                let n_xn_yp = self.is_solid(x - 1, y + 1, z);
                let n_xp_yn = self.is_solid(x + 1, y - 1, z);
                let n_xp_yp = self.is_solid(x + 1, y + 1, z);

                [
                    calculate_ao(n_xp, n_yn, n_xp_yn), // vertex at (+X, -Y)
                    calculate_ao(n_xn, n_yn, n_xn_yn), // vertex at (-X, -Y)
                    calculate_ao(n_xn, n_yp, n_xn_yp), // vertex at (-X, +Y)
                    calculate_ao(n_xp, n_yp, n_xp_yp), // vertex at (+X, +Y)
                ]
            }
        }
    }
}

/// Add crossed quads for foliage blocks (two quads forming an X pattern)
/// This creates a more natural look for grass, flowers, and other vegetation
fn add_crossed_quads(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    pos: Vec3,
    tex_layer: u32,
) {
    // Two diagonal quads forming an X when viewed from above
    // Offset slightly inward from block edges for better appearance
    let offset = 0.15; // Slight inset from block edges

    // First diagonal: from (0,0) to (1,1) in XZ plane
    add_foliage_quad(
        vertices,
        indices,
        [pos.x + offset, pos.y, pos.z + offset],
        [pos.x + 1.0 - offset, pos.y, pos.z + 1.0 - offset],
        [pos.x + 1.0 - offset, pos.y + 1.0, pos.z + 1.0 - offset],
        [pos.x + offset, pos.y + 1.0, pos.z + offset],
        tex_layer,
        [0.707, 0.0, 0.707], // Normal pointing diagonally
    );

    // Same quad, but facing the other way (for backface visibility)
    add_foliage_quad(
        vertices,
        indices,
        [pos.x + 1.0 - offset, pos.y, pos.z + 1.0 - offset],
        [pos.x + offset, pos.y, pos.z + offset],
        [pos.x + offset, pos.y + 1.0, pos.z + offset],
        [pos.x + 1.0 - offset, pos.y + 1.0, pos.z + 1.0 - offset],
        tex_layer,
        [-0.707, 0.0, -0.707], // Opposite normal
    );

    // Second diagonal: from (1,0) to (0,1) in XZ plane
    add_foliage_quad(
        vertices,
        indices,
        [pos.x + 1.0 - offset, pos.y, pos.z + offset],
        [pos.x + offset, pos.y, pos.z + 1.0 - offset],
        [pos.x + offset, pos.y + 1.0, pos.z + 1.0 - offset],
        [pos.x + 1.0 - offset, pos.y + 1.0, pos.z + offset],
        tex_layer,
        [-0.707, 0.0, 0.707], // Normal pointing diagonally
    );

    // Same quad, but facing the other way
    add_foliage_quad(
        vertices,
        indices,
        [pos.x + offset, pos.y, pos.z + 1.0 - offset],
        [pos.x + 1.0 - offset, pos.y, pos.z + offset],
        [pos.x + 1.0 - offset, pos.y + 1.0, pos.z + offset],
        [pos.x + offset, pos.y + 1.0, pos.z + 1.0 - offset],
        tex_layer,
        [0.707, 0.0, -0.707], // Opposite normal
    );
}

/// Add a single foliage quad with specified corners
fn add_foliage_quad(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    p0: [f32; 3],
    p1: [f32; 3],
    p2: [f32; 3],
    p3: [f32; 3],
    tex_layer: u32,
    normal: [f32; 3],
) {
    let base_index = vertices.len() as u32;

    // UVs for a 1x1 texture quad
    let uvs = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
    let positions = [p0, p1, p2, p3];

    // Use full brightness (no AO) for foliage - they're thin and shouldn't occlude themselves
    let ao = 1.0;

    for i in 0..4 {
        vertices.push(Vertex {
            position: positions[i],
            uv: uvs[i],
            normal,
            tex_layer,
            ao,
        });
    }

    // Two triangles for the quad
    indices.extend_from_slice(&[
        base_index,
        base_index + 1,
        base_index + 2,
        base_index,
        base_index + 2,
        base_index + 3,
    ]);
}

/// Add a merged face from greedy meshing with proper UV tiling
/// width and height are in blocks; UVs are scaled to tile the texture
fn add_greedy_face(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    pos: Vec3,
    tex_layer: u32,
    face: Face,
    width: usize,
    height: usize,
    random_rotation: bool,
    ao_values: [f32; 4],
) {
    let base_index = vertices.len() as u32;
    let w = width as f32;
    let h = height as f32;

    // Calculate rotation if enabled (only for 1x1 faces to preserve tiling)
    // For larger merged quads, rotation would break texture tiling
    let rotation = if random_rotation && width == 1 && height == 1 {
        let face_idx = match face {
            Face::Top => 0,
            Face::Bottom => 1,
            Face::Left => 2,
            Face::Right => 3,
            Face::Front => 4,
            Face::Back => 5,
        };
        position_hash(pos.x as i32, pos.y as i32, pos.z as i32, face_idx)
    } else {
        0
    };

    // Position, UV, and normal for each face
    // UVs are scaled by width/height to tile the texture across merged faces
    // The width/height dimensions depend on the face orientation:
    // - For axis-aligned faces, "width" extends along one axis, "height" along another
    // - We need to map row/col from greedy meshing to the correct world axes
    let (positions, uvs, normal) = match face {
        // Top/Bottom: greedy mesh iterates axis=Y, row=X, col=Z
        // So width extends along Z, height extends along X
        Face::Top => (
            [
                [pos.x, pos.y + 1.0, pos.z + w],           // row, col+w
                [pos.x + h, pos.y + 1.0, pos.z + w],       // row+h, col+w
                [pos.x + h, pos.y + 1.0, pos.z],           // row+h, col
                [pos.x, pos.y + 1.0, pos.z],               // row, col
            ],
            [[0.0, w], [h, w], [h, 0.0], [0.0, 0.0]],
            [0.0, 1.0, 0.0],
        ),
        Face::Bottom => (
            [
                [pos.x, pos.y, pos.z],
                [pos.x + h, pos.y, pos.z],
                [pos.x + h, pos.y, pos.z + w],
                [pos.x, pos.y, pos.z + w],
            ],
            [[0.0, 0.0], [h, 0.0], [h, w], [0.0, w]],
            [0.0, -1.0, 0.0],
        ),
        // Left/Right: greedy mesh iterates axis=X, row=Y, col=Z
        // So width extends along Z, height extends along Y
        Face::Left => (
            [
                [pos.x, pos.y, pos.z],
                [pos.x, pos.y, pos.z + w],
                [pos.x, pos.y + h, pos.z + w],
                [pos.x, pos.y + h, pos.z],
            ],
            [[w, h], [0.0, h], [0.0, 0.0], [w, 0.0]],
            [-1.0, 0.0, 0.0],
        ),
        Face::Right => (
            [
                [pos.x + 1.0, pos.y, pos.z + w],
                [pos.x + 1.0, pos.y, pos.z],
                [pos.x + 1.0, pos.y + h, pos.z],
                [pos.x + 1.0, pos.y + h, pos.z + w],
            ],
            [[w, h], [0.0, h], [0.0, 0.0], [w, 0.0]],
            [1.0, 0.0, 0.0],
        ),
        // Front/Back: greedy mesh iterates axis=Z, row=X, col=Y
        // So width extends along Y, height extends along X
        Face::Front => (
            [
                [pos.x, pos.y, pos.z + 1.0],
                [pos.x + h, pos.y, pos.z + 1.0],
                [pos.x + h, pos.y + w, pos.z + 1.0],
                [pos.x, pos.y + w, pos.z + 1.0],
            ],
            [[0.0, w], [h, w], [h, 0.0], [0.0, 0.0]],
            [0.0, 0.0, 1.0],
        ),
        Face::Back => (
            [
                [pos.x + h, pos.y, pos.z],
                [pos.x, pos.y, pos.z],
                [pos.x, pos.y + w, pos.z],
                [pos.x + h, pos.y + w, pos.z],
            ],
            [[0.0, w], [h, w], [h, 0.0], [0.0, 0.0]],
            [0.0, 0.0, -1.0],
        ),
    };

    for i in 0..4 {
        // Apply rotation to UVs if enabled
        let uv = if rotation > 0 {
            rotate_uv(uvs[i], rotation, w, h)
        } else {
            uvs[i]
        };
        vertices.push(Vertex {
            position: positions[i],
            uv,
            normal,
            tex_layer,
            ao: ao_values[i],
        });
    }

    // Two triangles per face
    indices.extend_from_slice(&[
        base_index,
        base_index + 1,
        base_index + 2,
        base_index,
        base_index + 2,
        base_index + 3,
    ]);
}
