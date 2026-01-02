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
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
        0 => Float32x3,  // position
        1 => Float32x2,  // uv
        2 => Float32x3,  // normal
        3 => Uint32,     // tex_layer
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
}

impl BlockType {
    /// Total number of block types (for array indexing)
    pub const COUNT: usize = 10;

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
        matches!(self, BlockType::Air | BlockType::Water | BlockType::Leaves)
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
        }
    }
}

impl Default for BlockType {
    fn default() -> Self {
        BlockType::Air
    }
}

pub const CHUNK_SIZE: usize = 16;

/// A 2D slice of block solidity data for a chunk boundary face
/// Used for parallel mesh generation where we can't hold chunk references
pub type BoundarySlice = [[bool; CHUNK_SIZE]; CHUNK_SIZE];

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
    /// Check if block at neighbor boundary position is solid
    fn is_solid_at(&self, dir: Face, local_x: usize, local_y: usize, local_z: usize) -> bool {
        match dir {
            Face::Right => self.pos_x.map(|b| b[local_y][local_z]).unwrap_or(true),
            Face::Left => self.neg_x.map(|b| b[local_y][local_z]).unwrap_or(true),
            Face::Top => self.pos_y.map(|b| b[local_x][local_z]).unwrap_or(true),
            Face::Bottom => self.neg_y.map(|b| b[local_x][local_z]).unwrap_or(true),
            Face::Front => self.pos_z.map(|b| b[local_x][local_y]).unwrap_or(true),
            Face::Back => self.neg_z.map(|b| b[local_x][local_y]).unwrap_or(true),
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
    /// Check if block at neighbor position is solid
    /// Returns true (solid) if neighbor doesn't exist - skip rendering until neighbor loads
    fn is_solid_at(&self, dir: Face, local_x: usize, local_y: usize, local_z: usize) -> bool {
        match dir {
            Face::Right => {
                // +X: check neg_x face of pos_x neighbor
                self.pos_x
                    .map(|c| c.blocks[0][local_y][local_z].is_solid())
                    .unwrap_or(true)
            }
            Face::Left => {
                // -X: check pos_x face of neg_x neighbor
                self.neg_x
                    .map(|c| c.blocks[CHUNK_SIZE - 1][local_y][local_z].is_solid())
                    .unwrap_or(true)
            }
            Face::Top => {
                // +Y: check neg_y face of pos_y neighbor
                self.pos_y
                    .map(|c| c.blocks[local_x][0][local_z].is_solid())
                    .unwrap_or(true)
            }
            Face::Bottom => {
                // -Y: check pos_y face of neg_y neighbor
                self.neg_y
                    .map(|c| c.blocks[local_x][CHUNK_SIZE - 1][local_z].is_solid())
                    .unwrap_or(true)
            }
            Face::Front => {
                // +Z: check neg_z face of pos_z neighbor
                self.pos_z
                    .map(|c| c.blocks[local_x][local_y][0].is_solid())
                    .unwrap_or(true)
            }
            Face::Back => {
                // -Z: check pos_z face of neg_z neighbor
                self.neg_z
                    .map(|c| c.blocks[local_x][local_y][CHUNK_SIZE - 1].is_solid())
                    .unwrap_or(true)
            }
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

    /// Extract the +X boundary (x = CHUNK_SIZE-1) as a 2D slice of solidity
    /// Used by -X neighbor to check if faces should render
    pub fn extract_pos_x_boundary(&self) -> BoundarySlice {
        let mut slice = [[false; CHUNK_SIZE]; CHUNK_SIZE];
        for y in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                slice[y][z] = self.blocks[CHUNK_SIZE - 1][y][z].is_solid();
            }
        }
        slice
    }

    /// Extract the -X boundary (x = 0) as a 2D slice of solidity
    /// Used by +X neighbor to check if faces should render
    pub fn extract_neg_x_boundary(&self) -> BoundarySlice {
        let mut slice = [[false; CHUNK_SIZE]; CHUNK_SIZE];
        for y in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                slice[y][z] = self.blocks[0][y][z].is_solid();
            }
        }
        slice
    }

    /// Extract the +Y boundary (y = CHUNK_SIZE-1) as a 2D slice of solidity
    pub fn extract_pos_y_boundary(&self) -> BoundarySlice {
        let mut slice = [[false; CHUNK_SIZE]; CHUNK_SIZE];
        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                slice[x][z] = self.blocks[x][CHUNK_SIZE - 1][z].is_solid();
            }
        }
        slice
    }

    /// Extract the -Y boundary (y = 0) as a 2D slice of solidity
    pub fn extract_neg_y_boundary(&self) -> BoundarySlice {
        let mut slice = [[false; CHUNK_SIZE]; CHUNK_SIZE];
        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                slice[x][z] = self.blocks[x][0][z].is_solid();
            }
        }
        slice
    }

    /// Extract the +Z boundary (z = CHUNK_SIZE-1) as a 2D slice of solidity
    pub fn extract_pos_z_boundary(&self) -> BoundarySlice {
        let mut slice = [[false; CHUNK_SIZE]; CHUNK_SIZE];
        for x in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                slice[x][y] = self.blocks[x][y][CHUNK_SIZE - 1].is_solid();
            }
        }
        slice
    }

    /// Extract the -Z boundary (z = 0) as a 2D slice of solidity
    pub fn extract_neg_z_boundary(&self) -> BoundarySlice {
        let mut slice = [[false; CHUNK_SIZE]; CHUNK_SIZE];
        for x in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                slice[x][y] = self.blocks[x][y][0].is_solid();
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
                    Some(!blocks[x + 1][y][z].is_solid())
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
                    Some(!blocks[x - 1][y][z].is_solid())
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
            |y, x, z| (x, y, z),  // axis=y, row=x, col=z
            |blocks, x, y, z| {
                if y == CHUNK_SIZE - 1 {
                    None
                } else {
                    Some(!blocks[x][y + 1][z].is_solid())
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
                    Some(!blocks[x][y - 1][z].is_solid())
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
                    Some(!blocks[x][y][z + 1].is_solid())
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
                    Some(!blocks[x][y][z - 1].is_solid())
                }
            },
            |tex| tex.sides,
        );

        self.mesh.dirty = false;
    }

    /// Greedy mesh one face direction across all slices
    fn greedy_mesh_axis_with_boundaries<F, G, T>(
        &mut self,
        texture_indices: &BlockTextureArray,
        neighbors: &BoundaryNeighbors,
        chunk_offset: Vec3,
        face: Face,
        coord_mapper: F,
        should_render_internal: G,
        get_texture: T,
    ) where
        F: Fn(usize, usize, usize) -> (usize, usize, usize),
        G: Fn(&[[[BlockType; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE], usize, usize, usize) -> Option<bool>,
        T: Fn(&crate::texture::BlockTextureIndices) -> u32,
    {
        // Mask stores texture layer for each cell, 0 means no face
        // We use u32::MAX as "no face" marker since 0 could be a valid texture
        const NO_FACE: u32 = u32::MAX;
        let mut mask = [[NO_FACE; CHUNK_SIZE]; CHUNK_SIZE];

        for axis in 0..CHUNK_SIZE {
            // Build mask for this slice
            for row in 0..CHUNK_SIZE {
                for col in 0..CHUNK_SIZE {
                    let (x, y, z) = coord_mapper(axis, row, col);
                    let block = self.blocks[x][y][z];

                    if !block.is_solid() {
                        mask[row][col] = NO_FACE;
                        continue;
                    }

                    // Check if face should render
                    let should_render = match should_render_internal(&self.blocks, x, y, z) {
                        Some(result) => result,
                        None => !neighbors.is_solid_at(face, x, y, z), // boundary case
                    };

                    if should_render {
                        let tex = &texture_indices[block.as_index()];
                        mask[row][col] = get_texture(tex);
                    } else {
                        mask[row][col] = NO_FACE;
                    }
                }
            }

            // Greedy merge the mask into quads
            let mut processed = [[false; CHUNK_SIZE]; CHUNK_SIZE];

            for row in 0..CHUNK_SIZE {
                for col in 0..CHUNK_SIZE {
                    if processed[row][col] || mask[row][col] == NO_FACE {
                        continue;
                    }

                    let tex_layer = mask[row][col];

                    // Find width (extend along col)
                    let mut width = 1;
                    while col + width < CHUNK_SIZE
                        && !processed[row][col + width]
                        && mask[row][col + width] == tex_layer
                    {
                        width += 1;
                    }

                    // Find height (extend along row)
                    let mut height = 1;
                    'height: while row + height < CHUNK_SIZE {
                        for c in col..col + width {
                            if processed[row + height][c] || mask[row + height][c] != tex_layer {
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

                    add_greedy_face(
                        &mut self.mesh.vertices,
                        &mut self.mesh.indices,
                        pos,
                        tex_layer,
                        face,
                        width,
                        height,
                        random_rotation,
                    );
                }
            }
        }
    }

    /// Generate mesh with neighbor-aware culling and cache it
    /// Uses direct array indexing for O(1) texture lookup
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

                    // +X face (Right)
                    let should_render_pos_x = if x == CHUNK_SIZE - 1 {
                        !neighbors.is_solid_at(Face::Right, x, y, z)
                    } else {
                        !self.blocks[x + 1][y][z].is_solid()
                    };
                    if should_render_pos_x {
                        add_face(&mut self.mesh.vertices, &mut self.mesh.indices, pos, tex.sides, Face::Right);
                    }

                    // -X face (Left)
                    let should_render_neg_x = if x == 0 {
                        !neighbors.is_solid_at(Face::Left, x, y, z)
                    } else {
                        !self.blocks[x - 1][y][z].is_solid()
                    };
                    if should_render_neg_x {
                        add_face(&mut self.mesh.vertices, &mut self.mesh.indices, pos, tex.sides, Face::Left);
                    }

                    // +Y face (Top)
                    let should_render_pos_y = if y == CHUNK_SIZE - 1 {
                        !neighbors.is_solid_at(Face::Top, x, y, z)
                    } else {
                        !self.blocks[x][y + 1][z].is_solid()
                    };
                    if should_render_pos_y {
                        add_face(&mut self.mesh.vertices, &mut self.mesh.indices, pos, tex.top, Face::Top);
                    }

                    // -Y face (Bottom)
                    let should_render_neg_y = if y == 0 {
                        !neighbors.is_solid_at(Face::Bottom, x, y, z)
                    } else {
                        !self.blocks[x][y - 1][z].is_solid()
                    };
                    if should_render_neg_y {
                        add_face(&mut self.mesh.vertices, &mut self.mesh.indices, pos, tex.bottom, Face::Bottom);
                    }

                    // +Z face (Front)
                    let should_render_pos_z = if z == CHUNK_SIZE - 1 {
                        !neighbors.is_solid_at(Face::Front, x, y, z)
                    } else {
                        !self.blocks[x][y][z + 1].is_solid()
                    };
                    if should_render_pos_z {
                        add_face(&mut self.mesh.vertices, &mut self.mesh.indices, pos, tex.sides, Face::Front);
                    }

                    // -Z face (Back)
                    let should_render_neg_z = if z == 0 {
                        !neighbors.is_solid_at(Face::Back, x, y, z)
                    } else {
                        !self.blocks[x][y][z - 1].is_solid()
                    };
                    if should_render_neg_z {
                        add_face(&mut self.mesh.vertices, &mut self.mesh.indices, pos, tex.sides, Face::Back);
                    }
                }
            }
        }

        self.mesh.dirty = false;
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

/// Add a single 1x1 face (used by non-greedy meshing)
fn add_face(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    pos: Vec3,
    tex_layer: u32,
    face: Face,
) {
    add_greedy_face(vertices, indices, pos, tex_layer, face, 1, 1, false);
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
