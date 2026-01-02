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

    /// Generate mesh using extracted boundary data (for parallel processing)
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

        for x in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for z in 0..CHUNK_SIZE {
                    let block = self.blocks[x][y][z];
                    if !block.is_solid() {
                        continue;
                    }

                    let pos = Vec3::new(x as f32, y as f32, z as f32) + chunk_offset;
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

fn add_face(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    pos: Vec3,
    tex_layer: u32,
    face: Face,
) {
    let base_index = vertices.len() as u32;

    // Position, UV, and normal for each face
    // UVs are 0-1 for a single voxel; will tile properly with greedy meshing
    let (positions, uvs, normal) = match face {
        Face::Top => (
            [
                [pos.x, pos.y + 1.0, pos.z + 1.0],
                [pos.x + 1.0, pos.y + 1.0, pos.z + 1.0],
                [pos.x + 1.0, pos.y + 1.0, pos.z],
                [pos.x, pos.y + 1.0, pos.z],
            ],
            [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
            [0.0, 1.0, 0.0],
        ),
        Face::Bottom => (
            [
                [pos.x, pos.y, pos.z],
                [pos.x + 1.0, pos.y, pos.z],
                [pos.x + 1.0, pos.y, pos.z + 1.0],
                [pos.x, pos.y, pos.z + 1.0],
            ],
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            [0.0, -1.0, 0.0],
        ),
        Face::Left => (
            [
                [pos.x, pos.y, pos.z],
                [pos.x, pos.y, pos.z + 1.0],
                [pos.x, pos.y + 1.0, pos.z + 1.0],
                [pos.x, pos.y + 1.0, pos.z],
            ],
            [[1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]],
            [-1.0, 0.0, 0.0],
        ),
        Face::Right => (
            [
                [pos.x + 1.0, pos.y, pos.z + 1.0],
                [pos.x + 1.0, pos.y, pos.z],
                [pos.x + 1.0, pos.y + 1.0, pos.z],
                [pos.x + 1.0, pos.y + 1.0, pos.z + 1.0],
            ],
            [[1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]],
            [1.0, 0.0, 0.0],
        ),
        Face::Front => (
            [
                [pos.x, pos.y, pos.z + 1.0],
                [pos.x + 1.0, pos.y, pos.z + 1.0],
                [pos.x + 1.0, pos.y + 1.0, pos.z + 1.0],
                [pos.x, pos.y + 1.0, pos.z + 1.0],
            ],
            [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
            [0.0, 0.0, 1.0],
        ),
        Face::Back => (
            [
                [pos.x + 1.0, pos.y, pos.z],
                [pos.x, pos.y, pos.z],
                [pos.x, pos.y + 1.0, pos.z],
                [pos.x + 1.0, pos.y + 1.0, pos.z],
            ],
            [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
            [0.0, 0.0, -1.0],
        ),
    };

    for i in 0..4 {
        vertices.push(Vertex {
            position: positions[i],
            uv: uvs[i],
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
