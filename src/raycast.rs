use glam::Vec3;

use crate::voxel::{BlockType, CHUNK_SIZE};
use crate::worldgen::World;

/// The face of a block that was hit by a raycast
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockFace {
    Top,    // +Y
    Bottom, // -Y
    North,  // -Z
    South,  // +Z
    East,   // +X
    West,   // -X
}

impl BlockFace {
    /// Get the normal vector for this face
    pub fn normal(&self) -> Vec3 {
        match self {
            BlockFace::Top => Vec3::Y,
            BlockFace::Bottom => Vec3::NEG_Y,
            BlockFace::North => Vec3::NEG_Z,
            BlockFace::South => Vec3::Z,
            BlockFace::East => Vec3::X,
            BlockFace::West => Vec3::NEG_X,
        }
    }
}

/// Result of a successful raycast hit
#[derive(Debug, Clone, Copy)]
pub struct RaycastHit {
    /// World position of the block that was hit
    pub block_pos: [i32; 3],
    /// The face of the block that was hit
    pub face: BlockFace,
    /// The exact world position where the ray hit
    pub hit_point: Vec3,
    /// Distance from ray origin to hit point
    pub distance: f32,
    /// The block type that was hit
    pub block_type: BlockType,
}

impl RaycastHit {
    /// Get the position where a new block should be placed (adjacent to hit face)
    pub fn placement_pos(&self) -> [i32; 3] {
        let normal = self.face.normal();
        [
            self.block_pos[0] + normal.x as i32,
            self.block_pos[1] + normal.y as i32,
            self.block_pos[2] + normal.z as i32,
        ]
    }
}

/// Performs a raycast through the voxel world using a DDA-style algorithm
///
/// This uses a 3D-DDA (Digital Differential Analyzer) algorithm for efficient
/// voxel traversal. It steps through exactly one block at a time in the optimal order.
///
/// # Arguments
/// * `world` - The voxel world to raycast against
/// * `origin` - Starting point of the ray (usually camera position)
/// * `direction` - Direction of the ray (should be normalized)
/// * `max_distance` - Maximum distance to check (in world units)
///
/// # Returns
/// `Some(RaycastHit)` if a solid block was hit, `None` otherwise
pub fn raycast(
    world: &World,
    origin: Vec3,
    direction: Vec3,
    max_distance: f32,
) -> Option<RaycastHit> {
    // Normalize direction
    let dir = direction.normalize();

    // Current voxel position (floored to get block coordinates)
    let mut pos = [
        origin.x.floor() as i32,
        origin.y.floor() as i32,
        origin.z.floor() as i32,
    ];

    // Direction signs for stepping
    let step = [
        if dir.x >= 0.0 { 1i32 } else { -1 },
        if dir.y >= 0.0 { 1i32 } else { -1 },
        if dir.z >= 0.0 { 1i32 } else { -1 },
    ];

    // Distance along ray to cross one voxel boundary in each direction
    let t_delta = [
        if dir.x.abs() < 1e-10 { f32::MAX } else { (1.0 / dir.x).abs() },
        if dir.y.abs() < 1e-10 { f32::MAX } else { (1.0 / dir.y).abs() },
        if dir.z.abs() < 1e-10 { f32::MAX } else { (1.0 / dir.z).abs() },
    ];

    // Distance to next voxel boundary in each direction
    let mut t_max = [
        if dir.x >= 0.0 {
            ((pos[0] + 1) as f32 - origin.x) * t_delta[0]
        } else {
            (origin.x - pos[0] as f32) * t_delta[0]
        },
        if dir.y >= 0.0 {
            ((pos[1] + 1) as f32 - origin.y) * t_delta[1]
        } else {
            (origin.y - pos[1] as f32) * t_delta[1]
        },
        if dir.z >= 0.0 {
            ((pos[2] + 1) as f32 - origin.z) * t_delta[2]
        } else {
            (origin.z - pos[2] as f32) * t_delta[2]
        },
    ];

    // Track which face we entered from (for the first block, use closest face)
    let mut last_face = BlockFace::Top;
    let mut t = 0.0f32;

    // DDA traversal
    while t < max_distance {
        // Check if current voxel is solid
        if let Some(block_type) = get_block_at_world_pos(world, pos) {
            if block_type.is_solid() && !block_type.is_foliage() {
                let hit_point = origin + dir * t;
                return Some(RaycastHit {
                    block_pos: pos,
                    face: last_face,
                    hit_point,
                    distance: t,
                    block_type,
                });
            }
        }

        // Step to next voxel (choose axis with smallest t_max)
        if t_max[0] < t_max[1] {
            if t_max[0] < t_max[2] {
                // Step X
                t = t_max[0];
                t_max[0] += t_delta[0];
                pos[0] += step[0];
                last_face = if step[0] > 0 { BlockFace::West } else { BlockFace::East };
            } else {
                // Step Z
                t = t_max[2];
                t_max[2] += t_delta[2];
                pos[2] += step[2];
                last_face = if step[2] > 0 { BlockFace::North } else { BlockFace::South };
            }
        } else {
            if t_max[1] < t_max[2] {
                // Step Y
                t = t_max[1];
                t_max[1] += t_delta[1];
                pos[1] += step[1];
                last_face = if step[1] > 0 { BlockFace::Bottom } else { BlockFace::Top };
            } else {
                // Step Z
                t = t_max[2];
                t_max[2] += t_delta[2];
                pos[2] += step[2];
                last_face = if step[2] > 0 { BlockFace::North } else { BlockFace::South };
            }
        }
    }

    None
}

/// Get block at world position by looking up the appropriate chunk
fn get_block_at_world_pos(world: &World, pos: [i32; 3]) -> Option<BlockType> {
    let chunk_size = CHUNK_SIZE as i32;

    // Convert world position to chunk position
    let chunk_pos = (
        pos[0].div_euclid(chunk_size),
        pos[1].div_euclid(chunk_size),
        pos[2].div_euclid(chunk_size),
    );

    // Get the chunk
    let chunk = world.get_chunk(chunk_pos)?;

    // Convert to local coordinates within the chunk
    let local_x = pos[0].rem_euclid(chunk_size) as usize;
    let local_y = pos[1].rem_euclid(chunk_size) as usize;
    let local_z = pos[2].rem_euclid(chunk_size) as usize;

    Some(chunk.get_block(local_x, local_y, local_z))
}

/// Get the camera's look direction from yaw and pitch
pub fn get_camera_direction(yaw: f32, pitch: f32) -> Vec3 {
    Vec3::new(
        yaw.cos() * pitch.cos(),
        pitch.sin(),
        yaw.sin() * pitch.cos(),
    ).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_face_normal() {
        assert_eq!(BlockFace::Top.normal(), Vec3::Y);
        assert_eq!(BlockFace::Bottom.normal(), Vec3::NEG_Y);
        assert_eq!(BlockFace::East.normal(), Vec3::X);
        assert_eq!(BlockFace::West.normal(), Vec3::NEG_X);
        assert_eq!(BlockFace::South.normal(), Vec3::Z);
        assert_eq!(BlockFace::North.normal(), Vec3::NEG_Z);
    }
}
