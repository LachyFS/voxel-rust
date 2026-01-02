use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

pub const TEXTURE_SIZE: u32 = 16;

#[derive(Debug, Deserialize)]
struct TextureConfig {
    blocks: HashMap<String, BlockTextures>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum BlockTextures {
    Uniform {
        all: String,
    },
    PerFace {
        top: String,
        bottom: String,
        sides: String,
    },
}

/// Texture indices for each face of a block
#[derive(Clone, Copy, Debug, Default)]
pub struct BlockTextureIndices {
    pub top: u32,
    pub bottom: u32,
    pub sides: u32, // Used for left, right, front, back
}

/// Fast texture lookup array indexed by BlockType discriminant
/// Avoids HashMap lookups during mesh generation
pub type BlockTextureArray = [BlockTextureIndices; 10]; // BlockType::COUNT

/// Manages texture loading and the texture array
pub struct TextureManager {
    /// Maps texture filename to layer index
    texture_indices: HashMap<String, u32>,
    /// Maps block name to its texture indices
    block_textures: HashMap<String, BlockTextureIndices>,
    /// Raw RGBA data for all textures (concatenated) - dropped after GPU upload
    texture_data: Option<Vec<u8>>,
    /// Number of texture layers
    pub layer_count: u32,
}

impl TextureManager {
    pub fn load(textures_dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let config_path = textures_dir.join("textures.toml");
        let config_str = fs::read_to_string(&config_path)?;
        let config: TextureConfig = toml::from_str(&config_str)?;

        let mut texture_indices: HashMap<String, u32> = HashMap::new();
        let mut texture_data: Vec<u8> = Vec::new();
        let mut current_index: u32 = 0;

        // First pass: collect all unique texture files using HashSet for O(1) lookup
        let mut unique_files_set: HashSet<String> = HashSet::new();
        for block_tex in config.blocks.values() {
            match block_tex {
                BlockTextures::Uniform { all } => {
                    unique_files_set.insert(all.clone());
                }
                BlockTextures::PerFace { top, bottom, sides } => {
                    unique_files_set.insert(top.clone());
                    unique_files_set.insert(bottom.clone());
                    unique_files_set.insert(sides.clone());
                }
            };
        }
        let unique_files: Vec<String> = unique_files_set.into_iter().collect();

        // Load each unique texture
        for filename in &unique_files {
            let path = textures_dir.join(filename);
            let img = image::open(&path)?;
            let rgba = img.to_rgba8();

            if img.width() != TEXTURE_SIZE || img.height() != TEXTURE_SIZE {
                return Err(format!(
                    "Texture {} is {}x{}, expected {}x{}",
                    filename,
                    img.width(),
                    img.height(),
                    TEXTURE_SIZE,
                    TEXTURE_SIZE
                )
                .into());
            }

            texture_data.extend_from_slice(&rgba);
            texture_indices.insert(filename.clone(), current_index);
            current_index += 1;
        }

        // Build block texture mappings
        let mut block_textures: HashMap<String, BlockTextureIndices> = HashMap::new();

        for (block_name, block_tex) in &config.blocks {
            let indices = match block_tex {
                BlockTextures::Uniform { all } => {
                    let idx = texture_indices[all];
                    BlockTextureIndices {
                        top: idx,
                        bottom: idx,
                        sides: idx,
                    }
                }
                BlockTextures::PerFace { top, bottom, sides } => BlockTextureIndices {
                    top: texture_indices[top],
                    bottom: texture_indices[bottom],
                    sides: texture_indices[sides],
                },
            };
            block_textures.insert(block_name.clone(), indices);
        }

        log::info!(
            "Loaded {} textures for {} block types",
            current_index,
            block_textures.len()
        );

        Ok(Self {
            texture_indices,
            block_textures,
            texture_data: Some(texture_data),
            layer_count: current_index,
        })
    }

    pub fn get_block_textures(&self, block_name: &str) -> Option<BlockTextureIndices> {
        self.block_textures.get(block_name).copied()
    }

    /// Create a fast lookup array indexed by BlockType discriminant
    /// This replaces HashMap lookups with O(1) array indexing during mesh generation
    pub fn create_texture_array_lookup(&self) -> BlockTextureArray {
        use crate::voxel::BlockType;

        let mut array = [BlockTextureIndices::default(); 10];

        // Map each block type to its texture indices
        let mappings = [
            (BlockType::Air, "air"),
            (BlockType::Grass, "grass"),
            (BlockType::Dirt, "dirt"),
            (BlockType::Stone, "stone"),
            (BlockType::Cobblestone, "cobblestone"),
            (BlockType::Sand, "sand"),
            (BlockType::Wood, "wood"),
            (BlockType::Leaves, "leaves"),
            (BlockType::Brick, "brick"),
            (BlockType::Water, "water"),
        ];

        for (block_type, name) in mappings {
            if let Some(indices) = self.block_textures.get(name) {
                array[block_type.as_index()] = *indices;
            }
        }

        array
    }

    pub fn texture_data(&self) -> &[u8] {
        self.texture_data.as_ref().expect("Texture data already consumed")
    }

    /// Create the wgpu texture array and drop CPU-side texture data to free memory
    pub fn create_texture_array(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> (wgpu::Texture, wgpu::TextureView, wgpu::Sampler) {
        let texture_data = self.texture_data.take().expect("Texture data already consumed");

        let size = wgpu::Extent3d {
            width: TEXTURE_SIZE,
            height: TEXTURE_SIZE,
            depth_or_array_layers: self.layer_count,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Block Texture Array"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Upload each layer
        let bytes_per_texture = (TEXTURE_SIZE * TEXTURE_SIZE * 4) as usize;
        for layer in 0..self.layer_count {
            let offset = layer as usize * bytes_per_texture;
            let layer_data = &texture_data[offset..offset + bytes_per_texture];

            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: layer,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                layer_data,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(TEXTURE_SIZE * 4),
                    rows_per_image: Some(TEXTURE_SIZE),
                },
                wgpu::Extent3d {
                    width: TEXTURE_SIZE,
                    height: TEXTURE_SIZE,
                    depth_or_array_layers: 1,
                },
            );
        }

        // texture_data is dropped here, freeing CPU memory
        log::debug!("Texture data uploaded to GPU and CPU memory freed ({} bytes)", texture_data.len());

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Block Texture Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest, // Pixelated look
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        (texture, view, sampler)
    }
}
