# Voxel Rust

A Minecraft-inspired voxel engine written in Rust using wgpu for GPU-accelerated rendering.

## Features

- **Procedural World Generation** - Infinite terrain with multiple biomes, cave systems, and ravines
- **Biome System** - Plains, Forest, Desert, Taiga, Tundra, Mountains, and Beach with smooth blending
- **Cave Generation** - Three cave types: worm caves (tunnels), cheese caves (caverns), and ravines
- **Structure Generation** - Oak trees, spruce trees, cacti, and vegetation
- **Greedy Meshing** - Optimized mesh generation for efficient GPU rendering
- **Dynamic Lighting** - Day/night cycle with sun movement and shadow mapping
- **Ambient Occlusion** - Per-vertex AO for realistic shadowing

## Screenshots

*Coming soon*

## Requirements

- Rust 1.70+
- Graphics driver supporting Metal (macOS), Vulkan (Linux), or DirectX 12 (Windows)

## Building and Running

```bash
# Build and run in release mode (recommended)
cargo run --release

# With embedded assets (standalone binary)
cargo run --release --features embed-assets

# Development with logging
RUST_LOG=info cargo run
```

## Controls

| Key | Action |
|-----|--------|
| W/A/S/D | Move forward/left/backward/right |
| Mouse | Look around |

FPS and time of day are displayed in the window title.

## Configuration

Edit `assets/config.toml` to customize:

```toml
[rendering]
render_distance = 10    # Chunks to render (16 blocks each)
height_chunks = 16      # Vertical chunks

[terrain]
seed = 42               # World generation seed
sea_level = 32
terrain_height = 64.0

[camera]
fov = 70.0
movement_speed = 20.0
mouse_sensitivity = 0.003
```

Biomes can be configured in `assets/biomes.toml`.

## Project Structure

```
src/
├── main.rs       # Application entry point and event loop
├── renderer.rs   # wgpu rendering pipeline and shaders
├── voxel.rs      # Block types, chunks, and mesh generation
├── worldgen.rs   # Procedural terrain and cave generation
├── camera.rs     # Camera controller and frustum culling
├── config.rs     # Configuration loading
└── texture.rs    # Texture atlas management
```

## Technical Details

- **Graphics API**: wgpu (Metal/Vulkan/DX12 abstraction)
- **Window Management**: winit
- **Math**: glam
- **Noise Functions**: noise crate (Perlin, Simplex)
- **Parallelism**: rayon for chunk generation

## License

MIT
