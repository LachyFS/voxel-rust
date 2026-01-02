//! Embedded assets for standalone release builds
//!
//! This module embeds all game assets (configs, textures) directly into the binary
//! at compile time using `include_str!` and `include_bytes!`.
//!
//! Enable the `embed-assets` feature to use embedded assets instead of loading from disk.

// Config files embedded as strings
pub const CONFIG_TOML: &str = include_str!("../assets/config.toml");
pub const BIOMES_TOML: &str = include_str!("../assets/biomes.toml");
pub const TEXTURES_TOML: &str = include_str!("../assets/textures/textures.toml");

// Texture files embedded as byte arrays
// Each texture is a 16x16 PNG file
pub struct EmbeddedTexture {
    pub name: &'static str,
    pub data: &'static [u8],
}

pub const EMBEDDED_TEXTURES: &[EmbeddedTexture] = &[
    EmbeddedTexture {
        name: "grass_top.png",
        data: include_bytes!("../assets/textures/grass_top.png"),
    },
    EmbeddedTexture {
        name: "grass_side.png",
        data: include_bytes!("../assets/textures/grass_side.png"),
    },
    EmbeddedTexture {
        name: "dirt.png",
        data: include_bytes!("../assets/textures/dirt.png"),
    },
    EmbeddedTexture {
        name: "stone.png",
        data: include_bytes!("../assets/textures/stone.png"),
    },
    EmbeddedTexture {
        name: "cobblestone.png",
        data: include_bytes!("../assets/textures/cobblestone.png"),
    },
    EmbeddedTexture {
        name: "sand.png",
        data: include_bytes!("../assets/textures/sand.png"),
    },
    EmbeddedTexture {
        name: "wood_side.png",
        data: include_bytes!("../assets/textures/wood_side.png"),
    },
    EmbeddedTexture {
        name: "wood_top.png",
        data: include_bytes!("../assets/textures/wood_top.png"),
    },
    EmbeddedTexture {
        name: "leaves.png",
        data: include_bytes!("../assets/textures/leaves.png"),
    },
    EmbeddedTexture {
        name: "brick.png",
        data: include_bytes!("../assets/textures/brick.png"),
    },
    EmbeddedTexture {
        name: "water.png",
        data: include_bytes!("../assets/textures/water.png"),
    },
    EmbeddedTexture {
        name: "snow.png",
        data: include_bytes!("../assets/textures/snow.png"),
    },
    EmbeddedTexture {
        name: "ice.png",
        data: include_bytes!("../assets/textures/ice.png"),
    },
    EmbeddedTexture {
        name: "gravel.png",
        data: include_bytes!("../assets/textures/gravel.png"),
    },
    EmbeddedTexture {
        name: "clay.png",
        data: include_bytes!("../assets/textures/clay.png"),
    },
    EmbeddedTexture {
        name: "cactus_side.png",
        data: include_bytes!("../assets/textures/cactus_side.png"),
    },
    EmbeddedTexture {
        name: "cactus_top.png",
        data: include_bytes!("../assets/textures/cactus_top.png"),
    },
    EmbeddedTexture {
        name: "dead_bush.png",
        data: include_bytes!("../assets/textures/dead_bush.png"),
    },
    EmbeddedTexture {
        name: "tall_grass.png",
        data: include_bytes!("../assets/textures/tall_grass.png"),
    },
    EmbeddedTexture {
        name: "podzol_top.png",
        data: include_bytes!("../assets/textures/podzol_top.png"),
    },
    EmbeddedTexture {
        name: "podzol_side.png",
        data: include_bytes!("../assets/textures/podzol_side.png"),
    },
    EmbeddedTexture {
        name: "spruce_wood_side.png",
        data: include_bytes!("../assets/textures/spruce_wood_side.png"),
    },
    EmbeddedTexture {
        name: "spruce_wood_top.png",
        data: include_bytes!("../assets/textures/spruce_wood_top.png"),
    },
    EmbeddedTexture {
        name: "spruce_leaves.png",
        data: include_bytes!("../assets/textures/spruce_leaves.png"),
    },
];
