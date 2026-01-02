//! CLI tool to generate simple 16x16 voxel textures using noise and patterns
//!
//! Usage: cargo run --bin generate-textures

use image::{Rgba, RgbaImage};
use noise::{NoiseFn, Perlin, Simplex};
use rand::Rng;
use std::fs;
use std::path::Path;

const TEXTURE_SIZE: u32 = 16;

fn main() {
    let textures_dir = Path::new("assets/textures");
    fs::create_dir_all(textures_dir).expect("Failed to create textures directory");

    println!("Generating textures in {:?}", textures_dir);

    // Generate various block textures
    generate_texture(textures_dir, "grass_top", |x, y| {
        grass_top(x, y)
    });

    generate_texture(textures_dir, "grass_side", |x, y| {
        grass_side(x, y)
    });

    generate_texture(textures_dir, "dirt", |x, y| {
        dirt(x, y)
    });

    generate_texture(textures_dir, "stone", |x, y| {
        stone(x, y)
    });

    generate_texture(textures_dir, "cobblestone", |x, y| {
        cobblestone(x, y)
    });

    generate_texture(textures_dir, "sand", |x, y| {
        sand(x, y)
    });

    generate_texture(textures_dir, "wood_side", |x, y| {
        wood_side(x, y)
    });

    generate_texture(textures_dir, "wood_top", |x, y| {
        wood_top(x, y)
    });

    generate_texture(textures_dir, "leaves", |x, y| {
        leaves(x, y)
    });

    generate_texture(textures_dir, "brick", |x, y| {
        brick(x, y)
    });

    generate_texture(textures_dir, "water", |x, y| {
        water(x, y)
    });

    // Generate the textures.toml config
    generate_config(textures_dir);

    println!("Done! Generated textures and config.");
}

fn generate_texture<F>(dir: &Path, name: &str, pixel_fn: F)
where
    F: Fn(u32, u32) -> Rgba<u8>,
{
    let mut img = RgbaImage::new(TEXTURE_SIZE, TEXTURE_SIZE);

    for y in 0..TEXTURE_SIZE {
        for x in 0..TEXTURE_SIZE {
            img.put_pixel(x, y, pixel_fn(x, y));
        }
    }

    let path = dir.join(format!("{}.png", name));
    img.save(&path).expect("Failed to save texture");
    println!("  Generated: {}", name);
}

// Helper to convert HSV to RGB
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = match (h as u32) % 360 {
        0..=59 => (c, x, 0.0),
        60..=119 => (x, c, 0.0),
        120..=179 => (0.0, c, x),
        180..=239 => (0.0, x, c),
        240..=299 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    (
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}

fn noise_at(noise: &impl NoiseFn<f64, 2>, x: u32, y: u32, scale: f64) -> f64 {
    noise.get([x as f64 * scale, y as f64 * scale])
}

// ---- Texture generators ----

fn grass_top(x: u32, y: u32) -> Rgba<u8> {
    let perlin = Perlin::new(42);
    let n = noise_at(&perlin, x, y, 0.3);

    // Green hue with variation
    let hue = 100.0 + (n * 20.0) as f32;
    let sat = 0.6 + (n * 0.2) as f32;
    let val = 0.5 + (n * 0.15) as f32;

    let (r, g, b) = hsv_to_rgb(hue, sat.clamp(0.0, 1.0), val.clamp(0.0, 1.0));
    Rgba([r, g, b, 255])
}

fn grass_side(x: u32, y: u32) -> Rgba<u8> {
    if y < 4 {
        // Top part is grass
        grass_top(x, y)
    } else {
        // Bottom part is dirt
        dirt(x, y)
    }
}

fn dirt(x: u32, y: u32) -> Rgba<u8> {
    let perlin = Perlin::new(123);
    let n = noise_at(&perlin, x, y, 0.4);

    // Brown hue
    let hue = 25.0 + (n * 10.0) as f32;
    let sat = 0.5 + (n * 0.15) as f32;
    let val = 0.35 + (n * 0.1) as f32;

    let (r, g, b) = hsv_to_rgb(hue, sat.clamp(0.0, 1.0), val.clamp(0.0, 1.0));
    Rgba([r, g, b, 255])
}

fn stone(x: u32, y: u32) -> Rgba<u8> {
    let perlin = Perlin::new(456);
    let simplex = Simplex::new(789);

    let n1 = noise_at(&perlin, x, y, 0.3);
    let n2 = noise_at(&simplex, x, y, 0.5);
    let n = (n1 + n2 * 0.5) / 1.5;

    // Gray with slight variation
    let val = 0.4 + (n * 0.15) as f32;
    let v = (val.clamp(0.3, 0.55) * 255.0) as u8;

    Rgba([v, v, (v as f32 * 0.95) as u8, 255])
}

fn cobblestone(x: u32, y: u32) -> Rgba<u8> {
    let perlin = Perlin::new(321);
    let n = noise_at(&perlin, x, y, 0.6);

    // Create a more blocky pattern
    let block_n = ((n * 3.0).floor() / 3.0) as f32;
    let val = 0.35 + block_n * 0.2;
    let v = (val.clamp(0.25, 0.55) * 255.0) as u8;

    // Add some edge darkening for depth
    let edge = ((x % 4 == 0) || (y % 4 == 0)) as u8 * 15;

    Rgba([v.saturating_sub(edge), v.saturating_sub(edge), v.saturating_sub(edge + 5), 255])
}

fn sand(x: u32, y: u32) -> Rgba<u8> {
    let perlin = Perlin::new(555);
    let n = noise_at(&perlin, x, y, 0.35);

    // Sandy yellow
    let hue = 45.0 + (n * 8.0) as f32;
    let sat = 0.35 + (n * 0.1) as f32;
    let val = 0.75 + (n * 0.1) as f32;

    let (r, g, b) = hsv_to_rgb(hue, sat.clamp(0.0, 1.0), val.clamp(0.0, 1.0));
    Rgba([r, g, b, 255])
}

fn wood_side(x: u32, y: u32) -> Rgba<u8> {
    let perlin = Perlin::new(777);

    // Vertical wood grain
    let grain = ((x as f64 * 0.8).sin() * 0.5 + 0.5) as f32;
    let n = noise_at(&perlin, x, y, 0.2) as f32;

    let hue = 25.0 + n * 5.0;
    let sat = 0.55 + grain * 0.1;
    let val = 0.35 + grain * 0.15 + n * 0.05;

    let (r, g, b) = hsv_to_rgb(hue, sat.clamp(0.0, 1.0), val.clamp(0.0, 1.0));
    Rgba([r, g, b, 255])
}

fn wood_top(x: u32, y: u32) -> Rgba<u8> {
    let cx = TEXTURE_SIZE as f32 / 2.0;
    let cy = TEXTURE_SIZE as f32 / 2.0;
    let dx = x as f32 - cx;
    let dy = y as f32 - cy;
    let dist = (dx * dx + dy * dy).sqrt();

    let perlin = Perlin::new(888);
    let n = noise_at(&perlin, x, y, 0.3) as f32;

    // Tree rings
    let ring = ((dist * 0.8 + n * 2.0).sin() * 0.5 + 0.5) * 0.15;

    let hue = 28.0;
    let sat = 0.5;
    let val = 0.4 + ring + n * 0.05;

    let (r, g, b) = hsv_to_rgb(hue, sat, val.clamp(0.0, 1.0));
    Rgba([r, g, b, 255])
}

fn leaves(x: u32, y: u32) -> Rgba<u8> {
    let perlin = Perlin::new(999);
    let n = noise_at(&perlin, x, y, 0.5);

    // Some pixels are transparent for a leafy look
    let mut rng = rand::thread_rng();
    let alpha = if rng.gen::<f32>() < 0.15 { 0 } else { 255 };

    let hue = 110.0 + (n * 25.0) as f32;
    let sat = 0.55 + (n * 0.2) as f32;
    let val = 0.4 + (n * 0.2) as f32;

    let (r, g, b) = hsv_to_rgb(hue, sat.clamp(0.0, 1.0), val.clamp(0.0, 1.0));
    Rgba([r, g, b, alpha])
}

fn brick(x: u32, y: u32) -> Rgba<u8> {
    let perlin = Perlin::new(111);
    let n = noise_at(&perlin, x, y, 0.25) as f32;

    // Brick pattern: offset every other row
    let row = y / 4;
    let offset = if row % 2 == 0 { 0 } else { 4 };
    let bx = (x + offset) % 8;

    // Mortar lines
    let is_mortar = (y % 4 == 0) || (bx == 0);

    if is_mortar {
        // Gray mortar
        let v = (0.6 + n * 0.1).clamp(0.0, 1.0);
        let gray = (v * 255.0) as u8;
        Rgba([gray, gray, gray, 255])
    } else {
        // Red brick with variation
        let hue = 8.0 + n * 8.0;
        let sat = 0.65 + n * 0.1;
        let val = 0.45 + n * 0.1;

        let (r, g, b) = hsv_to_rgb(hue, sat.clamp(0.0, 1.0), val.clamp(0.0, 1.0));
        Rgba([r, g, b, 255])
    }
}

fn water(x: u32, y: u32) -> Rgba<u8> {
    let simplex = Simplex::new(222);
    let n = noise_at(&simplex, x, y, 0.4);

    let hue = 200.0 + (n * 15.0) as f32;
    let sat = 0.6;
    let val = 0.5 + (n * 0.15) as f32;

    let (r, g, b) = hsv_to_rgb(hue, sat, val.clamp(0.0, 1.0));
    Rgba([r, g, b, 180]) // Semi-transparent
}

fn generate_config(dir: &Path) {
    let config = r#"# Texture configuration
# Maps block types to their texture files

[blocks]

[blocks.grass]
top = "grass_top.png"
bottom = "dirt.png"
sides = "grass_side.png"

[blocks.dirt]
all = "dirt.png"

[blocks.stone]
all = "stone.png"

[blocks.cobblestone]
all = "cobblestone.png"

[blocks.sand]
all = "sand.png"

[blocks.wood]
top = "wood_top.png"
bottom = "wood_top.png"
sides = "wood_side.png"

[blocks.leaves]
all = "leaves.png"

[blocks.brick]
all = "brick.png"

[blocks.water]
all = "water.png"
"#;

    let config_path = dir.join("textures.toml");
    fs::write(&config_path, config).expect("Failed to write config");
    println!("  Generated: textures.toml");
}
