// Vertex shader

struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// Texture array and sampler
@group(1) @binding(0)
var block_textures: texture_2d_array<f32>;
@group(1) @binding(1)
var block_sampler: sampler;

// Light uniform
struct LightUniform {
    light_space_matrix: mat4x4<f32>,
    sun_direction: vec3<f32>,
    _pad1: f32,
    sun_color: vec3<f32>,
    _pad2: f32,
    ambient_color: vec3<f32>,
    time_of_day: f32,
};

@group(2) @binding(0)
var<uniform> light: LightUniform;

// Shadow map
@group(3) @binding(0)
var shadow_map: texture_depth_2d;
@group(3) @binding(1)
var shadow_sampler: sampler_comparison;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tex_layer: u32,
    @location(4) ao: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) world_position: vec3<f32>,
    @location(3) @interpolate(flat) tex_layer: u32,
    @location(4) ao: f32,
    @location(5) light_space_position: vec4<f32>,
};

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);
    out.uv = model.uv;
    out.normal = model.normal;
    out.world_position = model.position;
    out.tex_layer = model.tex_layer;
    out.ao = model.ao;
    // Transform position to light space for shadow mapping
    out.light_space_position = light.light_space_matrix * vec4<f32>(model.position, 1.0);
    return out;
}

// Fragment shader

// Calculate shadow factor using PCF (Percentage Closer Filtering)
fn calculate_shadow(light_space_pos: vec4<f32>, normal: vec3<f32>) -> f32 {
    // Perform perspective divide
    let proj_coords = light_space_pos.xyz / light_space_pos.w;

    // Transform to [0,1] range for texture sampling
    let shadow_coords = vec2<f32>(
        proj_coords.x * 0.5 + 0.5,
        -proj_coords.y * 0.5 + 0.5  // Flip Y for texture coordinates
    );

    // Check if outside shadow map
    if shadow_coords.x < 0.0 || shadow_coords.x > 1.0 ||
       shadow_coords.y < 0.0 || shadow_coords.y > 1.0 ||
       proj_coords.z < 0.0 || proj_coords.z > 1.0 {
        return 1.0; // No shadow outside the shadow map
    }

    // Current depth from light's perspective
    let current_depth = proj_coords.z;

    // Bias based on surface angle to light (reduces shadow acne)
    let sun_dir = normalize(light.sun_direction);
    let bias = max(0.002 * (1.0 - dot(normal, sun_dir)), 0.001);

    // PCF: sample multiple points for softer shadows
    var shadow = 0.0;
    let texel_size = 1.0 / 2048.0; // Shadow map size

    // 3x3 PCF kernel
    for (var x = -1; x <= 1; x++) {
        for (var y = -1; y <= 1; y++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            shadow += textureSampleCompare(
                shadow_map,
                shadow_sampler,
                shadow_coords + offset,
                current_depth - bias
            );
        }
    }
    shadow /= 9.0;

    return shadow;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample from texture array using layer index
    let tex_color = textureSample(block_textures, block_sampler, in.uv, in.tex_layer);

    // Discard transparent pixels using alpha testing
    // Using 0.5 threshold for clean cutout rendering of leaves, etc.
    // This avoids transparency sorting issues that cause the "xray" look
    if tex_color.a < 0.5 {
        discard;
    }

    // Normalize sun direction
    let sun_dir = normalize(light.sun_direction);

    // Directional lighting from sun
    let ndotl = max(dot(in.normal, sun_dir), 0.0);
    let diffuse = ndotl * light.sun_color;

    // Calculate shadow (only when sun is up)
    var shadow = 1.0;
    if light.sun_direction.y > -0.1 {
        shadow = calculate_shadow(in.light_space_position, in.normal);
    }

    // Face-based shading: different brightness per face direction
    // Top faces are brightest, bottom faces are darkest
    var face_shade = 1.0;
    if in.normal.y > 0.5 {
        face_shade = 1.0;       // Top: full brightness
    } else if in.normal.y < -0.5 {
        face_shade = 0.6;       // Bottom: darker
    } else {
        // Side faces: vary by direction for visual interest
        if abs(in.normal.x) > 0.5 {
            face_shade = 0.85;   // East/West sides
        } else {
            face_shade = 0.75;   // North/South sides
        }
    }

    // Combine all lighting components
    // Ambient provides base illumination (even in shadow)
    // Diffuse provides directional sunlight
    // Shadow modulates the diffuse component
    // Face shade and AO provide local detail
    let ambient = light.ambient_color;
    let lit_color = ambient + diffuse * shadow;
    let final_lighting = lit_color * face_shade * in.ao;

    return vec4<f32>(tex_color.rgb * final_lighting, tex_color.a);
}
