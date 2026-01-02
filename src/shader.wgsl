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
    return out;
}

// Fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample from texture array using layer index
    let tex_color = textureSample(block_textures, block_sampler, in.uv, in.tex_layer);

    // Discard fully transparent pixels (for leaves, etc.)
    if tex_color.a < 0.1 {
        discard;
    }

    // Directional lighting
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let ambient = 0.5;
    let diffuse = max(dot(in.normal, light_dir), 0.0) * 0.5;

    // Face-based shading: different brightness per face direction
    // Top faces are brightest, bottom faces are darkest
    var face_shade = 1.0;
    if in.normal.y > 0.5 {
        face_shade = 1.0;       // Top: full brightness
    } else if in.normal.y < -0.5 {
        face_shade = 0.5;       // Bottom: darker
    } else {
        // Side faces: vary by direction for visual interest
        if abs(in.normal.x) > 0.5 {
            face_shade = 0.8;   // East/West sides
        } else {
            face_shade = 0.7;   // North/South sides
        }
    }

    // Combine lighting: ambient + directional, modulated by face shade and AO
    let lighting = (ambient + diffuse) * face_shade * in.ao;

    return vec4<f32>(tex_color.rgb * lighting, tex_color.a);
}
