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
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) world_position: vec3<f32>,
    @location(3) @interpolate(flat) tex_layer: u32,
};

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);
    out.uv = model.uv;
    out.normal = model.normal;
    out.world_position = model.position;
    out.tex_layer = model.tex_layer;
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

    // Simple directional lighting
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let ambient = 0.4;
    let diffuse = max(dot(in.normal, light_dir), 0.0);
    let lighting = ambient + diffuse * 0.6;

    return vec4<f32>(tex_color.rgb * lighting, tex_color.a);
}
