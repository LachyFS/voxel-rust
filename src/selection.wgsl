// Selection box shader for Minecraft-style block highlighting
// Renders a wireframe bounding box around the selected block

struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct SelectionUniform {
    // Block position (world coordinates)
    block_pos: vec3<f32>,
    // Line thickness (unused in line rendering, kept for compatibility)
    _padding: f32,
    // Color of the selection box (RGB + alpha)
    color: vec4<f32>,
};

@group(1) @binding(0)
var<uniform> selection: SelectionUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Scale box slightly larger than 1 voxel to avoid z-fighting
    let scale = 1.002;
    let offset = (scale - 1.0) / 2.0;

    // Scale and translate the unit cube to the block position
    let world_pos = in.position * scale - offset + selection.block_pos;

    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = selection.color;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
