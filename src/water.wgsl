// Water shader with animated waves, transparency, reflections, and caustics

struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// Water-specific uniform
struct WaterUniform {
    time: f32,
    wave_speed: f32,
    wave_height: f32,
    transparency: f32,
    // Water color properties
    shallow_color: vec3<f32>,
    _pad1: f32,
    deep_color: vec3<f32>,
    _pad2: f32,
    // Camera position for reflections
    camera_position: vec3<f32>,
    _pad3: f32,
};

@group(1) @binding(0)
var<uniform> water: WaterUniform;

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

// Texture array and sampler (for water texture/normal map)
@group(3) @binding(0)
var block_textures: texture_2d_array<f32>;
@group(3) @binding(1)
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
    @location(1) world_position: vec3<f32>,
    @location(2) @interpolate(flat) tex_layer: u32,
    @location(3) view_dir: vec3<f32>,
    @location(4) normal: vec3<f32>,
    @location(5) ao: f32, // Water level (0-1) or waterfall indicator (>1)
};

// Simplex noise for wave animation
fn mod289_3(x: vec3<f32>) -> vec3<f32> {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

fn mod289_4(x: vec4<f32>) -> vec4<f32> {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

fn permute(x: vec4<f32>) -> vec4<f32> {
    return mod289_4(((x * 34.0) + 1.0) * x);
}

fn taylor_inv_sqrt(r: vec4<f32>) -> vec4<f32> {
    return 1.79284291400159 - 0.85373472095314 * r;
}

// 2D simplex noise
fn snoise(v: vec2<f32>) -> f32 {
    let C = vec4<f32>(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);

    var i = floor(v + dot(v, C.yy));
    let x0 = v - i + dot(i, C.xx);

    var i1: vec2<f32>;
    if x0.x > x0.y {
        i1 = vec2<f32>(1.0, 0.0);
    } else {
        i1 = vec2<f32>(0.0, 1.0);
    }

    var x12 = x0.xyxy + C.xxzz;
    x12 = vec4<f32>(x12.xy - i1, x12.zw);

    i = i - floor(i * (1.0 / 289.0)) * 289.0;
    let p = permute(permute(i.y + vec4<f32>(0.0, i1.y, 1.0, 0.0)) + i.x + vec4<f32>(0.0, i1.x, 1.0, 0.0));

    var m = max(0.5 - vec4<f32>(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw), 0.0), vec4<f32>(0.0));
    m = m * m;
    m = m * m;

    let x = 2.0 * fract(p * C.wwww) - 1.0;
    let h = abs(x) - 0.5;
    let ox = floor(x + 0.5);
    let a0 = x - ox;

    m = m * (1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h));

    let g = vec3<f32>(
        a0.x * x0.x + h.x * x0.y,
        a0.y * x12.x + h.y * x12.y,
        a0.z * x12.z + h.z * x12.w
    );

    return 130.0 * dot(m.xyz, g);
}

// Fractional Brownian Motion for more natural waves
fn fbm(p: vec2<f32>) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;

    for (var i = 0; i < 4; i++) {
        value += amplitude * snoise(p * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    return value;
}

// Calculate wave displacement
fn wave_displacement(world_pos: vec3<f32>, time: f32) -> f32 {
    let wave_scale = 0.08; // Reduced for subtler waves
    let wave_freq = 0.5;

    // Multiple wave layers for more natural appearance
    let p1 = vec2<f32>(world_pos.x * wave_freq + time * 0.15, world_pos.z * wave_freq + time * 0.1);
    let p2 = vec2<f32>(world_pos.x * wave_freq * 0.7 - time * 0.12, world_pos.z * wave_freq * 0.7 + time * 0.08);

    let wave1 = snoise(p1) * 0.6;
    let wave2 = snoise(p2) * 0.4;

    return (wave1 + wave2) * wave_scale * water.wave_height;
}

// Calculate wave normal from displacement
fn wave_normal(world_pos: vec3<f32>, time: f32) -> vec3<f32> {
    let eps = 0.2;
    let h = wave_displacement(world_pos, time);
    let hx = wave_displacement(world_pos + vec3<f32>(eps, 0.0, 0.0), time);
    let hz = wave_displacement(world_pos + vec3<f32>(0.0, 0.0, eps), time);

    let dx = (hx - h) / eps;
    let dz = (hz - h) / eps;

    // Blend with up vector to reduce extreme normals
    let wave_n = normalize(vec3<f32>(-dx * 0.5, 1.0, -dz * 0.5));
    return wave_n;
}

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    var pos = model.position;

    // Only apply wave displacement to top faces of water
    if model.normal.y > 0.5 {
        // Check if this vertex is at a block edge (fractional part near 0 or 1)
        // Don't displace edge vertices to prevent gaps between water blocks
        let frac_x = fract(pos.x);
        let frac_z = fract(pos.z);
        let edge_threshold = 0.01;
        let is_edge_x = frac_x < edge_threshold || frac_x > (1.0 - edge_threshold);
        let is_edge_z = frac_z < edge_threshold || frac_z > (1.0 - edge_threshold);

        if !is_edge_x && !is_edge_z {
            // Interior vertex - full displacement
            pos.y += wave_displacement(pos, water.time);
        } else if is_edge_x && is_edge_z {
            // Corner vertex - no displacement to anchor corners
            // Keep pos.y unchanged
        } else {
            // Edge vertex - reduced displacement to smooth transition
            pos.y += wave_displacement(pos, water.time) * 0.3;
        }
    }

    out.clip_position = camera.view_proj * vec4<f32>(pos, 1.0);
    out.uv = model.uv;
    out.world_position = pos;
    out.tex_layer = model.tex_layer;
    out.view_dir = normalize(water.camera_position - pos);
    out.normal = model.normal;
    out.ao = model.ao;

    return out;
}

// Fresnel effect for realistic water reflections
fn fresnel(view_dir: vec3<f32>, normal: vec3<f32>) -> f32 {
    let base_reflectivity = 0.02; // Water's base reflectivity
    let cos_theta = max(dot(view_dir, normal), 0.0);
    return base_reflectivity + (1.0 - base_reflectivity) * pow(1.0 - cos_theta, 5.0);
}

// Caustics pattern
fn caustics(uv: vec2<f32>, time: f32) -> f32 {
    let scale = 3.0;
    let speed = 0.5;

    let p1 = uv * scale + vec2<f32>(time * speed, time * speed * 0.7);
    let p2 = uv * scale * 1.3 - vec2<f32>(time * speed * 0.8, time * speed * 0.5);

    let c1 = snoise(p1);
    let c2 = snoise(p2);

    // Create sharp caustic lines
    let caustic = pow(max(c1 + c2, 0.0), 2.0);
    return caustic * 0.3;
}

// Specular highlight
fn specular_highlight(view_dir: vec3<f32>, normal: vec3<f32>, light_dir: vec3<f32>) -> f32 {
    let reflect_dir = reflect(-light_dir, normal);
    let spec = pow(max(dot(view_dir, reflect_dir), 0.0), 64.0);
    return spec;
}

// Waterfall noise pattern
fn waterfall_noise(uv: vec2<f32>, time: f32) -> f32 {
    // Animated vertical streaks
    let scroll_speed = 3.0;
    let scrolled_uv = vec2<f32>(uv.x, uv.y + time * scroll_speed);

    // Multiple octaves of noise for detail
    let noise1 = snoise(scrolled_uv * vec2<f32>(2.0, 1.0)) * 0.5;
    let noise2 = snoise(scrolled_uv * vec2<f32>(4.0, 2.0) + vec2<f32>(100.0, 0.0)) * 0.25;
    let noise3 = snoise(scrolled_uv * vec2<f32>(8.0, 4.0) + vec2<f32>(200.0, 0.0)) * 0.125;

    return 0.5 + noise1 + noise2 + noise3;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Check if this is a waterfall (ao > 1.0)
    let is_waterfall = in.ao > 1.0;

    if is_waterfall {
        // Waterfall rendering
        let fall_height = (in.ao - 1.0) * 16.0; // Decode fall height

        // Animated scrolling effect using UV.y (which encodes vertical position)
        let noise = waterfall_noise(in.uv, water.time);

        // Foam/white water effect - more at top, fading down
        let foam_factor = max(0.0, 1.0 - in.uv.y / fall_height) * 0.5;

        // Base waterfall color - lighter/foamier
        let waterfall_base = mix(water.shallow_color, vec3<f32>(0.7, 0.85, 0.95), foam_factor);

        // Add noise variation
        let varied_color = waterfall_base * (0.8 + noise * 0.4);

        // Lighting
        let sun_dir = normalize(light.sun_direction);
        let ndotl = max(dot(in.normal, sun_dir), 0.0);
        let lit_color = varied_color * (light.ambient_color * 0.9 + light.sun_color * ndotl * 0.3);

        // More transparent at edges, opaque in middle
        let edge_fade = 1.0 - abs(in.uv.x - 0.5) * 2.0;
        let alpha = 0.6 + foam_factor * 0.3 + edge_fade * 0.1;

        return vec4<f32>(lit_color, alpha);
    }

    // Regular water rendering
    // Calculate animated wave normal for top faces
    var normal = in.normal;
    if in.normal.y > 0.5 {
        normal = wave_normal(in.world_position, water.time);
    }

    // Fresnel for reflectivity based on viewing angle
    let fresnel_factor = fresnel(in.view_dir, normal);

    // Sun direction and color
    let sun_dir = normalize(light.sun_direction);
    let sun_color = light.sun_color;

    // Basic lighting - softer
    let ndotl = max(dot(normal, sun_dir), 0.0);
    let diffuse = ndotl * sun_color * 0.3;

    // Specular highlight - much more controlled
    let spec = specular_highlight(in.view_dir, normal, sun_dir);
    let specular_color = sun_color * spec * 0.4; // Reduced intensity

    // Depth-based color (shallow vs deep)
    let depth_factor = 1.0 - fresnel_factor;
    let water_base_color = mix(water.shallow_color, water.deep_color, depth_factor * 0.6);

    // Apply lighting
    let ambient = light.ambient_color * 0.8;
    let lit_color = water_base_color * (ambient + diffuse);

    // Add subtle specular reflection (only when sun is visible)
    var final_color = lit_color;
    if light.sun_direction.y > 0.0 {
        final_color = lit_color + specular_color * fresnel_factor * 0.5;
    }

    // Sky reflection approximation - subtle
    let sky_color = mix(
        vec3<f32>(0.5, 0.7, 0.9),  // Day sky
        vec3<f32>(0.1, 0.1, 0.2),  // Night sky
        clamp(-light.sun_direction.y * 2.0, 0.0, 1.0)
    );
    let sky_reflect = sky_color * fresnel_factor * 0.25;
    let with_reflection = final_color + sky_reflect;

    // Final alpha - more transparent
    let alpha = mix(water.transparency, 0.95, fresnel_factor * 0.3);

    return vec4<f32>(with_reflection, alpha);
}
