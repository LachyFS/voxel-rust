// Sky shader with Rayleigh scattering and sun rendering

struct CameraUniform {
    view_proj: mat4x4<f32>,
};

struct LightUniform {
    light_space_matrix: mat4x4<f32>,
    sun_direction: vec3<f32>,
    _pad1: f32,
    sun_color: vec3<f32>,
    _pad2: f32,
    ambient_color: vec3<f32>,
    time_of_day: f32,
};

struct SkyUniforms {
    inv_view_proj: mat4x4<f32>,
    camera_position: vec3<f32>,
    _pad: f32,
    viewport_size: vec2<f32>,
    _pad2: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> sky: SkyUniforms;

@group(1) @binding(0)
var<uniform> light: LightUniform;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

// Fullscreen triangle vertices (no vertex buffer needed)
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    // Generate fullscreen triangle
    // Vertex 0: (-1, -1), Vertex 1: (3, -1), Vertex 2: (-1, 3)
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);

    out.clip_position = vec4<f32>(x, y, 0.9999, 1.0); // Near far plane

    return out;
}

// Calculate ray direction from screen position (done per-fragment to avoid interpolation issues)
fn get_ray_direction(frag_coord: vec4<f32>) -> vec3<f32> {
    // Convert fragment coordinates (pixels) to normalized device coordinates [-1, 1]
    // frag_coord.xy is in pixels [0, width] x [0, height]
    // NDC.x = (frag_coord.x / width) * 2 - 1
    // NDC.y = 1 - (frag_coord.y / height) * 2  (flip Y because frag_coord.y=0 is at top)
    let ndc_x = (frag_coord.x / sky.viewport_size.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (frag_coord.y / sky.viewport_size.y) * 2.0;

    // Transform from clip space to world space using inverse view-projection
    let near_clip = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    let far_clip = vec4<f32>(ndc_x, ndc_y, 1.0, 1.0);

    let near_world = sky.inv_view_proj * near_clip;
    let far_world = sky.inv_view_proj * far_clip;

    let near_pos = near_world.xyz / near_world.w;
    let far_pos = far_world.xyz / far_world.w;

    return normalize(far_pos - near_pos);
}

// Rayleigh scattering coefficients
const RAYLEIGH_BETA: vec3<f32> = vec3<f32>(5.5e-6, 13.0e-6, 22.4e-6); // Wavelength-dependent
const MIE_BETA: f32 = 21e-6;
const RAYLEIGH_HEIGHT: f32 = 8000.0; // Scale height for Rayleigh scattering
const MIE_HEIGHT: f32 = 1200.0; // Scale height for Mie scattering
const PLANET_RADIUS: f32 = 6371000.0; // Earth radius in meters
const ATMOSPHERE_RADIUS: f32 = 6471000.0; // Atmosphere top
const SUN_INTENSITY: f32 = 22.0;

// Phase function for Rayleigh scattering
fn rayleigh_phase(cos_theta: f32) -> f32 {
    return 3.0 / (16.0 * 3.14159265) * (1.0 + cos_theta * cos_theta);
}

// Phase function for Mie scattering (Henyey-Greenstein)
fn mie_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let num = (1.0 - g2);
    let denom = pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5);
    return num / (4.0 * 3.14159265 * denom);
}

// Ray-sphere intersection
fn ray_sphere_intersect(origin: vec3<f32>, dir: vec3<f32>, radius: f32) -> vec2<f32> {
    let a = dot(dir, dir);
    let b = 2.0 * dot(origin, dir);
    let c = dot(origin, origin) - radius * radius;
    let d = b * b - 4.0 * a * c;

    if d < 0.0 {
        return vec2<f32>(-1.0, -1.0);
    }

    let sqrt_d = sqrt(d);
    return vec2<f32>(
        (-b - sqrt_d) / (2.0 * a),
        (-b + sqrt_d) / (2.0 * a)
    );
}

// Calculate atmospheric density at a height
fn density_at_height(height: f32, scale_height: f32) -> f32 {
    return exp(-height / scale_height);
}

// Simplified atmospheric scattering
fn atmosphere(ray_dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    // Normalize sun direction
    let sun = normalize(sun_dir);

    // Camera at ground level (simplified)
    let ray_origin = vec3<f32>(0.0, PLANET_RADIUS + 100.0, 0.0);

    // Find intersection with atmosphere
    let atmo_hit = ray_sphere_intersect(ray_origin, ray_dir, ATMOSPHERE_RADIUS);

    if atmo_hit.y < 0.0 {
        return vec3<f32>(0.0); // No atmosphere hit
    }

    let t_start = max(0.0, atmo_hit.x);
    let t_end = atmo_hit.y;

    // Check if we hit the planet
    let planet_hit = ray_sphere_intersect(ray_origin, ray_dir, PLANET_RADIUS);
    var t_max = t_end;
    if planet_hit.x > 0.0 {
        t_max = min(t_max, planet_hit.x);
    }

    // Ray marching through atmosphere
    let num_samples = 8;
    let step_size = (t_max - t_start) / f32(num_samples);

    var total_rayleigh = vec3<f32>(0.0);
    var total_mie = vec3<f32>(0.0);
    var optical_depth_r = 0.0;
    var optical_depth_m = 0.0;

    for (var i = 0; i < num_samples; i++) {
        let t = t_start + (f32(i) + 0.5) * step_size;
        let sample_pos = ray_origin + ray_dir * t;
        let height = length(sample_pos) - PLANET_RADIUS;

        // Density at this point
        let density_r = density_at_height(height, RAYLEIGH_HEIGHT) * step_size;
        let density_m = density_at_height(height, MIE_HEIGHT) * step_size;

        optical_depth_r += density_r;
        optical_depth_m += density_m;

        // Light ray to sun (simplified - just use height-based attenuation)
        let sun_ray_optical_depth_r = density_r * 2.0;
        let sun_ray_optical_depth_m = density_m * 2.0;

        // Attenuation
        let tau = RAYLEIGH_BETA * (optical_depth_r + sun_ray_optical_depth_r) +
                  MIE_BETA * (optical_depth_m + sun_ray_optical_depth_m);
        let attenuation = exp(-tau);

        total_rayleigh += density_r * attenuation;
        total_mie += density_m * attenuation;
    }

    // Phase functions
    let cos_theta = dot(ray_dir, sun);
    let phase_r = rayleigh_phase(cos_theta);
    let phase_m = mie_phase(cos_theta, 0.758); // g = 0.758 for typical atmosphere

    // Final color
    let rayleigh_color = total_rayleigh * RAYLEIGH_BETA * phase_r;
    let mie_color = total_mie * MIE_BETA * phase_m;

    return SUN_INTENSITY * (rayleigh_color + mie_color);
}

// Render sun disc
fn sun_disc(ray_dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let sun = normalize(sun_dir);
    let cos_angle = dot(ray_dir, sun);

    // Sun disc (angular radius ~0.5 degrees = 0.0087 radians, cos ~ 0.99996)
    let sun_angular_radius = 0.0087;
    let sun_cos_angle = cos(sun_angular_radius);

    if cos_angle > sun_cos_angle {
        // Inside sun disc - bright yellow/white
        let edge_factor = (cos_angle - sun_cos_angle) / (1.0 - sun_cos_angle);
        let intensity = 50.0 * smoothstep(0.0, 1.0, edge_factor);
        return vec3<f32>(1.0, 0.95, 0.8) * intensity;
    }

    // Sun glow (corona)
    let glow_size = 0.05;
    let glow_cos = cos(glow_size);
    if cos_angle > glow_cos {
        let glow_factor = (cos_angle - glow_cos) / (sun_cos_angle - glow_cos);
        let glow_intensity = pow(glow_factor, 2.0) * 2.0;
        return vec3<f32>(1.0, 0.8, 0.5) * glow_intensity;
    }

    return vec3<f32>(0.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Calculate ray direction per-fragment to avoid interpolation artifacts
    let ray_dir = get_ray_direction(in.clip_position);
    let sun_dir = light.sun_direction;

    // Check if sun is below horizon
    let sun_y = sun_dir.y;

    // Base sky color from atmospheric scattering
    var sky_color = atmosphere(ray_dir, sun_dir);

    // Add sun disc (only when above horizon)
    if sun_y > -0.05 {
        let sun_visibility = smoothstep(-0.05, 0.1, sun_y);
        sky_color += sun_disc(ray_dir, sun_dir) * sun_visibility;
    }

    // Night sky - add stars and darker color
    if sun_y < 0.1 {
        let night_factor = smoothstep(0.1, -0.2, sun_y);

        // Simple star field using noise-like function
        let star_coord = ray_dir * 1000.0;
        let star_hash = fract(sin(dot(floor(star_coord), vec3<f32>(12.9898, 78.233, 45.164))) * 43758.5453);
        let star_brightness = smoothstep(0.997, 1.0, star_hash) * night_factor * 2.0;
        sky_color += vec3<f32>(star_brightness);

        // Night sky base color
        let night_color = vec3<f32>(0.02, 0.02, 0.05);
        sky_color = mix(sky_color, night_color + sky_color * 0.1, night_factor * 0.8);
    }

    // Horizon fog/haze
    let horizon_factor = 1.0 - abs(ray_dir.y);
    let horizon_power = pow(horizon_factor, 8.0);

    // Horizon color based on sun position
    var horizon_color = vec3<f32>(0.7, 0.8, 0.9); // Day
    if sun_y < 0.3 && sun_y > -0.1 {
        // Sunset/sunrise colors
        let sunset_factor = 1.0 - abs(sun_y - 0.1) / 0.2;
        horizon_color = mix(horizon_color, vec3<f32>(1.0, 0.5, 0.3), sunset_factor * 0.7);
    }

    sky_color = mix(sky_color, horizon_color, horizon_power * 0.5);

    // Tone mapping (simple Reinhard)
    sky_color = sky_color / (sky_color + vec3<f32>(1.0));

    // Gamma correction
    sky_color = pow(sky_color, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(sky_color, 1.0);
}
