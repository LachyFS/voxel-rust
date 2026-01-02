use std::sync::Arc;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::camera::{Camera, CameraUniform, Frustum};
use crate::texture::{BlockTextureArray, TextureManager};
use crate::voxel::Vertex;
use crate::worldgen::World;

/// Shadow map resolution (higher = better quality shadows but more expensive)
const SHADOW_MAP_SIZE: u32 = 2048;

/// Light uniform containing sun direction, color, and light-space matrix
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LightUniform {
    light_space_matrix: [[f32; 4]; 4],
    sun_direction: [f32; 3],
    _pad1: f32,
    sun_color: [f32; 3],
    _pad2: f32,
    ambient_color: [f32; 3],
    time_of_day: f32, // 0.0 = midnight, 0.5 = noon, 1.0 = midnight
}

/// Sky uniform for atmospheric scattering
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SkyUniform {
    inv_view_proj: [[f32; 4]; 4],
    camera_position: [f32; 3],
    _pad: f32,
    viewport_size: [f32; 2],
    _pad2: [f32; 2],
}

impl SkyUniform {
    pub fn new() -> Self {
        Self {
            inv_view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            camera_position: [0.0, 0.0, 0.0],
            _pad: 0.0,
            viewport_size: [1280.0, 720.0],
            _pad2: [0.0, 0.0],
        }
    }

    pub fn update(&mut self, camera: &Camera, viewport_width: f32, viewport_height: f32) {
        let view_proj = camera.view_projection_matrix();
        self.inv_view_proj = view_proj.inverse().to_cols_array_2d();
        self.camera_position = camera.position.to_array();
        self.viewport_size = [viewport_width, viewport_height];
    }
}

impl LightUniform {
    pub fn new() -> Self {
        Self {
            light_space_matrix: Mat4::IDENTITY.to_cols_array_2d(),
            sun_direction: [0.5, 1.0, 0.3],
            _pad1: 0.0,
            sun_color: [1.0, 1.0, 0.9],
            _pad2: 0.0,
            ambient_color: [0.3, 0.35, 0.4],
            time_of_day: 0.25, // Start at sunrise
        }
    }

    /// Update light based on time of day (0.0-1.0 where 0.5 is noon)
    pub fn update(&mut self, time: f32, camera_pos: Vec3) {
        self.time_of_day = time;

        // Sun angle: 0.0 = midnight (below horizon), 0.5 = noon (directly above), 1.0 = midnight
        // Convert to angle: sunrise at 0.25, noon at 0.5, sunset at 0.75
        let sun_angle = (time * std::f32::consts::TAU) - std::f32::consts::FRAC_PI_2;

        // Sun direction (orbits around the X axis for east-west movement)
        let sun_y = sun_angle.sin();
        let sun_xz = sun_angle.cos();
        self.sun_direction = [sun_xz * 0.7, sun_y, sun_xz * 0.3];

        // Sun color based on time (warm at sunrise/sunset, white at noon, dark at night)
        let day_factor = (sun_y + 0.2).clamp(0.0, 1.0); // Slight glow before fully risen

        // Interpolate between night (dark blue), sunset (orange), and day (white)
        let night_color = Vec3::new(0.1, 0.1, 0.2);
        let sunset_color = Vec3::new(1.0, 0.6, 0.3);
        let day_color = Vec3::new(1.0, 0.98, 0.95);

        let sun_color = if sun_y > 0.0 {
            // Day time: blend between sunset and day
            Vec3::lerp(sunset_color, day_color, (sun_y * 2.0).clamp(0.0, 1.0))
        } else {
            // Night time
            night_color
        };
        self.sun_color = [sun_color.x * day_factor, sun_color.y * day_factor, sun_color.z * day_factor];

        // Ambient color (brighter during day, darker at night)
        let ambient_day = Vec3::new(0.4, 0.45, 0.5);
        let ambient_night = Vec3::new(0.05, 0.05, 0.1);
        let ambient = Vec3::lerp(ambient_night, ambient_day, day_factor);
        self.ambient_color = [ambient.x, ambient.y, ambient.z];

        // Calculate light-space matrix for shadow mapping
        // Only calculate shadows when sun is above horizon
        if sun_y > -0.1 {
            let sun_dir = Vec3::from(self.sun_direction).normalize();
            let light_pos = camera_pos + sun_dir * 200.0;
            let light_view = Mat4::look_at_rh(light_pos, camera_pos, Vec3::Y);
            // Orthographic projection for directional light (covers a large area)
            let light_proj = Mat4::orthographic_rh(-150.0, 150.0, -150.0, 150.0, 1.0, 500.0);
            self.light_space_matrix = (light_proj * light_view).to_cols_array_2d();
        }
    }
}

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    sky_pipeline: wgpu::RenderPipeline,
    // Double-buffered vertex/index buffers to avoid GPU contention
    vertex_buffers: [wgpu::Buffer; 2],
    index_buffers: [wgpu::Buffer; 2],
    num_indices: [u32; 2],
    current_buffer: usize,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    texture_bind_group: wgpu::BindGroup,
    shadow_bind_group: wgpu::BindGroup,
    depth_texture: wgpu::TextureView,
    shadow_texture: wgpu::TextureView,
    light_buffer: wgpu::Buffer,
    light_bind_group: wgpu::BindGroup,
    light_uniform: LightUniform,
    sky_buffer: wgpu::Buffer,
    sky_bind_group: wgpu::BindGroup,
    sky_uniform: SkyUniform,
    /// Fast texture lookup array indexed by BlockType discriminant
    block_textures: BlockTextureArray,
    /// Current game time (in seconds, wraps at day length)
    game_time: f32,
    /// Day length in seconds
    day_length: f32,
    /// Whether time is frozen
    time_frozen: bool,
}

/// Calculate GPU buffer sizes based on render distance
/// Returns (vertex_buffer_size, index_buffer_size) in bytes
fn calculate_buffer_sizes(render_distance: i32, height_chunks: i32) -> (u64, u64) {
    // Each chunk is 16x16x16 blocks
    // Worst case: every block has 6 faces exposed = 6 * 4 vertices * 36 bytes/vertex per block
    // But in practice, most blocks are occluded. Estimate ~10% exposed faces on average.
    // Plus surface chunks have more exposed faces than underground chunks.

    let num_chunks = ((2 * render_distance + 1) * (2 * render_distance + 1) * height_chunks) as u64;

    // Estimated vertices per chunk (conservative estimate for surface chunks)
    // Surface: ~2000 vertices, underground: ~500, average ~1000
    let vertices_per_chunk: u64 = 1500;
    let indices_per_chunk: u64 = vertices_per_chunk * 6 / 4; // 6 indices per 4 vertices (2 triangles per quad)

    let vertex_size = std::mem::size_of::<Vertex>() as u64; // 36 bytes
    let index_size = std::mem::size_of::<u32>() as u64; // 4 bytes

    // Add 50% headroom for dense areas
    let vertex_buffer_size = num_chunks * vertices_per_chunk * vertex_size * 3 / 2;
    let index_buffer_size = num_chunks * indices_per_chunk * index_size * 3 / 2;

    // Ensure minimum sizes and round up to power of 2 MB for alignment
    let min_vertex = 16 * 1024 * 1024; // 16 MB minimum
    let min_index = 8 * 1024 * 1024;   // 8 MB minimum

    let vertex_buffer_size = vertex_buffer_size.max(min_vertex);
    let index_buffer_size = index_buffer_size.max(min_index);

    log::info!(
        "GPU buffer sizes for {} chunks: vertex={:.1}MB, index={:.1}MB",
        num_chunks,
        vertex_buffer_size as f64 / (1024.0 * 1024.0),
        index_buffer_size as f64 / (1024.0 * 1024.0)
    );

    (vertex_buffer_size, index_buffer_size)
}

impl Renderer {
    pub async fn new(
        window: Arc<Window>,
        texture_manager: &mut TextureManager,
        render_distance: i32,
        height_chunks: i32,
    ) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    label: None,
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Camera uniform
        let camera_uniform = CameraUniform::new();
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        // Texture array bind group
        let (_texture, texture_view, sampler) =
            texture_manager.create_texture_array(&device, &queue);

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("texture_bind_group"),
        });

        // Light uniform for sun direction, color, and shadow mapping
        let light_uniform = LightUniform::new();
        let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light Buffer"),
            contents: bytemuck::cast_slice(&[light_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let light_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("light_bind_group_layout"),
            });

        let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &light_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_buffer.as_entire_binding(),
            }],
            label: Some("light_bind_group"),
        });

        // Shadow map texture and sampler
        let shadow_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shadow Map"),
            size: wgpu::Extent3d {
                width: SHADOW_MAP_SIZE,
                height: SHADOW_MAP_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_texture_view = shadow_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Shadow Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        // Shadow bind group (for main shader to sample shadow map)
        let shadow_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Depth,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                        count: None,
                    },
                ],
                label: Some("shadow_bind_group_layout"),
            });

        let shadow_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &shadow_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&shadow_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&shadow_sampler),
                },
            ],
            label: Some("shadow_bind_group"),
        });

        // Shaders
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Main Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let shadow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shadow Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shadow.wgsl").into()),
        });

        // Depth texture
        let depth_texture = create_depth_texture(&device, &config);

        // Shadow pipeline (depth-only, renders from light's perspective)
        let shadow_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Shadow Pipeline Layout"),
                bind_group_layouts: &[&light_bind_group_layout],
                push_constant_ranges: &[],
            });

        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Shadow Pipeline"),
            layout: Some(&shadow_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shadow_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: None, // Depth-only pass
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2, // Slight bias to prevent shadow acne
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Main render pipeline with shadow map sampling
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &camera_bind_group_layout,
                    &texture_bind_group_layout,
                    &light_bind_group_layout,
                    &shadow_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Sky shader and pipeline
        let sky_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sky Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("sky.wgsl").into()),
        });

        // Sky uniform buffer
        let sky_uniform = SkyUniform::new();
        let sky_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sky Buffer"),
            contents: bytemuck::cast_slice(&[sky_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let sky_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("sky_bind_group_layout"),
            });

        let sky_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &sky_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: sky_buffer.as_entire_binding(),
            }],
            label: Some("sky_bind_group"),
        });

        let sky_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Sky Pipeline Layout"),
                bind_group_layouts: &[&sky_bind_group_layout, &light_bind_group_layout],
                push_constant_ranges: &[],
            });

        let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sky Pipeline"),
            layout: Some(&sky_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &sky_shader,
                entry_point: Some("vs_main"),
                buffers: &[], // No vertex buffer - fullscreen triangle
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &sky_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // No culling for fullscreen triangle
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false, // Don't write depth - sky is at infinity
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Calculate buffer sizes based on render distance
        let (vertex_buffer_size, index_buffer_size) = calculate_buffer_sizes(render_distance, height_chunks);

        // Create double-buffered vertex/index buffers to avoid GPU contention
        let vertex_buffers = [
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Vertex Buffer 0"),
                size: vertex_buffer_size,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Vertex Buffer 1"),
                size: vertex_buffer_size,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        ];

        let index_buffers = [
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Index Buffer 0"),
                size: index_buffer_size,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Index Buffer 1"),
                size: index_buffer_size,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        ];

        // Build fast block texture lookup array (O(1) indexing instead of HashMap)
        let block_textures = texture_manager.create_texture_array_lookup();

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            shadow_pipeline,
            sky_pipeline,
            vertex_buffers,
            index_buffers,
            num_indices: [0, 0],
            current_buffer: 0,
            camera_buffer,
            camera_bind_group,
            texture_bind_group,
            shadow_bind_group,
            depth_texture,
            shadow_texture: shadow_texture_view,
            light_buffer,
            light_bind_group,
            light_uniform,
            sky_buffer,
            sky_bind_group,
            sky_uniform,
            block_textures,
            game_time: 0.35 * 120.0, // Start at sunrise (0.25 of day cycle)
            day_length: 120.0, // 2 minutes per full day cycle
            time_frozen: true, // Start with time frozen
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture = create_depth_texture(&self.device, &self.config);
        }
    }

    pub fn update_camera(&mut self, camera: &Camera) {
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update(camera);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[camera_uniform]),
        );

        // Update sky uniform with inverse view-projection matrix and viewport size
        self.sky_uniform.update(camera, self.size.width as f32, self.size.height as f32);
        self.queue.write_buffer(
            &self.sky_buffer,
            0,
            bytemuck::cast_slice(&[self.sky_uniform]),
        );
    }

    /// Update time of day and lighting. Call this each frame with delta time.
    pub fn update_time(&mut self, dt: f32, camera_pos: Vec3) {
        // Advance game time (only if not frozen)
        if !self.time_frozen {
            self.game_time += dt;
            if self.game_time >= self.day_length {
                self.game_time -= self.day_length;
            }
        }

        // Convert to 0.0-1.0 range for time of day
        let time_normalized = self.game_time / self.day_length;

        // Update light uniform
        self.light_uniform.update(time_normalized, camera_pos);

        // Upload to GPU
        self.queue.write_buffer(
            &self.light_buffer,
            0,
            bytemuck::cast_slice(&[self.light_uniform]),
        );
    }

    /// Toggle time freeze on/off
    pub fn toggle_time_freeze(&mut self) {
        self.time_frozen = !self.time_frozen;
    }

    /// Check if time is frozen
    pub fn is_time_frozen(&self) -> bool {
        self.time_frozen
    }

    /// Get current time of day as a string for display
    pub fn get_time_string(&self) -> String {
        let time_normalized = self.game_time / self.day_length;
        // Convert 0.0-1.0 to 24-hour clock (0.0 = midnight, 0.5 = noon)
        let hours = (time_normalized * 24.0) % 24.0;
        let hour = hours.floor() as u32;
        let minutes = ((hours - hour as f32) * 60.0).floor() as u32;
        format!("{:02}:{:02}", hour, minutes)
    }

    /// Update mesh from entire world with neighbor-aware culling and frustum culling
    /// Uses double-buffering to avoid GPU contention
    pub fn update_world(&mut self, world: &mut World, frustum: &Frustum) {
        // Use World's mesh generation which handles neighbor culling and frustum culling
        // Returns slices into pre-allocated buffers (no allocation)
        let (all_vertices, all_indices) = world.generate_world_mesh(&self.block_textures, frustum);

        if !all_vertices.is_empty() {
            let vertex_bytes = bytemuck::cast_slice(all_vertices);
            let index_bytes = bytemuck::cast_slice(all_indices);

            // Write to the OTHER buffer (not the one being rendered)
            let write_buffer = 1 - self.current_buffer;

            // Check buffer sizes
            if vertex_bytes.len() as u64 > self.vertex_buffers[write_buffer].size() {
                log::warn!(
                    "Vertex buffer too small: need {} bytes, have {}",
                    vertex_bytes.len(),
                    self.vertex_buffers[write_buffer].size()
                );
                return;
            }
            if index_bytes.len() as u64 > self.index_buffers[write_buffer].size() {
                log::warn!(
                    "Index buffer too small: need {} bytes, have {}",
                    index_bytes.len(),
                    self.index_buffers[write_buffer].size()
                );
                return;
            }

            self.queue.write_buffer(&self.vertex_buffers[write_buffer], 0, vertex_bytes);
            self.queue.write_buffer(&self.index_buffers[write_buffer], 0, index_bytes);
            self.num_indices[write_buffer] = all_indices.len() as u32;

            // Swap to the newly written buffer for next render
            self.current_buffer = write_buffer;

            log::debug!(
                "Uploaded {} vertices, {} indices to buffer {}",
                all_vertices.len(),
                all_indices.len(),
                write_buffer
            );
        }
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Shadow pass: render scene from light's perspective to shadow map
        {
            let mut shadow_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Shadow Pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_texture,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            shadow_pass.set_pipeline(&self.shadow_pipeline);
            shadow_pass.set_bind_group(0, &self.light_bind_group, &[]);
            shadow_pass.set_vertex_buffer(0, self.vertex_buffers[self.current_buffer].slice(..));
            shadow_pass.set_index_buffer(
                self.index_buffers[self.current_buffer].slice(..),
                wgpu::IndexFormat::Uint32,
            );
            shadow_pass.draw_indexed(0..self.num_indices[self.current_buffer], 0, 0..1);
        }

        // Main render pass: sky first, then terrain with shadows
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            // Render sky first (fullscreen triangle, writes to depth buffer at far plane)
            render_pass.set_pipeline(&self.sky_pipeline);
            render_pass.set_bind_group(0, &self.sky_bind_group, &[]);
            render_pass.set_bind_group(1, &self.light_bind_group, &[]);
            render_pass.draw(0..3, 0..1); // Fullscreen triangle (3 vertices, no vertex buffer)

            // Render terrain with shadows
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(1, &self.texture_bind_group, &[]);
            render_pass.set_bind_group(2, &self.light_bind_group, &[]);
            render_pass.set_bind_group(3, &self.shadow_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffers[self.current_buffer].slice(..));
            render_pass.set_index_buffer(
                self.index_buffers[self.current_buffer].slice(..),
                wgpu::IndexFormat::Uint32,
            );
            render_pass.draw_indexed(0..self.num_indices[self.current_buffer], 0, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

fn create_depth_texture(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> wgpu::TextureView {
    let size = wgpu::Extent3d {
        width: config.width,
        height: config.height,
        depth_or_array_layers: 1,
    };
    let desc = wgpu::TextureDescriptor {
        label: Some("Depth Texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    };
    let texture = device.create_texture(&desc);
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}
