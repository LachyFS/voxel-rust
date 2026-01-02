use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::camera::{Camera, CameraUniform, Frustum};
use crate::texture::{BlockTextureArray, TextureManager};
use crate::voxel::Vertex;
use crate::worldgen::World;

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    // Double-buffered vertex/index buffers to avoid GPU contention
    vertex_buffers: [wgpu::Buffer; 2],
    index_buffers: [wgpu::Buffer; 2],
    num_indices: [u32; 2],
    current_buffer: usize,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    texture_bind_group: wgpu::BindGroup,
    depth_texture: wgpu::TextureView,
    /// Fast texture lookup array indexed by BlockType discriminant
    block_textures: BlockTextureArray,
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

        // Shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        // Depth texture
        let depth_texture = create_depth_texture(&device, &config);

        // Render pipeline with both bind group layouts
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout, &texture_bind_group_layout],
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
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
            vertex_buffers,
            index_buffers,
            num_indices: [0, 0],
            current_buffer: 0,
            camera_buffer,
            camera_bind_group,
            texture_bind_group,
            depth_texture,
            block_textures,
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

    pub fn update_camera(&self, camera: &Camera) {
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update(camera);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[camera_uniform]),
        );
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

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.5,
                            g: 0.7,
                            b: 1.0,
                            a: 1.0,
                        }),
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

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(1, &self.texture_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffers[self.current_buffer].slice(..));
            render_pass.set_index_buffer(self.index_buffers[self.current_buffer].slice(..), wgpu::IndexFormat::Uint32);
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
