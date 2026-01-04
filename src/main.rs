mod camera;
mod config;
#[cfg(feature = "embed-assets")]
mod embedded_assets;
mod fluid;
mod raycast;
mod renderer;
mod texture;
mod voxel;
mod worldgen;

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use camera::{Camera, CameraController};
use config::{CompiledBiomesConfig, Config};
use raycast::{get_camera_direction, raycast, RaycastHit};
use renderer::Renderer;
use texture::TextureManager;
use voxel::BlockType;
use worldgen::World;

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    camera: Option<Camera>,
    camera_controller: CameraController,
    world: Option<World>,
    texture_manager: Option<TextureManager>,
    fluid_simulator: fluid::FluidSimulator,
    config: Config,
    last_frame: Instant,
    // FPS tracking
    frame_count: u32,
    fps_timer: Instant,
    current_fps: f32,
    // Voxel selection and editing
    current_selection: Option<RaycastHit>,
    selected_block_type: BlockType,
}

impl App {
    fn new() -> Self {
        // Load configuration
        #[cfg(feature = "embed-assets")]
        let config = Config::load_embedded();
        #[cfg(not(feature = "embed-assets"))]
        let config = Config::load(Path::new("assets/config.toml"));

        // Load textures before window creation
        #[cfg(feature = "embed-assets")]
        let texture_manager = match TextureManager::load_embedded() {
            Ok(tm) => Some(tm),
            Err(e) => {
                log::error!("Failed to load embedded textures: {}", e);
                None
            }
        };
        #[cfg(not(feature = "embed-assets"))]
        let texture_manager = match TextureManager::load(Path::new("assets/textures")) {
            Ok(tm) => Some(tm),
            Err(e) => {
                log::error!("Failed to load textures: {}. Run `cargo run --bin generate-textures` first.", e);
                None
            }
        };

        Self {
            window: None,
            renderer: None,
            camera: None,
            camera_controller: CameraController::new(
                config.camera.movement_speed,
                config.camera.mouse_sensitivity,
            ),
            world: None,
            texture_manager,
            fluid_simulator: fluid::FluidSimulator::new(),
            config,
            last_frame: Instant::now(),
            frame_count: 0,
            fps_timer: Instant::now(),
            current_fps: 0.0,
            current_selection: None,
            selected_block_type: BlockType::Stone, // Default block to place
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            // Check if textures are loaded
            let Some(texture_manager) = &self.texture_manager else {
                log::error!("Cannot start without textures. Run: cargo run --bin generate-textures");
                event_loop.exit();
                return;
            };

            let window_attrs = Window::default_attributes()
                .with_title("Voxel World")
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

            let window = Arc::new(event_loop.create_window(window_attrs).unwrap());
            self.window = Some(window.clone());

            let size = window.inner_size();
            let aspect = size.width as f32 / size.height as f32;

            // Create camera positioned above the terrain (using config values)
            let mut camera = Camera::new(aspect, self.config.camera.fov);
            camera.position = glam::Vec3::new(
                self.config.camera.start_x,
                self.config.camera.start_y,
                self.config.camera.start_z,
            );
            camera.pitch = self.config.camera.start_pitch.to_radians();
            self.camera = Some(camera);

            // Generate procedural world with infinite terrain
            log::info!("Generating world...");
            let gen_start = Instant::now();

            // Create world with config values
            let render_distance = self.config.rendering.render_distance;
            let height_chunks = self.config.rendering.height_chunks;
            #[cfg(feature = "embed-assets")]
            let biomes_config = CompiledBiomesConfig::load_embedded();
            #[cfg(not(feature = "embed-assets"))]
            let biomes_config = CompiledBiomesConfig::load(Path::new("assets/biomes.toml"));
            let mut world = World::with_config(
                &self.config.terrain,
                biomes_config,
                render_distance,
                height_chunks,
            );

            // Generate initial chunks around player start position
            world.generate_initial(self.config.camera.start_x, self.config.camera.start_z);

            log::info!("World generated in {:?}", gen_start.elapsed());
            self.world = Some(world);

            // Initialize renderer with texture manager (takes ownership of texture data for GPU upload)
            // Pass render distance for dynamic GPU buffer sizing
            let mut renderer = pollster::block_on(Renderer::new(
                window.clone(),
                &mut self.texture_manager.as_mut().unwrap(),
                render_distance,
                height_chunks,
            ));

            // Upload world mesh (initial build with camera frustum)
            if let (Some(world), Some(camera)) = (&mut self.world, &self.camera) {
                log::info!("Building mesh...");
                let mesh_start = Instant::now();
                let frustum = camera.frustum();
                renderer.update_world(world, &frustum);
                log::info!("Mesh built in {:?}", mesh_start.elapsed());
            }

            self.renderer = Some(renderer);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(physical_size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(physical_size);
                    if let Some(camera) = &mut self.camera {
                        camera.aspect = physical_size.width as f32 / physical_size.height as f32;
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                self.camera_controller.process_keyboard(&event);

                // Block type selection with number keys (only on press, not release)
                if event.state == ElementState::Pressed {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::Digit1) => self.selected_block_type = BlockType::Stone,
                        PhysicalKey::Code(KeyCode::Digit2) => self.selected_block_type = BlockType::Dirt,
                        PhysicalKey::Code(KeyCode::Digit3) => self.selected_block_type = BlockType::Grass,
                        PhysicalKey::Code(KeyCode::Digit4) => self.selected_block_type = BlockType::Wood,
                        PhysicalKey::Code(KeyCode::Digit5) => self.selected_block_type = BlockType::Cobblestone,
                        PhysicalKey::Code(KeyCode::Digit6) => self.selected_block_type = BlockType::Sand,
                        PhysicalKey::Code(KeyCode::Digit7) => self.selected_block_type = BlockType::Brick,
                        PhysicalKey::Code(KeyCode::Digit8) => self.selected_block_type = BlockType::Leaves,
                        PhysicalKey::Code(KeyCode::Digit9) => self.selected_block_type = BlockType::Snow,
                        _ => {}
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                // Only handle left click for block operations (right click is for camera)
                if state == ElementState::Pressed {
                    match button {
                        MouseButton::Left => {
                            // Destroy block at selection
                            if let Some(hit) = &self.current_selection {
                                if let Some(world) = &mut self.world {
                                    let pos = hit.block_pos;
                                    if world.destroy_block(pos[0], pos[1], pos[2]).is_some() {
                                        log::debug!("Destroyed block at {:?}", pos);
                                        // Notify fluid simulator to activate adjacent water
                                        // This also registers any worldgen water that wasn't yet in the fluid system
                                        self.fluid_simulator.on_block_destroyed_with_world(pos[0], pos[1], pos[2], world);
                                    }
                                }
                            }
                        }
                        MouseButton::Middle => {
                            // Place block adjacent to selection
                            if let Some(hit) = &self.current_selection {
                                if let Some(world) = &mut self.world {
                                    let pos = hit.placement_pos();
                                    if world.place_block(pos[0], pos[1], pos[2], self.selected_block_type) {
                                        log::debug!("Placed {:?} at {:?}", self.selected_block_type, pos);
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
                // Still pass to camera controller for right-click camera movement
                self.camera_controller.process_mouse_button(button, state);
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last_frame).as_secs_f32();
                self.last_frame = now;

                // FPS calculation
                self.frame_count += 1;
                let elapsed = self.fps_timer.elapsed().as_secs_f32();
                if elapsed >= 1.0 {
                    self.current_fps = self.frame_count as f32 / elapsed;
                    self.frame_count = 0;
                    self.fps_timer = Instant::now();

                    // Update window title with FPS, time of day, and biome
                    if let Some(window) = &self.window {
                        let time_str = self.renderer.as_ref()
                            .map(|r| r.get_time_string())
                            .unwrap_or_default();
                        let biome_str = match (&self.camera, &self.world) {
                            (Some(camera), Some(world)) => {
                                world.get_biome_name(camera.position.x, camera.position.z).to_string()
                            }
                            _ => String::new(),
                        };
                        if biome_str.is_empty() {
                            window.set_title(&format!("Voxel World - {:.0} FPS - {}", self.current_fps, time_str));
                        } else {
                            window.set_title(&format!("Voxel World - {:.0} FPS - {} - {}", self.current_fps, time_str, biome_str));
                        }
                    }
                }

                // Update camera
                if let Some(camera) = &mut self.camera {
                    self.camera_controller.update_camera(camera, dt);
                    if let Some(renderer) = &mut self.renderer {
                        renderer.update_camera(camera);
                        // Update time of day and lighting
                        renderer.update_time(dt, camera.position);
                    }

                    // Update world chunks based on player position
                    if let Some(world) = &mut self.world {
                        let _chunks_changed = world.update_for_position(camera.position.x, camera.position.z);

                        // Update fluid simulation
                        self.fluid_simulator.update_with_world(dt, world);

                        // Perform raycast for block selection
                        let direction = get_camera_direction(camera.yaw, camera.pitch);
                        self.current_selection = raycast(world, camera.position, direction, 10.0);

                        // Update renderer with selection
                        if let Some(renderer) = &mut self.renderer {
                            renderer.update_selection(self.current_selection.as_ref());
                        }

                        // Rebuild mesh if chunks changed (always use current frustum for culling)
                        // Also update every frame for frustum culling when camera rotates
                        // Pass fluid simulator for water level rendering
                        if let Some(renderer) = &mut self.renderer {
                            let frustum = camera.frustum();
                            renderer.update_world_with_fluids(world, &frustum, Some(&self.fluid_simulator));
                        }
                    }
                }

                // Render
                if let Some(renderer) = &mut self.renderer {
                    match renderer.render() {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => {
                            renderer.resize(renderer.size);
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            log::error!("Out of memory!");
                            event_loop.exit();
                        }
                        Err(e) => {
                            log::warn!("Surface error: {:?}", e);
                        }
                    }
                }

                // Request next frame
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            self.camera_controller.process_mouse_motion(delta);
        }
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
