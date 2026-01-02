use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use winit::event::{ElementState, KeyEvent, MouseButton};
use winit::keyboard::{KeyCode, PhysicalKey};

pub struct Camera {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub aspect: f32,
    pub fov_y: f32,
    pub z_near: f32,
    pub z_far: f32,
}

impl Camera {
    pub fn new(aspect: f32, fov_degrees: f32) -> Self {
        Self {
            position: Vec3::new(8.0, 10.0, 25.0),
            yaw: -90.0_f32.to_radians(),
            pitch: -20.0_f32.to_radians(),
            aspect,
            fov_y: fov_degrees.to_radians(),
            z_near: 0.1,
            z_far: 1000.0,
        }
    }

    pub fn view_matrix(&self) -> Mat4 {
        let direction = Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize();

        Mat4::look_at_rh(self.position, self.position + direction, Vec3::Y)
    }

    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, self.aspect, self.z_near, self.z_far)
    }

    pub fn view_projection_matrix(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }

    pub fn forward(&self) -> Vec3 {
        Vec3::new(self.yaw.cos(), 0.0, self.yaw.sin()).normalize()
    }

    pub fn right(&self) -> Vec3 {
        Vec3::new(-self.yaw.sin(), 0.0, self.yaw.cos()).normalize()
    }

    /// Extract frustum planes from the view-projection matrix
    pub fn frustum(&self) -> Frustum {
        Frustum::from_view_projection(&self.view_projection_matrix())
    }
}

/// A 3D plane represented as ax + by + cz + d = 0
#[derive(Clone, Copy)]
pub struct Plane {
    pub normal: Vec3,
    pub d: f32,
}

impl Plane {
    /// Create a plane from a Vec4 (xyz = normal, w = d) and normalize it
    pub fn from_vec4(v: Vec4) -> Self {
        let length = Vec3::new(v.x, v.y, v.z).length();
        Self {
            normal: Vec3::new(v.x / length, v.y / length, v.z / length),
            d: v.w / length,
        }
    }

    /// Signed distance from point to plane (positive = in front, negative = behind)
    #[inline]
    pub fn distance(&self, point: Vec3) -> f32 {
        self.normal.dot(point) + self.d
    }
}

/// View frustum for culling (6 planes: left, right, bottom, top, near, far)
pub struct Frustum {
    planes: [Plane; 6],
}

impl Frustum {
    /// Extract frustum planes from a view-projection matrix
    /// Uses the Gribb-Hartmann method for plane extraction
    pub fn from_view_projection(vp: &Mat4) -> Self {
        // Get rows of the matrix (glam stores column-major)
        let row0 = Vec4::new(vp.x_axis.x, vp.y_axis.x, vp.z_axis.x, vp.w_axis.x);
        let row1 = Vec4::new(vp.x_axis.y, vp.y_axis.y, vp.z_axis.y, vp.w_axis.y);
        let row2 = Vec4::new(vp.x_axis.z, vp.y_axis.z, vp.z_axis.z, vp.w_axis.z);
        let row3 = Vec4::new(vp.x_axis.w, vp.y_axis.w, vp.z_axis.w, vp.w_axis.w);

        Self {
            planes: [
                Plane::from_vec4(row3 + row0), // Left
                Plane::from_vec4(row3 - row0), // Right
                Plane::from_vec4(row3 + row1), // Bottom
                Plane::from_vec4(row3 - row1), // Top
                Plane::from_vec4(row3 + row2), // Near
                Plane::from_vec4(row3 - row2), // Far
            ],
        }
    }

    /// Test if an axis-aligned bounding box intersects or is inside the frustum
    /// Returns true if the AABB is at least partially visible
    #[inline]
    pub fn intersects_aabb(&self, min: Vec3, max: Vec3) -> bool {
        for plane in &self.planes {
            // Find the corner of the AABB that is furthest in the direction of the plane normal
            let p = Vec3::new(
                if plane.normal.x >= 0.0 { max.x } else { min.x },
                if plane.normal.y >= 0.0 { max.y } else { min.y },
                if plane.normal.z >= 0.0 { max.z } else { min.z },
            );

            // If the furthest corner is behind the plane, the AABB is completely outside
            if plane.distance(p) < 0.0 {
                return false;
            }
        }
        true
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
        }
    }

    pub fn update(&mut self, camera: &Camera) {
        self.view_proj = camera.view_projection_matrix().to_cols_array_2d();
    }
}

pub struct CameraController {
    speed: f32,
    sensitivity: f32,
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    mouse_pressed: bool,
    mouse_delta: (f32, f32),
}

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            speed,
            sensitivity,
            forward: false,
            backward: false,
            left: false,
            right: false,
            up: false,
            down: false,
            mouse_pressed: false,
            mouse_delta: (0.0, 0.0),
        }
    }

    pub fn process_keyboard(&mut self, event: &KeyEvent) -> bool {
        let pressed = event.state == ElementState::Pressed;

        match event.physical_key {
            PhysicalKey::Code(KeyCode::KeyW) | PhysicalKey::Code(KeyCode::ArrowUp) => {
                self.forward = pressed;
                true
            }
            PhysicalKey::Code(KeyCode::KeyS) | PhysicalKey::Code(KeyCode::ArrowDown) => {
                self.backward = pressed;
                true
            }
            PhysicalKey::Code(KeyCode::KeyA) | PhysicalKey::Code(KeyCode::ArrowLeft) => {
                self.left = pressed;
                true
            }
            PhysicalKey::Code(KeyCode::KeyD) | PhysicalKey::Code(KeyCode::ArrowRight) => {
                self.right = pressed;
                true
            }
            PhysicalKey::Code(KeyCode::Space) => {
                self.up = pressed;
                true
            }
            PhysicalKey::Code(KeyCode::ShiftLeft) | PhysicalKey::Code(KeyCode::ShiftRight) => {
                self.down = pressed;
                true
            }
            _ => false,
        }
    }

    pub fn process_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        if button == MouseButton::Right {
            self.mouse_pressed = state == ElementState::Pressed;
        }
    }

    pub fn process_mouse_motion(&mut self, delta: (f64, f64)) {
        if self.mouse_pressed {
            self.mouse_delta.0 += delta.0 as f32;
            self.mouse_delta.1 += delta.1 as f32;
        }
    }

    pub fn update_camera(&mut self, camera: &mut Camera, dt: f32) {
        // Movement
        let forward = camera.forward();
        let right = camera.right();

        if self.forward {
            camera.position += forward * self.speed * dt;
        }
        if self.backward {
            camera.position -= forward * self.speed * dt;
        }
        if self.right {
            camera.position += right * self.speed * dt;
        }
        if self.left {
            camera.position -= right * self.speed * dt;
        }
        if self.up {
            camera.position.y += self.speed * dt;
        }
        if self.down {
            camera.position.y -= self.speed * dt;
        }

        // Mouse look (only when right mouse button is held)
        if self.mouse_delta.0 != 0.0 || self.mouse_delta.1 != 0.0 {
            camera.yaw += self.mouse_delta.0 * self.sensitivity;
            camera.pitch -= self.mouse_delta.1 * self.sensitivity;

            // Clamp pitch to avoid gimbal lock
            camera.pitch = camera.pitch.clamp(
                -89.0_f32.to_radians(),
                89.0_f32.to_radians(),
            );

            self.mouse_delta = (0.0, 0.0);
        }
    }
}
