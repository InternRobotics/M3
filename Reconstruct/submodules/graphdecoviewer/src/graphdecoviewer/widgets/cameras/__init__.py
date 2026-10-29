import numpy as np
from .. import Widget
from imgui_bundle import imgui
from ...types import ViewerMode, Texture2D

# Coordinate system is same as OpenCV
# Forward -> +Z
# Up -> -Y
# Right -> +X
class Camera(Widget):
    def __init__(
            self, mode: ViewerMode,
            res_x: int=1280, res_y: int=720, fov_y: float=30.0,
            z_near: float=0.001, z_far: float=100.0,
            to_world: np.ndarray=None
        ):
        super().__init__(mode)

        # Extrinsics
        self.origin = np.asarray([0.0, 0.0, 0.0])
        self.forward = np.asarray([0.0, 0.0, 1.0])
        self.up = np.asarray([0.0, -1.0, 0.0])
        self.right = np.asarray([1.0, 0.0, 0.0])

        self.last_frame_time = 0
        self.delta_time = 0

        if to_world is not None:
            self.update_pose(to_world)

        # Intrinsics
        self.res_x = res_x
        self.res_y = res_y
        self.fov_y = np.deg2rad(fov_y)
        self.fov_x = 2 * np.arctan(np.tan(self.fov_y / 2) * (res_x / res_y))
        self.z_near = z_near
        self.z_far = z_far

    def server_recv(self, _, text):
        self.res_x = text["res_x"]
        self.res_y = text["res_y"]
        self.fov_x = text["fov_x"]
        self.fov_y = text["fov_y"]
        self.z_near = text["z_near"]
        self.z_far = text["z_far"]
        self.update_pose(np.array(text["to_world"]))

    def client_send(self):
        return None, self.to_json()

    @classmethod
    def from_json(cls, mode: ViewerMode, json):
        to_world = np.array(json["to_world"])
        del json["to_world"]
        return cls(mode, to_world=to_world, **json)

    def to_json(self):
        return {
            "res_x": self.res_x,
            "res_y": self.res_y,
            "fov_x": self.fov_x,
            "fov_y": self.fov_y,
            "z_near": self.z_near,
            "z_far": self.z_far,
            "to_world": self.to_world.tolist()
        }

    def process_mouse_input(self):
        """ Child class should override this to navigate. """
        pass
    
    def process_keyboard_input(self):
        """ Child class should override this to navigate. """
        pass

    @property
    def to_world(self) -> np.ndarray:
        mat = np.identity(4, dtype=np.float32)
        mat[:3, 3] = self.origin
        mat[:3, 0] = self.right
        mat[:3, 1] = -self.up
        mat[:3, 2] = self.forward
        return mat

    @property
    def to_camera(self) -> np.ndarray:
        return np.linalg.inv(self.to_world)
    
    @property
    def projection(self) -> np.ndarray:
        tan_half_fov_y = np.tan(self.fov_y / 2)
        tan_half_fov_x = np.tan(self.fov_x / 2)

        top = tan_half_fov_y * self.z_near
        bottom = -top
        right = tan_half_fov_x * self.z_near
        left = -right

        P = np.zeros((4, 4), dtype=np.float32)

        z_sign = 1.0

        P[0, 0] = 2.0 * self.z_near / (right - left)
        P[1, 1] = 2.0 * self.z_near / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * self.z_far / (self.z_far - self.z_near)
        P[2, 3] = -(self.z_far * self.z_near) / (self.z_far - self.z_near)

        return P
    
    @property
    def full_projection(self) -> np.ndarray:
        return self.projection @ self.to_camera

    def show_gui(self) -> bool:
        updated, self.fov_y = imgui.slider_angle("FoV Y", self.fov_y, 5, 120)
        if updated:
            self.compute_fov_x()
        
        updated, [self.res_x, self.res_y] = imgui.slider_int2("Resolution", [self.res_x, self.res_y], 1, 4096)
        if updated:
            self.compute_fov_x()
        
        curr_time = imgui.get_time()
        self.delta_time = curr_time - self.last_frame_time
        self.last_frame_time = curr_time
        return False
    
    def compute_fov_x(self):
        self.fov_x = 2 * np.arctan(np.tan(self.fov_y / 2) * (self.res_x / self.res_y))
    
    def draw_camera(self, camera: 'Camera', texture: Texture2D, thickness: float=1.0, color: tuple=(1.0, 1.0, 1.0)):
        """
        Draw the camera onto texture as observed from another camera. The camera
        is drawn on a OpenGL texture with a fragment shader.

        Args:
            camera: The camera from which the current camera is observed.
            texture: The texture on which the camera is to be drawn.
            thickness: The thickness of the lines in pixels.
            color: The color (normalized) of the camera lines.
        """
        raise NotImplementedError()

    def apply_rotation(self, angle_forward: float, angle_right: float, angle_up: float):
        """
        Rotate the camera about its local axes (forward, right, up).
        Angles are in radians.
        """

        def rotate_vec(vec, axis, angle):
            """Rotate vector `vec` around normalized `axis` by `angle` (radians)."""
            axis = axis / np.linalg.norm(axis)
            c = np.cos(angle)
            s = np.sin(angle)
            dot = np.dot(axis, vec)
            cross = np.cross(axis, vec)
            return c * vec + s * cross + (1 - c) * dot * axis

        if abs(angle_forward) > 1e-7:
            self.up = rotate_vec(self.up, self.forward, angle_forward)
            self.right = rotate_vec(self.right, self.forward, angle_forward)
        if abs(angle_right) > 1e-7:
            self.forward = rotate_vec(self.forward, self.right, angle_right)
            self.up = rotate_vec(self.up, self.right, angle_right)
        if abs(angle_up) > 1e-7:
            self.forward = rotate_vec(self.forward, self.up, angle_up)
            self.right = rotate_vec(self.right, self.up, angle_up)

        # Re-orthonormalize (to handle floating-point drift)
        self.forward /= np.linalg.norm(self.forward)
        # Recompute right as cross of forward with global -Y or some logic:
        # But typically you'd just do cross of (forward, up)
        self.right = np.cross(self.forward, self.up)
        self.right /= np.linalg.norm(self.right)
        # Recompute up as cross of (right, forward)
        self.up = np.cross(self.right, self.forward)
        self.up /= np.linalg.norm(self.up)

    def update_pose(self, mat: np.ndarray):
        self.origin = mat[:3, 3]
        self.forward = mat[:3, 2]
        self.forward = self.forward / np.linalg.norm(self.forward)
        self.up = -mat[:3, 1]
        self.up = self.up / np.linalg.norm(self.up)
        self.right = mat[:3, 0]
        self.right = self.right / np.linalg.norm(self.right)