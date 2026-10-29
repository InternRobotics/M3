import glfw
import numpy as np
import logging
from . import Camera
from ...types import ViewerMode
from imgui_bundle import imgui


# TODO: Coalesce all camera types into a single class
class FPSCamera(Camera):
    def __init__(
        self,
        mode: ViewerMode,
        res_x: int = 1280,
        res_y: int = 720,
        fov_y: float = 30.0,
        z_near: float = 0.001,
        z_far: float = 100.0,
        to_world: np.ndarray = None,
    ):
        super().__init__(mode, res_x, res_y, fov_y, z_near, z_far, to_world)
        self.smoothness = 0.4
        self.origin_motion = np.zeros(3)
        self.rotation_motion = np.zeros(3)
        self.smoothed_origin_motion = np.zeros(3)
        self.smoothed_rotation_motion = np.zeros(3)
        self.speed = 1
        self.rot_speed = 1
        self.mouse_speed = 2
        self.radians_per_pixel = np.pi / 150
        self.invert_mouse = False
        self.current_type = "FPS"

    def setup(self):
        if self.mode != ViewerMode.SERVER:
            self.movement_keys = {
                "w": imgui.Key[glfw.get_key_name(glfw.KEY_UNKNOWN, glfw.get_key_scancode(glfw.KEY_W))],
                "a": imgui.Key[glfw.get_key_name(glfw.KEY_UNKNOWN, glfw.get_key_scancode(glfw.KEY_A))],
                "s": imgui.Key[glfw.get_key_name(glfw.KEY_UNKNOWN, glfw.get_key_scancode(glfw.KEY_S))],
                "d": imgui.Key[glfw.get_key_name(glfw.KEY_UNKNOWN, glfw.get_key_scancode(glfw.KEY_D))],
                "q": imgui.Key[glfw.get_key_name(glfw.KEY_UNKNOWN, glfw.get_key_scancode(glfw.KEY_Q))],
                "e": imgui.Key[glfw.get_key_name(glfw.KEY_UNKNOWN, glfw.get_key_scancode(glfw.KEY_E))],
                "j": imgui.Key[glfw.get_key_name(glfw.KEY_UNKNOWN, glfw.get_key_scancode(glfw.KEY_J))],
                "k": imgui.Key[glfw.get_key_name(glfw.KEY_UNKNOWN, glfw.get_key_scancode(glfw.KEY_K))],
                "l": imgui.Key[glfw.get_key_name(glfw.KEY_UNKNOWN, glfw.get_key_scancode(glfw.KEY_L))],
                "i": imgui.Key[glfw.get_key_name(glfw.KEY_UNKNOWN, glfw.get_key_scancode(glfw.KEY_I))],
                "o": imgui.Key[glfw.get_key_name(glfw.KEY_UNKNOWN, glfw.get_key_scancode(glfw.KEY_O))],
                "u": imgui.Key[glfw.get_key_name(glfw.KEY_UNKNOWN, glfw.get_key_scancode(glfw.KEY_U))],
            }

    def process_mouse_input(self) -> bool:
        if imgui.is_mouse_dragging(0):
            delta = imgui.get_mouse_drag_delta()
            delta.y *= -1 if self.invert_mouse else 1
            delta.x *= -1 if self.invert_mouse else 1
            angle_right = (
                -delta.y * self.radians_per_pixel * self.delta_time * self.mouse_speed
            )
            angle_up = (
                -delta.x * self.radians_per_pixel * self.delta_time * self.mouse_speed
            )
            self.apply_rotation(0, angle_right, angle_up)
            imgui.reset_mouse_drag_delta()
            return True

        return False

    def process_keyboard_input(self) -> bool:
        if self.mode != ViewerMode.SERVER:
            if imgui.is_key_down(self.movement_keys["w"]):
                self.origin_motion += self.forward
            if imgui.is_key_down(self.movement_keys["a"]):
                self.origin_motion -= self.right
            if imgui.is_key_down(self.movement_keys["q"]):
                self.origin_motion -= self.up
            if imgui.is_key_down(self.movement_keys["s"]):
                self.origin_motion -= self.forward
            if imgui.is_key_down(self.movement_keys["d"]):
                self.origin_motion += self.right
            if imgui.is_key_down(self.movement_keys["e"]):
                self.origin_motion += self.up

            if imgui.is_key_down(self.movement_keys["o"]):
                self.rotation_motion[0] += 50 * self.radians_per_pixel
            if imgui.is_key_down(self.movement_keys["u"]):
                self.rotation_motion[0] -= 50 * self.radians_per_pixel
            if imgui.is_key_down(self.movement_keys["i"]):
                self.rotation_motion[1] += 50 * self.radians_per_pixel
            if imgui.is_key_down(self.movement_keys["k"]):
                self.rotation_motion[1] -= 50 * self.radians_per_pixel
            if imgui.is_key_down(self.movement_keys["j"]):
                self.rotation_motion[2] += 50 * self.radians_per_pixel
            if imgui.is_key_down(self.movement_keys["l"]):
                self.rotation_motion[2] -= 50 * self.radians_per_pixel
        else:
            logging.warning("Unexpected keyboard input for server camera")

    def show_gui(self):
        # Sliders
        super().show_gui()
        _, self.smoothness = imgui.slider_float("Smoothness", self.smoothness, 0, 0.999)
        _, self.speed = imgui.slider_float("Speed", self.speed, 0.1, 10)
        _, self.rot_speed = imgui.slider_float(
            "Rotation Speed", self.rot_speed, 0.1, 10
        )
        _, self.invert_mouse = imgui.checkbox("Invert Mouse", self.invert_mouse)

        # Smooth motion
        weight = 1 - np.exp(-self.delta_time / (self.smoothness + 1e-6))
        self.smoothed_origin_motion = (
            self.smoothed_origin_motion * (1 - weight)
            + self.origin_motion * weight
        )
        self.smoothed_rotation_motion = (
            self.smoothed_rotation_motion * (1 - weight)
            + self.rotation_motion * weight
        )

        # Apply motion
        self.origin += self.smoothed_origin_motion * self.delta_time * self.speed
        self.apply_rotation(
            *(self.smoothed_rotation_motion * self.delta_time * self.rot_speed)
        )

        self.origin_motion = np.zeros(3)
        self.rotation_motion = np.zeros(3)