from OpenGL import *
from enum import IntFlag, auto
from imgui_bundle import imgui, __version__ as imgui_version

IMGUI_192 = tuple(map(int, imgui_version.split("."))) >= (1,92,0)

class ViewerMode(IntFlag):
    LOCAL = auto()
    SERVER = auto()
    CLIENT = auto()

# Aliases for easy access
LOCAL = ViewerMode.LOCAL
CLIENT = ViewerMode.CLIENT
SERVER = ViewerMode.SERVER
LOCAL_SERVER = ViewerMode.LOCAL | ViewerMode.SERVER
LOCAL_CLIENT = ViewerMode.LOCAL | ViewerMode.CLIENT

class Texture2D:
    """
    This is just a struct like class which holds the state of the texture which
    for now is only the resolution and OpenGL ID but more fields my be added as
    needed. It doesn't provide any methods to actually perform operations on the 
    exture, use the OpenGL API for that. It is the responsibility of the
    application to update the state of the texture when and if needed.
    """
    def __init__(self):
        self.res_x = -1
        self.res_y = -1
        # There was an API change in ImGui v1.92.0 where all texture related functions expect
        # ImTextureRef instead of ImTextureID.
        if IMGUI_192:
            self._id = imgui.ImTextureRef(-1)
        else:
            self._id = -1
    
    @property
    def tex_ref(self):
        """ Use this when passing to imgui.image """
        return self._id

    @property
    def id(self):
        """ Use this when passing to OpenGL related functions """
        if IMGUI_192:
            return self._id._tex_id
        else:
            return self._id

    @id.setter
    def id(self, tex_id):
        if IMGUI_192:
            self._id._tex_id = tex_id
        else:
            self._id = tex_id