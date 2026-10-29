import numpy as np
from . import Widget
from OpenGL.GL import *
from imgui_bundle import imgui
from abc import abstractmethod
from ..types import *

def _cudaGetErrorEnum(error):
    if isinstance(error, driver.CUresult):
        err, name = driver.cuGetErrorName(error)
        return name if err == driver.CUresult.CUDA_SUCCESS else "<unknown>"
    else:
        raise RuntimeError('Unknown error type: {}'.format(error))

def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError("CUDA error code={}({})".format(result[0].value, _cudaGetErrorEnum(result[0])))
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]

class Image(Widget):
    """
    Base class for the image viewer widget. Each child class must override
    the '_upload' method to upload their image to the OpenGL texture.
    """
    def __init__(self, mode: ViewerMode):
        self.texture = Texture2D()
        self.img = None
        self.step_called = False
        super().__init__(mode)

    def setup(self):
        """ Create OpenGL texture to be displayed. """
        if self.mode and LOCAL_CLIENT:
            self.texture.id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.texture.id)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    
    def destroy(self):
        """ Delete the texture. """
        if self.mode and LOCAL_CLIENT:
            glDeleteTextures(1, int(self.texture.id))

    def step(self, img):
        """
        This function just stores the references to the image to be displayed.
        The actual uploading of the image to the OpenGL texture is done in the
        the 'show_gui' function. If your application changes underlying memory
        of the image, then make sure to copy the image before passing it.
        """
        self.img = img
        self.step_called = True

    @abstractmethod
    def _upload(self):
        """
        Upload 'self.img' to the OpenGL texture. Each child class should override
        this method to define the upload procedure based upon the source.
        """

    def show_gui(self, draw_list: imgui.ImDrawList=None, res_x=0, res_y=0):
        if self.img is None:
            return

        glBindTexture(GL_TEXTURE_2D, self.texture.id)
        self._upload()

        if res_x <= 0:
            res_x = self.texture.res_x
        if res_y <= 0:
            res_y = self.texture.res_y

        if draw_list is not None:
            # Figure out
            draw_list.add_image(self.texture.tex_ref, (0, 0), (res_x, res_y))
        else:
            imgui.image(self.texture.tex_ref, (res_x, res_y))

class NumpyImage(Image):
    """ Image viewer where the image to be shown comes from NumPy array. """
    def _upload(self):
        img = self.img
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        if img.shape[1] != self.texture.res_x or img.shape[0] != self.texture.res_y:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, img.shape[1], img.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, img)
            self.texture.res_x = img.shape[1]
            self.texture.res_y = img.shape[0]
        else:
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, img.shape[1], img.shape[0], GL_RGB, GL_UNSIGNED_BYTE, img)
    
    def server_send(self):
        if not self.step_called:
            return None, None
        img = self.img
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        self.step_called = False
        return memoryview(np.ascontiguousarray(img.flatten())), {"shape": tuple(img.shape)}
    
    def client_recv(self, binary, text):
        self.img = np.frombuffer(binary, dtype=np.uint8).reshape(text["shape"])


# Check if 'cuda-python' and 'torch' are available
enable_torch_image = True
try:
    # import cuda.bindings as cuda
    from cuda.bindings import driver
except ImportError:
    print("WARNING: 'cuda-python' not found. Stubbing 'TorchImage' with 'NumpyImage'.")
    enable_torch_image = False

try:
    import torch
except ImportError:
    print("WARNING: 'torch' not found. Stubbing 'TorchImage' with 'NumpyImage'.")
    enable_torch_image = False
else:
    if not torch.cuda.is_available():
        print("WARNING: 'torch' is not compiled with CUDA support. Stubbing 'TorchImage' with 'NumpyImage'.")
        enable_torch_image = False

if enable_torch_image:
    class TorchImage(Image):
        """ Image viewer where the image to be shown comes from Torch tensor **on the GPU**. """
        _cuda_resource = None

        def _upload(self):
            img = self.img
            assert img.is_cuda, "'img' is not the GPU."
            if img.dtype != torch.uint8:
                img = (torch.clip(img, 0, 1) * 255).byte()

            if img.shape[1] != self.texture.res_x or img.shape[0] != self.texture.res_y:
                if self._cuda_resource is not None:
                    checkCudaErrors(driver.cuGraphicsUnregisterResource(self._cuda_resource))
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.shape[1], img.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, img.cpu().numpy())
                self._cuda_resource = checkCudaErrors(driver.cuGraphicsGLRegisterImage(self.texture.id, GL_TEXTURE_2D, driver.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD))
                self.texture.res_x = img.shape[1]
                self.texture.res_y = img.shape[0]
            else:
                # For some reason the copy only works for 4 byte pixels
                if img.shape[-1] == 3:
                    img = torch.cat([img, 255 * torch.ones((img.shape[0], img.shape[1], 1), device=img.device, dtype=torch.uint8)], -1)
                # Copy to OpenGL texture
                checkCudaErrors(driver.cuGraphicsMapResources(1, self._cuda_resource, 0))
                cuda_array = checkCudaErrors(driver.cuGraphicsSubResourceGetMappedArray(self._cuda_resource, 0, 0))
                copy_params = driver.CUDA_MEMCPY3D()
                copy_params.srcMemoryType = driver.CUmemorytype.CU_MEMORYTYPE_DEVICE
                copy_params.srcDevice = img.data_ptr()
                copy_params.srcPitch = self.texture.res_x * 4
                copy_params.dstMemoryType = driver.CUmemorytype.CU_MEMORYTYPE_ARRAY
                copy_params.dstArray = cuda_array
                copy_params.WidthInBytes = self.texture.res_x * 4
                copy_params.Height = self.texture.res_y
                copy_params.Depth = 1
                checkCudaErrors(driver.cuMemcpy3D(copy_params))
                checkCudaErrors(driver.cuGraphicsUnmapResources(1, self._cuda_resource, 0))

        def destroy(self):
            """ Delete the 'cuda_resource' then delete the OpenGL texture. """
            if self._cuda_resource is not None:
                checkCudaErrors(driver.cuGraphicsUnregisterResource(self._cuda_resource))
            super().destroy()
        
        def server_send(self):
            if not self.step_called:
                return None, None
            img = self.img
            if img.dtype != torch.uint8:
                img = (torch.clip(img, 0, 1) * 255).byte()
            self.step_called = False
            return memoryview(img.contiguous().flatten().cpu().numpy()), {"shape": tuple(img.shape)}

        def client_recv(self, binary, text):
            img = np.frombuffer(binary, dtype=np.uint8).reshape(text["shape"])
            self.img = torch.from_numpy(img).to(0)

else:
    class TorchImage(NumpyImage):
        # Update the step function to convert the input to a tensor to numpy array
        def step(self, img):
            if not isinstance(img, np.ndarray):
                # A tensor
                img = img.detach().cpu().numpy()
            # Otherwise it's already a numpy array
            self.img = img