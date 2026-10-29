import glfw
import json
import time
import threading
from typing import Optional
from collections import defaultdict
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK, ConnectionClosedError
from websockets.sync.server import serve, ServerConnection
from websockets.sync.client import connect, ClientConnection
from .types import *
from .widgets import Widget
from abc import ABC, abstractmethod
from imgui_bundle import immapp, hello_imgui

class Viewer(ABC):
    """
    Base class for viewer. This class setups up the relevant ImGui callbacks.
    The child class must override the 'show_gui' function to build the GUI.
    It can can also override the 'step' function to perform any per frame
    computations required.
    The '(server|client)_(send|recv)' should be used for message passing between
    the server and client for remote viewer support.
    """
    def __init__(self, mode: ViewerMode):
        if not hasattr(self, "window_title"):
            self.window_title = "Viewer"
        self.should_exit = False
        self.num_connections = 0

        # Only used in client mode
        self.websocket = None

        # Mapping of widget_id to widget
        self.widget_id_to_widget = {}

        self.mode = mode

        # Import server specific modules
        if self.mode and LOCAL_SERVER:
            self.import_server_modules()

    def _setup(self):
        """ Go over all of the widgets and initialize them """
        for _, widget in vars(self).items():
            if isinstance(widget, Widget):
                widget.setup()
                self.widget_id_to_widget[widget.widget_id] = widget

    def _destroy(self):
        """ Go over all of the widgets and free any manually allocated objects """
        for _, widget in vars(self).items():
            if isinstance(widget, Widget):
                widget.destroy()
    
    def _main(self, websocket=None):
        """
        TODO: Update
        Internal method which handles inputs, resize and calls
        backend computation and then creates the UI.
        """
        if self.mode is CLIENT and self.websocket is not None:
            try:
                self._client_send(self.websocket)
            except ConnectionClosed:
                print("INFO: Server disconnected")
                self.websocket.close()
                self.websocket = None

        if self.mode is SERVER:
            self._server_recv(websocket)

        if self.mode and LOCAL_SERVER:
            self.step()

        if self.mode is SERVER:
            self._server_send(websocket)

        if self.mode is CLIENT and self.websocket is not None:
            try:
                self._client_recv(self.websocket)
            except ConnectionClosed:
                print("INFO: Server disconnected")
                self.websocket.close()
                self.websocket = None

        if self.mode and LOCAL_CLIENT:
            self.show_gui()
    
    def _server_loop(self, websocket: ServerConnection):
        """ Internal method which runs the server loop. """
        # We only allow one client to connect at a time (for now).
        if self.num_connections > 0:
            print("INFO: Client already connected. Only one client is allowed.")
            websocket.close()
            return
        self.num_connections += 1

        glfw.make_context_current(self.window)
        self.onconnect(websocket)

        # Main Loop
        try:
            while True:
                self._main(websocket)
        except ConnectionClosedOK:
            print("INFO: Client disconnected.")
            self.num_connections -= 1
        except ConnectionClosedError as e:
            print(f"ERROR: Connection closed with error: {e}")
            self.num_connections -= 1

        glfw.make_context_current(None)

    def _server_send(self, websocket: ServerConnection):
        """
        Internal method which goes over all of the registered widgets to compile 
        and send the server state to the client.
        """
        metadata = {}   # Metadata for each widget (Should be JSON serializable)
        all_binaries = []   # List of all binaries to be sent
        binary_to_widget = []   # Mapping of which binary is for which widget
        for _, widget in vars(self).items():
            if isinstance(widget, Widget):
                binary, text = widget.server_send()
                if text is not None:
                    metadata[widget.widget_id] = text
                if binary is not None:
                    all_binaries.append(binary)
                    binary_to_widget.append(widget.widget_id)
        
        # Add global state (viewer)
        binary, text = self.server_send()
        if text is not None:
            metadata["viewer"] = text
        if binary is not None:
            all_binaries.append(binary)
            binary_to_widget.append("viewer")
        
        # Send metadata
        websocket.send(json.dumps(metadata), text=True)

        # Send binary mapping
        websocket.send(json.dumps(binary_to_widget), text=True)

        # Send binaries
        for binary in all_binaries:
            websocket.send(binary, text=False)

    def _server_recv(self, websocket: ServerConnection):
        """
        Internal method which receives state from the client and updates all of
        the widgets.
        """
        # Receive metadata
        metadata = json.loads(websocket.recv())

        # Receive binary mapping
        binary_to_widget = json.loads(websocket.recv())

        # Receive binaries
        all_binaries = []
        for _ in binary_to_widget:
            binary = websocket.recv()
            all_binaries.append(binary)

        # Assemble the data
        all_data = defaultdict(dict)
        for widget_id, metadata in metadata.items():
            if widget_id == "viewer":
                all_data["viewer"]["metadata"] = metadata
            else:
                all_data[int(widget_id)]["metadata"] = metadata
        for widget_id, binary in zip(binary_to_widget, all_binaries):
            all_data[widget_id]["binary"] = binary

        # Update the widgets
        for widget_id, data in all_data.items():
            if widget_id == "viewer":
                self.server_recv(data.get("binary", None), data.get("metadata", None))
            else:
                widget = self.widget_id_to_widget[int(widget_id)]
                widget.server_recv(data.get("binary", None), data.get("metadata", None))
    
    def _client_loop(self, ip: str, port: int):
        """
        Internal method which runs the client loop. This loop only deals with
        connecting to the server and handling reconnections. The '_main' method
        is run by the 'immapp.run' function.
        """
        while True:
            # Try to connect to the server
            if self.websocket is None:
                try:
                    websocket = connect(f"ws://{ip}:{port}", max_size=None, compression=None)
                    print("INFO: Connected to server.")
                    self.onconnect(websocket)
                    self.websocket = websocket  # Make websocket available after onconnect finishes to avoid the main thread from usinng it
                except Exception as e:
                    print(f"INFO: Failed to connect to server with error: {e}."
                        " Retrying in 2 seconds.")
                    self.websocket = None
            time.sleep(2)

    def _client_send(self, websocket: ClientConnection):
        """
        Internal method which goes over all of the registered widgets to compile
        and send the client state to the server.
        """
        metadata = {}   # Metadata for each widget (Should be JSON serializable)
        all_binaries = []   # List of all binaries to be sent
        binary_to_widget = []   # Mapping of which binary is for which widget
        for _, widget in vars(self).items():
            if isinstance(widget, Widget):
                binary, text = widget.client_send()
                if text is not None:
                    metadata[widget.widget_id] = text
                if binary is not None:
                    all_binaries.append(binary)
                    binary_to_widget.append(widget.widget_id)
        
        # Add global state (viewer)
        binary, text = self.client_send()
        if text is not None:
            metadata["viewer"] = text
        if binary is not None:
            all_binaries.append(binary)
            binary_to_widget.append("viewer")
        
        # Send metadata
        websocket.send(json.dumps(metadata), text=True)

        # Send binary mapping
        websocket.send(json.dumps(binary_to_widget), text=True)

        # Send binaries
        for binary in all_binaries:
            websocket.send(binary, text=False)

    def _client_recv(self, websocket: ClientConnection):
        """
        Internal method which receives state from the server and updates all of
        the widgets.
        """
        # Receive metadata
        metadata = json.loads(websocket.recv())

        # Receive binary mapping
        binary_to_widget = json.loads(websocket.recv())

        # Receive binaries
        all_binaries = []
        for _ in binary_to_widget:
            binary = websocket.recv()
            all_binaries.append(binary)

        # Assemble the data
        all_data = defaultdict(dict)
        for widget_id, metadata in metadata.items():
            if widget_id == "viewer":
                all_data["viewer"]["metadata"] = metadata
            else:
                all_data[int(widget_id)]["metadata"] = metadata
        for widget_id, binary in zip(binary_to_widget, all_binaries):
            all_data[widget_id]["binary"] = binary

        # Update the widgets
        for widget_id, data in all_data.items():
            if widget_id == "viewer":
                self.client_recv(data.get("binary", None), data.get("metadata", None))
            else:
                widget = self.widget_id_to_widget[int(widget_id)]
                widget.client_recv(data.get("binary", None), data.get("metadata", None))
    
    def onconnect(self, websocket: ClientConnection|ServerConnection):
        """ Called when a new connection is made. """
        pass

    def run(self, ip: str = "localhost", port: int = 6009):
        self.create_widgets()
        self.running = True

        # Run the client connection in a different thread, the main thread runs the GUI.
        if self.mode is CLIENT:
            connect_thread = threading.Thread(target=self._client_loop, args=(ip, port))
            # Make the thread a daemon so that it exits when the main thread exits.
            connect_thread.daemon = True
            connect_thread.start()
        if self.mode and LOCAL_CLIENT:
            self._runner_params = hello_imgui.RunnerParams()
            self._runner_params.fps_idling.enable_idling = False
            self._runner_params.app_window_params.window_geometry.window_size_state = hello_imgui.WindowSizeState.maximized
            self._runner_params.app_window_params.window_title = self.window_title
            self._runner_params.imgui_window_params.show_status_bar = True
            self._runner_params.imgui_window_params.show_menu_bar = True
            self._runner_params.callbacks.post_init = self._setup
            # self._runner_params.callbacks.before_exit = self._before_exit
            self._runner_params.callbacks.show_gui = self._main
            self._runner_params.callbacks.show_status = self.show_status
            # Disable VSync
            self._runner_params.callbacks.post_init_add_platform_backend_callbacks = lambda: glfw.swap_interval(0)
            self._runner_params.platform_backend_type = hello_imgui.PlatformBackendType.glfw
            self._addon_params = immapp.AddOnsParams(with_implot=True)

            # This is required to make 'want_capture_*' work. The default value is to create a full screen window,
            # but that would mean the 'want_capture_mouse' variable will always be set.
            self._runner_params.imgui_window_params.default_imgui_window_type = hello_imgui.DefaultImGuiWindowType.provide_full_screen_dock_space
            immapp.run(self._runner_params, self._addon_params)
        if self.mode is SERVER:
            # Initialize OpenGL and setup widgets
            glfw.init()
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            self.window = glfw.create_window(1920, 1080, "", None, None)
            glfw.make_context_current(self.window)
            self._setup()

            # Release window so that the server thread can use it
            glfw.make_context_current(None)

            # Start server
            with serve(self._server_loop, ip, port, max_size=None, compression=None) as server:
                server_thread = threading.Thread(target=server.serve_forever)
                server_thread.start()
                while True:
                    try:
                        time.sleep(1)
                    except KeyboardInterrupt:
                        print("INFO: Shutting down server.")
                        server.shutdown()
                        server_thread.join()
                        break
            
            # Reacquire GLFW context and free resources
            glfw.make_context_current(self.window)
            self._destroy()

        self.running = False

    def step(self):
        """ Your application logic goes here. """
        pass

    def create_widgets(self):
        """ Define stateful widgets here. """
    
    def server_send(self) -> tuple[Optional[bytes],Optional[dict]]:
        """ Send global viewer state to the client. """
        return None, None
    
    def server_recv(self, binary: Optional[bytes], text: Optional[dict]):
        """ Receive and process global viewer state from the client. """

    def client_send(self) -> tuple[Optional[bytes],Optional[dict]]:
        """ Send global viewer state to the server. """
        return None, None

    def client_recv(self, binary: Optional[bytes], text: Optional[dict]):
        """ Receive and process global viewer state from the server. """
        pass

    def show_status(self):
        """ Use this function to render status bar at the bottom. """

    def import_server_modules(self):
        """
        Import server specific modules here. We want the viewer to run without 
        any additional dependancies so we only import the server modules when
        on the server. The modules imported here can only be used in `step` and
        `server*` methods. Don't forget to declare the variables as `global` to 
        ensure that they are globally accesible.
        """

    @abstractmethod
    def show_gui(self) -> bool:
        """ Define the GUI here. """