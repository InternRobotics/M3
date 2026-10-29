import numpy as np
from . import Widget
from ..types import ViewerMode
from typing import List
from imgui_bundle import implot, imgui

class PerformanceMonitor(Widget):
    # TODO: Support single field
    def __init__(self, mode: ViewerMode, fields: List[str], add_other: bool=True, history: int=100):
        """
        Show the statistics and plot of time taken per frame as a realtime line plot.
        
        Args:
            fields: The list of passes that you want to plot.
            add_others: Add a new field named 'Other' which accounts for remaining frame
                        time which hasn't been measured. If it is `True` then the last
                        field is assumed to be the total frame time.
            history: Length of history to be kept.
        """
        super().__init__(mode)
        self.add_other = add_other
        if add_other:
            self.fields= fields[:-1] + ["Other"]
        else:
            self.fields = fields
        self.history = history

        self.times = {}
        for field in self.fields:
            self.times[field] = np.zeros(history)

        self.offset = 0
        self.fps = np.array([1000/60, 1000/30, 1000/16])

        # We need this because there is a bug in ImPlot which doesn't consider
        # the `offset` for the X-Axis.
        x = np.arange(self.history, dtype=np.float64)
        self._x = []
        for i in range(self.history):
            self._x.append(np.roll(x, i))

    def step(self, times: List[float]):
        if self.add_other:
            times[-1] = times[-1] - sum(times[:-1])

        for i in range(len(times)):
            self.times[self.fields[i]][self.offset] = times[i]
            # Cumsum
            if i:
                self.times[self.fields[i]][self.offset] += self.times[self.fields[i-1]][self.offset]

        self.offset += 1
        self.offset %= self.history

    def show_gui(self):
        if implot.begin_plot("Frame Time", imgui.ImVec2(-1, -1)):
            implot.setup_legend(implot.Location_.north_west, flags=implot.LegendFlags_.horizontal | implot.LegendFlags_.no_buttons)
            implot.setup_axis_limits(implot.ImAxis_.x1, 0, self.history)
            implot.setup_axis_limits(implot.ImAxis_.y1, 0, 1000 / 14)   # Anything above 16 FPS is bad
            implot.setup_axes("Frame", "Time (ms)", x_flags=implot.AxisFlags_.lock_min | implot.AxisFlags_.lock_max, y_flags=implot.AxisFlags_.lock_min)
            implot.push_style_var(implot.StyleVar_.fill_alpha, 0.25)
            for i in range(len(self.fields)):
                if i == 0:
                    implot.plot_shaded(self.fields[i], values=self.times[self.fields[i]], offset=self.offset)
                else:
                    implot.plot_shaded(self.fields[i], self._x[self.offset], ys2=self.times[self.fields[i]], ys1=self.times[self.fields[i-1]], offset=self.offset)
                implot.plot_line(self.fields[i], self.times[self.fields[i]], offset=self.offset)
            implot.plot_inf_lines("FPS", self.fps, flags=implot.InfLinesFlags_.horizontal)
            implot.end_plot()
    
    def server_send(self):
        return None, { "offset": self.offset, "times": { field: times[self.offset] for field, times in self.times.items() } }
    
    def client_recv(self, _, text):
        self.offset = text["offset"]
        for field, times in self.times.items():
            times[self.offset] = text["times"][field]