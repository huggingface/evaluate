import textwrap

import matplotlib.pyplot as plt
import numpy as np


class ComplexRadar:
    """
    Create a complex radar chart with different scales for each variable
    Source: https://towardsdatascience.com/how-to-create-and-visualize-complex-radar-charts-f7764d0f3652

    Parameters
    ----------
    fig : figure object
        A matplotlib figure object to add the axes on
    variables : list
        A list of variables
    ranges : list
        A list of tuples (min, max) for each variable
    n_ring_levels: int, defaults to 5
        Number of ordinate or ring levels to draw
    show_scales: bool, defaults to True
        Indicates if we the ranges for each variable are plotted
    format_cfg: dict, defaults to None
        A dictionary with formatting configurations

    """

    def __init__(self, fig, variables, ranges, n_ring_levels=5, show_scales=True, format_cfg=None):

        # Default formatting
        self.format_cfg = {
            # Axes
            # https://matplotlib.org/stable/api/figure_api.html
            "axes_args": {},
            # Tick labels on the scales
            # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.rgrids.html
            "rgrid_tick_lbls_args": {"fontsize": 8},
            # Radial (circle) lines
            # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.grid.html
            "rad_ln_args": {},
            # Angle lines
            # https://matplotlib.org/3.2.2/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D
            "angle_ln_args": {},
            # Include last value (endpoint) on scale
            "incl_endpoint": False,
            # Variable labels (ThetaTickLabel)
            "theta_tick_lbls": {"va": "top", "ha": "center"},
            "theta_tick_lbls_txt_wrap": 15,
            "theta_tick_lbls_brk_lng_wrds": False,
            "theta_tick_lbls_pad": 25,
            # Outer ring
            # https://matplotlib.org/stable/api/spines_api.html
            "outer_ring": {"visible": True, "color": "#d6d6d6"},
        }

        if format_cfg is not None:
            self.format_cfg = {
                k: (format_cfg[k]) if k in format_cfg.keys() else (self.format_cfg[k]) for k in self.format_cfg.keys()
            }

        # Calculate angles and create for each variable an axes
        # Consider here the trick with having the first axes element twice (len+1)
        angles = np.arange(0, 360, 360.0 / len(variables))
        axes = [
            fig.add_axes([0.1, 0.1, 0.9, 0.9], polar=True, label="axes{}".format(i), **self.format_cfg["axes_args"])
            for i in range(len(variables) + 1)
        ]

        # Ensure clockwise rotation (first variable at the top N)
        for ax in axes:
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_axisbelow(True)

        # Writing the ranges on each axes
        for i, ax in enumerate(axes):

            # Here we do the trick by repeating the first iteration
            j = 0 if (i == 0 or i == 1) else i - 1
            ax.set_ylim(*ranges[j])
            # Set endpoint to True if you like to have values right before the last circle
            grid = np.linspace(*ranges[j], num=n_ring_levels, endpoint=self.format_cfg["incl_endpoint"])
            gridlabel = ["{}".format(round(x, 2)) for x in grid]
            gridlabel[0] = ""  # remove values from the center
            lines, labels = ax.set_rgrids(
                grid, labels=gridlabel, angle=angles[j], **self.format_cfg["rgrid_tick_lbls_args"]
            )

            ax.set_ylim(*ranges[j])
            ax.spines["polar"].set_visible(False)
            ax.grid(visible=False)

            if show_scales is False:
                ax.set_yticklabels([])

        # Set all axes except the first one unvisible
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.xaxis.set_visible(False)

        # Setting the attributes
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
        self.ax1 = axes[1]
        self.plot_counter = 0

        # Draw (inner) circles and lines
        self.ax.yaxis.grid(**self.format_cfg["rad_ln_args"])
        # Draw outer circle
        self.ax.spines["polar"].set(**self.format_cfg["outer_ring"])
        # Draw angle lines
        self.ax.xaxis.grid(**self.format_cfg["angle_ln_args"])

        # ax1 is the duplicate of axes[0] (self.ax)
        # Remove everything from ax1 except the plot itself
        self.ax1.axis("off")
        self.ax1.set_zorder(9)

        # Create the outer labels for each variable
        l, text = self.ax.set_thetagrids(angles, labels=variables)

        # Beautify them
        labels = [t.get_text() for t in self.ax.get_xticklabels()]
        labels = [
            "\n".join(
                textwrap.wrap(
                    label,
                    self.format_cfg["theta_tick_lbls_txt_wrap"],
                    break_long_words=self.format_cfg["theta_tick_lbls_brk_lng_wrds"],
                )
            )
            for label in labels
        ]
        self.ax.set_xticklabels(labels, **self.format_cfg["theta_tick_lbls"])

        for t, a in zip(self.ax.get_xticklabels(), angles):
            if a == 0:
                t.set_ha("center")
            elif a > 0 and a < 180:
                t.set_ha("left")
            elif a == 180:
                t.set_ha("center")
            else:
                t.set_ha("right")

        self.ax.tick_params(axis="both", pad=self.format_cfg["theta_tick_lbls_pad"])

    def _scale_data(self, data, ranges):
        """Scales data[1:] to ranges[0]"""
        for d, (y1, y2) in zip(data[1:], ranges[1:]):
            assert (y1 <= d <= y2) or (y2 <= d <= y1)
        x1, x2 = ranges[0]
        d = data[0]
        sdata = [d]
        for d, (y1, y2) in zip(data[1:], ranges[1:]):
            sdata.append((d - y1) / (y2 - y1) * (x2 - x1) + x1)
        return sdata

    def plot(self, data, *args, **kwargs):
        """Plots a line"""
        sdata = self._scale_data(data, self.ranges)
        self.ax1.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kwargs)
        self.plot_counter = self.plot_counter + 1

    def fill(self, data, *args, **kwargs):
        """Plots an area"""
        sdata = self._scale_data(data, self.ranges)
        self.ax1.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kwargs)

    def use_legend(self, *args, **kwargs):
        """Shows a legend"""
        self.ax1.legend(*args, **kwargs)

    def set_title(self, title, pad=25, **kwargs):
        """Set a title"""
        self.ax.set_title(title, pad=pad, **kwargs)


def radar_plot(data, model_names, invert_range=[]):
    """
    `data`: list of `dict`s of metric + value pairs.
        E.g. data = [{"accuracy": 0.9, "precision":0.8},{"accuracy": 0.7, "precision":0.6},]
    `names`: list of `str`s with model names
        E.g. names = ["model1", "model 2", ...]
    `invert_range`: list of `str`s with the metrics to invert (in cases when lower is better, e.g. speed)
        E.g. invert_range=["latency_in_seconds"]
    """
    variables = data.keys()
    if all(x in variables for x in invert_range) is False:
        raise ValueError("All of the metrics in `invert_range` should be in the data provided.")
    if isinstance(data, list) is False:
        raise ValueError("The input must be a list of dicts of metric + value pairs")
    min_max_per_variable = data.describe().T[["min", "max"]]
    min_max_per_variable["min"] = min_max_per_variable["min"] - 0.1 * (
        min_max_per_variable["max"] - min_max_per_variable["min"]
    )
    min_max_per_variable["max"] = min_max_per_variable["max"] + 0.1 * (
        min_max_per_variable["max"] - min_max_per_variable["min"]
    )

    ranges = list(min_max_per_variable.itertuples(index=False, name=None))
    ranges = [
        (max_value, min_value) if var in invert_range else (min_value, max_value)
        for var, (min_value, max_value) in zip(variables, ranges)
    ]
    format_cfg = {
        "rad_ln_args": {"visible": True},
        "outer_ring": {"visible": True},
        "angle_ln_args": {"visible": True},
        "rgrid_tick_lbls_args": {"fontsize": 12},
        "theta_tick_lbls": {"fontsize": 12},
        "theta_tick_lbls_pad": 3,
        "theta_tick_lbls_brk_lng_wrds": True,
    }
    fig = plt.figure(figsize=(6, 12), dpi=300)
    radar = ComplexRadar(fig, variables, ranges, n_ring_levels=3, show_scales=True, format_cfg=format_cfg)
    for g in zip(data.index):
        radar.plot(data.loc[g].values, label=g, marker="o", markersize=3)
        radar.use_legend(**{"loc": "upper right", "bbox_to_anchor": (2, 1)})
    return fig
