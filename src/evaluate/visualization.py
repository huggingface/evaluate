import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ComplexRadar:
    """Create a complex radar chart with different scales for each variable
    Args:
    fig (`matplotlib.figure`) :  A matplotlib figure object to add the axes on.
    variables (`list`) : a list of variables to. plot
    ranges (`list` of `tuples`): A list of ranges (min, max) for each variable
    n_ring_levels (`int): Number of ordinate or ring levels to draw.
        Default: 5.
    show_scales (`bool`): Indicates if we the ranges for each variable are plotted.
        Default: True.
    format_cfg (`dict`): A dictionary with formatting configurations.
        Default: None.
    Returns:
    `matplotlib.figure.Figure`: a radar plot.
    """

    def __init__(self, fig, variables, ranges, n_ring_levels=5, show_scales=True, format_cfg=None):

        self.format_cfg = format_cfg

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

    def use_legend(self, *args, **kwargs):
        """Shows a legend"""
        self.ax1.legend(*args, **kwargs)


def radar_plot(data, model_names, invert_range=[], config=None, fig=None):
    """Create a complex radar chart with different scales for each variable
    Source: https://towardsdatascience.com/how-to-create-and-visualize-complex-radar-charts-f7764d0f3652

    Args:
        data (`List[dict]`): the results (list of metric + value pairs).
            E.g. data = [{"accuracy": 0.9, "precision":0.8},{"accuracy": 0.7, "precision":0.6}]
        names (`List[dict]`): model names.
            E.g. names = ["model1", "model 2", ...]
        invert_range (`List[dict]`, optional): the metrics to invert (in cases when smaller is better, e.g. speed)
            E.g. invert_range=["latency_in_seconds"]
        config (`dict`, optional) : a specification of the formatting configurations, namely:

            - rad_ln_args (`dict`, default `{"visible": True}`): The visibility of the radial (circle) lines.

            - outer_ring (`dict`, default `{"visible": True}`): The visibility of the outer ring.

            - angle_ln_args (`dict`, default `{"visible": True}`): The visibility of the angle lines.

            - rgrid_tick_lbls_args (`dict`, default `{"fontsize": 12}`): The font size of the tick labels on the scales.

            - theta_tick_lbls (`dict`, default `{"fontsize": 12}`): The font size of the variable labels on the plot.

            - theta_tick_lbls_pad (`int`, default `3`): The padding of the variable labels on the plot.

            - theta_tick_lbls_brk_lng_wrds (`bool`, default `True` ): Whether long words in the label are broken up or not.

            - theta_tick_lbls_txt_wrap (`int`, default `15`): Text wrap for tick labels

            - incl_endpoint (`bool`, default `False`): Include value endpoints on calse

            - marker (`str`, default `"o"`): the shape of the marker used in the radar plot.

            - markersize (`int`, default `3`): the shape of the marker used in the radar plot.

            - legend_loc (`str`, default `"upper right"`): the location of the legend in the radar plot. Must be one of: 'upper left', 'upper right', 'lower left', 'lower right'.

            - bbox_to_anchor (`tuple`, default `(2, 1)`: anchor for the legend.
        fig (`matplotlib.figure.Figure`, optional): figure used to plot the radar plot.

    Returns:
        `matplotlib.figure.Figure`
    """
    data = pd.DataFrame(data)
    data.index = model_names
    variables = data.keys()
    if all(x in variables for x in invert_range) is False:
        raise ValueError("All of the metrics in `invert_range` should be in the data provided.")
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
        "axes_args": {},
        "rad_ln_args": {"visible": True},
        "outer_ring": {"visible": True},
        "angle_ln_args": {"visible": True},
        "rgrid_tick_lbls_args": {"fontsize": 12},
        "theta_tick_lbls": {"fontsize": 12},
        "theta_tick_lbls_pad": 3,
        "theta_tick_lbls_brk_lng_wrds": True,
        "theta_tick_lbls_txt_wrap": 15,
        "incl_endpoint": False,
        "marker": "o",
        "markersize": 3,
        "legend_loc": "upper right",
        "bbox_to_anchor": (2, 1),
    }
    if config is not None:
        format_cfg.update(config)
    if fig is None:
        fig = plt.figure()
    radar = ComplexRadar(
        fig,
        variables,
        ranges,
        n_ring_levels=3,
        show_scales=True,
        format_cfg=format_cfg,
    )
    for g in zip(data.index):
        radar.plot(data.loc[g].values, label=g, marker=format_cfg["marker"], markersize=format_cfg["markersize"])
        radar.use_legend(**{"loc": format_cfg["legend_loc"], "bbox_to_anchor": format_cfg["bbox_to_anchor"]})
    return fig
