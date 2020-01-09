import decimal
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Rectangle
import numpy
import os
import pandas
import time
import warnings

# Setting pdf.fonttype as 42 so that exported plots' fonts are editable in
# Adobe Illustrator/Photoshop, etc. Default fonttype is 3, which is NOT
# editable as normal text (rather, it appears as shapes, etc.; extremely
# tiresome to touch up for presentations, publications, etc.)
rcParams['pdf.fonttype'] = 42


def plot(
        trajectory_df,
        induction_record,
        fig=None,
        ax=None,
        exp_kind=None,
        exp_idx=None,
        cmap=None,
        save=False,
        save_to=None,
        legend=True,
        plot_title=None,
        plot_title_save=None,
        style=None,
        fig_dim=(7, 4.2),
        time_unit="h",
        x_label=None,
        y_label=None,
        x_label_fontsize=12,
        y_label_fontsize=12,
        xticks_fontsize=12,
        yticks_fontsize=12,
        title_fontize=16,
        legend_fontsize=12,
        num_legendcol=1,
        induction_label_fontsize=12,
        induction_label_x_pos=1.2,
        induction_label_y_pos=0.8,
        induction_label_rotation=90,
        linewidth=1,
        shade_inductions=True,
        is_classification=False,
        shade_clogs=False
):
    """
    Returns, displays, and saves a matplotlib.pyplot.plot()-generated,
    seaborn-styled figure with specified trajectories (strain by index,
    gene name, position, etc.) and trajectory types (cell_trap, cell_trap_raw,
    cell_trap_zscore, etc.) with induction regions shaded and annotated with
    inducer type and concentration.

    :param trajectory_df: dynomics.Experiment trajectory object.
        Object containing experimental data of interest.
    :param idxlist: list or list-like, default None
        A list of desired strains, etc. If left as None, a random subset of
        100 strains will be plotted.
    :param plot_title: str, default None
        Custom plot title. If None, plot() will add the expID and
        the experiment's title (self.record.title()) as the plot title.
    :param traj_type: str
        A string referring to a pandas.DataFrame attribute of box.Experiment
        that contains trajectories for plotting.
        Examples are "cell_trap", "background", etc.
    :param show: boolean, default True
        Whether or not to show the plot inline.
    :param save: boolean, default False
        Indicates whether or not to save the plot. If True, saves it to
        ../plots/expID_####/PLOT_TITLE.pdf.
    :param legend: boolean, default True
        Indicates whether or not to include a legend. Helpful to turn legends
        off for plots with many strains.
    :param context: str, default "talk"
        A seaborn figure option that reformats figure proportions to match
        various display contexts. Options include
        "talk", "notebook", "paper", and "poster".
    :param style: str, default "white"
        A seaborn figure option that formats figure style themes. Options
        include "darkgrid", "whitegrid", "dark",
        "white", and "ticks".
    :param fig_dim: tuple, default (15, 10)
        Figure dimensions in inches for the matplotlib figure canvas.
    :param time_unit: str, default "h"
        The time unit to display on the plot's x-axis. Options are "h" for
        hour and "m" for minute.
    :param x_label_fontsize: float, default 20
        Fontsize for x-axis' title.
    :param y_label_fontsize: float, default 20
        Fontsize for y-axis' title.
    :param xticks_fontsize: float, default 17
        Fontsize for x-axis' ticks' labels.
    :param yticks_fontsize: float, default 17
        Fontsize for y-axis' ticks' labels.
    :param title_fontize: float, default 30
        Fontsize for plot title.
    :return: plt: matplotlib.pyplot.plot object
        Figure and axes objects for the desired plot are returned for further
        customization or later display.
    """

    # Set plot style.
    if style is None:
        style = "seaborn-white"
    plt.style.use([style])

    # Set colormap as colormap_1 (one hundred distinct colors).
    if cmap is None:
        cmap = colormaps[1]

    # Initialize the index for unique color values.
    color_idx = 0

    # Initialize figure object and its axes attributes.
    if fig is None and ax is None:
        fig, ax =\
                plt.subplots(1, 1, figsize=fig_dim)
    elif fig is not None and ax is None:
        ax = fig.add_subplot(111, figsize=fig_dim)
    
    if is_classification:
        ax.set_ylim(bottom=1, top=9)

        toxin_labels = [
            'Water',
            'Arsenic',
            'Cadmium',
            'Cobalt',
            'Chromium',
            'Copper',
            'Mercury',
            'Malathion',
            'Lead'
        ]

        ax.set_yticklabels(toxin_labels)

    # Iterate through subsetted trajectories' data and plot on fig.axes.
    times = trajectory_df.columns.values
    if time_unit is "d":
        times = (times - times.min())/86400
    elif time_unit is "h":
        times = (times - times.min())/3600
    elif time_unit is "m":
        times = (times - times.min())/60
    else:
        warnings.warn("Please enter an acceptable time unit.")
        return

    # Loop through the unique device positions and plot.
    for strain in trajectory_df.index.values:

        # Plot the strain's trajectories with appropriate legend label.
        ax.plot(
            times,
            trajectory_df.loc[strain].values,
            label="{}".format(strain),
            color=cmap[color_idx],
            linewidth=linewidth
        )

        # Draw the matplotlib.Canvas. This method renders it possible to get
        # the axis' ticklabels for formatting.
        # plt.draw()
        fig.canvas.draw_idle()

        # Increment the color index for the next position and/or
        # trajectory type.
        color_idx = (color_idx + 1) % len(cmap)

    # Create major axis labels for both x- and y-axis.
    if x_label is None:
        x_label = "Time ({0})".format(time_unit)
    if y_label is None:
        y_label = "Intensity (a.u.)"

    # # Draw the matplotlib.Canvas. This method renders it possible to get
    # # the axis' ticklabels for formatting.
    # plt.draw()

    # Annotate y-axis label's properties and tick labels' fontsize.
    # ytick_labels = [label.get_text() for label in ax.get_yticklabels()]
    # print(ytick_labels)
    # ax.set_yticklabels(ytick_labels, fontsize=yticks_fontsize)

    ax.set_ylabel(y_label, fontsize=y_label_fontsize)

    # Get the x-axis' tick labels for formatting.
    x_tick_labels = ax.get_xticks()

    # Add x-axis labels, tick labels, and format tick fontsize.
    annotate_axis(
        ax.xaxis,
        tick_labels=x_tick_labels,
        axis_label=x_label,
        ticklabels_fontsize=xticks_fontsize,
        label_fontsize=x_label_fontsize,
        rotation=45
    )

    # Add title to figure.
    add_ax_title(
        ax,
        exp_kind=exp_kind,
        exp_idx=exp_idx,
        plot_title=plot_title,
        title_fontsize=title_fontize
    )

    # Overlay grey, transparent rectangles to denote inductions.
    if shade_inductions:

        # If the current experiment is ongoing, then set the most recent
        # induction's end time as the most recently imaged time point.
        if induction_record.end_time.isna().iloc[-1]:
            induction_record.end_time.iloc[-1] = times[-1]

        shade_induction_regions(
            trajectory_df,
            induction_record,
            ax,
            induction_label_fontsize,
            induction_label_x_pos,
            induction_label_y_pos,
            induction_label_rotation,
            time_unit=3600
        )
        fig.canvas.draw_idle()

    # Add legend if specified.
    if legend:
        lgd = ax.legend(
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                fontsize=legend_fontsize,
                ncol=num_legendcol
        )
    else:
        lgd = None

    # Create filename for saving figure.
    if plot_title_save is None:
        if exp_kind is None:
            exp_kind = ""
        if exp_idx is None:
            exp_idx = ""
        plot_title_save = "%s_%s_%s" % (exp_kind, exp_idx, now())

    # Recompute the ax.dataLim.
    ax.relim()

    # Update ax.viewLim using the new dataLim.
    ax.autoscale_view(True, True, True)
    # plt.draw()
    fig.canvas.draw_idle()

    # Save figure, if desired.
    if save:
        save_figure(
            fig,
            exp_idx,
            save_to=save_to,
            plot_title_save=plot_title_save,
            lgd=lgd
        )

    # # Display figure, if desired.
    # if show:
    #     fig.show()

    return fig, ax


def plot_experiment(
        experiment,
        idxlist,
        traj_type=None,
        **plot_kwargs
):

    # Set default trajectory type if none is passed and end the function if
    # a specified trajectory is not an attribute of the passed
    # dynomics.Experiment object.
    if traj_type is None:
        traj_type = ["cell_trap"]

    # Point to Experiment's induction record.
    induction_record = experiment.induction_record.copy()

    # Get info for plotting title.
    if plot_kwargs["exp_kind"] is None:
        plot_kwargs["exp_kind"] = experiment.kind
    elif plot_kwargs["exp_idx"] is None:
        plot_kwargs["exp_idx"] = experiment.idx
    elif plot_kwargs["exp_kind"] is None and plot_kwargs["exp_idx"] is None:
        plot_kwargs["exp_kind"] = experiment.kind
        plot_kwargs["exp_idx"] = experiment.idx


    # Loop through trajectories, isolate the desired trajectory's values,
    # and then loop through the desired device positions to plot.
    for traj in traj_type:

        # If a subset of strains from experiment.trajectories.traj_type are
        # desired, subset only the specified rows.
        if idxlist is None:
            trajectory_df = experiment.__get_attribute__(traj).copy()
        else:
            trajectory_df = get_plotting_data(experiment, idxlist, traj)

    fig, ax =\
        plot(
            trajectory_df,
            induction_record,
            **plot_kwargs
        )

    return fig, ax


def save_figure(fig, exp_idx=None, save_to=None, plot_title_save=None,
                lgd=None):

    if exp_idx is None:
        exp_idx = ""

    if save_to is None:
        save_to = "./plots/exp{}/".format(exp_idx)

    if not os.path.exists(save_to):
        os.makedirs(save_to)

    save_to += "{}.pdf".format(plot_title_save)

    if lgd is None:
        fig.savefig(
            save_to,
            bbox_inches='tight'
        )
    else:
        fig.savefig(
            save_to,
            bbox_extra_artists=(lgd,),
            bbox_inches='tight'
        )

    return


def add_ax_title(
        ax,
        exp_kind="Exp",
        exp_idx=0,
        plot_title=None,
        title_fontsize=16
):
    # Add default or custom plot title.
    if plot_title is None:
        plot_title_annotate = "%s %s" % (exp_kind, exp_idx)
    else:
        plot_title_annotate = plot_title

    # Add title to figure.
    ax.set_title(
        plot_title_annotate,
        fontsize=title_fontsize,
        y=1.02
    )

    return


def get_plotting_data(self, idxlist, traj):
    """
    Returns subsetted plotting data of the desired trajectory type and
    strains.

    Parameters
    ----------
    self: pandas.DataFrame
        DataFrame of trajectories from dynomics.Experiment or
        dynomics.Induction classes.
    idxlist: list or list-like
        List of desired strains by device position, row_column index,
        strain index, or gene name.

    Returns
    -------
    data: pandas.DataFrame
        DataFrame with appropriately subsetted trajectories for the
        desired strains, displaying the desired trajectories.
    """

    # If a subset of strains from self.trajectories.traj_type are desired,
    # subset only the specified rows.
    if idxlist is None:
        # If no strains are specified, return all strains.
        data = self.__getattribute__(traj).copy()
    else:
        # Use subset rows to return strains. Must assign copy of
        # trajectories to data outside of function statement below.
        data = self.__getattribute__(traj).copy()
        data = data.loc[idxlist]
    return data


def get_strain_description(self, device_position, col_name="gene_name"):

    # Relabel if mean or standard deviation are requested.
    if device_position == 0:
        return "mean"

    elif device_position == -1:
        return "std_dev"

    # In case of mushrooms, get the device position only.
    elif type(device_position) is tuple:

        device_position, shroom = device_position[0], device_position[1]

        strain_name = \
            self.loading_record[
                col_name
            ].loc[
                self.loading_record.device_position == device_position
                ].values[0]

        return "{0!s}: trap {1!s}".format(strain_name, shroom)

    # Return the desired strain descriptor for plot labeling.
    else:
        return self.loading_record[col_name].loc[
                    self.loading_record.device_position == device_position
               ].values[0]


def make_time_axis(t_list, t_unit):
    """
    Creates a time axis for annotating a plot from a list of reference unix
    times.

    Parameters
    ----------
    t_list: list or list-like
        A list of reference unix times to convert into times of t_unit (min, h,
        days, etc.).
    t_unit: str
        The unit of time to which to convert.

    Returns
    -------
    List of times converted to t_units.
    """
    # Set t_unit to the appropriate integer value.
    t_unit = convert_time_unit(t_unit)

    # Convert list of times to be in terms of desired time units.
    t_list = [t/t_unit for t in t_list]
    # decimal.getcontext().prec = 0
    return [float(decimal.Decimal("%i" % t)) for t in t_list]


def convert_time_unit(time_unit):
    """Retrieves the appropriate integer for converting a unix time stamp
    (in seconds) to minutes, hours, days, etc."""

    # Set t_unit to the appropriate integer value.
    if time_unit is "s":
        return 1.
    elif time_unit is "m":
        return 60.
    elif time_unit is "h":
        return 60.**2
    elif time_unit is "d":
        return 24.*60**2
    elif time_unit is "w":
        return 7*24.*60**2
    else:
        warnings.warn(
            "Passed argument time_unit is not viable."
        )
        return None


def now():
    year = time.localtime()[0]
    month = time.localtime()[1]
    day = time.localtime()[2]
    hour = time.localtime()[3]
    minute = time.localtime()[4]
    second = time.localtime()[5]

    return "{0}{1}{2}{3}{4}{5}".format(year, month, day, hour, minute, second)


def convert_trajectory_time_units(trajectory_df, time_unit):

    trajectory_times = trajectory_df.columns.copy().values
    trajectory_times =\
        (trajectory_times - trajectory_times.min()) / time_unit
    trajectory_df.columns = trajectory_times

    return trajectory_df


def shade_induction_regions(
        trajectory_df,
        induction_record,
        ax,
        induction_label_fontsize,
        induction_label_x_pos,
        induction_label_y_pos,
        induction_label_rotation,
        time_unit=3600
):

    """
    Marks induction regions in matplotlib.axes instance by shading them grey.

    :param trajectory_df: pandas.DataFrame
        The trajectories dataframe containing the data to be shaded over.
    :param induction_record: pandas.DataFrame
        The induction record associated with the trajectories data,
        containing transition and end times for the dataframe.
    :param ax: matplotlib.axes
        The axes instance that contains the plotted trajectories and on which
        the shaded regions are to be drawn.
    :param induction_label_fontsize: float
    :param induction_label_x_pos: float
        Position of the label's upper-left corner, as a fraction of
        the distance from the figure's upper-left corner.
    :param induction_label_y_pos: float
        Position of the label's upper-left corner, as a fraction of
        the distance from the figure's upper-left corner.
    :param induction_label_rotation: float
    :param time_unit: float

    :return: None
    """

    for induction_idx in induction_record.index:
        start = induction_record.transition_time.loc[induction_idx]
        end = induction_record.end_time.loc[induction_idx]

        start = start - trajectory_df.columns.values[0]
        end = end - trajectory_df.columns.values[0]

        induction_record.transition_time.loc[induction_idx] = start/time_unit
        induction_record.end_time.loc[induction_idx] = end/time_unit


    # Redefine the induction record's time axis to be in terms of time_units.
    trajectory_df = convert_trajectory_time_units(trajectory_df, time_unit)

    # Iterate through exp/ind object"s induction indices and shade/annotate
    # regions where toxin (ie, non-DI-H2O inductions occur).
    induction_list = induction_record.index.tolist()

    for induction in induction_list:

        current_induction_record = induction_record.loc[induction].copy()

        # If the induction is a water induction, continue to the next iteration
        # of the loop.
        if current_induction_record.stock_idx == 46:
            continue

        # Define start and end times for induction region shading.
        ind_slices = get_induction_slices(
            trajectory_df,
            current_induction_record
        )

        try:
            start = ind_slices[0]
        except IndexError:
            continue
        try:
            stop = ind_slices[-1]
        except IndexError:
            stop = trajectory_df.columns.aslist()[-1]

        # Get label for toxins to annotate shaded induction region.
        toxin_label = current_induction_record.combined_concentrations

        # Shade the induction regions.
        shade_region(
            ax,
            start,
            stop,
            label=toxin_label,
            label_fontsize=induction_label_fontsize,
            label_x_pos=induction_label_x_pos,
            label_y_pos=induction_label_y_pos,
            label_rotation=induction_label_rotation
        )

    return


def get_clog_transitions(clogs_df, algorithim_id=5, noise_threshold=20):

    clog_transitions = \
        clogs_df.loc[algorithim_id].loc[noise_threshold]

    return clog_transitions.diff(axis=1).dropna(axis=1)


def calculate_clog_transition_times(device_position, clogs_df, **clogs_kwargs):

    position_clog_indicators = get_clog_transitions(
        clogs_df,
        **clogs_kwargs
    )

    clog_transitions = position_clog_indicators.loc[device_position].copy()
    clog_transitions = clog_transitions[clog_transitions.values == 1]

    clog_transition_times = clog_transitions.index.values

    if clog_transition_times.shape[0] % 2 != 0:
        final_time = position_clog_indicators.columns.values[-1]
        clog_transition_times = numpy.append(clog_transition_times, final_time)

    num_clogs = clog_transition_times.shape[0] / 2
    clog_transition_times = numpy.reshape(clog_transition_times,
                                          (num_clogs, 2))

    return pandas.DataFrame(
        data=clog_transition_times,
        columns=[
            "transition_time",
            "end_time"
        ]
    )


def shade_clog_regions(
        trajectory_df,
        clogs,
        ax,
        clog_label_fontsize=12,
        clog_label_x_pos=1.2,
        clog_label_y_pos=0.5,
        clog_label_rotation=0,
        time_unit=3600
):

    clogs = convert_trajectory_time_units(clogs, time_unit)

    device_positions = trajectory_df.index.get_level_values(0)

    for device_position in device_positions:

        clog_transition_times = calculate_clog_transition_times(
            device_position,
            clogs
        )

        for clog_idx in clog_transition_times.index:

            start =\
                clog_transition_times.transition_time.loc[clog_idx]
            stop =\
                clog_transition_times.end_time.loc[clog_idx]

            # Shade the induction regions.
            shade_region(
                ax,
                start,
                stop,
                label="clogged",
                label_fontsize=clog_label_fontsize,
                label_x_pos=clog_label_x_pos,
                label_y_pos=clog_label_y_pos,
                label_rotation=clog_label_rotation,
                shade_color="red",
                shade_clog=True
            )

    return


def shade_region(
        ax,
        start,
        stop,
        label=None,
        label_fontsize=12,
        label_x_pos=1.2,
        label_y_pos=0.8,
        label_rotation=90,
        shade_color="black",
        shade_alpha=0.25,
        shade_clog=False
):
    # Assign default label value.
    if label is None:
        label = ""

    # Calculate coordinates for aesthetically-pleasing shaded grey inducer-
    # region rectangle.
    rect_height = numpy.diff(ax.get_ylim())[0]
    rect_width = stop - start
    rect_x_coord = start
    rect_y_coord = ax.get_ylim()[0]

    # Set the relative height at which the inducer label will be placed.
    label_y_coord = ax.get_ylim()[0] + \
        (label_y_pos * numpy.diff(ax.get_ylim())[0])

    label_x_coord =\
        (rect_x_coord + (label_x_pos * rect_width / 2.))

    # Add a grey rectangle over the specified region.
    rectangle = ax.add_patch(Rectangle((rect_x_coord, rect_y_coord),
                                       rect_width,
                                       rect_height,
                                       fill=True,
                                       color=shade_color,
                                       alpha=shade_alpha,
                                       edgecolor="black",
                                       lw=1))
    if shade_clog:
        rectangle.set_label("clogged")
    else:
        # Label to the shaded region.
        ax.annotate(label,
                    (label_x_coord,
                     label_y_coord),
                    color="black",
                    fontsize=label_fontsize,
                    ha="center",
                    va="center",
                    rotation=label_rotation)

    return


def get_induction_slices(trajectory_df, induction_record):
    """
    Retrieves all timepoints between the start and finish of an induction.
    Returns these timepoints as a numpy.ndarray.
    """

    # Store induction start and end times.
    start = induction_record.transition_time
    finish = induction_record.end_time

    # Store all times.
    t_list = trajectory_df.columns.tolist()

    # Isolate induction times.
    first = numpy.where(t_list > start)
    second = numpy.where(t_list < finish)
    t_idx = numpy.intersect1d(first, second)

    return [t_list[idx] for i, idx in enumerate(t_idx)]


def annotate_axis(
        axis,
        tick_labels=None,
        axis_label=None,
        ticklabels_fontsize=12,
        label_fontsize=12,
        rotation=0
):
    """
    Annotate the passed axis object of a matplotlib.Axes instance.
    """

    # Set default parameters if none are passed.
    if tick_labels is None:
        tick_labels = ""

    if axis_label is None:
        axis_label = ""

    # Modify properties of axis' tick labels.
    axis.set_ticklabels(tick_labels, **{'fontsize': ticklabels_fontsize})

    # Modify font properties of axis' main label.
    axis.set_label_text(axis_label, fontdict={'fontsize': label_fontsize})

    # Set angle of rotation for tick labels.
    # plt.setp(axis.get_majorticklabels(), rotation=rotation)
    axis.set_tick_params(
        **{'which':'major', 'labelrotation':rotation}
    )

    return

# Primary colormaps generated by mindboggle.mio.colors.distinguishable_colors.
colormaps ={ 1:
    [[ 0.62068966,  0.06896552,  1.        ],
    [ 0.        ,  0.5862069 ,  0.        ],
    [ 0.75862069,  0.20689655,  0.        ],
    [ 0.03448276,  0.51724138,  0.72413793],
    [ 0.68965517,  0.5862069 ,  0.        ],
    [ 0.72413793,  1.        ,  0.10344828],
    [ 0.        ,  0.86206897,  0.79310345],
    [ 0.96551724,  0.48275862,  0.65517241],
    [ 0.34482759,  0.31034483,  0.03448276],
    [ 0.44827586,  0.03448276,  0.24137931],
    [ 0.03448276,  0.24137931,  0.5862069 ],
    [ 0.        ,  0.31034483,  0.24137931],
    [ 0.62068966,  0.65517241,  1.        ],
    [ 0.48275862,  0.44827586,  0.48275862],
    [ 1.        ,  0.68965517,  0.48275862],
    [ 0.51724138,  0.5862069 ,  0.51724138],
    [ 0.51724138,  0.82758621,  1.        ],
    [ 0.27586207,  0.13793103,  0.        ],
    [ 0.13793103,  0.        ,  0.24137931],
    [ 1.        ,  0.93103448,  0.5862069 ],
    [ 0.79310345,  0.96551724,  0.79310345],
    [ 0.10344828,  0.89655172,  0.37931034],
    [ 0.        ,  0.55172414,  0.51724138],
    [ 0.93103448,  0.79310345,  1.        ],
    [ 0.75862069,  0.        ,  0.51724138],
    [ 0.34482759,  0.37931034,  0.34482759],
    [ 0.62068966,  0.48275862,  0.34482759],
    [ 0.51724138,  0.65517241,  0.        ],
    [ 0.27586207,  0.4137931 ,  1.        ],
    [ 0.37931034,  0.27586207,  0.48275862],
    [ 0.13793103,  0.17241379,  0.        ],
    [ 0.72413793,  0.65517241,  0.65517241],
    [ 0.06896552,  0.27586207,  0.34482759],
    [ 0.65517241,  0.48275862,  0.72413793],
    [ 0.24137931,  0.37931034,  0.06896552],
    [ 1.        ,  0.37931034,  0.34482759],
    [ 1.        ,  0.        ,  1.        ],
    [ 0.5862069 ,  0.34482759,  0.34482759],
    [ 0.72413793,  0.68965517,  0.55172414],
    [ 1.        ,  0.72413793,  0.13793103],
    [ 0.31034483,  0.27586207,  0.27586207],
    [ 0.17241379,  0.06896552,  0.10344828],
    [ 0.51724138,  0.31034483,  0.        ],
    [ 0.48275862,  0.48275862,  0.27586207],
    [ 0.31034483,  0.        ,  0.79310345],
    [ 0.34482759,  0.62068966,  1.        ],
    [ 0.51724138,  0.        ,  0.06896552],
    [ 0.37931034,  0.68965517,  0.48275862],
    [ 0.68965517,  1.        ,  1.        ],
    [ 1.        ,  0.75862069,  0.75862069],
    [ 0.82758621,  0.48275862,  0.        ],
    [ 0.        ,  0.68965517,  0.79310345],
    [ 0.72413793,  0.79310345,  0.79310345],
    [ 0.31034483,  0.48275862,  0.34482759],
    [ 0.        ,  0.10344828,  0.20689655],
    [ 0.        ,  0.37931034,  0.65517241],
    [ 0.86206897,  0.10344828,  0.34482759],
    [ 0.34482759,  0.37931034,  0.48275862],
    [ 0.51724138,  0.55172414,  0.5862069 ],
    [ 0.96551724,  0.86206897,  0.75862069],
    [ 0.79310345,  0.79310345,  0.        ],
    [ 0.62068966,  0.72413793,  0.48275862],
    [ 0.82758621,  0.86206897,  1.        ],
    [ 0.48275862,  0.48275862,  0.68965517],
    [ 0.03448276,  0.13793103,  0.13793103],
    [ 0.51724138,  0.72413793,  0.68965517],
    [ 0.20689655,  0.17241379,  0.06896552],
    [ 0.44827586,  0.37931034,  0.31034483],
    [ 0.24137931,  0.4137931 ,  0.4137931 ],
    [ 0.72413793,  0.51724138,  0.48275862],
    [ 0.48275862,  1.        ,  0.75862069],
    [ 0.51724138,  0.4137931 ,  0.        ],
    [ 0.37931034,  0.55172414,  0.5862069 ],
    [ 0.5862069 ,  0.        ,  0.62068966],
    [ 0.68965517,  0.65517241,  0.79310345],
    [ 1.        ,  0.37931034,  0.03448276],
    [ 0.62068966,  0.44827586,  1.        ],
    [ 0.34482759,  0.        ,  0.34482759],
    [ 0.03448276,  0.24137931,  0.        ],
    [ 0.17241379,  0.20689655,  0.24137931],
    [ 0.89655172,  1.        ,  0.62068966],
    [ 0.44827586,  0.31034483,  0.37931034],
    [ 1.        ,  1.        ,  0.86206897],
    [ 0.20689655,  0.17241379,  0.24137931],
    [ 0.        ,  0.        ,  0.51724138],
    [ 0.        ,  0.44827586,  0.17241379],
    [ 0.37931034,  0.51724138,  0.13793103],
    [ 1.        ,  0.5862069 ,  0.96551724],
    [ 0.62068966,  0.93103448,  0.51724138],
    [ 0.17241379,  0.89655172,  1.        ],
    [ 0.4137931 ,  0.79310345,  0.06896552],
    [ 0.68965517,  0.4137931 ,  0.51724138],
    [ 0.44827586,  0.34482759,  0.75862069],
    [ 0.4137931 ,  0.27586207,  0.24137931],
    [ 0.79310345,  0.62068966,  0.72413793],
    [ 0.24137931,  0.31034483,  0.20689655],
    [ 0.79310345,  0.62068966,  0.37931034],
    [ 0.        ,  0.68965517,  1.        ],
    [ 0.93103448,  0.86206897,  0.89655172],
    [ 0.65517241,  0.06896552,  0.31034483]]
    }
