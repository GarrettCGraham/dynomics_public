import decimal
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from numpy import diff, intersect1d, where
from scipy.cluster import hierarchy
from scipy.spatial import distance
from seaborn import diverging_palette, clustermap

from dynomics.plot import get_plotting_data


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
    decimal.getcontext().prec = 2
    return [float(decimal.Decimal("%.2f" % t)) for t in t_list]


def get_induction_slices(self, induction_record):
    """ Retrieves all timepoints between the start and finish of an induction.
    Returns these timepoints as a numpy.ndarray."""

    # Store induction start and end times.
    start = induction_record.transition_time
    finish = induction_record.end_time

    # Store
    t_list = self.cell_trap.columns.tolist()
    first = where(t_list > start)
    second = where(t_list < finish)

    induction_slices = intersect1d(first, second)

    return induction_slices


def shade_induction_regions(self,
                            ax,
                            induction_label_fontsize,
                            induction_label_x_pos,
                            induction_label_y_pos,
                            induction_label_rotation):

    """"
    Shades induction regions in matplotlib.axes instance, ax.
    """

    # Iterate through exp/ind object"s induction indices and shade/annotate
    # regions where toxin (ie, non-DI-H2O inductions occur).
    induction_list = self.induction_record.index.tolist()

    for induction in induction_list:

        induction_record = self.induction_record.loc[induction].copy()

        # If the induction is a water induction, continue to the next iteration
        # of the loop.
        if induction_record.stock_idx == 46:
            continue

        # Define start and end times for induction region shading.
        ind_slices = get_induction_slices(self, induction_record)
        start = ind_slices[0]
        stop = ind_slices[-1]
        
        # Define start and end times for induction region shading.
        ind_slices = get_induction_slices(self, induction_record)
        try:
            start = ind_slices[0]
        except IndexError:
            return
        try:
            stop = ind_slices[-1]
        except IndexError:
            stop = self.cell_trap.columns.aslist()[-1]

        # Get label for toxins to annotate shaded induction region.
        toxin_label = induction_record.combined_concentrations

        # Calculate coordinates for aesthetically-pleasing shaded grey inducer-
        # region rectangle.
        rect_height = diff(ax.get_ylim())[0]
        rect_width = stop - start
        rect_x_coord = start
        rect_y_coord = ax.get_ylim()[0]

        # Set the relative height at which the inducer label will be placed.
        label_y_coord = ax.get_ylim()[0] + \
            (induction_label_y_pos * diff(ax.get_ylim())[0])
        label_x_coord =\
            (rect_x_coord + (induction_label_x_pos * rect_width / 2.))

        # Add a shaded grey rectangle over the inducer region.
        ax.add_patch(Rectangle((rect_x_coord, rect_y_coord),
                               rect_width,
                               rect_height,
                               fill=True,
                               color="grey",
                               alpha=0.25,
                               edgecolor="grey",
                               lw=0.01))

        # Add inducer labels to inducer region after it has been shaded grey.
        ax.annotate(toxin_label,
                    (label_x_coord,
                     label_y_coord),
                    color="black",
                    fontsize=induction_label_fontsize,
                    ha="center",
                    va="center",
                    rotation=induction_label_rotation)
    return


def calculate_pdist(data_array, metric="correlation", **kwargs):
    """
    Calculates pairwise-distance between all rows in data_array using the
    defined metric. From scipy.spatial.distance.pdist documentation.

    Parameters
    ----------
    data_array: numpy.ndarray
        A 2D M x N array of M real-valued N-dimensional data points.
    metric: str
        Distance metric to be used. Can be "braycurtis", "canberra",
        "chebyshev", "cityblock", "correlation", "cosine", "dice", "euclidean",
        "hamming", "jaccard", "kulsinski", "mahalanobis", "matching",
        "minkowski", "rogerstanimoto", "russellrao", "seuclidean",
        "sokalmichener", "sokalsneath", "sqeuclidean", "yule".

    Returns
    -------
    Y : numpy.ndarray
        Returns a condensed distance matrix Y. For each i and j (where i<j<n),
        the metric dist(u=X[i], v=X[j]) is computed and stored in entry ij.
        """
    return distance.pdist(data_array, metric=metric, **kwargs)


def calculate_linkage(pairwise_dists, method="average", **kwargs):
    """
    Calculates clusters using a condensed matrix of pairwise distances using the defined method.
    See scipy.cluster.hierarchy.linkage for additional documentation. From scipy.cluster.hierarchy.linkage
    documentation.
    :param pairwise_dists: numpy.ndarray
        A condensed distance matrix as returned by scipy.spatial.distance.pdist.
    :param method: str, default "average".
        Valid methods are 'single', 'complete', 'weighted', 'centroid', 'median' and 'ward'.
    :param kwargs: dict
        Keyword arguments for scipy.cluster.hierarchy.linkage.
    :return:
        Hierarchical clustering encoded as a linkage matrix.
    """

    return hierarchy.linkage(pairwise_dists, method=method, **kwargs)


def convert_time_unit(time_unit):
    """Retrieves the appropriate integer for converting a unix time stamp
    (in seconds) to minutes, hours, days, etc."""

    # Set t_unit to the appropriate integer value.
    if time_unit is "s":
        t_unit = 1.
    elif time_unit is "m":
        t_unit = 60.
    elif time_unit is "h":
        t_unit = 60.**2
    elif time_unit is "d":
        t_unit = 24.*60**2
    elif time_unit is "w":
        t_unit = 7*24.*60**2

    return t_unit


def reset_axes_pos_size(ax, x0=None, y0=None, width=0.85, height=0.85):
    """ Resets position and size of matplotlib.Axes """
    # Get current position of axes object.
    pos = ax.get_position()

    # Assign current values to x0 and y0.
    if x0 is None:
        x0 = pos.x0
    if y0 is None:
        y0 = pos.y0

    # Redraw axes object to fit its figure.
    ax.set_position([x0, y0, width, height])

    return


def annotate_axis(axis,
                  tick_labels="",
                  label="",
                  ticklabels_fontsize=12,
                  label_fontsize=12,
                  rotation=0):
    """Annotate the passed axis object of a matplotlib.Axes instance."""

    axis.set_ticklabels(tick_labels, **{'fontsize': ticklabels_fontsize})
    axis.set_label_text(label, fontdict={'fontsize': label_fontsize})
    # Set angle of rotation for tick labels.
    plt.setp(axis.get_majorticklabels(), rotation=rotation)
    return


def heatmap(
        self,
        idxlist=None,
        traj_type=["cell_trap"],
        pdist_metric="correlation",
        pdist_kwargs={},
        linkage_method="average",
        linkage_kwargs={},
        row_cluster=True,
        level_name=None,
        plot_title=None,
        figsize=None,
        time_unit="h",
        yticklabels=False,
        x_label_fontsize=20,
        y_label_fontsize=20,
        xticks_fontsize=10,
        yticks_fontsize=3,
        induction_label_fontsize=12,
        induction_label_x_pos=1.2,
        induction_label_y_pos=0.8,
        induction_label_rotation=90):

    """
    Creates heatmap of specified box.Experiment or box.Induction trajectories.
    Clusters trajectories with most similar trajectories by default, but this
    feature can be turned off by setting row_cluster = False.

    :param self: box.Experiment or box.Induction instance.
    :param idxlist:list or list-like, default None
        A list of desired strains, etc. If left as None, trajectories for all
        strains will be clustered.
    :param traj_type: str
        A string referring to a pandas.DataFrame attribute of box.Experiment that contains trajectories for plotting.
        Examples are "cell_trap", "background", etc.
    :param pdist_metric: str
        The metric with which to calculate pairwise-distance. See cluster.calculate_pdist
        documentation for valid metrics.
    :param pdist_kwargs: dict
        Dict of keyword arguments accepted by scipy.spatial.distance.pdist.
    :param linkage_method: str, default "average"
        The linkage method to use.  See cluster.calculate_pdist for more information.
    :param linkage_kwargs: dict
        Dict of keyword arguments accepted by scipy.cluster.hierarchy.linkage.
    :param row_cluster: bool, default True
        True indicates trajectories will be clustered and then plotted together.
    :param level_name: str, default None
        The level_name from which to retrieve the strains indicated in idxlist.
    :param plot_title: str, default None
        Desired plot title.
    :param figsize: tuple, default None
        Desired figure size in inches.
    :param time_unit: str, default "h"
        The unit of time to use for x-axis labeling.
    :param yticklabels: bool, default False
        True indicates that strain identifying information will be plotted
        on the y-axis major ticks.
    :param x_label_fontsize: int, default 20
        Fontsize for x-axis' label.
    :param y_label_fontsize: int, default 20
        Fontsize for y-axis' label.
    :param xticks_fontsize: int, default 10
        Fontsize for x-axis' ticks' labels.
    :param yticks_fontsize: int, default 3
        Fontsize for y-axis' ticks' labels.
    :param induction_label_fontsize: int, default 12
        Fontsize for induction region labels.
    :param induction_label_x_pos: float, default 1.2
    :param induction_label_y_pos: float, default 0.8
    :param induction_label_rotation: float, default 90

    :return: n: seaborn.ClusterGrid instance
        See seaborn.clustermap for documentation.
    """

    # Define custom divergent color palette.
    cmap = diverging_palette(h_neg=259,
                             h_pos=0,
                             s=90,
                             sep=16,
                             n=12,
                             as_cmap=True)

    # Get trajectory type (ie, cell_trap, cell_trap_raw, background, etc.).
    traj = traj_type[0]

    # Subset data to the desired strains and trajectories.
    if idxlist is None:
        data = self.__getattribute__(traj)
    else:
        data = get_plotting_data(self, idxlist, traj)

    # Eliminate missing values.
    data.interpolate(inplace=True)

    # Convert to numpy.ndarray.
    data_array = data.values

    if row_cluster:
        # Calculate pairwise distance and linkage.
        pairwise_dists = calculate_pdist(data_array,
                                         metric=pdist_metric,
                                         **pdist_kwargs)
        linkage = calculate_linkage(pairwise_dists,
                                    method=linkage_method,
                                    **linkage_kwargs)
    else:
        linkage = None

    # Customize time labels for x-axis.
    t_list = data.columns.tolist()
    t_list_zeroed = t_list - min(t_list)
    x_axis_labels = make_time_axis(t_list_zeroed, time_unit)
    for i, l in enumerate(x_axis_labels):
        if i % 50 != 0:
            x_axis_labels[i] = ""

    # Plot heatmap with desired clustering.
    n = clustermap(data,
                   standard_scale=0,
                   figsize=figsize,
                   row_cluster=row_cluster,
                   col_cluster=False,
                   row_linkage=linkage,
                   cmap=cmap,
                   xticklabels=True,
                   yticklabels=yticklabels,
                   cbar=False)

    # Eliminate scaling markers for color bar.
    n.cax.set_visible(False)

    # Set current axes object to be the heatmap.
    ax = n.ax_heatmap

    # Overlay grey, transparent rectangles to denote inductions.
    shade_induction_regions(self,
                            ax,
                            induction_label_fontsize,
                            induction_label_x_pos,
                            induction_label_y_pos,
                            induction_label_rotation)

    # Delete row dendrogram subplot axes instance.
    plt.delaxes(plt.gcf().axes[0])
    # Delete column dendrogram subplot axes instance.
    plt.delaxes(plt.gcf().axes[0])

    # Reset current axes object to be the heatmap.
    ax = n.ax_heatmap

    # Add a plot title.
    if plot_title is not None:
        ax.set_title(plot_title, fontdict={"fontsize": 25})

    # Annotate x-axis and y-axis.
    xlabel = 'Time (%s)' % time_unit
    ylabel = 'Gene name'
    y_axis_labels = data.index.tolist()

    annotate_axis(ax.xaxis,
                  tick_labels=x_axis_labels,
                  label=xlabel,
                  ticklabels_fontsize=xticks_fontsize,
                  label_fontsize=x_label_fontsize,
                  rotation=45)

    if yticklabels:
        annotate_axis(ax.yaxis,
                      tick_labels=y_axis_labels,
                      label=ylabel,
                      ticklabels_fontsize=yticks_fontsize,
                      label_fontsize=y_label_fontsize,
                      rotation=0)
        reset_axes_pos_size(ax, x0=0.025, width=0.85)
    # Clear y-axis ticklabels if not desired.
    else:
        ax.yaxis.label.set_text("")
        plt.draw()  # Redraw the plot without y-axis label.
        reset_axes_pos_size(ax, x0=0.025, width=0.95)

    ax.set_title("%s %s: %s" % (self.kind, self.idx, self.organism.split("-")[0]),
                 fontsize=y_label_fontsize)

    return n
