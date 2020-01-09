import math

import numpy as np
import pandas as pd
import scipy


def get_histogram(values, bin_size=400, **calchist_kwargs):
    """
    Calculates the histogram for an array or set of arrays of signal values.
    """

    # Convert values to flattened numpy.array.
    values = np.array(values)

    # Flatten values to 1-D array.
    flattened_values = flatten_values(values)

    # Calculate number of bins to use.
    num_bins = calculate_number_bins(bin_size)

    # Calculate histogram with bins and values.
    hist, bin_edges =\
        calculate_histogram(
            flattened_values,
            num_bins=num_bins,
            **calchist_kwargs
        )

    return hist, bin_edges


def calculate_histogram(flattened_values, num_bins=0, **calchist_kwargs):
    """
    Calculates the histogram for an array or set of arrays of signal values.
    """

    # If no bins given, then allow numpy to optimize number of bins.
    if num_bins:
        num_bins = "auto"

    # Calculate histogram.
    hist, bin_edges = np.histogram(
        flattened_values,
        bins=num_bins,
        **calchist_kwargs
    )

    return hist, bin_edges


def flatten_values(values):
    """
    Flattens a nested array or array-like.

    :param values: array or array-like
    :return: numpy.array
        A flattened array with the same values as the passed array or
        array-like.
    """

    # Convert values to flattened numpy.array.
    values = np.array(values)

    # Flatten values to 1-D array.
    flattened_values = values.flatten()

    return flattened_values


def calculate_number_bins(flattened_values, bin_size=400):

    bin_size = float(bin_size)

    minimum = flattened_values.min()
    maximum = flattened_values.max()

    values_range = maximum - minimum

    num_bins = int(math.ceil(np.divide(values_range, bin_size)))

    return num_bins


def calculate_mode(flattened_values, max_ffctl_threshold=18000):
    """
    Calculates the major mode of the 1D list or list-like values passed to
    it and returns both the mode's value and the counts of the mode.

    :param flattened_values: numpy.array
    :param max_ffctl_threshold: int
        The maximum values to consider when calculating the dominant mode.
    :return: tuple of int
        The dominant mode and the number of time it occurs in the data.
    """

    flattened_values = [
        value for value in flattened_values if value < max_ffctl_threshold
    ]

    fulltrap_mean, count = scipy.stats.mstats.mode(flattened_values)

    return fulltrap_mean[0], count[0]


def test_trap_fullness(
        trap_values_df,
        fulltrap_mean,
        fulltrap_stdev,
        alpha_level=0.05
):
    """
    Tests whether or not a trap is a member of the normally-distributed
    population of full traps that are defined according to the passed mean
    and standard deviation. The test is performed using a hypothesis test
    with the passed significance level; the null hypothesis is that the trap
    is full.

    :param trap_values_df: pandas.DataFrame
    :param fulltrap_mean: float
    :param fulltrap_stdev: float
    :param alpha_level: float
    :return: pandas.DataFrame
        Multi-indexed the same as dynomics.Experiment.cell_trap_mushrooms.
        Contains a fullness binary indicator of the hypothesis-testing
        outcome, the associated p-values for those traps, the associated
        z-score for the trap, and the trap's TL values.
    """

    # Calculate the z-score at each time point for the trap's TL values,
    # assuming their a member of the full-trap normal distribution's
    # population, which is centered at fulltrap_mean and has standard
    # deviation fulltrap_stdev.
    trap_zscores =\
        np.divide((trap_values_df.values - fulltrap_mean), fulltrap_stdev)

    trap_pvalues = scipy.stats.norm.sf(abs(trap_zscores))

    trap_fullness = np.zeros(trap_values_df.shape[0])

    for i, zero in enumerate(trap_fullness):

        if trap_pvalues[i] >= alpha_level:

            trap_fullness[i] = 1

    return pd.DataFrame(
        data=[trap_fullness, trap_pvalues, trap_zscores, trap_values_df.values],
        columns=trap_values_df.index,
        index=["fullness", "pvalue", "zscore", "values"]
    )


def calculate_stdev(
        mean,
        fulltrap_threshold,
        num_samples,
        alpha_threshold=0.05
):

    stdev = -1*np.divide(
        np.multiply(
            (fulltrap_threshold - mean),
            np.sqrt(num_samples)
        ),
        scipy.special.erfinv(2.*alpha_threshold - 1)
    )

    return stdev


def calculate_fulltrap_threshold(position_df):
    """
    Calculates the TL threshold below which a trap is considered full. This
    value corresponds to the maximum value of the time series composed of
    the minimum values at each time point.  This value is the value of the
    fullest trap when the fullest trap is the least full of all the fullest
    traps.  The passed dataframe should only include data from equilibriated
    growth periods (ie, approx. 6 hours before the first toxin induction and
    after).

    :param position_df:
    :return: pandas.DataFrame
        The minimum values at each time point for the given device
        position's dataframe, along with the index of the trap with the
        lowest value.
    """

    # Initialize dataframe to store maximum value at each timepoint and owner's
    # trap index.
    minimums = pd.DataFrame(
        columns=["minimum", "trap_idx"],
        index=position_df.columns.copy()
    )

    # Loop through all time points and find max value between all four traps
    # for current time point.
    for time_idx in position_df.columns:

        current_min_value = \
            position_df[time_idx].values.min()

        current_min_trap_idx = \
            position_df[time_idx][
                position_df[time_idx] == current_min_value
                ].index.values

        minimums["minimum"].loc[time_idx] = \
            current_min_value

        minimums["trap_idx"].loc[time_idx] = \
            current_min_trap_idx

    return minimums


def test_device_position_fullness(
        position_df,
        growup_phase_endtime,
        alpha_level=0.05,
        alpha_threshold=0.01
):
    """
    Calculates a distribution for when the given device position's traps are
    full and then tests that positions traps' TL values at every time point to
    determine whether or not the specified trap is full at that moment.

    :param position_df: pandas.DataFrame
        A pandas.DataFrame holding the values of the TL data for all traps
        in the specified device position.
    :param growup_phase_endtime: int (reference unix time)
        The time point in reference unix time after which all device
        positions are assumed to have equilibriated in growth.
    :param alpha_level: float
        The statistical significance level against which the TL values
        should be tested.
    :param alpha_threshold: float
        The statistical significance value to be equated with the
        trap-is-full threshold value.
    :return: device_position_fullness_df: pandas.DataFrame
        A dataframe that contains, for each trap at each time point,
        a trap-is-full boolean, that trap's associated p-value, z-score,
        and current TL value.
    """

    # Flatten values.
    flattened_values = flatten_values(position_df.values)

    # Establish Gaussian distribution for position.
    # Calculate mode of position's traps.
    fulltrap_mean, count = calculate_mode(flattened_values)

    # Determine times occurring after growth equilibration.
    postgrowth_times =\
        [
            time for time in position_df.columns if time > growup_phase_endtime
        ]

    # Subset values occurring after equilibration.
    postgrowth_df = position_df[postgrowth_times].copy()

    # Calculate TL threshold below which a trap is full, assuming there is
    # always one full trap post-growth phase.
    trap_minimums =\
        calculate_fulltrap_threshold(postgrowth_df)
    fulltrap_threshold_value = \
        trap_minimums["minimum"].max()

    # Store the number of time points in the postgrowth data. This value
    # will be used for calculating a standard deviation for the
    # trap-is-full normal distribution.
    num_samples = postgrowth_df.shape[1]

    # Calculate variance for the given significance level, alpha_level.
    fulltrap_stdev =\
        calculate_stdev(
            fulltrap_mean,
            fulltrap_threshold_value,
            num_samples,
            alpha_threshold=alpha_threshold
        )

    # Initialize pandas.MultiIndex object to store TL values, fullness,
    # p-values, and z-scores.
    multiindex =\
        pd.MultiIndex.from_product(
            [[1, 2, 3, 4], ["fullness", "pvalue", "zscore", "values"]],
            names=["trap_idx", "attribute"]
        )

    # Initialize pandas.DataFrame with multiindex to store TL values, fullness,
    # p-values, and z-scores.
    device_position_fullness_df = pd.DataFrame(
        columns=position_df.columns.copy(),
        index=multiindex
    )

    # Loop through individual traps and test their fullness with respect to
    # their device position's trap-is-full distribution.
    for trap_idx in position_df.index:

        # Determine when the trap is full using the test_trap_fullness
        # function.
        trap_fullness_df =\
            test_trap_fullness(
                position_df.loc[trap_idx],
                fulltrap_mean,
                fulltrap_stdev,
                alpha_level=alpha_level
            )

        # Record the outcomes of test_trap_fullness for the current trap in
        # the predefined pandas.DataFrame object.
        device_position_fullness_df.loc[trap_idx] =\
            trap_fullness_df.values

    return device_position_fullness_df, fulltrap_mean, fulltrap_stdev


# List of tuples of engineered strains with strain index, device position for
# experiment 1102, and gene name.  Obtained using get_engineered_strains.
engineered_strains = [
    (16069.0, 1557, 'cadC(Cd1)'),
    (16070.0, 1559, 'arsR(As7)'),
    (16071.0, 1561, 'merR(Hg3)'),
    (16072.0, 1563, 'zntA(Pb7)'),
    (16073.0, 1565, 'zraP(Zn6)'),
    (16074.0, 1567, 'cusC(Cu3)'),
    (16069.0, 1575, 'cadC(Cd1)'),
    (16070.0, 1577, 'arsR(As7)'),
    (16071.0, 1579, 'merR(Hg3)'),
    (16072.0, 1581, 'zntA(Pb7)'),
    (16073.0, 1583, 'zraP(Zn6)'),
    (16074.0, 1585, 'cusC(Cu3)')
]


def get_transmittance(self):

    """
    Calculates the transmittance for each individual mushroom for the passed
    dynomics.Experiment.

    :param self: dynomics.Experiment
    :return: dynomics.Experiment
        Returns a new experiment instance with the transmittance calculated
        for each trap.
    """

    # Store the loaded device positions.
    pos_idxs_full = self.loading_record.device_position[
        ~self.loading_record.strain_idx.isnull()].values
    pos_idxs_all = self.loading_record.device_position.values

    # Make copies of the mushrooms' TL and the background TL.
    # tl = self.tlffc_mushrooms.copy()
    # background = self.tl_background.copy()
    tl = self.tl_mushrooms.copy()
    background = self.tl_background.copy()

    # Identify shared times.
    tl_times = set(tl.columns.tolist())
    background_times = set(background.columns.tolist())
    times = np.array(list(tl_times & background_times))

    # Python sets are unordered, so we need to re-order the times in
    # ascending order.
    times.sort()

    # Reset the TL dataframes to only include the shared times.
    tl = tl[times].copy()
    background = background[times].copy()

    # Initialize a dataframe to hold transmittance.
    transmittance = pd.DataFrame(
        data=tl.values,
        columns=tl.columns,
        index=tl.index
    )

    # Loop through all positions and calculate the transmittance.
    for pos_idx in pos_idxs_all:
        # Get the current position's TLs' values.
        current_trans = transmittance.loc[pos_idx].values
        current_background = background.loc[pos_idx].values

        # Replicate the background so that it is of the correct dimensions
        # for division (ie, the same shape as current_trans).
        current_backgrounds = np.array([current_background for i in range(4)])

        # Calculate transmittance and store.
        transmittance.loc[pos_idx] = \
            np.divide(current_trans, current_backgrounds)

    # Add transmittance as a new Experiment attribute.
    return self.set(transmittance=transmittance)


def get_od(self):
    """
    Calculates the optical density for each trap in a dynomics.Experiment's
    TL trap data.

    :param self: dynomics.Experiment
    :return: dynomics.Experiment
        Returns a new experiment instance with the optical density calculated
        for each trap.
    """

    # Get a copy of the experiment's transmittance data.
    try:
        transmittance = self.transmittance.copy()
    except AttributeError:
        self = get_transmittance(self)
        transmittance = self.transmittance.copy()

    # Initialize a dataframe to hold OD.
    od = pd.DataFrame(
        data=transmittance.values,
        columns=transmittance.columns,
        index=transmittance.index

    )

    # Calculate OD using Beer-Lambert Law.
    od = od.apply(lambda t: np.log10(np.divide(1., t)), axis=1)

    # Set OD dataframe as a new attribute in a new experiment object.
    return self.set(od=od)


def normalize_fl_by_od(self, od_power=1):
    """
    Normalizes a dynomics.Experiment's traps' signal trajectories by their
    optical densities.

    :param self: dynomics.Experiment
    :return: dynomics.Experiment
        Returns a new experiment instance with optical density-normalized
        signal intensity calculated for each trap.
    """

    # Get a copy of the experiment's OD data.
    try:
        od = self.od.copy()
    except AttributeError:
        self = get_od(self)
        od = self.od.copy()

    # Initialize dataframe to hold normalized FL values.
    fl_traps = self.cell_trap_mushrooms.copy()

    # Normalize background-subtracted FL by OD and set as a new attribute.
    fl_traps = fl_traps.divide(od**od_power)

    # Set OD-normalized FL dataframe as a new attribute in a new experiment
    # object.
    return self.set(fl_traps=fl_traps)
