import pandas
import scipy
import numpy


def identity(x):
    return x


def rolling_apply(df, window_size, f, trim=True, **kargs):
    ''' A function that will apply a window wise operation to the rows of a
        dataframe, df.
        input:
            df - a pandas dataframe
            window_size - The size of windows to extract from each row series
            f - A function mapping f(numpy.ndarray) -> scalar
            trim - a boolean, if true, columns with null values will be trimmed
            kargs - Other parameters to pass to the Series.rolling constructor
        output:
            a dataframe
    '''
    data = df.apply(lambda x: x.rolling(window_size, **kargs).apply(f), axis=1)
    if(trim):
        data = data.dropna(axis=1, how='any')
    return data


def transform_standardized(cell_trap_frame, axis=1):
    """ Transforms a dataframe into a feature representation of the data
        standardizing each element by row. Use axis=0 to perform the
        transformation by column as a form of time normalization.
        ret = ( t(raw) - min(t_axis) ) / max(t_axis)
    """
    mins = cell_trap_frame.min(axis=axis)
    maxs = cell_trap_frame.max(axis=axis)
    temp = cell_trap_frame.sub(mins, axis=abs(axis - 1))
    return temp.div(maxs, axis=abs(axis - 1))


def resample(cell_trap_frame, resample_freq=60):
    cell_trap = cell_trap_frame.transpose()
    cell_trap.index = pandas.to_datetime(cell_trap.index, unit="s")
    cell_trap_resamp = cell_trap.resample("{0}T".format(resample_freq)).mean()
    data_plot = cell_trap_resamp.transpose()
    data_plot.columns = data_plot.columns.values.astype(numpy.int64) // 10**9
    return data_plot


def holt_winters_second_order_ewma(x, alpha, beta):

    """
    A smoothing function that takes a weighted mean of a point in a time
    series with respect to its history. alpha is the weight (ie, the relative
    importance) with which the time series most recent behavior is valued.
    Similarly, beta is the weight given to the most recent 1st derivative, w,
    when averaging the trend.
    input
    :param x: array or array-like
        Time series to be smoothed.
    :param alpha: float, (0,1)
        Weight given to most recent time point.
    :param beta: float, (0, 1)
        Weight given to most recent linear slope/trend.
    :return: s: numpy.array
        The smoothed time series.
    """
    N = x.size
    s = numpy.zeros((N, ))
    b = numpy.zeros((N, ))
    s[0] = x[0]
    for i in range(1, N):
        if(numpy.isnan(s[i-1])):
            s[i] = x[i]
        s[i] = alpha * x[i] + (1 - alpha) * (s[i - 1] + b[i - 1])
        b[i] = beta * (s[i] - s[i - 1]) + (1 - beta) * b[i - 1]
    return s


def smooth(cell_trap_frame, alpha=0.1, beta=0.001):
    return cell_trap_frame.apply(
        lambda x: holt_winters_second_order_ewma(x, alpha, beta),
        axis=1,
        raw=True)


def rolling_standardized_center(cell_trap_frame, window_size):
    ''' ret[a, b] = ( ctf[a,b] - min(ctf[a, b-ws/2:b+ws/2]) / max(ctf[a, b-ws/2:b+ws/2])
    '''
    return rolling_apply(
        cell_trap_frame,
        window_size,
        lambda x: (x[window_size / 2] - x.min()) / x.max(),
        center=True)


def rolling_standardized_right(cell_trap_frame, window_size):
    ''' let ctf = cell_trap_frame - min(cell_trap_frame) + 1
        ret[a,b] = ( ctf[a,b] - min(ctf[a, b-ws:b]) ) / max(ctf[a,b-ws:b])
    '''
    cell_trap_frame = cell_trap_frame - cell_trap_frame.min().min() + 1
    return rolling_apply(
        cell_trap_frame,
        window_size,
        lambda x: (x[-1] - x.min()) / x.max() )


def transform_z_score(cell_trap_frame, axis=0):
    ''' z_score the columns (axis=0) or rows(axis=1) of a dataframe.
    '''
    return cell_trap_frame.apply(
        scipy.stats.mstats.zscore,
        raw=True,
        axis=axis)


def rolling_z_score_center(cell_trap_frame, window_size):
    ''' ret[a,b] = zscore(ctf[a, b-ws/2:b+ws/2])[b]
    '''
    return rolling_apply(
        cell_trap_frame,
        window_size,
        lambda x: scipy.stats.mstats.zscore(x)[window_size / 2],
        center=True)


def rolling_z_score_right(cell_trap_frame, window_size):
    ''' ret[a.b] = zscore(ctf[a,b-ws:b])[b]
    '''
    return rolling_apply(
        cell_trap_frame,
        window_size,
        lambda x: (x[-1] - x.mean()) / x.std())


def normalize_time(cell_trap_frame):
    """ Transforms a dataframe by dividing each element by its column average.
        Applying this transformation effectively means that each element is
        now scaled in comparison to other elements observed at same time
        ret = t(raw) / mean(t_column)
    """
    means = cell_trap_frame.mean(axis=0)
    return cell_trap_frame.div(means, axis=1)


def transform_delta(cell_trap_frame, shift=False):
    ''' Computes the delta across rows, delta(T, T-1). By default, the values
        will associate with the t-1 label. Use shift=True to associate deltas
        with the T label
        ret = t(raw) - t-1(raw)
    '''
    deltas = cell_trap_frame.diff(axis=1)
    if(shift):
        deltas = deltas.shift(-1, axis=1)
        return deltas.iloc[:, :-1]

    else:
        return deltas.iloc[:, 1:]


def transform_derivative(cell_trap_frame):
    ''' computes first derivative of the cell trap data
        delta(signal)/delta(time)
    '''
    deltas = cell_trap_frame.diff(axis=1)
    times = pandas.Series(cell_trap_frame.keys())
    label_map = {k1: k2 for k1, k2 in zip(times.keys(), deltas.keys())}
    times = times.rename(label_map)
    delta_times = times.diff()
    ret = deltas.apply(lambda c: c / delta_times, axis=1)
    remap = {v: (i + v) / 2 for i,v in zip(ret.keys(), ret.keys()[1:])}
    ret = ret.rename(columns=remap)
    return ret


def transform_rof(cell_trap_frame):
    middle_values = rolling_apply(
        cell_trap_frame,
        2,
        lambda x: float(x[0] + x[1]) / 2)
    deltas = transform_delta(cell_trap_frame)
    return middle_values + deltas


#TODO: NH: make the interior print statement a warnings.waring(), instead of a print().
def drop_duplicates(df, subset=None, keep='first'):
    """ Drops duplicates from a DataFrame df.
        Params :
            df : DataFrame
                A dataFrame duplicates should be removed from
            subset : [obj]
                A list of column keys in df to care about when establishing
                if two entries are duplicates. Defaults to all column keys
            keep : 'first' or 'last' or False
                A rule to determine which duplicate entries should be kept.
                'first' means that the first entry of a duplicate should be
                kept while 'last' means that the last entry of a duplicate
                should be kept. If rule=False, then all duplicated entries
                will be dropped.
        Returns: DataFrame
    """
    data = df.loc[numpy.invert(df.duplicated(subset=subset, keep=keep))]
    if data.shape != df.shape:
        print('Dropped duplicates : Original - {0!s} : New - {1!s}'.format(
            df.shape, data.shape))
    return data


def add_mean_and_std(df):
    """ Return a copy of df with rows representing mean and standard
        deviation added. Mean corrosponds to index 0 and stddev is -1
    """
    mean_series = df.mean(axis=0)
    std_series = df.std(axis=0)
    ret = df.copy()
    ret.loc[0] = mean_series
    ret.loc[-1] = std_series
    return ret.sort_index()


feature_types = {
    'raw': lambda x: x,
    'normalized': normalize_time,
    'z_scored': lambda x: transform_z_score(x, axis=0),
    'raw`': transform_derivative,
    'raw`^2': lambda x: transform_derivative(x) ** 2,
    'raw``': lambda x: transform_derivative(transform_derivative(x)),
    'raw``^2': lambda x: transform_derivative(transform_derivative(x)) ** 2,
    'formation_rate': transform_rof,
    'standardized_10': lambda x: rolling_standardized_right(x, 10),
    'standardized_20': lambda x: rolling_standardized_right(x, 20),
    'standardized_30': lambda x: rolling_standardized_right(x, 30),
    'rolling_z-score_10': lambda x: rolling_z_score_right(x, 10),
    'rolling_z-score_20': lambda x: rolling_z_score_right(x, 20),
    'rolling_z-score_30': lambda x: rolling_z_score_right(x, 30),
    'smooth_rolling_z_score_30': lambda x: rolling_z_score_right(smooth(x),30),
    'smooth': lambda x: smooth(x),
    'smooth`': lambda x: transform_derivative(smooth(x)),
    'smooth`^2': lambda x: transform_derivative(smooth(x)) ** 2,
    'smooth``':
        lambda x: transform_derivative(transform_derivative(smooth(x))),
    'smooth``^2':
        lambda x: transform_derivative(transform_derivative(smooth(x))) ** 2,
    'smooth_resampled': lambda x: smooth(resample(x)),
    'smooth_resampled``':
        lambda x: smooth(resample(x)).diff(axis=1).diff(axis=1).dropna(axis=1, how='any'),
    'smooth_resampled``^2':
        lambda x: smooth(resample(x)).diff(axis=1).diff(axis=1).dropna(axis=1, how='any') ** 2,
    'transform_derivative': lambda x: transform_derivative(x),
    'sqrt': lambda x: numpy.sqrt(x),
    'norm_mean': lambda x: x.divide(x.mean().mean()),
    'abs': lambda x: x.abs(),
    'square': lambda x: x.multiply(x),
}
