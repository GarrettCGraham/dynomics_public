import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import numpy
import pandas
from . import models


def get_as_strain_indexed(e):
    """ Return a DataFrame reprsenting the cell_trap of e using strain_idx as
        index, merging data for equivalent strain_idx
    """
    # copy cell trap dataframe to ensure that original experiment is
    # unaffected by a call to get_as_strain_indexed
    df = e.cell_trap.copy()

    # Create a map from device position to strain id
    position_to_strain_map = pandas.Series(
        data=e.loading_record.strain_idx.values,
        index=e.loading_record.device_position.values
    )

    # Add these strain ids to the cell trap dataframe to associate the device
    # positions with their strain ids
    df['strain_idx'] = position_to_strain_map

    # Filter positions with nan strain index values (meaning they are empty)
    df = df.loc[numpy.invert(df.strain_idx.isnull())]

    # use strain index as new index for cell trap
    df = df.set_index('strain_idx')

    # merge entries with identical strain ids
    df = df.groupby(df.index).mean()

    return df


def dim_reduction_plot(exps, reduct_type=PCA, reduct_args=None):
    ''' Generates 2d Scatter_Plots of the data for a list of Experiments.
        A call to this function will generate 2 plots. Before reduction, tf
        will be applied to the cell_trap data. After reduction, two scatter
        plots will be generated. The first ('{prefix}_classes.png')
        colors time points according to what induction class they belong to.
        The second ('{prefix}_experiments.png') colors time points
        accordingto what experiment they belonged too

        Input
            exps - [Experiment]
                A list of Experiment objects to analyze. Experiments should
                share strains but don't need to have the same mapping
            reduct_type -
                A reference to an sklearn object that will perform the
                reduction. (Optional: Default = sklearn.decomposition.PCA)
            reduct_args -
                A dict of args to pass to the sklearn object that will be
                performing the reduction.
                (Optional: Default = {})
        Returns - fig1, fig2
    '''
    # set args to be appropriate if None provided
    if(reduct_args is None):
        reduct_args = {}

    # get data from cell_trap and construct targets
    feats, targs = models.extract_features_targets(exps, as_1d=True, as_class=True)
    Y_exps = feats.index.get_level_values(0)
    le = LabelEncoder()
    targs = le.fit_transform(targs.values.ravel())


    # Apply standard (z-score) scaling to data
    scaler = StandardScaler()
    X = scaler.fit_transform(feats.values)

    # apply dimensionality reduction to data
    reducer = reduct_type(n_components=2, **reduct_args)
    X = reducer.fit_transform(X)

    x1 = X[:, 0]
    x2 = X[:, 1]
    cm = plt.cm.get_cmap('plasma')

    # plot the dimensionality reduced data points coloring points according
    # to toxin label
    fig1 = plt.figure()
    plt.scatter(x1, x2, c=targs, lw=0, cmap=cm)

    # plot the dimensionality reduced data points coloring points according
    # to experiment label
    fig2 = plt.figure()
    plt.scatter(x1, x2, c=Y_exps, lw=0, cmap=cm)

    return (fig1, fig2)
