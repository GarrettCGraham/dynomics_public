from matplotlib.colors import LogNorm
from sklearn.model_selection import LeavePGroupsOut
from sklearn.metrics import confusion_matrix

import pandas
import numpy
from matplotlib import patches, pyplot
from collections import Counter

from . import plot


def extract_features_targets(
    processed_exps,
    dropna=True,
    as_class=False,
    use_tox_ids=False,
    as_1d=False
):
    """ Extract data and targets from a set of experiments. Features will
        be extracted from each experiment's cell trap attributes. Targets are
        computed using the induction_states method.
        Input:
            processed_exps : [dynomics.Experiment]
                A list of experiment objects to extract data from. Data will
                be extracted from the cell_trap attribute of each experiment
            dropna : bool
                whether or not to drop time points with non-finite values from
                the dataset
            as_class : bool
                If true, then the targets will be presence/absence of each toxin,
                rather than the concentration of each toxin.
            use_tox_ids : bool
                if true, toxin ids will be used in the returned target_frame
                instead of toxin_short_names
            as_1d : bool
                if true, return targets as Nx1 dimensional dataframe. This flag
                is useful for non-multilabel problems
        Returns: (DataFrame, DataFrame)
        A dataframe respresenting the features and a dataframe representing the
        targets for for all experiments
    """
    # extract data from cell_trap attributes
    feature_frame = [e.cell_trap.transpose() for e in processed_exps]
    feature_frame = pandas.concat(
        feature_frame,
        axis=0,
        keys=[e.idx for e in processed_exps],
        names=['exp_id']
    )
    # extract targets from experiments
    target_frame = [
        e.induction_states(
            as_class=as_class,
            use_tox_ids=use_tox_ids,
            as_1d=as_1d
        )
        for e in processed_exps
    ]
    target_frame = pandas.concat(
        target_frame,
        axis=0,
        keys=[e.idx for e in processed_exps],
        names=['exp_id']
    )
    target_frame = target_frame.fillna(False if as_class else 0.0)
    # Drop nan/inf values if desired
    if(dropna):
        pre_shape = feature_frame.shape
        feature_frame[feature_frame == numpy.inf] = numpy.nan
        feature_frame = feature_frame.dropna(axis=0)
        if(feature_frame.shape != pre_shape):
            fdoc = 'Data shape before and after nan/inf removal : {0!s} {1!s}'
            print(fdoc.format(pre_shape, feature_frame.shape))
        target_frame = target_frame.loc[feature_frame.index]
    return feature_frame, target_frame


def construct_exp_splits(feature_frame, leave_n_out=1):
    """ Constructs a list of (train,test) splits for a feature_frame
        representing a set of experiments. These splits used integer
        based (as opposed to label based) indexing of feature_frame.
        Input:
            feature_frame : DataFrame
                A pandas dataframe returned by extract_features_targets
                representing multiple experiments
            leave_n_out : int
                The number of experiments to leave out in each cross validation
                fold
        Returns: [(Array, Array)]
            A list of (train index, test index) splits
    """
    groups = feature_frame.index.get_level_values(0)
    logo = LeavePGroupsOut(n_groups=leave_n_out)
    df_mat = feature_frame.values
    cv_splits = [
        (train_index, test_index)
        for train_index, test_index in logo.split(df_mat, groups=groups)
    ]
    return cv_splits


def extract_features_targets_splits(
    processed_exps,
    dropna=True,
    as_class=False,
    use_tox_ids=False,
    as_1d=False,
    leave_n_out=1
):
    """ Convenience function allowing for extracting features, targets,
        and expeirment wise cv splits for a set of experiments.
        Input:
            processed_exps : [dynomics.Experiment]
                A list of experiment objects to extract data from. Data will
                be extracted from the cell_trap attribute of each experiment
            dropna : bool
                whether or not to drop time points with non-finite values from
                the dataset
            as_class : bool
                if true targets will be pressence/absence of each toxin
                instead of concentration for each toxin
            use_tox_ids : bool
                if true, toxin ids will be used in the returned target_frame
                instead of toxin_short_names
            as_1d : bool
                if true, return targets as Nx1 dimensional dataframe. This flag
                is useful for non-multilabel problems
            leave_n_out : int
                The number of experiments to leave out in each cross validation
                fold
        Returns : (DataFrame, DataFrame, [(Array, Array)])
            A tuple representing the extracted features, the extracted targets,
            and the experiment wise cross validation splits.
    """
    feature_frame, target_frame = extract_features_targets(
        processed_exps,
        dropna=dropna,
        as_class=as_class,
        use_tox_ids=use_tox_ids,
        as_1d=as_1d
    )
    splits = construct_exp_splits(feature_frame, leave_n_out)
    return (feature_frame, target_frame, splits)


def gen_predict_vs_time_plots(classer, feature_frame, target_frame, exps):
    """ Generates a set of plots depeciting the performance of classer on each
        of the experiments during cross validation
        Input:
            classer : sklearn.Estimator
                Some classifier object to train and test on
            feature_frame : DataFrame
                A dataframe representing the features extracted from a set of
                expeirments.
            target_frame : DataFrame
                a 1d representation of induction state at each time point in
                feature_frame
            exps : [Experiment]
                A set of experiment objects with idx vals matching those found
                in data. The only Experiment attributes used by
                gen_predict_vs_time_plots are idx and induction_record
        Return : [pyplot.figure]
            A list of figures representing each of the plots
    """
    # A map from experiment ID to expeirment induction_records
    eid_induct_map = {e.idx: e.induction_record for e in exps}

    # The set of toxins seen across all experiments
    toxins = [
        short for short, _ in Counter(
            target_frame.induction_state.values
        ).most_common()
    ]

    # A map from toxin short name to an int
    tid_short_name_map = {short: i + 1 for i, short in enumerate(toxins)}

    # a list of experiment ids in Data
    exp_ids = feature_frame.index.get_level_values(0).unique()
    ret = []

    for eid in exp_ids:
        # Use all experiments that aren't eid as training data
        train_X = feature_frame.drop(eid, axis=0)
        train_y = target_frame.drop(eid, axis=0)

        # Fit the classifier on the training data
        classer.fit(train_X.values, train_y.values.ravel())

        # Use the data from experiment eid for validation
        test_X = feature_frame.loc[eid]

        # run classifier on validaiton data and map to ints
        test_p = classer.predict(test_X)
        test_pt = [tid_short_name_map[n] for n in test_p]

        # overwrite the first two values to ensure maximum range in data
        # this is a hack to make the experiment.plot function work better
        test_pt[0] = tid_short_name_map[toxins[-1]] + 1
        test_pt[1] = tid_short_name_map[toxins[0]] - 1

        # Map predictions and unix time stamps
        pred_as_frame = pandas.Series(
            data=test_pt,
            index=test_X.index.values
        ).to_frame('prediction').transpose()

        # generate the appropriate plot
        f, ax = pyplot.subplots(1,1)
        plot.plot(
            pred_as_frame,
            eid_induct_map[eid],
            ax=ax,
            shade_clogs=False,
            plot_title='prediction_vs_time : {0!s}'.format(eid)
        )
        pyplot.yticks(range(len(toxins) + 2), [''] + toxins + [''])
        pyplot.ylabel('Toxin Prediction', fontsize=12)
        ret.append(f)
    return ret


def confusion_score_model(
    classer,
    feature_frame,
    target_frame,
    cv_splits=None,
    prefit=False
):
    ''' Construct the training and testing confusion matrixes for classer on
        data
        Input:
            classer : sklearn.Estimator
                a classifiers to train and score
            feature_frame : DataFrame
                A dataframe representing the features extracted from a set of
                expeirments.
            target_frame : DataFrame
                a 1d representation of induction state at each time point in
                feature_frame
            cv_splits : [([],[])]
                A list of (train,test) splits which should be used for cross
                validation
        Return: (pandas.DataFrame, pandas.DataFrame)
            A tuple containing a DataFrame representing the confusion matrix
            during cross validation and a DataFrame representing the
            confusion matrix during training. i.e. (test_scores, train_scores)
    '''
    # Create variables to store conf matrix values during cross validation
    unique_labels = target_frame.iloc[:, 0].unique()
    num_labels = len(unique_labels)
    test_mat = numpy.zeros((num_labels, num_labels))
    train_mat = numpy.zeros((num_labels, num_labels))
    # perform cross validation
    for train, test in cv_splits:
        # get train and test splits
        train_X = feature_frame.values[train]
        train_y = target_frame.values.ravel()[train]
        test_X = feature_frame.values[test]
        test_y = target_frame.values.ravel()[test]
        if not prefit:
            # fit training data
            classer.fit(train_X, train_y)
        # Predict for training data and update confusion matrix
        train_p = classer.predict(train_X)
        temp = confusion_matrix(train_y, train_p, labels=unique_labels)
        train_mat += temp / temp.sum().sum() / len(cv_splits)
        # Predict for testing data and update confusion matrix
        test_p = classer.predict(test_X)
        temp = confusion_matrix(test_y, test_p, labels=unique_labels)
        test_mat += temp / temp.sum().sum() / len(cv_splits)
    # define helper to handle converting matrixes to dataframes
    def to_frame(mat):
        mat = pandas.DataFrame(data=mat, index=unique_labels, columns=unique_labels)
        mat.loc[:, 'sum'] = mat.sum(axis=1)
        mat.loc['sum', :] = mat.sum(axis=0)
        mat.index.name='True Label'
        mat.columns.name='Predicted Label'
        return mat
    # convert matrixes to dataframes and return
    return (to_frame(train_mat), to_frame(test_mat))


def plot_confusion_matrix(
        cm_df,
        title=None,
        cmap=pyplot.cm.Blues,
        fontsize=6,
        fontcolor=None,
        num_round=4,
        plot_top=0.88,
        cbar_ticks=None,
        cbar_min_divisor=2
):
    """
    Create and return a matplotlib figure representing a confusion matrix.

    Input:
        cm_df : pandas.DataFrame
            a pandas dataframe representing a confusion matrix
        title : str
            a plot title
        cmap : color map
            some pyplot colormap to use in plotting
        fontsize : int
            how large the text in each posititon of the matrix should be
        fontcolor : str
            the color that the text in each position of the matrix
    Return: pyplot.figure
        a figure object representing the plot
"""

    # Set figure title.
    if title is None:
        title = 'Confusion matrix'

    # Set figure fontcolor.
    if fontcolor is None:
        fontcolor = "black"

    conf_mat = cm_df.as_matrix()
    conf_mat_nozeros = cm_df.copy()
    #     conf_mat_nozeros['Sum'] = 0
    #     conf_mat_nozeros.loc['Sum'] = 0
    conf_mat_nozeros = conf_mat_nozeros.as_matrix()

    # Get class names.
    classes = cm_df.index

    # Set color bar ticks and format their labels.
    if cbar_ticks is None:
        cbar_ticks = [0.001, 0.01, 0.1, 0.2, 0.4, 0.8, 1.0]
    cbar_tick_labels = [str(label) for label in cbar_ticks]

    # Set color bar minimum and maximum.
    cbar_min = numpy.min(
        [i for i in cm_df.values.ravel() if i > 0]) / cbar_min_divisor
    cbar_max = numpy.max([i for i in cm_df.values.ravel() if i < 1])

    # Eliminate actual zeros from plotting data.
    for i, row in enumerate(conf_mat):
        for j, col in enumerate(row):
            if col < cbar_min:
                conf_mat_nozeros[i, j] = cbar_min

    # Initialize figure and axes objects and plot colored cells.
    fig, ax = pyplot.subplots(
        1,
        1,
        figsize=(7, 5)
    )
    cax = ax.imshow(
        conf_mat_nozeros,
        interpolation='nearest',
        cmap=cmap,
        norm=LogNorm(vmin=cbar_min, vmax=cbar_max)
    )

    # Add color bar, figure title, labels, and axis ticks.
    cbar = fig.colorbar(cax, ax=ax, ticks=cbar_ticks)
    cbar.ax.set_yticklabels(cbar_tick_labels)
    fig.suptitle(
        title,
        **{'x': 0.53, 'y': 0.97}
    )
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ticks = list(range(len(classes)))
    pyplot.xticks(ticks, classes, rotation=45)
    pyplot.yticks(ticks, classes)
    ax.tick_params(axis=u'both', which=u'both', length=0)

    # Add numerical values to the matrix's cells.
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax.text(
                j,
                i,
                # conf_mat[i, j],
                '{results:.{digits}f}'.format(
                    results=conf_mat[i, j],
                    digits=num_round
                    ),
                horizontalalignment="center",
                color=fontcolor,
                fontsize=fontsize
            )

    # Add borders to the summation row and column.
    p = patches.Rectangle(
        (-0.45, 6.5),
        7.9,
        0.95,
        fill=False,
        linewidth=3,
        edgecolor='black'
    )
    ax.add_patch(p)
    p = patches.Rectangle(
        (6.5, -0.45),
        0.95,
        7.9,
        fill=False,
        linewidth=3,
        edgecolor='black'
    )
    ax.add_patch(p)

    # Format final image for saving.
    pyplot.tight_layout()
    pyplot.subplots_adjust(top=plot_top)

    return fig, ax
