import pandas
import numpy

from copy import copy, deepcopy
from scipy import stats

from . import models


def score_strains(exps, comp):
    """ Use anova analysis to score strains according their ability to seperate
        induction states of interest.
        Input:
            exps : [dynomics.Experiment]
                a list of experiments to analyze.
            comp : f(DataFrame) -> Series
                A function that will be applied to the extracted induction
                state dataframe. It should return a Series mapping each
                time point to a single value. Time points which share an
                induction state should be mapped to the same value. Time
                points mapped to NaN values will be ignored.
    """
    # Reindex experiments to be in terms of (device_position, strain_idx), time
    strain_idxed_exps = []
    for exp in exps:
        e = deepcopy(exp)
        m = pandas.Series(
            index=e.loading_record.device_position.values,
            data=e.loading_record.strain_idx.values
        )
        ct = e.cell_trap
        ct['strain_idx'] = m
        ct['device_position'] = ct.index.values
        ct = ct.dropna(axis=0, subset=['strain_idx'])
        ct = ct.set_index(['strain_idx', 'device_position'])
        strain_idxed_exps.append(e.set(cell_trap=ct))

    # extract data and induction state
    feats, targs = models.extract_features_targets(
        strain_idxed_exps,
        as_class=True,
        dropna=False,
        use_tox_ids=True
    )
    targs = comp(targs)

    # split data into a set of tables, one for each induction state
    # and filter nan strids
    tables = [feats.loc[targs == t] for t in set(targs) if(t == t)]

    # Create a dict to hold the anova scores for each strain
    scores = {}
    for strid in feats.columns.get_level_values(0).unique():
        # unravel relevant data to handle replicates and filter nan values
        data = [t.loc[:, strid].values.ravel() for t in tables]
        data = [s[numpy.isfinite(s)] for s in data]
        data = [s for s in data if(len(s) > 0)]

        # perform anova test
        if(len(data) < 1):
            scores[strid] = numpy.nan
        else:
            scores[strid] = stats.f_oneway(*data)[1]
    return pandas.Series(scores)


if(__name__ == '__main__'):
    print('test')
