import numpy
import os
import pickle
import time

from fuzzywuzzy import fuzz
from PIL import Image

from dynomics import dynomics


def pickle_load(path):
    """ Loads data in from .pickle file as specified in path.
        Params:
            path : str
                Path to a pickle file
        Returns: object
            the data encoded by the pickle file
    """
    try:
        with open(path, "rb") as input_file:
            data = pickle.load(input_file)
        return data
    except ModuleNotFoundError:
        split_path = path.split('/')
        current_dir = '/'.join(split_path[:-1])
        old_data_dir = current_dir + '/OLD_' + time.strftime('%Y%m%d_%H%M%S') + '/'
        os.makedirs(old_data_dir)
        os.rename(old_data_dir + split_path[-1])
        return


def pickle_dump(path, obj):
    """ Pickles obj and saves it to path
        Params :
            path : str
                Path to where the pickle file should be saved
            obj : object
                object to be pickled and saved
        Returns : None
    """
    dirs = os.path.dirname(path)
    if dirs != '' and not os.path.isdir(dirs):
        os.makedirs(dirs)
    with open(path, 'wb') as outfile:
        pickle.dump(obj, outfile)


def array_to_image(nda, path):
    ''' Convert a 2d array of values (scaled between 0 and 1) into an image and
        save to path.
        Input:
            nda - numpy.ndarray
                A 2d numpy array of values between 0 and 1
            path - str or None
                An optional path to a location where the image should be saved.
                If None, the image won't be saved
        Return: PIL.Image
            an image object representing the nda
    '''
    data = nda * 255
    data = data.astype(numpy.uint8)
    im = Image.fromarray(data)
    if(path is not None):
        im.save(path)
    return im


def get_engineered_strains(exp_idx):
    """
    Gets a list of the engineered strains' strain indices, device positions,
    and gene names for a given experiment.

    :param exp_idx: int
        The experiment index number for the experiment of interest.
    :return: matching_gene_names: list of tuples
        List of length-3 tuples with strain index, device position, and gene
        name.
    """
    return get_strains_of_interest("(", exp_idx)


def get_strains_of_interest(matching_pattern, exp_idx, matching_score=50):

    """
    Performs fuzzy string-matching on an experiment's loaded strains to find
    strains with gene names similar to the desired pattern.

    :param matching_pattern: str
        A pattern sought in the gene name of interest.
    :param exp_idx: int
        The experiment index number for the experiment of interest.
    :return: matching_gene_names: list of tuples
        List of length-3 tuples with strain index, device position, and gene
        name.
    """

    e = dynomics.Experiment(exp_idx)

    matching_gene_names =\
        [
            (
                strain_idx,
                device_position,
                gene_name,
                position_name
            )
            for strain_idx, device_position, gene_name, position_name in
            zip(
                e.loading_record.strain_idx,
                e.loading_record.device_position,
                e.loading_record.gene_name,
                e.loading_record.position_name
            )
            if fuzz.partial_ratio(gene_name, matching_pattern) > matching_score
        ]

    return matching_gene_names


def shift_induction_times(exp, min_start=20, min_end=30):
    """
    Shifts the induction start and end times forward or backward by separately
    specified numbers of minutes.  Returns an experiment object with a new
    induction record.

    :param min_start: int
        Number of minutes to shift induction start times forward.
    :param min_end: int
        Number of minutes to shift induction end times forward.
    :return: new_exp: dynomics.Experiment
        An experiment object with a new induction record, where the start and
        end times have been shifted by the desired amount.
    """
    new_induction_record = exp.induction_record.copy()

    new_transition_times =\
        new_induction_record.transition_time.add(min_start * 60)
    new_transition_times.iloc[0] =\
            new_transition_times.iloc[0] - min_start * 60

    new_end_times = new_induction_record.end_time.add(min_end * 60)
    new_end_times.iloc[-1] = new_end_times.iloc[-1] - min_end * 60

    new_induction_record.transition_time = new_transition_times
    new_induction_record.end_time = new_end_times

    new_exp = exp.set(induction_record=new_induction_record)

    return new_exp


if(__name__ == '__main__'):
    pass
