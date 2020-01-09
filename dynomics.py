import argparse
import copy
import datetime
import itertools
import json
import numpy
import os
import pandas
import sqlalchemy
import sys
import time
import warnings

from sklearn.decomposition import PCA
from collections import deque
from functools import reduce

from . import dim_reductions
from . import plot
from . import responder_tests
from . import transformations
from . import misc


# NOTE : This is one among potentially many params that we should
#      : encapsulate in a settings file somewhere.
HOST_SETTING = None


def create_db_engine(
    user=None,
    pw=None,
    host=HOST_SETTING
):
    """ Create and return a sqlalchemy engine for the darpa_data table.
    """
    # Create connection to MySQL database.
    # Only works with Biodynamics Lab-associated IP addresses.
    engine = sqlalchemy.create_engine(
        "mysql+pymysql://{0!s}:{1!s}@{2!s}/darpa_data".format(user, pw, host)
    )
    return engine


def import_query(query, engine=create_db_engine(), params=None, verbose=True):
    """ Perform some sql query and return results.
        Params:
            query : str
                Some string representing a sql query.
            engine : sqlalchemy.Engine
                Some sqlalchemy engine to execute the query.
            params : dict, default None
                A dictionary of parameters to add to the query.  This
                keyword argument is useful when the query includes
                characters such as "%", which PyMySQL then interprets as a
                Python string-formatting instance, rather than as part of the
                SQL query. For instance, the query snippet "AND subpath NOT
                LIKE '%PurgeStaging%' " throws an error when passed within
                in the query itself.  To avoid this error, replace
                "'%PurgeStaging%'" with "%(string_is_like)s" and pass the
                dictionary "{'string_is_like':str('%PurgeStaging%')}" as
                the keyword argument "params".
            verbose: bool, default False
                If True, print the timestamp, query and time elapsed to
                standard output.
        Returns: Pandas.DataFrame
            A dataframe representing the results of the query
    """
    # Print a message stating the query and when the query was started
    start = time.time()

    if params is not None:
        params_list = [
            "{key} : {value}".format(key=key, value=params[key])
            for key in params
        ]
        params_string = ", ".join(params_list)
    else:
        params_string = ""

    if verbose:
        print("{1!s}: {2!s}: {0!s}".format(
            query,
            time.strftime("%H:%M:%S"),
            params_string
        ))
        sys.stdout.flush()

    # Perform the query
    if "analysis_data_json" in query:
        # Open a connection to darpa_data with that engine.
        connection = engine.connect()

        # Via the connection object, send the query to darpa_data.
        results = connection.execute(query)

        # Fetch the rows returned by the query.  This method returns
        # a list of 2-tuples: the first element is reference_unix_time
        # and the second element is a JSON of the device position data
        # at that timepoint.
        data = results.fetchall()

    else:
        data = pandas.read_sql(query, con=engine, params=params)

    # Print a message stating how long the query took
    tot_time = time.time() - start
    m, s = divmod(tot_time, 60)
    h, m = divmod(m, 60)

    if verbose:
        print("{0!s}: Query finished in {1!s}h {2!s}m {3!s}s".format(
            time.strftime("%H:%M:%S"),
            int(h),
            int(m),
            int(s))
        )

    return data


def query_templater(
    table_name,
    columns=None,
    conds=None,
    db_name="darpa_data"
):
    """ Generates an appropriate SQL query for db_name
        Params:
            table_name : str
                Some darpa_data table
            columns : [str] or None
                A list of column names to fetch from table_name.
            conds : [str]
                A set of conditions
            db_name : str
                some database to query
        Returns: str
            A string of form "select {columns} from {db_name}.{table_name}
            where {conds}"
    """
    if(columns is None):
        columns = ["*"]
    ret = "select {1!s} from {2!s}.{0!s} ".format(
        table_name, ",".join(columns), db_name
    )
    if(conds is not None and len(conds) > 0):
        ret += "where " + " and ".join(conds)
    return ret


def getter_generator(attrib_name, priv_attr, query, fun, updatable):
    """ Returns a getter function for a given attribute name
        Params:
            attrib_name : str
                The name of the attribute
            priv_attr : str
                the name of the private attribute that will store the
                loaded data
            query : str
                A query to fetch appropriate data
            fun : f(DataFrame, self) -> DataFrame
                A transformation to apply to the data prior to cacheing.
                "self" is the Experiment object that the returned function
                is called from
            updatable: bool
                Whether or not to attempt to dynamically update the variable.
                Only works for data pulled from analysis_data table
        Returns: f(self) -> DataFrame
    """

    def attrib_getter(self):
        refresh_flag = attrib_name in self.refresh
        var_flag = getattr(self, priv_attr) is not None

        # If data is available and doesn't need to be refreshed
        if not refresh_flag and var_flag:
            return getattr(self, priv_attr)

        file_name = attrib_name + ".pickle"
        path = os.path.join(self.path, file_name)
        is_cached = os.path.isfile(path)

        # If data needs to be refreshed or cache isn't available
        if(refresh_flag or not var_flag and not is_cached):
            data = import_query(query.format(exp_idx=self.idx))
            if(type(data) is pandas.DataFrame and data.size > 0) or (type(
                    data) is list and len(data) > 0):
                data = fun(data, self)
                setattr(self, priv_attr, data)
                misc.pickle_dump(path, data)
                self.refresh.discard(attrib_name)
                return data
            else:
                warnings.warn(
                    "IDX {0!s}'s {1!s} is empty.".format(
                        self.idx,
                        attrib_name
                    )
                )
                return None
        # If data needs to be loaded
        else:
            data = misc.pickle_load(path)
            # if update needs to be requested
            if(updatable and self.update):
                temp = query.format(exp_idx=self.idx)
                temp += " and reference_unix_time>{0!s}".format(
                    data.columns[-1]
                )
                update = import_query(temp)
                # if update is actually available
                if(type(update) is pandas.DataFrame and update.size > 0) or\
                        (type(update) is list and len(update) > 0):
                    update = fun(update, self)
                    data = pandas.concat([data, update], axis=1)
                    misc.pickle_dump(path, data)
            setattr(self, priv_attr, data)
            return data
    return attrib_getter


def cell_trap_getter_generator(priv_attr):
    """ Generates a getter function for the cell_trap property.
    """

    def getter(self):
        if getattr(self, priv_attr) is None:
            data =\
                (
                    self.gfpffc_bulb_1 - self.gfpffc_bulb_bg
                )/self.gfpffc_bulb_bg
            setattr(self, priv_attr, data)
        return getattr(self, priv_attr)

    return getter


def json_data_reshape(data):
    """
    This function reshapes positional JSON data from the MySQL DB into a
    pivot table pandas.DataFrame.  This function is applied to all signal
    data and replaces tabular_data_reshape().

    :param data:
    :return:
    """
    # Sort in-place. The built-in list "sort" method sorts the tuples first by
    # the time and then by the elements of the JSON strings.  That second
    # ordering principal matters when there are multiple tuples that share the
    # same time value.
    data.sort()

    # Make a list of all reference_unix_time timepoints and then sort them (
    # ascending).
    reference_unix_time = list(
        {time_point[0] for time_point in data})
    reference_unix_time.sort()

    data_dict = {
        timepoint: dict() for timepoint in reference_unix_time
        }
    for t, d in data:
        d = {int(k): v for k, v in json.loads(d).items()}
        data_dict[t].update(d)

    return pandas.DataFrame(data_dict)


def clog_table_reshape(data):
    """
    Reshape the inferred_clogs table data.
    """
    thresholds = set(data.noise_threshold)
    algorithm_ids = set(data.algorithm_id)

    pairs = [
        (aid, t) for aid in sorted(algorithm_ids) for t in sorted(thresholds)
    ]

    tables = [
        data[
            (data.algorithm_id == aid) & (data.noise_threshold == t)
        ]
        for aid, t in pairs
    ]

    pivots = [
        pandas.pivot_table(
            t, index=["algorithm_id", "noise_threshold", "device_position"],
            columns="unix_timestamp", values="clog_flag"
        )
        for t in tables
    ]
    return pandas.concat(pivots)


def positional_attrib_desc(channel, desc):
    """ Create a description for an attribute with one value per position per
        time
        Params:
            channel : str
                Some value in darpa_data.analysis_data.channel_name_acquired
            desc : str
                Some value in darpa_data.analysis_data.sub_mask_desc
        returns: (str, str, str, f, str, bool)
            A tuple representing a dynomics attribute. Values are described as
            (public_name, private_name, query, transformation, doc_string, True)
    """

    base_name = desc.lower()
    name = channel + "_" + base_name
    conds = [
        "expID='{exp_idx}'",
        "channel_name_acquired='{0!s}'".format(channel),
        "sub_mask_desc='{0!s}'".format(desc)
    ]
    columns = ["reference_unix_time", "data"]
    query = query_templater("analysis_data_json", conds=conds, columns=columns)
    fun = lambda d, s: json_data_reshape(d)
    doc = "DataFrame constructed from {0!s} channel at {1!s}".format(
        channel, desc
    )
    return (name, "_" + name, query, fun, doc, True)


def detect_clogs(
    clog_data,
    background,
    non_clogged=60,
    threshold=-5,
    media_subtract=0.1,
    smooth_window=6
):
    """ Infer clogs
        Input :
            clog_data : DataFrame
                extracted clog mask data for an experiment
            background : DataFrame
                extracted background data for the an experiment
            non_clogged : int
                The number of time points in the beggining of the
                experiment to use to estabilish an unclogged distribution
            threshold : float
                Zscored values below threshold will be flagged as clogged
            media_subtract : float
                A value used when subtacting out changes in media flourescence.
                Should be a float between 0 and 1 or an int between 0 and
                clog_data.shape[0]
            smooth_window : int
                A number of time points to propogate clogs through. if a
                position is marked as clogged at time ti, then all points
                between ti and ti + smooth_window will be marked as clogged
        Returns : DataFrame
            A boolean dataframe flagging position/times as clogged (True) or
            unclogged (False)
    """
    # Normalize by background
    table = clog_data / background
    table = table.dropna(how="all", axis=1)

    # If desired, attempt to normalize for media brightness changes
    if(media_subtract is not None):
        if(media_subtract < 1):
            media_subtract = int(table.shape[0] * media_subtract)
        table = table.apply(
            lambda x: x - numpy.sort(x.values)[-media_subtract]
        )

    # Compute mean and std of early portion of data and zscore data
    m = numpy.nanmean(table.iloc[:, :non_clogged].values.ravel())
    std = numpy.nanstd(table.iloc[:, :non_clogged].values.ravel())
    table = (table - m) / std

    # apply threshold
    table = table < threshold

    # apply smoothing
    table = table.rolling(axis=1, window=smooth_window).mean().fillna(0) > 0
    return table


def flow_graphs(loading_record):
    """ Construct dictionaries representing flow paths as described in
        loading_record
        Input:
            loading_record : DataFrame
                The loading_record for an experiment
        Returns: (dict, dict)
            A tuple of dictionaries representing flow through the chip. The
            first maps device positions to their (upstream) parents. The
            second maps device positions to their (downstream) children
    """
    # Adjacency graph mapping X: [downstream of X]
    down_flow = {}
    # Adjacency graph mapping X: [upstream of X]
    up_flow = {}
    # populate the graphs
    for dp, ds_dps in zip(
        loading_record.device_position,
        loading_record.downstream_device_position
    ):
        if(ds_dps is not None):
            ds_dps = [int(x) for x in ds_dps.split(",")]
            if(dp not in down_flow):
                down_flow[dp] = []
            down_flow[dp].extend(ds_dps)
            for ds_dp in ds_dps:
                if(ds_dp not in up_flow):
                    up_flow[ds_dp] = []
                up_flow[ds_dp].append(dp)
    return (up_flow, down_flow)


def prop_clogs(df, up_flow, down_flow):
    """ Propogate clogs through df according to flow pattern descrbied by
        up_flow and down_flow
        Params :
            df : Pandas.DataFrame
                A boolean dataframe
            up_flow : dict(i: [i])
                A graph describing how clogs should be propogated. It
                should map each device position to a list of device positions
                that are uptream
            down_flow : dict(i: [i])
                A graph describing how clogs should be propogated. It
                should map each device position to a list of device positions
                that are downstream
        Returns : pandas.DataFrame
    """
    ret = {dp: df.loc[dp] for dp in df.index}
    pos_queue = deque([x for x in down_flow if x not in up_flow])
    finished = set()
    while(len(pos_queue) > 0):

        curr = pos_queue.popleft()
        # If nothing upstream of current, mark as finished and add
        # all positions downstream of current to pos_queue
        if(curr not in up_flow):
            finished.add(curr)
            if(curr in down_flow):
                for x in down_flow[curr]:
                    pos_queue.append(x)
        # Check that all positions upstream of current are finished
        # If they are then propogate clogs and add downstream positions
        # to pos_queue
        elif(all(x in finished for x in up_flow[curr])):
            up_stream_signal = reduce(
                numpy.logical_and,
                (ret[p] for p in up_flow[curr])
            )
            ret[curr] = ret[curr] | up_stream_signal
            if(curr in down_flow):
                for x in down_flow[curr]:
                    pos_queue.append(x)
            finished.add(curr)
        # If not all upstream positions have been finished, add current
        # position back to end of queue
        else:
            pos_queue.append(curr)
    return pandas.DataFrame(ret).transpose()


class Experiment(object):
    """ Experiment is a high level interface for analysis of dynomics data.

        Attributes:
            idx : int
                This Experiments ID.
            kind : str
                A string representing what kind of Experimnet object this is.
            path : str
                The path to a folder where attributes of this Experiment will
                be cached. Defaults to "./data/exp{idx}".
            refresh : set([str])
                A set of attributes for which the local cache should be
                ignored, meaning the first call to these attributes will pull
                fresh data from sql tables. This attribute will be modified
                as attributes are updated.
    """

    #TODO: preload is probably not going to be used any longer. Potentially delete.
    def __init__(
        self,
        expid,
        cache_path=None,
        preload=False,
        refresh=None,
        update=False
    ):
        """ Initializer for Experiment objects
            Params:
                expid : int
                    An experiment ID to use when acessing sql tables
                cache_path : str
                    A path to a folder where attributes of experiment should
                    be locally cached. Default = "./data/exp{expid}"
                preload : bool
                    A flag that if true will result in all tables being
                    pulled and cached during object initialization
                refresh : set([str])
                    A set of atributes for which local cache should be
                    ignored, meaning fresh data will be pulled from sql
                    tables. If "all" in refresh, then it will be interpretted
                    as meaning all attributes should be in refresh
                update : bool
                    if true, some attributes will be updated before use even
                    if data is locally cached. Only new data will be fetched
                    and then merged with local cache
        """
        self.idx = expid
        self.kind = "exp"
        self.update = update
        if(refresh is None):
            self.refresh = set()
        else:
            self.refresh = set(r.lower() for r in refresh)
        if "all" in self.refresh:
            self.refresh = set(Experiment.public_to_private_map.keys())

        default_path = "data/exp{0!s}".format(expid)
        self.path = default_path if (cache_path is None) else cache_path

        # if preload, then load all atributes, triggering downloads
        if (preload):
            for t in Experiment.attrib_descriptions:
                getattr(self, t[0])

    # A list of tuples describing most Experiment properties. Each tuple is
    # (pub_name, priv_name, query, transformation, doc_string, updatable).
    attrib_descriptions = [
        (
            "record",
            "_record",
            query_templater("experiment_record", conds=["expID='{exp_idx}'"]),
            lambda d, s: d,
            "DataFrame pulled from experiment_record table.",
            False
        ),
        (
            "loading_record",
            "_loading_record",
            query_templater("analysis_loading_record", conds=["expID='{exp_idx}'"]),
            lambda d, s: transformations.drop_duplicates(d, ["device_position"]),
            "DataFrame pulled from analysis_loading_record table.",
            False
        ),
        (
            "induction_record",
            "_induction_record",
            query_templater("analysis_induction_record", conds=["expID='{exp_idx}'"]),
            lambda d, s: d.sort_values("transition_time").set_index("induction_idx"),
            "DataFrame pulled from analysis_induction_record table",
            False
        ),
        (
            "toxin_record",
            "_toxin_record",
            query_templater("toxin_record"),
            lambda d, s: d.set_index("idx"),
            "DataFrame pulled from toxin_record table.",
            False
        ),
        (
            "inducer_stock_record_combined",
            "_inducer_stock_record_combined",
            query_templater("inducer_stock_record_combined"),
            lambda d, s: d.set_index("stock_idx"),
            "DataFrame pulled from inducer_stock_record_combined table.",
            False
        ),
        (
            "inducer_stock_toxin_mapping",
            "_inducer_stock_toxin_mapping",
            query_templater("inducer_stock_toxin_mapping"),
            lambda d, s: d.set_index("idx"),
            "DataFrame pulled from inducer_stock_toxin_mapping table.",
            False
        ),
        (
            "clogs",
            "_clogs",
            query_templater("inferred_clogs", conds=["expID='{exp_idx}'"]),
            lambda d, s: clog_table_reshape(d),
            "DataFrame pulled from inferred_clogs table.",
            False
        ),
        (
            "refill_record",
            "_refill_record",
            query_templater("refill_record"),
            lambda d, s: d.set_index("idx"),
            "DataFrame pulled from refill_record table.",
            False
        ),
    ]
    # Add bulb regions, background, and raw attributes for all channels.
    channels = ["gfpffc", "tlffc", ]
    descs = [
        "CELL_TRAP",
        "BULB_1",
        "BULB_2",
        "BULB_3",
        "BULB_4",
        "BULB_5",
        "BULB_12",
        "BULB_23",
        "BULB_34",
        "BULB_123",
        "BULB_234",
        "BULB_1234",
        "BULB_45",
        "BULB_BG",
        "BULB_PLUME",
        "ROOM_CHANNEL_LEFT",
        "ROOM_CHANNEL_RIGHT"
    ]
    attrib_descriptions.extend(
        [
            positional_attrib_desc(chan, desc)
            for chan, desc in itertools.product(channels, descs)
        ]
    )

    # attrib_descriptions converted to a list of tuples described as
    # (attrib_name, priv_attrib, getter, doc_string)
    attrib_descriptions = [
        (
            attr,
            priv_attr,
            getter_generator(attr, priv_attr, query, tf, updatable),
            doc_string,
        )
        for attr, priv_attr, query, tf, doc_string, updatable in attrib_descriptions
    ]

    # append the special cell_trap property
    attrib_descriptions.append(
        (
            "cell_trap",
            "_cell_trap",
            cell_trap_getter_generator("_cell_trap"),
            (
                "A placeholder attribute meant to be overwritten with "
                "whatever view of the data is needed. Defaults to "
                "(gfpffc_bulb_1 - gfpffc_bulb_bg) / gfpffc_bulb_bg."
                "Position 0 represents mean and Position -1 represents "
                "standard deviation."
            )
        )
    )

    # a dict that maps public property names to private attribute names
    public_to_private_map = {t[0]: t[1] for t in attrib_descriptions}


    # Methods
    def set(self, **kargs):
        """ Return a copy of self such that each attrib_name contained in
            kargs is  set to kargs[attrib_name].
        """
        ret = copy.deepcopy(self)
        for attrib_name in kargs:
            if attrib_name in Experiment.public_to_private_map:
                priv_attr = Experiment.public_to_private_map[attrib_name]
                setattr(ret, priv_attr, kargs[attrib_name])
            else:
                setattr(ret, attrib_name, kargs[attrib_name])
        return ret

    #TODO: NH: add caching functionality to this method.
    def get_position_wise_data(self, sub_mask_desc, channel):
        """ Import some data for some specific sub_mask_desc and channel
            for this experiment
            Params:
                sub_mask_desc : str
                    A sub mask to extract data for
                channel : str
                    A channel to extract data for
            returns: DataFrame
        """
        query = positional_attrib_desc(channel, sub_mask_desc)[2]
        query = query.format(exp_idx=self.idx)
        data = import_query(query)
        return json_data_reshape(data)

    def apply_mask(self, mask):
        """ Apply a boolean mask to self.cell_trap and return results in a new
            expeirment object
            Params:
                mask : DataFrame
                    A boolean dataframe with similar index/columns to
                    self.cell_trap. True values will be set to nan.
            Returns : Experiment
                A new experiment object that is equivalent to self except for
                its cell trap which has beeen masked
        """
        if(not all(x in mask.columns for x in self.cell_trap.columns)):
            warnings.warn(
                "Mask Data isn't available for some number of time "
                "points in cell_trap"
            )
        mask = mask.applymap(bool)
        new_trap = self.cell_trap.mask(mask)
        return self.set(cell_trap=transformations.add_mean_and_std(new_trap))

    def get_induction(self, attrib_name, *induction_idx):
        """
        Trim all the time points, as defined by exp.induction_record[induction_idx],
        from exp.attrib_name that aren't contained within the enumerated induction
        or inductions, as specified by the argument(s) *induction_idx.
            Input:
                exp : dynomics.Experiment
                    Some experiment with data we want to subset
                attrib_name : str
                    The attribute of exp to subset. This attribute should have unix
                    time in seconds as column labels
                induction_idx : int
                    Some induction index or indices that can be found in
                    exp.induction_record. Any number of indices may be specified.
            Returns:
                dynomics.Experiment
                A new experiment object.
        """
        df = getattr(self, attrib_name)
        induction_id = list(induction_idx)
        starts = self.induction_record.loc[induction_id].transition_time
        ends = self.induction_record.loc[induction_id].end_time
        new_induction_record = self.induction_record.loc[induction_id]

        # Check if the final induction's end time is present or not.
        # If not, it's reasonable to assume the experiment is ongoing and
        # to set the end time to the most recently imaged timepoint.
        if numpy.isnan(ends.values[-1]):
            new_end_time = self.cell_trap.columns.values[-1]
            ends.iloc[-1] = new_end_time
            new_induction_record.end_time.iloc[-1] = new_end_time

        cols = [
            x for x in df.columns
            if (any(s <= x < e for s, e in zip(starts, ends)))
            ]
        df = df.loc[:, cols]

        new_record = self.record.copy()
        new_record["exp_start_date_on_box"] = \
            datetime.datetime.fromtimestamp(
                starts.iloc[0]
            ).strftime("%Y-%m-%d %H:%M:%S")
        new_record["exp_end_date_on_box"] = datetime.datetime.fromtimestamp(
            ends.iloc[-1]
        ).strftime("%Y-%m-%d %H:%M:%S")

        return self.set(
            **{
                attrib_name: df,
                "induction_record": new_induction_record,
                "record": new_record,
            }
        )

    def infer_clogs(
        self,
        non_clogged=60,
        threshold=-5,
        media_subtract=0.1,
        propogate=True,
        smooth_window=6
    ):
        """ Flag each position at each time True if its flow is interrupted
            by clogs.
            Params:
                non_clogged : float
                    A positive value indicating the number of time points in
                    the start of the experiment to assume are totally
                    unclogged. Observation of these time points will be used to
                    establish an unclogged distribution. Values less than 1 are
                    interpretted as percent of time points. Else it is that
                    many time points.
                threshold : int
                    A Z-Score threshold. Hihger values will result in fewer
                    positions being flagged
                media_subtract : float
                    A positive value. Should be some value below the number
                    of clogged positions at any given time point. If the
                    input is less than 1, it is interpretted as a percent of
                    device_positions. Else, it is that many device positions
                propogate : bool
                    Whether or not clog flags in global clog positions should
                    be propogated
                smooth_window : int
                    The number of time points to smooth over.
            Returns: DataFrame
                A dataframe marking each time/position as clogged(True) or
                unclogged(False)
        """
        partial_detect_clogs = lambda x: detect_clogs(
            x,
            self.tl_background,
            non_clogged,
            threshold,
            media_subtract,
            smooth_window
        )

        local_up = partial_detect_clogs(self.tl_upstream_local_clog)
        global_up = partial_detect_clogs(self.tl_upstream_global_clog)
        local_down = partial_detect_clogs(self.tl_downstream_local_clog)
        global_down = partial_detect_clogs(self.tl_downstream_global_clog)

        if(propogate):
            up_flow, down_flow = flow_graphs(self.loading_record)
            global_up = prop_clogs(global_up, up_flow, down_flow)
            global_down = prop_clogs(global_down, down_flow, up_flow)

        # or-wise join each table and return
        return global_up | global_down | local_up | local_down

    def infer_empty(
        self,
        data=None,
        threshold=None,
        background=None
    ):
        """ Infer whether or not positions are empty at any given time
            and return a mask representing, for each position, for each
            time point, whether or not that position at that time is
            empty.
            Input:
                data : DataFrame or None
                    Either a dataframe containing trap signals or None. If
                    None, self.tl_raw will be used.
                threshold : int or None
                    A background normalized threshold to use. positions times
                    with data/self.tl_background greater than threshold will be
                    marked as empty. If None is provided, threshold will be
                    0.937 for 2k chips which seems to work well for 2k chips
                background : DataFrame or None
                    a tl background signal to use to normalize the provided
                    data. If None, self.tl_background will be used.
            Return : DataFrame
                A dataframe marking each time/position as either empty (True)
                or full (False)
        """
        if(data is None):
            data = self.tl_raw
        if(threshold is None):
            threshold = 0.937
        if(background is None):
            background = self.tl_background
        data = data / background
        data = data.dropna(how="all", axis=1)
        return data > threshold

    def top_k_strains(
        self,
        other_exps=None,
        tf=transformations.identity,
        k=10,
        toxins=None
    ):
        """ Scores each strain using anova for the set of toxins and return
            the scores for the top k strains.
            Params:
                other_exps : [Experiment]
                    A list of Expeirment objects to analyze along with this one
                tf : f(pandas.DataFrame) -> pandas.DataFrame
                    a transformation function to apply to the dataset before
                    performing analysis. Defaults to the identity function
                    (lambda x: x)
                k : int
                    the number of strain_idx numbers to return. if K greater
                    than actual number of strains, scores for all strains will
                    be returned
                toxins : [int] or None
                    a list of toxin ids to try to find responders for.
            Returns : Series
                a pandas Series mapping strain id to p_val scores for the top K
                strains
        """
        if (other_exps is None):
            other_exps = []
        all_exps = [self] + other_exps

        # apply transformation to each experiment
        transformed_exps = [e.set(cell_trap=tf(e.cell_trap)) for e in all_exps]
        # set default comparisons if none provided
        if (toxins is None):
            comp = lambda x: x.apply(str, raw=True, axis=1)
        else:
            toxins = set(toxins)
            toxins = list(toxins)
            comp = lambda x: x[toxins].apply(
                lambda x: "Water" if sum(x) == 0 else str(x),
                raw=True,
                axis=1
            )
        # Get anova scores and return
        scores = responder_tests.score_strains(
            transformed_exps,
            comp
        )
        return scores.sort_values().iloc[:k]

    def dim_reduction_plot(
        self,
        other_exps=None,
        reduct_type=PCA,
        tf=transformations.identity,
        reduct_args=None
    ):
        """ Perform a dimensionality reduction on the cell_trap data and plot
            results in two charts, one coloring points by induction class, and
            the other coloring points by experiment id.
            Params:
                other_exps : [Experiment]
                    optional list of other experiments to plot along with this
                    experiment. Default = []
                reduct_type : Class
                    A class reference that allows for dimensionality reduction.
                    For example: sklearn.decomposition.PCA. Defaults to PCA
                tf : f(DataFrame) -> DataFrame
                    a transformation to apply to the data before performing PCA
                reduct_args : {}
                    an optional dict of arguments to pass to the reduct_type
                    constructor. Default = {}
            Returns: Fig1, Fig2
        """
        if (other_exps is None):
            other_exps = []
        all_exps = [self]
        all_exps.extend(other_exps)
        all_exps = [e.set(cell_trap=tf(e.cell_trap)) for e in all_exps]
        f1, f2 = dim_reductions.dim_reduction_plot(
            all_exps, reduct_type, reduct_args
        )
        return (f1, f2)

    def drop_empty(self):
        """ Returns a copy of self with empty positions according
            to the loading record removed from cell_trap
        """
        return self.set(cell_trap=self.cell_trap.drop(self.empty_positions))

    def plot(
        self,
        idxlist,
        traj_type=None,
        shade_inductions=True,
        shade_clogs=False,
        exp_kind="Exp",
        exp_idx=None,
        **plot_kwargs
    ):
        if exp_idx is None:
            exp_idx = self.idx

        if shade_clogs:
            clogs = self.clogs.copy()
        else:
            clogs = None

        return plot.plot_experiment(
            self,
            idxlist,
            traj_type=traj_type,
            shade_inductions=shade_inductions,
            # shade_clogs=shade_clogs,
            # clogs=clogs,
            exp_kind=exp_kind,
            exp_idx=exp_idx,
            **plot_kwargs
        )

    def trim_growth(self, attribute="cell_trap", hours=18,
                    starting_induction=0):
        """ Trims away from the desired attribute the early part of an
            experiment when cells are growing up.
                hours : int
                    Some number of hours prior to the first toxin induction
                    which specifies how much data should be kept. I.E. if
                    hours=5, then all data before 5hrs before the first
                    toxin induction will be dropped.
                starting_induction : int
                    An integer specifying the induction from which to trim.
                    Defaults to the first toxin induction.
            Return: Experiment
                An experiment object with a smaller cell_trap field.
        """

        ind1_start = self.induction_record.end_time.iloc[starting_induction]
        start = ind1_start - 3600 * hours

        return eval(
            "self.set(" + attribute + "=self." + attribute + ".loc[:, " + str(
                start) + ":])")

    def trim_tail(self, attribute="cell_trap", hours=4, final_induction=-2):
        """ Trims away from the desired attribute the late part of an
            experiment where clogging may cause signal abnormalities.
                hours : int
                    Some number of hours after to the final toxin induction
                    which specifies how much data should be dropped. I.E. if
                    hours=5, then all data before 5 hrs after the final
                    toxin induction will be dropped.
                starting_induction : int
                    An integer specifying the induction from which to trim.
                    Defaults to the first toxin induction.
            Return: Experiment
                An experiment object with a smaller cell_trap field.
        """

        ind_final_end = self.induction_record.end_time.iloc[final_induction]
        end = ind_final_end + 3600 * hours

        return eval(
            "self.set(" + attribute + "=self." + attribute + ".loc[:, :" + str(
                end) + "])"
        )

    def induction_states(
        self,
        time_stamps=None,
        as_class=False,
        use_tox_ids=False,
        as_1d=False
    ):
        """ Return a dataframe representing the induction state at each time
            point.
            Input
                time_stamps : [int] or None
                    A list of unix_time_stamps in seconds for which induction
                    state information is desired. If None, then induction state
                    information will be provided for each time point in
                    self.cell_trap
                as_class : bool
                    if true, return pressence/absence instead of concentration
                    for each toxin
                use_tox_ids : bool
                    if true, use toxin ids instead of toxin names in the
                    returned pandas dataframe
                as_1d : bool
                    if true, induction states will be represented using an Nx1
                    dimensional frame with strings as values
            Return : pandas.DataFrame
        """
        # if no time stamps are provided, use time stamps from cell_trap
        if(time_stamps is None):
            time_stamps = self.cell_trap.columns

        # look up induction start times and combined_concentrations
        rstarts = self.induction_record.transition_time.values[::-1]
        conc = self.induction_record.combined_concentrations.values[::-1]

        # for each time point, find the last start time before that time point
        # and assign combined_concentrations for that time_point
        conce_states = [
            conc[(rstarts < t).argmax()]
            for t in time_stamps
        ]
        # split combined concentration fields
        conce_states = [
            [s.split(" ") for s in e.split(", ")]
            for e in conce_states
        ]
        # Parse combined concentration field and represent in dict.
        tox_conce_maps = [
            {" ".join(sub_e[2:]): float(sub_e[0]) for sub_e in e}
            for e in conce_states
        ]
        # convert toxin short names to toxin ids if desired
        if(use_tox_ids):
            toxin_short_name_id_map = {
                name: tid for name, tid in zip(
                    self.toxin_record.toxin_short_name,
                    self.toxin_record.index
                )
            }
            tox_conce_maps = [
                {
                    toxin_short_name_id_map[tox_name]: e[tox_name]
                    for tox_name in e
                }
                for e in tox_conce_maps
            ]
        # If return should be 1 dimensional
        if(as_1d):
            # convert dicts to tuples of tuples
            tox_conce_maps = [
                tuple((tox, e[tox]) for tox in e)
                for e in tox_conce_maps
            ]
            # If concentration isn't wanted drop concentration values
            if(as_class):
                tox_conce_maps = [
                    tuple(sub_e[0] for sub_e in e)
                    for e in tox_conce_maps
                ]
            # convert to dataframe and return
            ret = pandas.Series(data=tox_conce_maps, index=time_stamps)
            return ret.to_frame("induction_state").applymap(str)
        # If return should be multi dimensional
        else:
            # define the set of toxins present in dataset
            toxins = list(set(t for e in tox_conce_maps for t in e))

            # represent concentration for each time point for each toxin
            tox_arrays = [
                pandas.Series(
                    data=[e[t] if t in e else 0.0 for e in tox_conce_maps],
                    index=time_stamps,
                    name=t
                )
                for t in toxins
            ]

            # concat the series into a dataframe
            df = pandas.concat(tox_arrays, axis=1)

            # if pressence/absence is wanted instead of concentrations
            if(as_class):
                df = df > 0

            return df

    def delay(self, seconds=0, minutes=0, hours=0):
        """ Return a copy of self with self.induction_record having been delayed by
            the requested amount of time
            Input:
                seconds - int
                    A number of seconds to delay by
                minutes - int
                    A number of minutes to delay by
                hours - int
                    A number of hours to delay by
            Returns : Experiment
                A copy of self except for self.induction_record which has had its
                transition_time and end_time values shifted
        """
        total = seconds + 60 * minutes + hours * 3600
        ind_rec = self.induction_record.copy()
        ind_rec["transition_time"] += total
        ind_rec["end_time"] += total
        return self.set(induction_record=ind_rec)

    # Derived Attributes Defined Below
    @property
    def organism(self):
        """ A string representing the species used on the chip.
        """
        host_ids = self.loading_record.host_idx.unique()
        host_ids = [idx for idx in host_ids if (not numpy.isnan(idx))]
        host_id_name = {
            8.0: "yeast"
        }
        ret = [
            host_id_name[idx] if (idx in host_id_name) else "ecoli-{0!s}".format(idx)
            for idx in host_ids
        ]
        return "_".join(ret)

    @property
    def inductions(self):
        """ A list of induction ids for this Experiment
        """
        return self.induction_record.index.tolist()

    @property
    def empty_positions(self):
        """ A list of empty positions for this Experiment according to the
            loading record
        """
        return self.loading_record.device_position.loc[
            self.loading_record.strain_idx.isnull()].tolist()

    @property
    def toxins(self):
        """ The set of toxin used during the inductions of this
            Experiment
        """
        return set(self.induction_states().columns)


# loop through all attribute descriptions and add them to Experiment
for attrib_description in Experiment.attrib_descriptions:
    attrib_name, priv_attr, getter, doc_string = attrib_description
    # add the "private" attributes to Experiment and initialize them to none
    setattr(Experiment, priv_attr, None)
    # add the public properties to Experiment
    setattr(Experiment, attrib_name, property(getter, None, None, doc_string))


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(
        description="download and cache Experiment objects")
    parser.add_argument(
        "exp",
        nargs="*",
        help="an experiment ID to download and cache locally"
    )
    parser.add_argument(
        "-r",
        "--refresh",
        help=(
            "a flag if set will force downloading of data even if already "
            "cached"
        ),
        action="store_true",
        default=False
    )

    args = parser.parse_args()
    for e in args.exp:
        if args.refresh:
            Experiment(int(e), refresh=["all"], preload=True)
        else:
            Experiment(int(e), preload=True)
