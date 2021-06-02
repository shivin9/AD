# Authors: Shubhomoy Das (based on scikit-learn tree based APIs)
# License: BSD 3 clause

from __future__ import division

import copy

import numpy as np

from scipy.sparse import lil_matrix, csr_matrix

from scipy.sparse import issparse

import numbers
from sklearn.utils import check_random_state, check_array

from sklearn.ensemble import IsolationForest

from multiprocessing import Pool

import numpy as np



import pickle
import logging
import random
import os
import os.path
import errno
import pandas as pd
import numpy as np
import sys
from timeit import default_timer as timer
from datetime import timedelta
import traceback
import pkgutil
import io

from statsmodels.distributions.empirical_distribution import ECDF
import ranking
from ranking import Ranking

import scipy.sparse
from scipy.sparse import csr_matrix
import scipy.stats as stats
import scipy.optimize as opt

from sklearn.linear_model import LogisticRegression as LR
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RF

from argparse import ArgumentParser


logger = logging.getLogger(__name__)


def get_option_list():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="airline", required=False,
                        help="Dataset name")
    parser.add_argument("--algo", type=str, default="", required=False,
                        help="Algorithm to apply")
    parser.add_argument("--explore_only", action="store_true", default=False,
                        help="Perform exploratory analysis only instead of more expensive model fitting.")
    parser.add_argument("--budget", type=int, default=1, required=False,
                        help="Budget for feedback")
    parser.add_argument("--n_epochs", type=int, default=200, required=False,
                        help="Max training epochs")
    parser.add_argument("--train_batch_size", type=int, default=25, required=False,
                        help="Batch size for stochastic gradient descent based training methods")
    parser.add_argument("--n_lags", type=int, default=12, required=False,
                        help="Number of time lags for timeseries models")
    parser.add_argument("--normalize_trend", action="store_true", default=False,
                        help="Whether to remove trend in timeseries by successive difference")
    parser.add_argument("--log_transform", action="store_true", default=False,
                        help="Whether to apply element-wise log transform to the timeseries")
    parser.add_argument("--n_anoms", type=int, default=10, required=False,
                        help="Number of top anomalies to report")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Whether to enable output of debug statements")
    parser.add_argument("--plot", action="store_true", default=False,
                        help="Whether to plot figures")
    parser.add_argument("--log_file", type=str, default="", required=False,
                        help="File path to debug logs")
    parser.add_argument("--randseed", action="store", type=int, default=42,
                        help="Random seed so that results can be replicated")
    parser.add_argument("--results_dir", action="store", default="./temp",
                        help="Folder where the generated metrics will be stored")
    return parser


def get_command_args(debug=False, debug_args=None, parser=None):
    if parser is None:
        parser = get_option_list()

    if debug:
        unparsed_args = debug_args
    else:
        unparsed_args = sys.argv
        if len(unparsed_args) > 0:
            unparsed_args = unparsed_args[1:len(unparsed_args)]  # script name is first arg

    args = parser.parse_args(unparsed_args)
    return args


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed + 32767)


def matrix(d, nrow=None, ncol=None, byrow=False):
    """Returns the data as a 2-D matrix

    A copy of the same matrix will be returned if input data dimensions are
    same as output data dimensions. Else, a new matrix will be created
    and returned.

    Example:
        d = np.reshape(range(12), (6, 2))
        matrix(d[0:2, :], nrow=2, byrow=True)

    Args:
        d:
        nrow:
        ncol:
        byrow:

    Returns: np.ndarray
    """
    if byrow:
        # fill by row...in python 'C' fills by the last axis
        # therefore, data gets populated one-row at a time
        order = 'C'
    else:
        # fill by column...in python 'F' fills by the first axis
        # therefore, data gets populated one-column at a time
        order = 'F'
    if len(d.shape) == 2:
        d_rows, d_cols = d.shape
    elif len(d.shape) == 1:
        d_rows, d_cols = (1, d.shape[0])
    else:
        raise ValueError("Dimensions more than 2 are not supported")
    if nrow is not None and ncol is None:
        ncol = int(d_rows * d_cols / float(nrow))
    elif ncol is not None and nrow is None:
        nrow = int(d_rows * d_cols / float(ncol))
    if len(d.shape) == 2 and d_rows == nrow and d_cols == ncol:
        return d.copy()
    if not d_rows * d_cols == nrow * ncol:
        raise ValueError("input dimensions (%d, %d) not compatible with output dimensions (%d, %d)" %
                         (d_rows, d_cols, nrow, ncol))
    if isinstance(d, csr_matrix):
        return d.reshape((nrow, ncol), order=order)
    else:
        return np.reshape(d, (nrow, ncol), order=order)


# Ranks in decreasing order
def rank(x, ties_method="average"):
    ox = np.argsort(-x)
    sx = np.argsort(ox)
    if ties_method == "average":
        strategy = ranking.FRACTIONAL
    else:
        strategy = ranking.COMPETITION
    r = Ranking(x[ox], strategy=strategy, start=1)
    rnks = list(r.ranks())
    return np.array(rnks)[sx]


def nrow(x):
    if len(x.shape) == 2:
        return x.shape[0]
    return None


def ncol(x):
    if len(x.shape) == 2:
        return x.shape[1]
    return None


def rbind(m1, m2):
    if m1 is not None and m2 is not None and isinstance(m1, csr_matrix) and isinstance(m2, csr_matrix):
        return scipy.sparse.vstack([m1, m2])
    if m1 is None:
        return np.copy(m2)
    return np.append(m1, m2, axis=0)


def cbind(m1, m2):
    if len(m1.shape) == 1 and len(m2.shape) == 1:
        if len(m1) == len(m2):
            mat = np.empty(shape=(len(m1), 2))
            mat[:, 0] = m1
            mat[:, 1] = m2
            return mat
        else:
            raise ValueError("length of arrays differ: (%d, %d)" % (len(m1), len(m2)))
    return np.append(m1, m2, axis=1)


def sample(x, n):
    shuffle = np.array(x)
    np.random.shuffle(shuffle)
    return shuffle[0:n]


def get_sample_feature_ranges(x):
    min_vals = np.min(x, axis=0)
    max_vals = np.max(x, axis=0)
    return np.hstack([np.transpose([min_vals]), np.transpose([max_vals])])


def append(a1, a2):
    if isinstance(a1, np.ndarray) and len(a1.shape) == 1:
        return np.append(a1, a2)
    a = a1[:]
    if isinstance(a2, list):
        a.extend(a2)
    else:
        a.append(a2)
    return a


def rep(val, n, dtype=float):
    return np.ones(n, dtype=dtype) * val


def power(x, p):
    if isinstance(x, scipy.sparse.csr_matrix):
        return np.sqrt(x.power(p).sum(axis=1))
    else:
        return np.sqrt(np.power(x, p).sum(axis=1))


def quantile(x, q):
    return np.percentile(x, q)


def difftime(endtime, starttime, units="secs"):
    if units == "secs":
        t = timedelta(seconds=endtime-starttime)
    else:
        raise ValueError("units '%s' not supported!" % (units,))
    return t.seconds


def order(x, decreasing=False):
    if decreasing:
        return np.argsort(-x)
    else:
        return np.argsort(x)


def runif(n, min=0.0, max=1.0):
    return stats.uniform.rvs(loc=min, scale=min+max, size=n)


def rnorm(n, mean=0.0, sd=1.0):
    return stats.norm.rvs(loc=mean, scale=sd, size=n)


def pnorm(x, mean=0.0, sd=1.0):
    return stats.norm.cdf(x, loc=mean, scale=sd)


def ecdf(x):
    return ECDF(x)


def matrix_rank(x):
    return np.linalg.matrix_rank(x)


def normalize(w):
    # normalize ||w|| = 1
    w_norm = np.sqrt(w.dot(w))
    if w_norm > 0:
        w = w / w_norm
    return w


def get_random_item(samples, random_state):
    i = random_state.randint(0, samples.shape[0])
    return samples[i]


class SetList(list):
    """ A list class with support for rudimentary set operations
    This is a convenient class when set operations are required while
    preserving data ordering
    """
    def __init__(self, args):
        super(SetList, self).__init__(args)
    def __sub__(self, other):
        return self.__class__([item for item in self if item not in other])


class InstanceList(object):
    def __init__(self, x=None, y=None, ids=None, x_transformed=None):
        self.x = x
        self.y = y
        self.ids = ids
        # support for feature transform
        self.x_transformed = x_transformed
        if self.x is not None and self.x_transformed is not None:
            if self.x.shape[0] != self.x_transformed.shape[0]:
                raise ValueError("number of instances in x (%d) and x_transformed (%d) are not same" %
                                 (self.x.shape[0], self.x_transformed.shape[0]))

    def __len__(self):
        if self.x is not None:
            return self.x.shape[0]
        return 0

    def __repr__(self):
        return "instances(%s, %s, %s, %s)" % (
            "-" if self.x is None else str(self.x.shape),
            "-" if self.y is None else str(len(self.y)),
            "-" if self.x_transformed is None else str(self.x_transformed.shape),
            "-" if self.ids is None else str(len(self.ids))
        )

    def __str__(self):
        return repr(self)

    def add_instances(self, x, y, ids=None, x_transformed=None):
        if self.x is None:
            self.x = x
        else:
            self.x = rbind(self.x, x)

        if self.y is None:
            self.y = y
        elif y is not None:
            self.y = append(self.y, y)

        if self.ids is None:
            self.ids = ids
        elif ids is not None:
            self.ids = append(self.ids, ids)

        if self.x_transformed is None:
            self.x_transformed = x_transformed
        elif x_transformed is not None:
            self.x_transformed = rbind(self.x_transformed, x_transformed)

    def get_instances_at(self, indexes):
        insts_x = self.x[indexes, :]
        insts_y = None
        insts_id = None
        insts_transformed = None
        if self.y is not None:
            insts_y = self.y[indexes]
        if self.ids is not None:
            insts_id = self.ids[indexes]
        if self.x_transformed is not None:
            insts_transformed = self.x_transformed[indexes, :]
        return insts_x, insts_y, insts_id, insts_transformed

    def add_instance(self, x, y=None, id=None, x_transformed=None):
        if self.x is not None:
            self.x = rbind(self.x, x)
        else:
            self.x = x
        if y is not None:
            if self.y is not None:
                self.y = np.append(self.y, [y])
            else:
                self.y = np.array([y], dtype=int)
        if id is not None:
            if self.ids is not None:
                self.ids = np.append(self.ids, [id])
            else:
                self.ids = np.array([id], dtype=int)
        if x_transformed is not None:
            if self.x_transformed is not None:
                self.x_transformed = rbind(self.x_transformed, x_transformed)
            else:
                self.x_transformed = x_transformed

    def retain_with_mask(self, mask):
        self.x = self.x[mask]
        if self.y is not None:
            self.y = self.y[mask]
        if self.ids is not None:
            self.ids = self.ids[mask]
        if self.x_transformed is not None:
            self.x_transformed = self.x_transformed[mask]

    def remove_instance_at(self, index):
        mask = np.ones(self.x.shape[0], dtype=bool)
        mask[index] = False
        self.retain_with_mask(mask)


def append_instance_lists(list1, list2):
    """Merge two instance lists

    Args:
        list1: InstanceList
        list2: InstanceList
    """
    x = None
    if list1.x is not None and list2.x is not None:
        x = np.vstack([list1.x, list2.x])
    y = None
    if list1.y is not None and list2.y is not None:
        y = append(list1.y, list2.y)
    ids = None
    if list1.ids is not None and list2.ids is not None:
        ids = append(list1.ids, list2.ids)
    x_transformed = None
    if list1.x_transformed is not None and list2.x_transformed is not None:
        x_transformed = rbind(list1.x_transformed, list2.x_transformed)
    return InstanceList(x=x, y=y, ids=ids, x_transformed=x_transformed)


class SKLClassifier(object):
    def __init__(self):
        self.clf = None

    def predict(self, x, type="response"):
        if self.clf is None:
            raise ValueError("classifier not initialized/trained...")
        if type == "response":
            y = self.clf.predict_proba(x)
        else:
            y = self.clf.predict(x)
        return y

    def predict_prob_for_class(self, x, cls):
        if self.clf is None:
            raise ValueError("classifier not initialized/trained...")
        clsindex = np.where(self.clf.classes_ == cls)[0][0]
        # logger.debug("class index: %d" % (clsindex,))
        y = self.clf.predict_proba(x)[:, clsindex]
        return y


class DTClassifier(SKLClassifier):
    def __init__(self):
        SKLClassifier.__init__(self)
        self.max_depth = 5

    @staticmethod
    def fit(x, y, max_depth=5):
        classifier = DTClassifier()
        classifier.max_depth = max_depth

        classifier.clf = DT(max_depth=classifier.max_depth)

        classifier.clf.fit(x, y)
        return classifier


class RFClassifier(SKLClassifier):
    def __init__(self):
        SKLClassifier.__init__(self)

    @staticmethod
    def fit(x, y, n_estimators=10, max_depth=None):
        classifier = RFClassifier()
        classifier.clf = RF(n_estimators=n_estimators, max_depth=max_depth)
        classifier.clf.fit(x, y)
        return classifier


class SVMClassifier(SKLClassifier):
    def __init__(self):
        SKLClassifier.__init__(self)
        self.kernel = 'rbf'  # 'linear', 'poly', 'rbf', 'sigmoid'
        self.degree = 3
        self.C = 1.
        self.gamma = 'auto'

    @staticmethod
    def fit(x, y, C=1., kernel='rbf', degree=3, gamma='auto'):
        classifier = SVMClassifier()
        classifier.C = C
        classifier.kernel = kernel
        classifier.degree = degree
        classifier.gamma = gamma

        classifier.clf = svm.SVC(C=classifier.C, kernel=classifier.kernel,
                                 degree=classifier.degree,
                                 gamma=gamma, coef0=0.0, shrinking=True,
                                 probability=True, tol=0.001, cache_size=200,
                                 class_weight=None, verbose=False, max_iter=-1,
                                 random_state=None)

        classifier.clf.fit(x, y)
        return classifier


class LogisticRegressionClassifier(SKLClassifier):
    """
    see:
        http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """
    def __init__(self):
        SKLClassifier.__init__(self)

    @staticmethod
    def fit(x, y, penalty='l2', C=1., fit_intercept=True):
        classifier = LogisticRegressionClassifier()
        classifier.clf = LR(penalty=penalty, dual=False, tol=0.0001, C=C,
                            fit_intercept=fit_intercept, intercept_scaling=1,
                            class_weight=None, random_state=None, solver='liblinear',
                            max_iter=100, multi_class='ovr', verbose=0)
        classifier.clf.fit(x, y)
        return classifier


def read_csv(file, header=None, sep=',', index_col=None, skiprows=None, usecols=None, encoding='utf8'):
    """Loads data from a CSV

    Returns:
        DataFrame
    """

    if header is not None and header:
        header = 0 # first row is header

    data_df = pd.read_csv(file, header=header, sep=sep, index_col=index_col, skiprows=skiprows, usecols=usecols, encoding=encoding)

    return data_df


def read_resource(resource_path, package_name):
    """ Reads resource files packaged with the python library """
    data = None
    # print("trying resource package: {}; {}".format(package_name, resource_path))
    try:
        data = pkgutil.get_data(package_name, resource_path)
        # print("data type: {}".format(type(data)))
    except:
        pass
    return data


def read_resource_csv(resource_path, package_name='ad_examples.datasets',
                      header=None, sep=',', skiprows=None, usecols=None, encoding='utf8'):
    """ First tries to load the resource from package; if fail, try path as file in 'datasets' folder in source """
    data = read_resource(resource_path=resource_path, package_name=package_name)
    if data is not None:
        # print("Found resource {}".format(resource_path))
        data_df = read_csv(io.BytesIO(data), header=header, sep=sep, skiprows=skiprows, index_col=None, usecols=usecols, encoding=encoding)
    else:
        # perhaps the code is being run from source 'python folder; offer legacy support
        file_resource_path = "ad_examples/datasets/" + resource_path
        print("Loading file (legacy support) '%s' instead of package resource '%s' ..." % (file_resource_path, resource_path))
        data_df = read_csv(file_resource_path, header=header, sep=sep, skiprows=skiprows, index_col=None, usecols=usecols, encoding=encoding)
    return data_df


def dataframe_to_matrix(df, labelindex=0, startcol=1):
    """ Converts a python dataframe in the expected anomaly dataset format to numpy arrays.

    The expected anomaly dataset format is a CSV with the label ('anomaly'/'nominal')
    as the first column. Other columns are numerical features.

    Note: Both 'labelindex' and 'startcol' are 0-indexed.
        This is different from the 'read_data_as_matrix()' method where
        the 'opts' parameter has same-named attributes but are 1-indexed.

    :param df: Pandas dataframe
    :param labelindex: 0-indexed column number that refers to the class label
    :param startcol: 0-indexed column number that refers to the first column in the dataframe
    :return: (np.ndarray, np.array)
    """
    cols = df.shape[1] - startcol
    x = np.zeros(shape=(df.shape[0], cols))
    for i in range(cols):
        x[:, i] = df.iloc[:, i + startcol]
    labels = np.array([1 if df.iloc[i, labelindex] == "anomaly" else 0 for i in range(df.shape[0])], dtype=int)
    return x, labels


def read_data_as_matrix(opts):
    """ Reads data from CSV file and returns numpy matrix.

    Important: Assumes that the first column has the label \in {anomaly, nominal}

    :param opts: AadOpts
        Supplies parameters like file name, whether first row contains header, etc...
    :return: numpy.ndarray
    """
    if opts.labelindex != 1:
        raise ValueError("Invalid label index parameter %d" % opts.labelindex)

    data = read_csv(opts.datafile, header=opts.header, sep=',')
    labelindex = opts.labelindex - 1
    startcol = opts.startcol - 1
    return dataframe_to_matrix(data, labelindex=labelindex, startcol=startcol)


def save(obj, filepath):
    filehandler = open(filepath, 'w')
    pickle.dump(obj, filehandler)
    return obj


def load(filepath):
    filehandler = open(filepath, 'r')
    obj = pickle.load(filehandler)
    return obj


def dir_create(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def exception_to_string(exc):
    exc_type, exc_value, exc_traceback = exc
    return (str(exc_type) + os.linesep + str(exc_value)
            + os.linesep + str(traceback.extract_tb(exc_traceback)))


def configure_logger(args):
    global logger
    logger_format = "%(levelname)s [%(asctime)s]: %(message)s"
    logger_level = logging.DEBUG if args.debug else logging.ERROR
    if args.log_file is not None and args.log_file != "":
        # print "configuring logger to file %s" % (args.log_file,)
        logging.basicConfig(filename=args.log_file,
                            level=logger_level, format=logger_format,
                            filemode='w') # use filemode='a' for APPEND
    else:
        logging.basicConfig(level=logger_level, format=logger_format)
    logger = logging.getLogger("default")


class Timer(object):
    def __init__(self):
        self.start_time = timer()
        self.end_time = None

    def start(self):
        self.start_time = timer()
        self.end_time = None

    def end(self):
        self.end_time = timer()

    def elapsed(self):
        etime = self.end_time
        if etime is None:
            etime = timer()
        return difftime(etime, self.start_time, units="secs")

    def message(self, msg):
        if self.end_time is None:
            self.end_time = timer()
        tdiff = self.elapsed()
        return "%s %f sec(s)" % (msg, tdiff)


def constr_optim(theta, f, grad=None, ui=None, ci=None, a=None, b=None,
                 hessian=None, bounds=None, method="BFGS",
                 outer_iterations=500, debug=False, args=None):
    """solve non-linear constraint optimization with scipy.optimize

    problems have the form:
        minimize f_0(x)
        s.t.
            ui * x >= ci             --> Note: this is opposite of cvxopt
            a * x = b                --> Supported
            #f_k(x) <= 0, k=1..m     --> Not supported

    :param theta: np.array
            initial values. Must be in the domain of f()
    :param f: function that is being minimized
            returns the function evaluation
    :param grad: function
            returns the first derivative
    :param ui: np.ndarray
    :param ci: np.array
    :param a: np.ndarray
    :param b: np.array
    :param mu:
    :param control:
    :param method:
    :param hessian:
    :param outer_iterations:
    :param outer_eps:
    :param debug:
    :param bounds:
    :param args:
    :return:
    """
    x0 = np.array(theta)
    # build the constraint set
    cons = ()
    if ui is not None:
        for i in range(nrow(ui)):
            # cons += ({'type': 'ineq', 'fun': lambda x: x.dot(u_) - c_},)
            def fcons_ineq(x, i=i):
                return x.dot(ui[i, :]) - ci[i]
            cons += ({'type': 'ineq', 'fun': fcons_ineq},)
    if a is not None:
        for i in range(nrow(a)):
            def fcons_eq(x, i=i):
                return x.dot(a[i, :]) - b[i]
            cons += ({'type': 'eq', 'fun': fcons_eq},)
    res = opt.minimize(f, x0,
                       args=() if args is None else args,
                       method=method, jac=grad,
                       hess=hessian, hessp=None, bounds=bounds,
                       constraints=cons, tol=1e-6, callback=None,
                       #options={'gtol': 1e-6, 'maxiter': outer_iterations, 'disp': True}
                       options={'maxiter': outer_iterations, 'disp': debug}
                       )
    if not res.success:
        logger.debug("Optimization Failure:\nStatus: %d; Msg: %s" % (res.status, res.message))
    return res.x, res.success


class IdServer(object):
    def __init__(self, initial=0):
        self.curr = initial

    def get_next(self, n=1):
        """Returns n ids and adjusts self.curr"""
        ids = np.arange(self.curr, self.curr+n)
        self.curr += n
        return ids


class DataStream(object):
    def __init__(self, X, y=None, id_server=None):
        self.X = X
        self.y = y
        self.id_server = id_server

    def read_next_from_stream(self, n=1):
        """Returns first n instances from X and removes these instances from X"""
        n = min(n, self.X.shape[0])
        # logger.debug("DataStream.read_next_from_stream n: %d" % n)
        if n == 0:
            return None
        mask = np.zeros(self.X.shape[0], dtype=bool)
        mask[np.arange(n)] = True
        instances = self.X[mask]
        self.X = self.X[~mask]
        labels = None
        if self.y is not None:
            labels = self.y[mask]
            self.y = self.y[~mask]
        ids = None
        if self.id_server is not None:
            ids = self.id_server.get_next(n)
        # logger.debug("DataStream.read_next_from_stream instances: %s" % str(instances.shape))
        return InstanceList(instances, labels, ids)

    def empty(self):
        return self.X is None or self.X.shape[0] == 0


class StreamingSupport(object):

    def supports_streaming(self):
        """Whether the stream updating APIs are supported"""
        return False

    def add_samples(self, X, current=True):
        """Updates the count of samples at the temporary buffer or at the nodes"""
        raise NotImplementedError("add_samples() has not been implemented.")

    def update_model_from_stream_buffer(self):
        """Moves the sample counts from the temporary buffer to the current nodes.
        The buffer sample counts are not used in anomaly score computation.
        The buffer counts are updated when data streams in, but the node
        counts are not updated immediately. This method explicitly updates
        the node counts.
        """
        raise NotImplementedError("update_model_from_stream_buffer() has not been implemented.")


def get_rearranging_indexes(add_pos, move_pos, n):
    """Creates an array 0...n-1 and moves value at 'move_pos' to 'add_pos', and shifts others back
    Useful to reorder data when we want to move instances from unlabeled set to labeled.
    TODO:
        Use this to optimize the API StreamingAnomalyDetector.get_query_data()
        since it needs to repeatedly convert the data to transformed [node] features.
    Example:
        get_rearranging_indexes(2, 2, 10):
            array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        get_rearranging_indexes(0, 1, 10):
            array([1, 0, 2, 3, 4, 5, 6, 7, 8, 9])
        get_rearranging_indexes(2, 9, 10):
            array([0, 1, 9, 2, 3, 4, 5, 6, 7, 8])
    :param add_pos:
    :param move_pos:
    :param n:
    :return:
    """
    if add_pos > move_pos:
        raise ValueError("add_pos must be less or equal to move_pos")
    rearr_idxs = np.arange(n)
    if add_pos == move_pos:
        return rearr_idxs
    rearr_idxs[(add_pos + 1):(move_pos + 1)] = rearr_idxs[add_pos:move_pos]
    rearr_idxs[add_pos] = move_pos
    return rearr_idxs

__all__ = ["get_tree_partitions", "RandomSplitTree", "RandomSplitForest",
           "ArrTree", "HSSplitter", "HSTree", "HSTrees",
           "RSForestSplitter", "RSTree", "RSForest",
           "IForest", "StreamingSupport",
           "TREE_UPD_OVERWRITE", "TREE_UPD_INCREMENTAL", "tree_update_types"]

INTEGER_TYPES = (numbers.Integral, np.int)

IS_FIRST = 1
IS_NOT_FIRST = 0
IS_LEFT = 1
IS_NOT_LEFT = 0

TREE_LEAF = -1
TREE_UNDEFINED = -2
INFINITY = np.inf
EPSILON = np.finfo('double').eps

TREE_UPD_OVERWRITE = 0
TREE_UPD_INCREMENTAL = 1
tree_update_types = ["ovr", "incr"]


def get_tree_partitions(n_trees, n_views):
    """Returns an array with (almost) equal values representing an uniform partition"""
    # assume equal number of trees per view
    n_trees_per_view = int(n_trees / n_views)
    n_estimators_view = np.ones(n_views, dtype=int) * n_trees_per_view
    # adjust the number of trees for the last view so that the total is n_trees
    n_estimators_view[n_views - 1] = n_trees - np.sum(n_estimators_view[:-1])
    return n_estimators_view


class SplitContext(object):
    def __init__(self, min_vals=None, max_vals=None, r=1.):
        self.min_vals = min_vals
        self.max_vals = max_vals
        self.r = r

    def clone(self):
        sd = copy.deepcopy(self)
        return sd

    def __str__(self):
        tmp = cbind(self.min_vals, self.max_vals)
        return "r: %f, ranges:\n%s" % (self.r, str(np.transpose(tmp)))


class SplitRecord(object):
    def __init__(self, feature=0, threshold=0, pos=0, impurity_right=0, impurity_left=0):
        self.feature = feature
        self.threshold = threshold
        self.pos = pos
        self.impurity_right = impurity_right
        self.impurity_left = impurity_left
        self.left_context = None
        self.right_context = None


class StackRecord(object):
    def __init__(self, start, end, depth, parent, is_left,
                 impurity=0.0, n_constant_features=0, split_context=None):
        self.start = start
        self.end = end
        self.depth = depth
        self.parent = parent
        self.is_left = is_left
        self.impurity = impurity
        self.n_constant_features = n_constant_features
        self.split_context = split_context


class Node(object):
    def __init__(self):
        self.left_child = -1
        self.right_child = -1
        self.feature = -1
        self.threshold = -1
        self.impurity = -1
        self.n_node_samples = -1
        self.weighted_n_node_samples = -1

    def __str__(self):
        return "feature: %d, thres: %3.8f, n_node_samples: %3.2f, left: %d, right: %d" % \
                (self.feature, self.threshold, self.n_node_samples, self.left_child, self.right_child)

    def __repr__(self):
        return "feature[%d], thres[%3.8f], n_node_samples[%3.2f], left[%d], right[%d]" % \
                (self.feature, self.threshold, self.n_node_samples, self.left_child, self.right_child)


class ArrTree(object):
    """
    Array-based representation of a binary decision tree.
    
        Attributes
        ----------
        node_count : int
            The number of nodes (internal nodes + leaves) in the tree.
    
        capacity : int
            The current capacity (i.e., size) of the arrays, which is at least as
            great as `node_count`.
    
        max_depth : int
            The maximal depth of the tree.

        update_type: int
            Specifies how to update the tree node counts.

        incremental_update_weight: float
            For incremental weight update, specifies the weight given to current counts.
            Should be in range [0.0, 1.0]
    
        children_left : array of int, shape [node_count]
            children_left[i] holds the node id of the left child of node i.
            For leaves, children_left[i] == TREE_LEAF. Otherwise,
            children_left[i] > i. This child handles the case where
            X[:, feature[i]] <= threshold[i].
    
        children_right : array of int, shape [node_count]
            children_right[i] holds the node id of the right child of node i.
            For leaves, children_right[i] == TREE_LEAF. Otherwise,
            children_right[i] > i. This child handles the case where
            X[:, feature[i]] > threshold[i].
    
        feature : array of int, shape [node_count]
            feature[i] holds the feature to split on, for the internal node i.
    
        threshold : array of double, shape [node_count]
            threshold[i] holds the threshold for the internal node i.
    
        value : array of double, shape [node_count, n_outputs, max_n_classes]
            Contains the constant prediction value of each node.
    
        impurity : array of double, shape [node_count]
            impurity[i] holds the impurity (i.e., the value of the splitting
            criterion) at node i.
    
        n_node_samples : array of int, shape [node_count]
            n_node_samples[i] holds the number of training samples reaching node i.
    
        weighted_n_node_samples : array of int, shape [node_count]
            weighted_n_node_samples[i] holds the weighted number of training samples
            reaching node i.
    """
    def __init__(self, n_features, max_depth=0, update_type=TREE_UPD_OVERWRITE,
                 incremental_update_weight=0.5):
        self.n_features = n_features
        self.max_depth = max_depth
        self.update_type = update_type
        self.incremental_update_weight = incremental_update_weight

        self.node_count = 0
        self.capacity = 0

        self.nodes = None
        self.children_left = None
        self.children_right = None
        self.feature = None
        self.threshold = None
        self.v = None  # fraction of feature length relative to feature length at parent node
        self.acc_log_v = None  # log-scaled ratio of the current-node volume to the volume of entire feature space.
        self.value = None
        self.impurity = None
        self.n_node_samples = None
        self.n_node_samples_buffer = None
        self.weighted_n_node_samples = None

        self.value_stride = None

        self.clear()

    def clear(self):
        self.nodes = np.zeros(0, dtype=int)
        self.children_left = np.zeros(0, dtype=int)
        self.children_right = np.zeros(0, dtype=int)
        self.feature = np.zeros(0, dtype=int)
        self.threshold = np.zeros(0, dtype=float)
        self.v = np.zeros(0, dtype=float)
        self.acc_log_v = np.zeros(0, dtype=float)
        self.value = np.zeros(0, dtype=float)
        self.impurity = np.zeros(0, dtype=float)
        self.n_node_samples = np.zeros(0, dtype=float)
        self.n_node_samples_buffer = np.zeros(0, dtype=float)
        self.weighted_n_node_samples = np.zeros(0, dtype=float)

    def str_node(self, node_id):
        return "[%04d] feature: %d, thres: %3.8f, v: %3.8f, acc_log_v: %3.8f, n_node_samples: %3.2f, left: %d, right: %d" % \
               (node_id, self.feature[node_id], self.threshold[node_id],
                self.v[node_id], self.acc_log_v[node_id],
                self.n_node_samples[node_id],
                self.children_left[node_id], self.children_right[node_id])

    def resize(self, capacity=-1):
        """Resize all inner arrays to `capacity`, if `capacity` == -1, then
           double the size of the inner arrays.
        """
        # below code is from Cython implementation in sklearn
        self.resize_c(capacity)

    def resize_c(self, capacity=-1):
        """ Guts of resize """

        # below code is from Cython implementation in sklearn
        if capacity == self.capacity and self.nodes is not None:
            return 0

        if capacity == -1:
            if self.capacity == 0:
                capacity = 3  # default initial value
            else:
                capacity = 2 * self.capacity

        if self.nodes is None:
            self.nodes = np.zeros(capacity, dtype=int)
        else:
            self.nodes = np.resize(self.nodes, capacity)
        self.children_left = np.resize(self.children_left, capacity)
        self.children_right = np.resize(self.children_right, capacity)
        self.feature = np.resize(self.feature, capacity)
        self.threshold = np.resize(self.threshold, capacity)
        self.v = np.resize(self.v, capacity)
        self.acc_log_v = np.resize(self.acc_log_v, capacity)
        self.value = np.resize(self.value, capacity)
        self.impurity = np.resize(self.impurity, capacity)
        self.n_node_samples = np.resize(self.n_node_samples, capacity)
        self.n_node_samples_buffer = np.resize(self.n_node_samples_buffer, capacity)
        self.weighted_n_node_samples = np.resize(self.weighted_n_node_samples, capacity)

        # if capacity smaller than node_count, adjust the counter
        if capacity < self.node_count:
            self.node_count = capacity

        self.capacity = capacity

        return 0

    def reset_n_node_samples(self):
        self.n_node_samples[:] = 0

    def add_node(self, parent, is_left, is_leaf, feature,
                 threshold, v, impurity, n_node_samples,
                 weighted_n_node_samples):
        """Add a node to the tree.

        The new node registers itself as the child of its parent.

        Returns (size_t)(-1) on error.
        """
        node_id = self.node_count

        # below is from Cython implementation
        if node_id >= self.capacity:
            if self.resize_c() != 0:
                return -1

        self.nodes[node_id] = node_id
        self.impurity[node_id] = impurity
        self.n_node_samples[node_id] = n_node_samples
        self.weighted_n_node_samples[node_id] = weighted_n_node_samples

        self.v[node_id] = v
        self.acc_log_v[node_id] = np.log(v)

        if parent != TREE_UNDEFINED:
            self.acc_log_v[node_id] += self.acc_log_v[parent]
            if is_left:
                self.children_left[parent] = node_id
            else:
                self.children_right[parent] = node_id

        if is_leaf:
            self.children_left[node_id] = TREE_LEAF
            self.children_right[node_id] = TREE_LEAF
            self.feature[node_id] = TREE_UNDEFINED
            self.threshold[node_id] = TREE_UNDEFINED
        else:
            # left_child and right_child will be set later
            self.feature[node_id] = feature
            self.threshold[node_id] = threshold

        self.node_count += 1

        return node_id

    def add_samples(self, X, current=True):
        if self.node_count < 1:
            # no nodes; likely tree has not been constructed yet
            raise ValueError("Tree not constructed yet")
        for i in np.arange(X.shape[0]):
            node = 0  # start at root
            while node >= 0:
                if current:
                    self.n_node_samples[node] += 1
                else:
                    self.n_node_samples_buffer[node] += 1
                val = X[i, self.feature[node]]
                if self.children_left[node] == -1 and self.children_right[node] == -1:
                    # reached leaf
                    # self.n_node_samples[node] += 1
                    break
                if val <= self.threshold[node]:
                    next_node = self.children_left[node]
                else:
                    next_node = self.children_right[node]
                node = next_node

    def get_all_leaf_nodes(self):
        leaves = np.zeros(self.node_count, dtype=int)
        i = 0
        for node in np.arange(self.node_count):
            if self.children_left[node] == -1 and self.children_right[node] == -1:
                leaves[i] = node
                i += 1
        return leaves[0:i]

    def update_model_from_stream_buffer(self):
        if False:
            # debug
            leaves = self.get_all_leaf_nodes()
            logger.debug("buffer:\n%s" % str(list(self.n_node_samples_buffer[leaves])))
            n_prev_buffer = np.sum(self.n_node_samples_buffer[leaves])
            n_prev_curr = np.sum(self.n_node_samples[leaves])
        if self.update_type == TREE_UPD_OVERWRITE:
            # logger.debug("update overwrite")
            np.copyto(self.n_node_samples, self.n_node_samples_buffer)
        elif self.update_type == TREE_UPD_INCREMENTAL:
            # logger.debug("update incremental (%f)" % self.incremental_update_weight)
            self.n_node_samples *= (1. - self.incremental_update_weight)
            self.n_node_samples += (self.incremental_update_weight * self.n_node_samples_buffer)
        else:
            raise ValueError("Invalid tree update type: %d" % self.update_type)
        self.n_node_samples_buffer[:] = 0
        if False:
            # debug
            n_curr_buffer = np.sum(self.n_node_samples_buffer[leaves])
            n_curr_curr = np.sum(self.n_node_samples[leaves])
            logger.debug("update_model_from_stream_buffer() (%d, %d) -> (%d, %d)" %
                         (n_prev_curr, n_prev_buffer, n_curr_curr, n_curr_buffer))

    def apply(self, X, getleaves=True, getnodeinds=False):
        """Returns the nodes and/or the leaves through which the instances pass.

        :param X: matrix (might be sparse)
            Input instances where each row is an instance
        :param getleaves: boolean
            If True, then the final leaf node index for each input instance will be returned
        :param getnodeinds:
            If True, each node through which an instance passes from root to leaf will be returned.
        :return: tuple
        """
        if self.node_count < 1:
            # no nodes; likely tree has not been constructed yet
            raise ValueError("Tree not constructed yet")
        n = X.shape[0]
        leaves = None
        if getleaves:
            leaves = np.zeros(n, dtype=int)
        x_tmp = None
        if getnodeinds:
            nodeinds = csr_matrix((0, self.node_count), dtype=float)
            x_tmp = lil_matrix((n, self.node_count), dtype=nodeinds.dtype)
        for i in np.arange(n):
            node = 0  # start at root
            while node >= 0:
                if getnodeinds:
                    x_tmp[i, node] = 1
                v = X[i, self.feature[node]]
                if self.children_left[node] == -1 and self.children_right[node] == -1:
                    # reached leaf
                    if getleaves:
                        leaves[i] = node
                    break
                if v <= self.threshold[node]:
                    next_node = self.children_left[node]
                else:
                    next_node = self.children_right[node]
                node = next_node
        if getnodeinds:
            nodeinds = x_tmp.tocsr()
            return leaves, nodeinds
        return leaves

    def __repr__(self):
        s = ''
        pfx = '-'
        stack = list()
        stack.append((0, 0))
        while len(stack) > 0:
            node_id, depth = stack.pop()
            # logger.debug(node_id)
            s = s + "%s%s\n" % (pfx*depth, self.str_node(node_id))
            if self.children_right[node_id] != -1:
                stack.append((self.children_right[node_id], depth + 1))
            if self.children_left[node_id] != -1:
                stack.append((self.children_left[node_id], depth + 1))
        return s

    def __str__(self):
        return self.__repr__()


def HPDByInverseCDF(x, p=0.90, sigs=0):
    """Highest probability density by inverse cumulative distribution function

    Args:
        x: np.array

    Returns: (float, float, float)
        lower interval, upper interval, variance of x
    """
    v = np.var(x)
    if v == 0:
        return x[0], x[0], v
    x_cdf = ecdf(x)
    lc = (1 - p) / 2
    uc = p + lc
    cis = x_cdf([lc, uc])
    return cis[0] - sigs * v, cis[1] + sigs * v, v


class RandomTreeBuilder(object):
    """
    Attributes:
        splitter: HSSplitter
        max_depth: int
    """
    def __init__(self, splitter,
                 max_depth):
        self.splitter = splitter
        self.max_depth = max_depth

    def build(self, tree, X, y, sample_weight=None, X_idx_sorted=None):
        """Build a decision tree from the training set (X, y).
        
        Args:
            tree: ArrTree
            X: numpy.ndarray
            y: numpy.array
            sample_weight: numpy.array
            X_idx_sorted: numpy.array
        """

        if tree.max_depth <= 10:
            init_capacity = (2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047

        tree.resize(init_capacity)

        splitter = self.splitter
        max_depth = self.max_depth
        sample_weight_ptr = None

        # Recursive partition (without actual recursion)
        splitter.init(X, y, sample_weight_ptr, X_idx_sorted)

        n_node_samples = splitter.n_samples
        weighted_n_node_samples = None

        first = 1
        max_depth_seen = -1
        split_record = SplitRecord()
        stack = list()

        stack.append(StackRecord(0, n_node_samples, 0, TREE_UNDEFINED, 0,
                                 INFINITY, 0, splitter.split_context))

        while len(stack) > 0:
            stack_record = stack.pop()

            start = stack_record.start
            end = stack_record.end
            depth = stack_record.depth
            parent = stack_record.parent
            is_left = stack_record.is_left
            impurity = stack_record.impurity
            n_constant_features = stack_record.n_constant_features
            split_context = stack_record.split_context

            # logger.debug("feature ranges:\n%s" % str(split_context))

            n_node_samples = 0
            splitter.node_reset(split_context)

            if first:
                first = 0

            is_leaf = (depth >= max_depth)

            if not is_leaf:
                splitter.node_split(impurity, split_record, n_constant_features)

            node_id = tree.add_node(parent, is_left, is_leaf, split_record.feature,
                                    split_record.threshold, split_context.r,
                                    impurity, n_node_samples,
                                    weighted_n_node_samples)
            # logger.debug("Node: %s" % str(tree.nodes[node_id]))

            if not is_leaf:
                # Push right child on stack
                stack.append(StackRecord(split_record.pos, end, depth + 1, node_id, 0,
                             split_record.impurity_right, n_constant_features, split_record.right_context))

                # Push left child on stack
                stack.append(StackRecord(start, split_record.pos, depth + 1, node_id, 1,
                             split_record.impurity_left, n_constant_features, split_record.left_context))

            if False and parent >= 0:
                logger.debug("Parent Node: %s" % str(tree.nodes[parent]))

            if depth > max_depth_seen:
                max_depth_seen = depth

            # tree.resize_c(tree.node_count)
            tree.max_depth = max_depth_seen

        tree.reset_n_node_samples()
        tree.add_samples(X)


class RandomSplitTree(object):
    def __init__(self,
                 criterion=None,
                 splitter=None,
                 max_depth=10,
                 max_features=1,
                 random_state=None,
                 update_type=TREE_UPD_OVERWRITE,
                 incremental_update_weight=0.5):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.update_type = update_type
        self.incremental_update_weight = incremental_update_weight

        self.n_features_ = None
        self.n_outputs_ = None
        self.classes_ = None
        self.n_classes_ = None

        self.tree_ = None
        self.max_features_ = None

    def get_splitter(self, splitter=None):
        raise NotImplementedError("get_splitter() has not been implemented")

    def get_builder(self, splitter, max_depth):
        return RandomTreeBuilder(splitter, max_depth)

    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted=None):

        n_samples, self.n_features_ = X.shape

        max_depth = ((2 ** 31) - 1 if self.max_depth is None
                     else self.max_depth)

        self.n_outputs_ = 1
        self.n_classes_ = [1] * self.n_outputs_
        self.n_classes_ = np.array(self.n_classes_, dtype=np.intp)
        self.tree_ = ArrTree(self.n_features_, update_type=self.update_type,
                             incremental_update_weight=self.incremental_update_weight)

        splitter = self.get_splitter(self.splitter)
        builder = self.get_builder(splitter, max_depth)
        builder.build(self.tree_, X, y)

    def apply(self, X):
        return self.tree_.apply(X, getleaves=True, getnodeinds=False)

    def decision_function(self, X):
        """Average anomaly score of X (smaller values are more anomalous).

        This score ordering has been maintained such that it is compatible
        with the scikit-learn Isolation Forest API.
        """
        raise NotImplementedError("decision_function() has not been implemented.")


class RandomSplitForest(StreamingSupport):
    """Logic for Half-Space Trees (HSTrees) by default

    Return the anomaly score of each sample using the HSTrees algorithm

    Parameters
    ----------
    n_estimators : int, optional (default=100)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default="auto")
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
            - If "auto", then `max_samples=min(256, n_samples)`.
        If max_samples is larger than the number of samples provided,
        all samples will be used for all trees (no sampling).

    max_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.

            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

    min_vals : list of float, optional (default=None)
        The minimum value for each feature/dimension
        This list must be of the same length as the number of data dimensions
    
    max_vals : list of float, optional (default=None)
        The maximum value for each feature/dimension.
        This list must be of the same length as the number of data dimensions.
    
    max_depth: int
        The maximum depth to which to grow the tree
    
    bootstrap : boolean, optional (default=False)
        If True, individual trees are fit on random subsets of the training
        data sampled with replacement. If False, sampling without replacement
        is performed.
    
    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    update_type : int
        Specifies how to update the tree node counts.

    incremental_update_weight : float
        For incremental weight update, specifies the weight given to current counts.
        Should be in range [0.0, 1.0]

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.


    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator.

    max_samples_ : integer
        The actual number of samples

    References
    ----------
    .. [1] 

    """

    def __init__(self,
                 n_estimators=100,
                 max_samples="auto",
                 max_features=1.,
                 min_vals=None,
                 max_vals=None,
                 max_depth=10,
                 bootstrap=False,
                 n_jobs=1,
                 random_state=None,
                 update_type=TREE_UPD_OVERWRITE,
                 incremental_update_weight=0.5,
                 verbose=0):
        self.max_samples=max_samples
        self.max_features=max_features
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.min_vals = min_vals
        self.max_vals = max_vals
        self.max_depth = max_depth
        self.random_state = random_state
        self.update_type = update_type
        self.incremental_update_weight = incremental_update_weight
        self.estimators_ = None

    def _set_oob_score(self, X, y):
        raise NotImplementedError("OOB score not supported by iforest")

    def get_fitting_function(self):
        raise NotImplementedError("get_fiting_function() not implemented")

    def get_decision_function(self):
        raise NotImplementedError("get_decision_function() not implemented")

    def _fit(self, X, y, max_samples, max_depth, sample_weight=None):
        n_trees = self.n_estimators
        n_pool = self.n_jobs

        p = Pool(n_pool)
        rnd_int = self.random_state.randint(42)
        if isinstance(max_samples, str):
            max_samples = min(256, X.shape[0])
        logger.debug("max_samples: %d" % max_samples)
        trees = p.map(self.get_fitting_function(),
                      [(max_depth, X, max_samples, rnd_int + i, self.update_type, self.incremental_update_weight) for i in range(n_trees)])
        return trees

    def fit(self, X, y=None, sample_weight=None):
        """Fit estimator.

        Parameters
        ----------
        X : numpy.ndarray
            array-like or sparse matrix, shape (n_samples, n_features)
            The input samples. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csc_matrix`` for maximum efficiency.

        Returns
        -------
        self : object
            Returns self.
        """
        # ensure_2d=False because there are actually unit test checking we fail
        # for 1d.
        X = check_array(X, accept_sparse=['csc'], ensure_2d=True)
        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        self.random_state = check_random_state(self.random_state)
        y = self.random_state.uniform(size=X.shape[0])

        # ensure that max_sample is in [1, n_samples]:
        n_samples = X.shape[0]

        self.max_samples_ = n_samples

        self.estimators_ = self._fit(X, y, self.max_samples,
                                     max_depth=self.max_depth,
                                     sample_weight=sample_weight)

        if False:
            for i, estimator in enumerate(self.estimators_):
                logger.debug("Estimator %d:\n%s" % (i, str(estimator.tree_)))
                logger.debug("Node samples:\n%s" % str(estimator.tree_.n_node_samples))

        return self

    def predict(self, X):
        """Predict if a particular sample is an outlier or not."""
        raise NotImplementedError("predict() is not supported for RandomTrees")

    def decision_function(self, X):
        """Average anomaly score of X of the base classifiers."""
        scores = np.zeros((1, X.shape[0]))
        tm = Timer()
        if True:
            n_pool = self.n_jobs
            p = Pool(n_pool)
            hst_scores = p.map(self.get_decision_function(), [(X, hst, i) for i, hst in enumerate(self.estimators_)])
        else:
            hst_scores = list()
            for tree_id, hst in enumerate(self.estimators_):
                tm_tree = Timer()
                hst_scores.append(hst.decision_function(X))
                logger.debug(tm_tree.message("completed Tree[%d] decision function" % tree_id))
        logger.debug(tm.message("completed Trees decision_function"))
        for s in hst_scores:
            scores += s
        scores /= len(hst_scores)
        return scores.reshape((scores.shape[1],))

    def supports_streaming(self):
        return True

    def add_samples(self, X, current=True):
        for tree in self.estimators_:
            tree.tree_.add_samples(X, current)

    def get_node_ids(self, X, getleaves=True):
        if not getleaves:
            raise ValueError("Operation supported for leaf level only")
        forest_nodes = list()
        for estimator in self.estimators_:
            tree_nodes = estimator.apply(X)
            forest_nodes.append(tree_nodes)
        return forest_nodes

    def update_trees_by_replacement(self, X=None, replace_trees=None):
        """ Replaces current trees with new ones constructed from X

        :param X: numpy.ndarray
            Data matrix from which new trees will be constructed
        :param replace_trees: numpy.array(dtype=int)
            The indexes of trees to be replaced
        :return:
        """
        raise NotImplementedError("update_trees_by_replacement() is not implemented")

    def update_model_from_stream_buffer(self, replace_trees=None):
        """ Updates the model from current internal buffer

        :param replace_trees: numpy.array(dtype=int)
            The indexes of trees to be replaced
        :return:
        """
        for tree in self.estimators_:
            tree.tree_.update_model_from_stream_buffer()

        return None


class HSSplitter(object):
    """
    Attributes:
        split_context: SplitContext
    """
    def __init__(self, random_state=None):
        self.n_samples = 0
        self.weighted_n_samples = None
        self.split_context = None
        if random_state is None: print("No random state")
        self.random_state = check_random_state(random_state)

    def get_feature_ranges(self, X, rnd=None):
        """
        :param X: np.ndarray
        :return: (np.array, np.array)
        """
        rnd = self.random_state if rnd is None else rnd
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        diff = max_vals - min_vals
        sq = rnd.uniform(0, 1, len(min_vals))
        # logger.debug("sq: %s" % (str(sq)))
        sq_mn = sq - 2 * np.maximum(sq, 1 - sq)
        sq_mx = sq + 2 * np.maximum(sq, 1 - sq)
        mn = min_vals + diff * sq_mn
        mx = min_vals + diff * sq_mx
        return mn, mx

    def init(self, X, y, sample_weight_ptr, X_idx_sorted):
        self.n_samples = X.shape[0]
        min_vals, max_vals = self.get_feature_ranges(X, self.random_state)
        self.split_context = SplitContext(min_vals=min_vals, max_vals=max_vals, r=1.0)
        # logger.debug("root feature ranges:\n%s" % str(self.split_context))

    def node_reset(self, split_context, weighted_n_node_samples=None):
        self.split_context = split_context

    def node_split(self, impurity, split_record, n_constant_features):
        # select a random feature and split it in half
        feature = self.random_state.randint(0, len(self.split_context.min_vals))
        # logger.debug("splitting %d [%f, %f]" % (feature, self.split_context.min_vals[feature], self.split_context.max_vals[feature]))
        threshold = 0.5 * (self.split_context.min_vals[feature] + self.split_context.max_vals[feature])
        split_record.feature = feature
        split_record.threshold = threshold
        split_record.r = 0.5  # deterministic in case of HS Trees

        split_record.left_context = self.split_context.clone()
        split_record.left_context.max_vals[feature] = threshold
        split_record.left_context.r = split_record.r

        split_record.right_context = self.split_context.clone()
        split_record.right_context.min_vals[feature] = threshold
        split_record.right_context.r = 1 - split_record.r


class HSTree(RandomSplitTree):
    def __init__(self,
                 splitter=None,
                 max_depth=10,
                 max_features=1,
                 random_state=None,
                 update_type=TREE_UPD_OVERWRITE,
                 incremental_update_weight=0.5):
        RandomSplitTree.__init__(self,
                                 splitter=splitter,
                                 max_depth=max_depth,
                                 max_features=max_features,
                                 random_state=random_state,
                                 update_type=update_type,
                                 incremental_update_weight=incremental_update_weight)

    def get_splitter(self, splitter=None):
        return splitter if splitter is not None else HSSplitter(random_state=self.random_state)

    def decision_function(self, X):
        """Average anomaly score of X (smaller values are more anomalous).

        This score ordering has been maintained such that it is compatible
        with the scikit-learn Isolation Forest API.
        """
        if False:
            # Process all instances at once.
            leaves, nodeinds = self.tree_.apply(X, getleaves=True, getnodeinds=True)
            depths = np.array(np.transpose(nodeinds.sum(axis=1)))
            scores = self.tree_.n_node_samples[leaves] * (2. ** depths)
        else:
            # Process instances in batches. This saves some memory when the number of
            # nodes is very high and so is the number of instances.
            batch_size = 1000
            n = X.shape[0]
            scores = np.zeros(n, dtype=np.float32)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                x = X[start:end, :]
                leaves, nodeinds = self.tree_.apply(x, getleaves=True, getnodeinds=True)
                depths = np.array(np.transpose(nodeinds.sum(axis=1)))
                scores[start:end] = self.tree_.n_node_samples[leaves] * (2. ** depths)
        return scores


class HSTrees(RandomSplitForest):
    def __init__(self,
                 n_estimators=100,
                 max_features=1.,
                 min_vals=None,
                 max_vals=None,
                 max_depth=10,
                 n_jobs=1,
                 random_state=None,
                 update_type=TREE_UPD_OVERWRITE,
                 incremental_update_weight=0.5):
        RandomSplitForest.__init__(self, n_estimators=n_estimators,
                                   max_features=max_features,
                                   min_vals=min_vals,
                                   max_vals=max_vals,
                                   max_depth=max_depth,
                                   n_jobs=n_jobs,
                                   random_state=random_state,
                                   update_type=update_type,
                                   incremental_update_weight=incremental_update_weight)

    def get_fitting_function(self):
        return hstree_fit

    def get_decision_function(self):
        return hstree_decision


def hstree_fit(args):
    max_depth = args[0]
    X = args[1]
    max_samples = args[2]
    rnd = args[3]
    update_type = args[4]
    incremental_update_weight = args[5]
    random_state = check_random_state(rnd)
    n = X.shape[0]
    max_samples = min(max_samples, n)
    sample_idxs = np.arange(n)
    random_state.shuffle(sample_idxs)
    X_sub = X[sample_idxs[0:max_samples]]
    hst = HSTree(splitter=HSSplitter(random_state=random_state),
                 max_depth=max_depth, max_features=X_sub.shape[1],
                 random_state=random_state, update_type=update_type,
                 incremental_update_weight=incremental_update_weight)
    hst.fit(X_sub, None)
    return hst


def hstree_decision(args):
    X = args[0]
    hst = args[1]
    tree_id = args[2]
    tm = Timer()
    scores = hst.decision_function(X)
    # logger.debug(tm.message("completed HSTree[%d] decision function" % tree_id))
    return scores


class RSForestSplitter(HSSplitter):
    """
    Attributes:
        split_context: SplitContext
    """
    def __init__(self, random_state=None):
        HSSplitter.__init__(self, random_state=random_state)

    def get_feature_ranges(self, X, rnd=None):
        """
        :param X: np.ndarray
        :return: (np.array, np.array)
        """
        rnd = self.random_state if rnd is None else rnd
        d = X.shape[1]
        mn = np.zeros(d, dtype=float)
        mx = np.zeros(d, dtype=float)
        for i in np.arange(d):
            mn[i], mx[i], _ = HPDByInverseCDF(X[:, i], p=0.9, sigs=3)
        return mn, mx

    def node_split(self, impurity, split_record, n_constant_features):
        # select a random feature and split it in half
        feature = self.random_state.randint(0, len(self.split_context.min_vals))
        # logger.debug("splitting %d [%f, %f]" % (feature, self.split_context.min_vals[feature], self.split_context.max_vals[feature]))
        r = self.random_state.uniform(low=0., high=1., size=1)
        # for interval [a, b], and a random value r
        # the split is: a + r.(b - a) = (1 - r).a + r.b
        threshold = (1 - r) * self.split_context.min_vals[feature] + r * (self.split_context.max_vals[feature])
        split_record.feature = feature
        split_record.threshold = threshold
        split_record.r = r

        split_record.left_context = self.split_context.clone()
        split_record.left_context.max_vals[feature] = threshold
        split_record.left_context.r = split_record.r

        split_record.right_context = self.split_context.clone()
        split_record.right_context.min_vals[feature] = threshold
        split_record.right_context.r = 1 - split_record.r


class RSTree(RandomSplitTree):
    def __init__(self,
                 criterion=None,
                 splitter=None,
                 max_depth=10,
                 max_features=100,
                 random_state=None,
                 update_type=TREE_UPD_OVERWRITE,
                 incremental_update_weight=0.5):
        RandomSplitTree.__init__(self, criterion=criterion,
                                 splitter=splitter,
                                 max_depth=max_depth,
                                 max_features=max_features,
                                 random_state=random_state,
                                 update_type=update_type,
                                 incremental_update_weight=incremental_update_weight)

    def get_splitter(self, splitter=None):
        return splitter if splitter is not None else RSForestSplitter(random_state=self.random_state)

    def decision_function(self, X):
        """Average anomaly score of X (smaller values are more anomalous).

        This score ordering has been maintained such that it is compatible
        with the scikit-learn Isolation Forest API.
        """
        leaves, nodeinds = self.tree_.apply(X, getleaves=True, getnodeinds=True)
        scores = self.tree_.n_node_samples[leaves] * np.exp(-self.tree_.acc_log_v[leaves])
        return scores


class RSForest(RandomSplitForest):
    def __init__(self,
                 n_estimators=100,
                 max_features=1.,
                 min_vals=None,
                 max_vals=None,
                 max_depth=10,
                 n_jobs=1,
                 random_state=None,
                 update_type=TREE_UPD_OVERWRITE,
                 incremental_update_weight=0.5):
        RandomSplitForest.__init__(self, n_estimators=n_estimators,
                                   max_features=max_features,
                                   min_vals=min_vals,
                                   max_vals=max_vals,
                                   max_depth=max_depth,
                                   n_jobs=n_jobs,
                                   random_state=random_state,
                                   update_type=update_type,
                                   incremental_update_weight=incremental_update_weight)

    def get_fitting_function(self):
        return rsforest_fit

    def get_decision_function(self):
        return rsforest_decision


def rsforest_fit(args):
    max_depth = args[0]
    X = args[1]
    max_samples = args[2]
    rnd = args[3]
    update_type = args[4]
    incremental_update_weight = args[5]
    random_state = check_random_state(rnd)
    n = X.shape[0]
    max_samples = min(max_samples, n)
    sample_idxs = np.arange(n)
    random_state.shuffle(sample_idxs)
    X_sub = X[sample_idxs[0:max_samples]]
    rsf = RSTree(splitter=RSForestSplitter(random_state=random_state),
                 max_depth=max_depth, max_features=X_sub.shape[1],
                 random_state=random_state, update_type=update_type,
                 incremental_update_weight=incremental_update_weight)
    rsf.fit(X_sub, None)
    return rsf


def rsforest_decision(args):
    X = args[0]
    hst = args[1]
    tree_id = args[2]
    tm = Timer()
    scores = hst.decision_function(X)
    # logger.debug(tm.message("completed HSTree[%d] decision function" % tree_id))
    return scores


class IForest(RandomSplitForest):
    def __init__(self,
                 n_estimators=100,
                 max_samples="auto",
                 contamination=0.1,
                 max_features=1.,
                 bootstrap=False,
                 n_jobs=1,
                 replace_frac=0.2,
                 random_state=None,
                 verbose=0):
        RandomSplitForest.__init__(self, n_estimators=n_estimators,
                                   max_samples=max_samples,
                                   max_features=max_features,
                                   bootstrap=bootstrap,
                                   n_jobs=n_jobs,
                                   random_state=random_state,
                                   verbose=verbose)
        self.contamination = contamination
        # The fraction of trees replaced when new window of data arrives
        self.replace_frac = replace_frac
        self.ifor = None
        self.estimators_features_ = None
        self.buffer = None
        self.updated = False

    def fit(self, X, y=None, sample_weight=None):
        self.ifor = IsolationForest(n_estimators=self.n_estimators,
                                    max_samples=self.max_samples,
                                    contamination=self.contamination,
                                    max_features=self.max_features,
                                    bootstrap=self.bootstrap,
                                    n_jobs=self.n_jobs,
                                    random_state=self.random_state,
                                    verbose=self.verbose)
        self.ifor.fit(X, y, sample_weight)
        self.estimators_ = self.ifor.estimators_
        self.estimators_features_ = self.ifor.estimators_features_
        self.updated = False

    def _fit(self, X, y, max_samples, max_depth, sample_weight=None):
        raise NotImplementedError("method _fit() not supported")

    def decision_function(self, X):
        if self.updated:
            logger.debug("WARN: The underlying isolation forest was updated and " +
                         "using calling decision_function() on it will likely return inconsistent results.")
        return self.ifor.decision_function(X)

    def supports_streaming(self):
        return True

    def add_samples(self, X, current=True):
        if current:
            raise ValueError("IForest does not support adding to current instance set.")
        if self.buffer is None:
            self.buffer = X
        else:
            self.buffer = np.vstack([self.buffer, X])

    def update_trees_by_replacement(self, X=None, replace_trees=None):
        if X is None:
            X = self.buffer
        if X is None:
            logger.warning("No new data for update")
            return None

        if replace_trees is not None:
            replace_set = set(replace_trees)
            n_new_trees = len(replace_set)
            if n_new_trees < 0:
                raise ValueError("Replacement set is larger than allowed")
            old_tree_indexes_replaced = replace_trees
            old_tree_indexes_retained = np.array([i for i in range(len(self.estimators_)) if i not in replace_set], dtype=int)
        else:
            n_new_trees = int(self.replace_frac * len(self.estimators_))
            old_tree_indexes_replaced = np.arange(0, n_new_trees, dtype=int)
            old_tree_indexes_retained = np.arange(n_new_trees, len(self.estimators_))

        if n_new_trees > 0:
            new_ifor = IsolationForest(n_estimators=n_new_trees,
                                       max_samples=self.max_samples,
                                       contamination=self.contamination,
                                       max_features=self.max_features,
                                       bootstrap=self.bootstrap,
                                       n_jobs=self.n_jobs,
                                       random_state=self.random_state,
                                       verbose=self.verbose)
            new_ifor.fit(X, y=None, sample_weight=None)

            # retain estimators and features
            self.estimators_ = [self.estimators_[i] for i in old_tree_indexes_retained]
            self.estimators_features_ = [self.estimators_features_[i] for i in old_tree_indexes_retained]
            # append the new trees at the end of the list of older trees
            for estimator, features in zip(new_ifor.estimators_, new_ifor.estimators_features_):
                self.estimators_.append(estimator)
                self.estimators_features_.append(features)

            # Now, update the underlying isolation forest
            # NOTE: This might make the model inconsistent
            self.ifor.estimators_ = self.estimators_
            self.ifor.estimators_features_ = self.estimators_features_

            new_estimators = new_ifor.estimators_
        else:
            new_estimators = None

        self.updated = True
        self.buffer = None

        if False:
            logger.debug("IForest update_trees_by_replacement(): n_new_trees: %d, samples: %s" %
                         (n_new_trees, str(X.shape)))

        # we return lists in order to support feature groups in multiview forest (see IForestMultiview)
        return [old_tree_indexes_replaced], [old_tree_indexes_retained], [new_estimators]

    def update_model_from_stream_buffer(self, replace_trees=None):
        return self.update_trees_by_replacement(self.buffer)