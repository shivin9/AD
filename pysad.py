import numpy as np
from abc import ABC, abstractmethod
import abc
import numpy as np


class BaseStreamer(abc.ABC):
    """Abstract base class to simulate the streaming data.
    Args:
        shuffle (bool): Whether shuffle the data initially (Optional, default=False).
    """

    def __init__(self, shuffle=False):
        self.shuffle = shuffle

    @abstractmethod
    def iter(self, X, y=None):
        """Method that iterates array of data and (optionally) labels.
        Args:
            X (np.array of shape (num_instances, num_features)): The features of instances to iterate.
            y: (Optional, default=None) If not None, iterates labels with the same order.
        """
        pass

class ArrayStreamer(BaseStreamer):
    """Simulator class to iterate array(s).

    Args:
        shuffle (bool): Whether shuffle the data initially (Default=False).
    """

    def __init__(self, shuffle=False):
        self.shuffle = shuffle

    def iter(self, X, y=None):
        """Iterates array of features and possibly labels.

        Args:
            X (np.array of shape (num_instances, num_features)): The features array.
            y (np.array of shape (num_instances, ): The array containing labels (Default=None).
        """
        indices = list(range(len(X)))
        if self.shuffle:
            np.random.shuffle(indices)

        if y is None:
            for i in indices:
                yield X[i]
        else:
            assert len(X) == len(y)
            for i in indices:
                yield X[i], y[i]

def _iterate(X, y=None):
    """Iterates array of features and possibly labels.
    Args:
        X (np.array of shape (num_instances, num_features)): The features array.
        y (np.array of shape (num_instances, ): The array containing labels (Default=None).
    """

    iterator = ArrayStreamer(shuffle=False)

    if y is None:
        for xi in iterator.iter(X):
            yield xi, None
    else:
        for xi, yi in iterator.iter(X, y):
            yield xi, yi

class BaseModel(ABC):
    """Abstract base class for the models.
    """

    @abstractmethod
    def fit_partial(self, X, y=None):
        """Fits the model to next instance.

        Args:
            X (np.float array of shape (num_features,)): The instance to fit.
            y (int): The label of the instance (Optional for unsupervised models, default=None).

        Returns:
            object: Returns the self.
        """
        pass

    @abstractmethod
    def score_partial(self, X):
        """Scores the anomalousness of the next instance.

        Args:
            X (np.float array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        pass

    def fit_score_partial(self, X, y=None):
        """Applies fit_partial and score_partial to the next instance, respectively.

        Args:
            X (np.float array of shape (num_features,)): The instance to fit and score.
            y (int): The label of the instance (Optional for unsupervised models, default=None).

        Returns:
            float: The anomalousness score of the input instance.
        """
        return self.fit_partial(X, y).score_partial(X)

    def fit(self, X, y=None):
        """Fits the model to all instances in order.

        Args:
            X (np.float array of shape (num_instances, num_features)): The instances in order to fit.
            y (int): The labels of the instances in order to fit (Optional for unsupervised models, default=None).

        Returns:
            object: Fitted model.
        """
        for xi, yi in _iterate(X, y):
            self.fit_partial(xi, yi)

        return self

    def score(self, X):
        """Scores all instaces via score_partial iteratively.

        Args:
            X (np.float array of shape (num_instances, num_features)): The instances in order to score.

        Returns:
            np.float array of shape (num_instances,): The anomalousness scores of the instances in order.
        """
        y_pred = np.empty(X.shape[0], dtype=np.float)
        for i, (xi, _) in enumerate(_iterate(X)):
            y_pred[i] = self.score_partial(xi)

        return y_pred

    def fit_score(self, X, y=None):
        """This helper method applies fit_score_partial to all instances in order.

        Args:
            X (np.float array of shape (num_instances, num_features)): The instances in order to fit.
            y (np.int array of shape (num_instances, )): The labels of the instances in order to fit (Optional for unsupervised models, default=None).

        Returns:
            np.float array of shape (num_instances,): The anomalousness scores of the instances in order.
        """
        y_pred = np.zeros(X.shape[0], dtype=np.float)
        for i, (xi, yi) in enumerate(_iterate(X, y)):
            y_pred[i] = self.fit_score_partial(xi, yi)

        return y_pred

class RSHash(BaseModel):
    """Subspace outlier detection in linear time with randomized hashing :cite:`sathe2016subspace`. This implementation is adapted from `cmuxstream-baselines <https://github.com/cmuxstream/cmuxstream-baselines/blob/master/Dynamic/RS_Hash/sparse_stream_RSHash.py>`_.

        Args:
            feature_mins (np.float array of shape (num_features,)): Minimum boundary of the features.
            feature_maxes (np.float array of shape (num_features,)): Maximum boundary of the features.
            sampling_points (int): The number of sampling points (Default=1000).
            decay (float): The decay hyperparameter (Default=0.015).
            num_components (int): The number of ensemble components (Default=100).
            num_hash_fns (int): The number of hashing functions (Default=1).
    """

    def __init__(
            self,
            feature_mins,
            feature_maxes,
            sampling_points=1000,
            decay=0.015,
            num_components=100,
            num_hash_fns=1):
        self.minimum = feature_mins
        self.maximum = feature_maxes

        self.m = num_components
        self.w = num_hash_fns
        self.s = sampling_points
        self.dim = len(self.minimum)
        self.decay = decay
        self.scores = []
        self.num_hash = num_hash_fns
        self.cmsketches = []
        self.effS = max(1000, 1.0 / (1 - np.power(2, -self.decay)))

        self.f = np.random.uniform(
            low=1.0 / np.sqrt(self.effS), high=1 - (1.0 / np.sqrt(self.effS)), size=self.m)

        for i in range(self.num_hash):
            self.cmsketches.append({})

        self._sample_dims()

        self.alpha = self._sample_shifts()

        self.index = 0 + 1 - self.s

        self.last_score = None


    def fit_partial(self, X, y=None):
        """Fits the model to next instance.

        Args:
            X (np.float array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: Returns the self.
        """
        score_instance = 0
        for r in range(self.m):
            Y = -1 * np.ones(len(self.V[r]))
            Y[range(len(self.V[r]))] = np.floor(
                (X[np.array(self.V[r])] + np.array(self.alpha[r])) / float(self.f[r]))

            mod_entry = np.insert(Y, 0, r)
            mod_entry = tuple(mod_entry.astype(np.int))

            c = []
            for w in range(len(self.cmsketches)):
                try:
                    value = self.cmsketches[w][mod_entry]
                except KeyError:
                    value = (self.index, 0)

                # Scoring the Instance
                tstamp = value[0]
                wt = value[1]
                new_wt = wt * np.power(2, -self.decay * (self.index - tstamp))
                c.append(new_wt)

                # Update the instance
                new_tstamp = self.index
                self.cmsketches[w][mod_entry] = (new_tstamp, new_wt + 1)

            min_c = min(c)
            c = np.log(1 + min_c)
            score_instance = score_instance + c

        self.last_score = score_instance / self.m

        self.index += 1

        return self


    def score_partial(self, X):
        """Scores the anomalousness of the next instance. Outputs the last score. Note that this method must be called after the fit_partial

        Args:
            X (any): Ignored.
        Returns:
            float: The anomalousness score of the last fitted instance.
        """
        return self.last_score


    def _sample_shifts(self):
        alpha = []
        for r in range(self.m):
            alpha.append(
                np.random.uniform(
                    low=0,
                    high=self.f[r],
                    size=len(self.V[r])))

        return alpha

    def _sample_dims(self):
        max_term = np.max((2 * np.ones(self.f.size), list(1.0 / self.f)), axis=0)
        common_term = np.log(self.effS) / np.log(max_term)
        low_value = 1 + 0.5 * common_term
        high_value = common_term

        self.r = np.empty([self.m, ], dtype=int)
        self.V = []
        for i in range(self.m):
            if np.floor(low_value[i]) == np.floor(high_value[i]):
                self.r[i] = 1
            else:
                self.r[i] = min(
                    np.random.randint(
                        low=low_value[i],
                        high=high_value[i]),
                    self.dim)
            all_feats = np.array(list(range(self.dim)), dtype=np.int)

            choice_feats = all_feats[np.where(self.minimum != self.maximum)]
            sel_V = np.random.choice(
                choice_feats, size=self.r[i], replace=False)
            self.V.append(sel_V)
