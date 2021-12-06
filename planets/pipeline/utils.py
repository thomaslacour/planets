from sklearn.metrics import pairwise_distances
from scipy.spatial import distance
import numpy as np
import pandas as pd
import scipy


def cust_dist_mix_int_float(
    X, Y, int_pos, mink_pow=2, int_dist_prop=0.1, int_dist='manh'):

    """
    Combine p-minkowski distance for continuuous real values
    with integer designed distance for integer values.
    Params are:
        - X, Y
        - int_pos, integer positions
        - int_dist_prop, integer proportion in combined distance computation
        - int_dist, integer distance type
        - mink_pow (1 for L1, 2 for L2, etc.)
    """

    # input checks
    if not isinstance(X, np.ndarray) or \
        not isinstance(Y, np.ndarray):
        raise TypeError

    if X.shape != Y.shape:
        raise ValueError

    for pos in int_pos:
        if pos not in range(len(X)):
            raise ValueError

    if mink_pow < 1:
        raise ValueError

    if int_dist_prop > 1 or int_dist_prop < 0:
        raise ValueError

    if int_dist not in ['manh', 'hamm', 'canber', 'braycurt']:
        raise ValueError

    # --------------------------------------------------------------------------
    X_int = []
    X_real = []
    Y_int = []
    Y_real = []
    for i in range(len(X)):
        if i in int_pos:
            if X[i] != int(X[i]) or Y[i] != int(Y[i]):
                raise TypeError('Probable error while passing \'int_pos\'')
            X_int.append(X[i])
            Y_int.append(Y[i])
        else:
            X_real.append(X[i])
            Y_real.append(Y[i])

    X_int = np.asarray(X_int)
    X_real = np.asarray(X_real)
    Y_int = np.asarray(Y_int)
    Y_real = np.asarray(Y_real)

    # computing integer relative part of distance
    if int_dist == 'manh':
        int_dist_ = np.sum(np.abs(X_int - Y_int))
    elif int_dist == 'hamm':
        int_dist_ = np.sum(X_int == Y_int)
    elif int_dist == 'canber':
        int_dist_ = np.sum(np.abs(X_int - Y_int) / np.abs(X_int) + np.abs(Y_int))
    elif int_dist == 'braycurt':
        int_dist_ = np.sum(np.abs(X_int - Y_int)) / (np.sum(np.abs(X_int)) + np.sum(np.abs(Y_int)))

    # computing minkowski part of distance
    mink_dist = np.sum(((X_real - Y_real)**mink_pow)**(1/mink_pow))

    # combining both distance with given proportions
    cust_dist = int_dist_prop * int_dist_ + (1 - int_dist_prop) * mink_dist

    return cust_dist


def checks_visually_dist_matrix(X, dist_func, **kwargs):

    """
    Displays a distance matrix for visual check.
    """

    df = scipy.spatial.distance.cdist(X, X, metric=dist_func,**kwargs)
    pd.DataFrame(df).style.applymap(lambda x:'color:red' if x==0 else 'color: black')

    return df


def generate_random_mix_int_float_vector(int_min, int_max, n_int, n_float):

    """
    Generate a vector with n integer and n float. Integer are first in order.
    Integers and floats are stack, and numpy cast them all into float.
    """

    X_int = np.random.randint(int_min, int_max, n_int)
    X_float = np.random.random(n_float)

    X = np.hstack((X_int, X_float))

    return X


def is_distance_valid(dist_func, X, **kwargs):

    """
    Checks if a custom distance matches distance properties, and therefore is valid.
    """

    # compute distance matrix
    dist_mtx = scipy.spatial.distance.cdist(X, X, metric=dist_func,**kwargs)

    # non-negativity test
    if not np.isnan(np.sum(np.where(dist_mtx < 0, True, np.nan))):
        return(False, 'Custom distance returns negative value !')

    # identity test
    if not np.sum(dist_mtx.diagonal()) == 0:
        return(False, 'Custom distance does not respect identity property !')

    # symmetry test
    if not (dist_mtx.transpose() - dist_mtx).sum().sum() == 0:
        return(False, 'Custom distance does not respect symmetry property !')

    # # check triangle inequality
    # TODO test triangle inequality with matrixes
    # for sample in samples:
    #     if dist_func (X, Y) + dist_func (Y, Z) >= dist_func (X, Z):
    #         return(False, 'Custom distance does not respect triangle inequality property !')

    return (True,)
