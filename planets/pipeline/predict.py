import joblib
import pandas as pd
import numpy as np
import json
import os
from planets.pipeline import predict
# from google.cloud import storage


ROOT = '../planets/pipeline/'

PATH_TO_LOCAL_MODEL = ROOT + 'model.joblib'
AWS_BUCKET_TEST_PATH = ''
GCOULD_BUCKET_NAME=''


def _get_model_from_local(path_to_joblib=PATH_TO_LOCAL_MODEL):

    pipeline = joblib.load(path_to_joblib)

    return pipeline


def custom_predict(input_, n_neighs_shown=0, radius=0):

    """
    Returns not only prediction, but also neighbors et various infos in a
    FastAPI compatible format.
    """

    # !!! types casting !!! -- FastAPI passes str and not int
    n_neighs_shown = int(n_neighs_shown)
    radius = float(radius)

    dict_prediction = {}

    # load model from local
    model = _get_model_from_local()

    # temp loading index
    X_train = joblib.load(ROOT + 'X_train.joblib')
    X_index = joblib.load(ROOT + 'X_index.joblib')
    dist_norm_fact = joblib.load(ROOT + 'dist_norm_fact.joblib')

    # stock prediction and predictions probabilities
    dict_prediction['prediction'] = model.predict(input_)[0]
    dict_prediction['probabilities'] = \
        {class_:round(proba,3) for class_, proba in zip(model.classes_, model.predict_proba(input_)[0])}

    # transform imput to be able to get kneighbors
    transf_input = model.steps[1][1].transform(model.steps[0][1].transform(input_))

    # get kneighbors either k used by model or user defined k (if > k-model)
    model_n_neighs = model.get_params().get('kneighborsclassifier__n_neighbors')
    n_neighs_shown = n_neighs_shown
    if n_neighs_shown > model_n_neighs:
        model_n_neighs = n_neighs_shown
    neighbors = model.named_steps.get('kneighborsclassifier').kneighbors(
        transf_input, n_neighbors=model_n_neighs, return_distance=True)

    # get infos about kneighbors
    i = 0
    for neigh, dist in zip(neighbors[1][0].tolist(), neighbors[0][0].tolist()):

        norm_dist = dist/dist_norm_fact

        # filter on radius -- skip neighbor if dist > radius (defined by user)
        if radius !=0 and norm_dist > radius:
            continue

        temp_dict = {}
        true_idx = X_train.index[neigh]

        temp_dict['pl_name'] = X_index.iloc[true_idx].pl_name
        temp_dict['pl_type'] = X_index.iloc[true_idx].pl_type
        temp_dict['is_in_CHZ'] = str(X_index.iloc[true_idx].pl_orb_is_in_CHZ)
        dict_vect = X_train.iloc[neigh].to_dict()
        temp_vect = {}
        for key, val in dict_vect.items():
            if str(val) != 'nan':
                temp_vect[key] = val
            else:
                temp_vect[key] = None
        temp_dict['vector'] = temp_vect
        temp_dict['distance'] = dist
        temp_dict['norm_distance'] = norm_dist

        if i == 0 and norm_dist > 0.8:
            dict_prediction['pred_reliability'] = 'low'
        elif i == 0 and norm_dist > 0.2:
            dict_prediction['pred_reliability'] = 'avg'
        elif i ==0 and norm_dist < 0.2:
            dict_prediction['pred_reliability'] = 'high'

        dict_prediction[f'neigh_{i}'] = temp_dict
        i += 1

    return dict_prediction


def generate_random_planet(pl_type, sy_snum, sy_pnum, pl_orbper, pl_rade, pl_bmasse,
    pl_orbeccen, pl_insol, pl_eqt, st_teff, st_rad, st_mass,
    st_logg, reliability='avg', max_iter=1000, req_files_path=''):

    """
    Generates a random planet with given target, using knn trained model as
    validator. Returns -1 if not planet can be found.
    """

    if pl_type not in ['gas giant', 'neptune-like', 'terrestrial', 'super earth']:
        return -1

    # load model from local
    model = _get_model_from_local()

    # gets intervals from inputed X_train
    ipt_X_train = joblib.load(f'{req_files_path}ipt_X_train.joblib')

    # gets intervals from inputed X_train
    pl_type_frame = {}
    for pl_type_ in ipt_X_train.pl_type.unique():
        if pl_type_ != pl_type:
            continue

        col_intervals = {}
        for col in ipt_X_train.columns:
            if col == 'pl_type':
                continue
            filt = ipt_X_train.pl_type == pl_type_
            col_intervals[col] = [ipt_X_train[filt][col].min(), ipt_X_train[filt][col].max()]

        # returns a dict of dicts with pl_type as key, then features as key
        pl_type_frame[pl_type] = col_intervals

    # random iteration until coherent classification
    i = 0
    while True:
        rand_pl_specs = {}
        for key in pl_type_frame.get(pl_type).keys():

            spec_frame = pl_type_frame.get(pl_type)

            # if any feature has been given by user, keep it
            if locals().get(key) is not None:
                rand_pl_specs[key] = float(locals().get(key))

            # else randomly generate the other features

            # ints specific random generator
            elif key in ['sy_snum', 'sy_pnum']:
                rand_pl_specs[key] = np.random.randint(spec_frame.get(key)[0], spec_frame.get(key)[1])

            # floats specific random generator
            else:
                if spec_frame.get(key)[0] == 0 and spec_frame.get(key)[1] < 1:
                    rand_pl_specs[key] = np.random.random()
                elif spec_frame.get(key)[0] == 0:
                    rand_pl_specs[key] = np.random.randint(1, spec_frame.get(key)[1]) + np.random.random()
                else:
                    rand_pl_specs[key] = np.random.randint(spec_frame.get(key)[0], spec_frame.get(key)[1]) + np.random.random()

        # test random planet classification
        temp_prediction = custom_predict(pd.DataFrame([rand_pl_specs]))
        if temp_prediction.get('prediction') == pl_type \
            and temp_prediction.get('pred_reliability') == reliability:
            return rand_pl_specs

        # leave loop if maxi_iter is reached
        if i == max_iter:
            return -1
        else:
            i += 1

    return rand_pl_specs



if __name__ == '__main__':
    pass
