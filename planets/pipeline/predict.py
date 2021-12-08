import joblib
import pandas as pd
import numpy as np
import json
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
        print(radius, norm_dist)
        print(type(radius), type(norm_dist))
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

        if i == 0 and norm_dist > 1:
            dict_prediction['pred_reliability'] = 'low'
        elif i == 0 and norm_dist > 0.5 and norm_dist < 1:
            dict_prediction['pred_reliability'] = 'avg'
        else:
            dict_prediction['pred_reliability'] = 'high'

        dict_prediction[f'neigh_{i}'] = temp_dict
        i += 1

    return dict_prediction



if __name__ == '__main__':
    pass
