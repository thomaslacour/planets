import streamlit as st

import requests

from planets.preproc.masseffect import MassEffect
from planets.preproc.solarsystem import SolarSys
from planets.preproc.nasa import Nasa
from planets.pipeline.utils import ColumnTransformerWithNames

import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, FunctionTransformer
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.decomposition import PCA

import plotly
import plotly.express as px
import plotly.graph_objs as go

col_dict = Nasa().col_dict


class Load:
    """
    Loader for scrapped data.
    """

    def masseffect(self):
        ""
        ""
        masseffect = MassEffect()
        file = '../../raw_data/masseffect.csv'
        masseffect.load_raw_data('../../raw_data/masseffect.csv')
        data = masseffect.cleaned_data.dropna(subset=['pl_name'])
        data.pl_name = data[ ['pl_name'] ].apply(lambda x: x + ' [MassEffect]')

        return data

def make_pca(X, n=3):
    """
    """
    pca = PCA(n_components=n, svd_solver='auto')
    model = pca.fit(X)
    pca_cols = [f'PC{i}' for i in range(1, len(model.components_)+1)]
    X_proj = pd.DataFrame(pca.transform(X), columns=pca_cols)

    return X_proj, model



class DataViz:
    """
    Class to load and visualize dataset from Nasa
    """

    # raw data
    raw_exo = pd.read_csv('./../data/nasa_clean.csv')
    raw_sol = SolarSys().load_raw_data('./../../raw_data/solarsystem_wiki.csv').cleaned_data
    # concatenation of exoplanet dataset and solar system dataset
    raw = pd.concat(
        [raw_sol, raw_exo.rename(columns={'pl_bmasse': 'pl_masse'})],
        axis=0,
        join='outer'
    ).drop_duplicates().reset_index(drop=True)

    # tag for potentially livable planets
    is_livable=raw.is_pot_livable

    raw.pl_type = raw.pl_type.apply(
            lambda x: x
                if x in ['super earth', 'neptune-like', 'gas giant', 'terrestrial']
                else 'unknown'
            )

    # tag for planet names
    pl_name = raw.pl_name

    # set the Earth to be livable !
    raw.loc[2,'pl_orb_is_in_CHZ'] = True

    # features engineering
    # ·· we take the logarithm for the column whose values are on multiple
    #    order of magnitue
    cols_to_log=[
        'pl_masse', 'pl_rade', 'st_rad', 'st_mass', 'pl_orbper', 'pl_orbsmax',
        'pl_radj', 'pl_bmassj', 'pl_dens', 'pl_insol', 'pl_trandep', 'pl_trandur',
        'pl_ratror', 'st_dens', 'sy_dist', 'st_lum_y', 'OHZIN', 'CHZIN', 'CHZOUT', 'OHZOUT'
    ]
    raw[cols_to_log] = raw[cols_to_log].apply(np.log)

    # features selection
    cols_to_keep = cols_to_log
    cols_to_keep += ['pl_eqt','sy_snum', 'sy_pnum', 'sy_mnum', 'st_age', 'st_logg', 'pl_tranmid' ]

    # masking for numerical data
    mask = ((raw.dtypes != 'object') & (raw.dtypes != 'bool')).to_numpy()

    # seperation between num and obj
    raw_obj = raw.iloc[:,~mask]
    raw_num = raw[cols_to_keep]

    # preprocessing
    Imputer    = [SimpleImputer(), KNNImputer()][0]
    DataFramer = FunctionTransformer(lambda arr: pd.DataFrame(arr))

    imputer = make_column_transformer(
        (Imputer, make_column_selector(dtype_exclude=['object','bool'])),
        remainder='passthrough'
    )
    scaler = make_column_transformer(
        (RobustScaler(), make_column_selector(dtype_include=['float64'])),
        (MinMaxScaler(), make_column_selector(dtype_include=['int64'])),
        remainder='drop'
    )
    pipeline_preproc = make_pipeline( Imputer, DataFramer , scaler)

    # pipeline_preproc = ColumnTransformerWithNames([
    #     ('a', Imputer, make_column_selector(dtype_exclude=['object','bool'])),
    #     ('b', DataFramer, make_column_selector(dtype_exclude=['bool'])),
    #     ('c', RobustScaler(), make_column_selector(dtype_include=['float64'])),
    #     ('d', MinMaxScaler(), make_column_selector(dtype_include=['int64'])),
    # ], remainder='drop')
    # st.write(nasa.pipeline_preproc.get_feature_names())

    X = pd.DataFrame(pipeline_preproc.fit_transform(raw_num))


    def make_figure(*args, **kwargs):
        """
        """
        X = DataViz.X

        # projected data on the principal components
        X_proj, _ = make_pca(X)

        if len(args)!=0:
            mask_pl_name = DataViz.pl_name.isin(args)
        else:
            mask_pl_name = True

        # filters
        if kwargs['Livable']:
            # all planets tagged livable and Unknown
            liv_filt = ~( DataViz.raw_obj.pl_orb_is_in_CHZ == False ).to_numpy()

            # only tagged as livable
            liv_filt = ( DataViz.raw_obj.pl_orb_is_in_CHZ == True ).to_numpy()
        else:
            liv_filt = True
        # nliv = (data.df_obj.pl_orb_is_in_CHZ==False).to_numpy()
        # uliv = (data.df_obj.pl_orb_is_in_CHZ.isna()).to_numpy()

        pl_type_filt = lambda x: ( DataViz.raw.pl_type == x if x!='all' else DataViz.raw.pl_type.apply(lambda x: True) )

        px_style = { 'x':'x', 'y':'y', 'hover_name':'pl_name', 'hover_data':{'x':False, 'y':False} }
        planets_filt = lambda X, t, y: pd.DataFrame({
                'pl_name' : DataViz.pl_name,
                'x' : X.iloc[:,1][ y & pl_type_filt(t) ],
                'y' : X.iloc[:,2][ y & pl_type_filt(t) ],
                })

        # intialize scatter plot
        fig = px.scatter(
                planets_filt(X_proj, '', True), **px_style
        )

        # plot all planets in grey
        # ------------------------------------------------------------------
        if len(args)!=0 or kwargs['Livable'] or 'type' in kwargs.keys():
            all_in_grey = px.scatter(
                    planets_filt(X_proj, 'all', True),
                    **px_style,
                    color_discrete_sequence = [ 'grey' ],
                    opacity = 0.05,
            ).data[0]
            fig.add_trace(all_in_grey)
            fig.update_traces(
                    marker        = dict(color='grey'),
                    selector      = dict(mode='markers'),
                    hovertemplate = None,
                    hoverinfo     = 'skip'
            )
        # ------------------------------------------------------------------

        if 'neighbors' in kwargs.keys():
            mask_neighbors = DataViz.pl_name.isin(kwargs['neighbors'])
        else:
            mask_neighbors = np.zeros(X_proj.shape, dtype=bool)[:,0]

        type_filt = 'all'
        if 'type' in kwargs.keys():
            type_filt = kwargs['type'].lower()

        # plot all planets filtered
        # ------------------------------------------------------------------
        not_neighbors = px.scatter(
                planets_filt(X_proj[~mask_neighbors & mask_pl_name], type_filt, liv_filt),
                **px_style,
                color_discrete_sequence = [ 'rgb(31, 219, 180)' ],
                opacity = 0.8,
        ).data[0]
        # plot neighbors
        # ------------------------------------------------------------------
        neighbors = px.scatter(
                planets_filt(X_proj[mask_neighbors], 'all', liv_filt),
                **px_style,
                color_discrete_sequence = [ 'yellow' ],
        ).data[0]

        fig.add_trace(not_neighbors)
        fig.add_trace(neighbors)

        fig.update_xaxes(title='', visible=False, showticklabels=False)
        fig.update_yaxes(title='', visible=False, showticklabels=False)

        return fig

# pipeline.py
# X_train et model dans un dossier
# api fast api
# uvicorn fast_api:app 127.0.0.800

# TODO
URL_API = 'http://127.0.0.1:8000'
# /predict?sy_snum=2&sy_pnum=2&pl_orbper=2&pl_rade=2&pl_bmasse=2&pl_orbeccen=2&pl_insol=2&pl_eqt=2&st_teff=2&st_rad=2&st_mass=2&st_logg=2&n_neighs_shown=0&radius=0
def api_predict(X):
    """
    Get prediction from the api
    """

    features = ['sy_snum','sy_pnum','pl_orbper','pl_rade','pl_bmasse','pl_orbeccen','pl_insol','pl_eqt','st_teff','st_rad','st_mass','st_logg']
    hyperparams = ['n_neighs_shown','radius']

    url = URL_API + '/predict?'

    for feat in features:
        val = np.nan
        if feat in X.columns:
            val = round(X[feat].to_numpy()[0], 3)
        url += f"{feat}={val}&"
    url = url[:-1]

    resp = requests.get(url).json()

    prediction    = resp.pop('prediction', '?').capitalize()
    probabilities = resp.pop('probabilities', [0, 0, 0, 0])
    reliability   = resp.pop('pred_reliability', 0)
    neighbors     = [ neighbor['pl_name'] for neighbor in resp.values() ]

    return neighbors