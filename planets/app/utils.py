import os

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
        masseffect.load_raw_data('../../raw_data/masseffect.csv')
        masseffect = masseffect.cleaned_data.dropna(subset=['pl_name'])
        masseffect.pl_name = masseffect[ ['pl_name'] ].apply(lambda x: x + ' [MassEffect]')

        starwars = pd.read_csv('../../raw_data/kaggle_sw_planets_formatted.csv').iloc[:,1:]
        starwars.pl_name = starwars[ ['pl_name'] ].apply(lambda x: x + ' [StarWars]')

        data = pd.concat( (masseffect, starwars) ).reset_index()

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
        'pl_masse', 'pl_rade', 'st_rad', 'st_mass',
        'pl_orbper', 'pl_orbsmax',
        'pl_radj', 'pl_bmassj', 'pl_dens', 'pl_insol', 'pl_trandep', 'pl_trandur',
        'pl_ratror', 'st_dens', 'sy_dist', 'st_lum_y', 'OHZIN', 'CHZIN', 'CHZOUT', 'OHZOUT'
    ]
    raw[cols_to_log] = raw[cols_to_log].apply(np.log)

    # features selection
    cols_to_keep = cols_to_log
    # cols_to_keep += ['pl_eqt','sy_snum', 'sy_pnum', 'sy_mnum', 'st_age', 'st_logg', 'pl_tranmid' ]

    # masking for numerical data
    mask = ((raw.dtypes != 'object') & (raw.dtypes != 'bool')).to_numpy()

    # seperation between num and obj
    raw_obj = raw.iloc[:,~mask]
    raw_num = raw[cols_to_keep]

    # preprocessing
    Imputer    = [SimpleImputer(strategy='median'), KNNImputer()][1]
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

    nasa_filename = 'nasa_data_imputed_scaled.csv'

    if not os.path.isfile(nasa_filename) or True:
        X = pd.DataFrame(pipeline_preproc.fit_transform(raw_num))
        X.to_csv(nasa_filename)

    X = pd.read_csv(nasa_filename).iloc[:,1:]

    def make_figure(*args, **kwargs):
        """
        """
        X = DataViz.X

        X_proj = X.copy()
        X_proj.columns=DataViz.cols_to_keep

        # projected data on the principal components
        if kwargs['use_pca']:
            X_proj, _ = make_pca(X, n=2)
            u, v = 0, 1
            xlabel = 'First principal component'
            ylabel = 'Second principal component'
        else:
            u = DataViz.cols_to_keep.index(kwargs['x'])
            v = DataViz.cols_to_keep.index(kwargs['y'])
            xlabel = col_dict[kwargs['x']]
            ylabel = col_dict[kwargs['y']]

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
                'x' : X.iloc[:,u][ y & pl_type_filt(t) ],
                'y' : X.iloc[:,v][ y & pl_type_filt(t) ],
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

        fig.update_xaxes(title=xlabel, visible=True, showticklabels=False, showgrid=False, zeroline = False)
        fig.update_yaxes(title=ylabel, visible=True, showticklabels=False, showgrid=False, zeroline = False)

        fig.update_layout(title="Exoplanets collection from NASA 🟢 and your nearest candidate(s) 🟡", title_x=0.5)

        return fig

# pipeline.py
# X_train et model dans un dossier
# api fast api
# uvicorn fast_api:app 127.0.0.800

# URL_API = 'http://127.0.0.1:8000'
# /predict?sy_snum=2&sy_pnum=2&pl_orbper=2&pl_rade=2&pl_bmasse=2&pl_orbeccen=2&pl_insol=2&pl_eqt=2&st_teff=2&st_rad=2&st_mass=2&st_logg=2&n_neighs_shown=0&radius=0
URL_API = 'https://planetsuapi.herokuapp.com'

def api_predict(X, **kwargs):
    """
    Get prediction from the api
    """

    features = ['sy_snum','sy_pnum','pl_orbper','pl_rade','pl_bmasse','pl_orbeccen','pl_insol','pl_eqt','st_teff','st_rad','st_mass','st_logg']
    hyperparams = ['n_neighs_shown','radius']

    url = URL_API + '/predict?'

    for feat in features:
        val = np.nan
        if feat in X.columns:
            val = X[feat].to_numpy()[0]
            if val:
                val = round(val, 3)
            else:
                val = np.nan
        url += f"{feat}={val}&"
    if 'n_neighs_shown' in kwargs.keys():
        url += f"n_neighs_show={kwargs['n_neighs_shown']}&"
    if 'radius' in kwargs.keys():
        url += f"radius={kwargs['radius']}&"
    url = url[:-1]

    # st.write(url)

    resp = requests.get(url)
    # st.write(resp)

    resp = resp.json()
    # st.write(resp)

    prediction    = resp.pop('prediction', '?').capitalize()
    probabilities = resp.pop('probabilities', [0, 0, 0, 0])
    reliability   = resp.pop('pred_reliability', 0)
    neighbors     = [ neighbor['pl_name'] for neighbor in resp.values() ]

    return neighbors, (prediction, probabilities, reliability)



def api_generate(*args, **kwargs):

    # st.write(kwargs)

    if not 'pl_type' in kwargs.keys():
        return {
            'pl_rade':     [''],
            'pl_masse':    [''],
            'pl_eqt':      [''],
            'pl_orbper':   [''],
            'st_mass':     [''],
            'st_rad':      [''],
            'st_teff':     [''],
            'sy_pnum':     [''],
            'pl_orbeccen': [''],
            'pl_insol':    [''],
            'sy_snum':     [''],
            'st_logg':     [''], #  [log(cm/s²)]
        }

    URL_GENERATE_API = 'https://planetsuapi.herokuapp.com'
    URL_GENERATE_API += '/generate?'
    # URL_GENERATE_API += f'pl_type=gas%20giant&reliability=avg&max_iter=1000&sy_snum=null&sy_pnum=null&pl_orbper=null&pl_rade=null&pl_bmasse=null&pl_orbeccen=null&pl_insol=null&pl_eqt=null&st_teff=null&st_rad=null&st_mass=null&st_logg=null'

    URL_GENERATE_API += f'pl_type={kwargs["pl_type"]}'
    URL_GENERATE_API += f'&reliability=avg'
    URL_GENERATE_API += f'&max_iter=1000'
    URL_GENERATE_API += f'&sy_snum={kwargs["sy_snum"][0]}'
    URL_GENERATE_API += f'&sy_pnum={kwargs["sy_pnum"][0]}'
    URL_GENERATE_API += f'&pl_orbper={kwargs["pl_orbper"][0]}'
    URL_GENERATE_API += f'&pl_rade={kwargs["pl_rade"][0]}'
    URL_GENERATE_API += f'&pl_bmasse={kwargs["pl_masse"][0]}'
    URL_GENERATE_API += f'&pl_orbeccen={kwargs["pl_orbeccen"][0]}'
    URL_GENERATE_API += f'&pl_insol={kwargs["pl_insol"][0]}'
    URL_GENERATE_API += f'&pl_eqt={kwargs["pl_eqt"][0]}'
    URL_GENERATE_API += f'&st_teff={kwargs["st_teff"][0]}'
    URL_GENERATE_API += f'&st_rad={kwargs["st_rad"][0]}'
    URL_GENERATE_API += f'&st_mass={kwargs["st_mass"][0]}'
    URL_GENERATE_API += f'&st_logg={kwargs["st_logg"][0]}'

    resp = requests.get(URL_GENERATE_API)
    resp = resp.json()

    return resp
