from planets.preproc.nasa import Nasa
import streamlit as st

st.set_page_config(
     page_title="Planet U",
     page_icon="🪐",
     layout="wide",
 )

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from utils import Load, DataViz
import utils

data = Load().masseffect()
dataviz = DataViz
opt = {}
infos=None

# empty = pd.DataFrame([[np.nan] * len(data.columns)], columns=data.columns)
# empty.pl_name = ''
# data = empty.append(data, ignore_index=True)

# selector
# @st.cache
def get_select_box_data(reversed_mode):
    """
    """
    if reversed_mode:
        return pd.DataFrame({ 'pl_type': ['Terrestrial', 'Super Earth', 'Neptune-Like', 'Gas Giant'] })
    return pd.DataFrame({ 'pl_name': data.pl_name, })

def make_features_input(features, dat):
    columns = stc.columns(len(features))
    for i, (key, text) in enumerate(features.items()):
        input_label = features[key][0]
        input_default_value = ''
        if key in dat.columns:
            if str(dat[key][0])!='nan':
                input_default_value = np.round(dat[key][0], 2)
        if features[key][1]==float:
            features[key] += [ columns[i].text_input(input_label, value=input_default_value) ]
        if features[key][1]=='cat':
            min, max = features[key][2]
            features[key] += [ columns[i].number_input(input_label, min_value=min, max_value=max, step=1) ]
        if features[key][1]==int:
            features[key] += [ columns[i].number_input(input_label, step=1, min_value=0) ]
    return features


def print_generated_features(features, dat):
    """
    """
    columns = stc.columns(len(features))
    generated_features = dat
    for i, (key, text) in enumerate(features.items()):
        input_label = features[key][0]
        features[key] += [ generated_features[key].to_numpy()[0] ]
        columns[i].text_input(input_label, value=features[key][-1])

    return features


def get_features_data(generated=False):
    return pd.DataFrame(utils.api_generate())


features = {
        'pl_rade':['🌎 Radius', float],
        'pl_masse':['🌎 Masse', float],
        'pl_eqt':['🌎 Temp. (K)', float],
        'pl_orbper':['🌎 Orbit. Per.', float],
        'st_mass':['☀️  Masse', float],
        'st_rad':['☀️  Radius', float],
        'st_teff':['☀️  Temp. (K)', float],
}

advanced_features = {
        'sy_pnum':['Num. of 🌎 ', 'cat', (0,9)],
        'pl_orbeccen':['🌎 Eccentricity', float],
        'pl_insol':['🌎 Insol. Flux', float],
        'sy_snum':['Num. of ☀️ ', 'cat', (0,3) ],
        'st_logg':['☀️  Surf. Grav.', float], #  [log(cm/s²)]
}


# ==:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::== -

stc = st
st = st.sidebar

reverse_mode = stc.checkbox('Reverse mode', value=False)

df = get_select_box_data(reverse_mode)

if not reverse_mode:
    selected_planet_to_classify = st.selectbox('Select a preloaded planet', df['pl_name'])
    selected_data = data[df.pl_name == selected_planet_to_classify].assign(hack='').set_index('hack')
    features = make_features_input(features, selected_data)
else:
    selected_class_to_generate = st.selectbox('Select a target planet type', df['pl_type'])
    selected_data = get_features_data()
    features = print_generated_features(features, selected_data)

if st.checkbox("Advanced Features"):
    advanced_features = make_features_input(advanced_features, selected_data)
    features.update(advanced_features)

X = selected_data.copy()
for feat, val in features.items():
    if val[-1]=='':
        val[-1] = np.nan
    X[feat] = float(val[-1])
X.rename(columns={"pl_masse": "pl_bmasse"}, inplace=True)

if not reverse_mode:
    if stc.button('Find my planet !'):
        opt['neighbors'], infos = utils.api_predict(X)
else:
    selected_data = get_features_data(generated = stc.button('Generate a planet !'))
    # if stc.button('Generate a planet !'):
        # selected_data = pd.DataFrame(utils.api_generate('gas giant'))

# reset = st.button('Clear Cache')

st.markdown("""
        #
        #
""")

# if stc.button('Find'):
#     opt['neighbors'] = utils.api_predict(X)

opt['Livable'] = st.checkbox('Only Potentialy Livable Planets')

opt['type'] = st.selectbox(
     "Filter by a specific planet type",
     ('All', 'Terrestrial', 'Super Earth', 'Neptune-Like', 'Gas Giant')
)

pl_name = DataViz.raw.copy().pl_name

if opt['Livable']:
    mask_livable = ( DataViz.raw_obj.pl_orb_is_in_CHZ == True ).to_numpy()
    pl_name = pl_name[mask_livable]

if 'type' in opt.keys():
    if opt['type'].lower() != 'all':
        mask_type = DataViz.raw.pl_type == opt['type'].lower()
        pl_name = pl_name[mask_type]


filters = st.multiselect(
     label   = 'Filter by planets name',
     options = pl_name,
     # help    = 'Select planets you want to display specificaly.',
)


st.markdown(""" # """)

opt['use_pca'] = st.checkbox('Use PCA representation', value=True)
axes = {}

_dict = Nasa().col_dict
_feat = DataViz.cols_to_keep

axes_choices = { val:key for key, val in _dict.items() if key in _feat }

if not opt['use_pca']:
    axes['x'] = axes_choices.keys()
    axes['y'] = axes_choices.keys()

    columns = st.columns(2)

    opt['x'] = st.selectbox('x', axes.get('x', ('PC1',)), index=4)
    opt['y'] = st.selectbox('y', axes.get('y', ('PC2',)), index=2)

    opt['x'] = axes_choices[opt['x']]
    opt['y'] = axes_choices[opt['y']]

import streamlit as st

# fig = plt.figure()
# xxx = DataViz.raw
# fig.add_subplot(111).scatter(xxx['pl_masse'], xxx['pl_rade'])
# st.pyplot(fig)

columns = st.columns(4)
if infos:
    for i, (type, pred) in enumerate(infos[1].items()):
        columns[i].metric(label=type, value=f"{pred*100:.2f}%")

fig = dataviz.make_figure(*filters, **opt)
columns = st.columns([1,4,1])
columns[1].plotly_chart(fig, use_container_width=True)
