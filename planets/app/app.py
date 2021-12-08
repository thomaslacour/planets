import streamlit as st

st.set_page_config(
     page_title="Ex-stream-ly Cool App",
     page_icon="🪐",
     layout="centered",
 )

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from utils import Load, DataViz
import utils

data = Load().masseffect()
dataviz = DataViz
opt = {}

empty = pd.DataFrame([[np.nan] * len(data.columns)], columns=data.columns)
empty.pl_name = '--'
data = empty.append(data, ignore_index=True)

# selector
@st.cache
def get_select_box_data():
    """
    """
    return pd.DataFrame({
          'pl_name': data.pl_name,
        })

df = get_select_box_data()

stc = st
st = st.sidebar

selected_planet_to_classify = st.selectbox('Select a preloaded planet', df['pl_name'])
reset = st.button('Clear Cache')

st.markdown("""
        #
        #
""")

selected_data = data[df.pl_name == selected_planet_to_classify].assign(hack='').set_index('hack')

# st.write(selected_data.drop(columns='pl_name'))


features = {
        'pl_rade':['🌎 Radius'],
        'pl_masse':['🌎 Masse'],
        'pl_eqt':['🌎 Temp. (K)'],
        'pl_orbper':['🌎 Orbit. Per.'],
        'st_mass':['☀️  Masse'],
        'st_rad':['☀️  Radius'],
}

columns = stc.columns(len(features))

for i, (key, text) in enumerate(features.items()):
    input_label = features[key][0]
    input_default_value = ''
    if key in selected_data.columns:
        if str(selected_data[key][0])!='nan':
            input_default_value = np.round(selected_data[key][0], 2)
    features[key] += [ columns[i].text_input(input_label, value=input_default_value) ]


if stc.button('Find'):
    opt['neighbors'] = utils.predict()

opt['Livable'] = st.checkbox('Only Potentialy Livable Planets')

opt['type'] = st.radio(
     "Filter by a specific planet type",
     ('All', 'Super Earth', 'Neptune-Like', 'Gas Giant', 'Unknown')
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


import streamlit as st

fig = dataviz.make_figure(*filters, **opt)
st.plotly_chart(fig)
