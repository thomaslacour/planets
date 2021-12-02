#!/usr/bin/env python

import pandas as pd
import numpy as np

from planets.preproc.solarsystem import SolarSys
from planets.preproc.nasa import Nasa

def exoplanet_solarsystem():
    """ Plot the mass of a planets vs their radius for the dataset of exoplanet
        concatenate with the dataset of solar system.
    """
    # load nasa dataset for exoplanets
    df_nasa = Nasa().load_clean_data(filt='relevant').reset_index()
    # load solar system dataset
    SolarSys().scrap_data_to_csv()
    df_solar = SolarSys().load_raw_data()
    # TODO ...
