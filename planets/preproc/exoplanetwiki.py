#!/usr/bin/env python

import pandas as pd
import numpy as np
import re
# from planets.scrapper.fandom import Fandom
from planets.scrapper.utils import TableScrapper


RAW_DATA = '../raw_data/exoplanet_wiki.csv'


class ExoPlanetWiki:

    def __init__(self):
        self.raw_data=None
        self._cleaned_data=None

    def scrap_data_to_csv(self, filepath=RAW_DATA):
        """Scrap data from wiki
        """
        url='https://en.wikipedia.org/wiki/List_of_potentially_habitable_exoplanets'
        exoplanet_wiki = TableScrapper(url=url)
        data = exoplanet_wiki.scrap()[0]
        data.to_csv(filepath)


    def load_raw_data(self, filepath=RAW_DATA) -> pd.DataFrame:
        """
        """
        self.raw_data = pd.read_csv(filepath)
        return self


    def _clean_mass(self, text):
        """Clean mass column. Take the floor value.
        """
        if type(text) is float:
            return
        x = [ float(i) for i in re.findall('[-+]?([0-9]*\.[0-9]+|[0-9]+)', text) ]
        if x:
            return np.mean(x)


    @property
    def cleaned_data(self):
        """Cleaned data
        """

        df=self.raw_data
        df_clean=pd.DataFrame([])

        df_clean['pl_name'] = df.Object
        df_clean['st_name'] = df.Star
        df_clean['st_spectype'] = df['Star type']
        df_clean['pl_masse'] = df['Mass (M⊕)'].apply(self._clean_mass)
        df_clean['pl_rade'] = df['Radius (R⊕)'].apply(self._clean_mass)
        df_clean['pl_dens'] = df['Density (g/cm3)'].apply(self._clean_mass)
        df_clean['pl_insol'] = df['Flux (F⊕)'].apply(self._clean_mass)
        df_clean['pl_eqt'] = df['Teq (K)'].apply(self._clean_mass)
        df_clean['pl_orbper'] = df['Period (days)']

        # conversion from light year to pc
        df_clean['sy_dist'] = df['Distance (ly)']*0.3066

        df_clean['ref'] = df['Refs/Notes']

        columns_to_drop = []

        return df_clean


if __name__ == '__main__':
    pass

