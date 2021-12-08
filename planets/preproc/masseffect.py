#!/usr/bin/env python

import pandas as pd
import numpy as np
import re
from planets.scrapper.fandom import Fandom


RAW_DATA = './../../raw_data/masseffect.csv'


class MassEffect:

    def __init__(self):
        self.raw_data=None
        self._cleaned_data=None

    def scrap_data_to_csv(self, filepath=RAW_DATA):
        """Scrap data from fandom
        """
        masseffect_fandom = Fandom(page="masseffect")
        data = masseffect_fandom.scrap()
        data.to_csv(filepath)


    def load_raw_data(self, filepath=RAW_DATA) -> pd.DataFrame:
        """
        """
        self.data = pd.read_csv(filepath)
        return self


    def _clean_orbital_distance(self, text):
        """
        """
        if type(text) is float:
            return

        return float(text.replace('AU', ''))


    def _clean_orbital_period(self, text):
        """Orbital period in days (one year = 365.25 days)
        """
        if type(text) is float:
            return
        # handle the thousand seperator
        text = text.replace(',', '').lower()
        conversion_dict = {
            'earth years':1,
            'earth year':1,
            'years':1,
            'earth days':1/365,
            'earth hours':1/365/24
        }
        for substr, factor in conversion_dict.items():
            if text.endswith(substr):
                text = str(factor*float(text.replace(substr, '')))

        return 365.25*float(text)


    def _clear_radius(self, text):
        """Planet radius in earth radius (6371 km)
        """
        if type(text) is float:
            return
        # handle thousand seperator
        text = text.replace(',', '').lower()
        # remove 'km'
        return int(text[:-2])/6371


    def _clean_day_length(self, text):
        if type(text) is float:
            return
        text = text.replace(',', '').lower()
        conversion_dict = {
            'earth hours':1,
            'earth years':24*365.25,
            'earth days':24,
            'earth years (tidal lock)':np.nan
        }
        for substr, factor in conversion_dict.items():
            if text.endswith(substr):
                text = str(factor*float(text.replace(substr, '')))
        return float(text)

        return text


    def _clean_atm_press(self, text):
        """Atmospheric pressure in earth atm.
        """
        if type(text) is float:
            return
        text=text.lower()
        if text.endswith('trace'):
            return None
        text=text.replace('atm', '')

        return float(text)


    def _clean_surf_temp(self, text):
        """Surface temperature in Kelvin.
        """
        if type(text) is float:
            return
        x = [ int(i) for i in re.findall('[-+]?[0-9]+', text) ]

        return np.mean(x) + 273


    def _clean_gravity(self, text):
        if type(text) is float:
            return
        x = [ float(i) for i in re.findall('[-+]?([0-9]*\.[0-9]+|[0-9]+)', text) ]

        return np.mean(x)


    def _clean_mass(self, text):
        """
        """
        if type(text) is float:
            return
        x = [ float(i) for i in re.findall('[-+]?([0-9]*\.[0-9]+|[0-9]+)', text) ]

        return np.mean(x)


    def clean_satellites(self, text):
        # TODO
        return text


    @property
    def cleaned_data(self):
        """Cleaned data
        """

        df=self.data
        df_clean=pd.DataFrame([])

        columns_to_drop = []

        df_clean['pl_name'] = df['pl_name']
        # keplerian ratio
        df_clean['Keplerian Ratio'] = df['Keplerian Ratio']
        # orbital distance
        df_clean['Orbital Distance [AU]'] = df['Orbital Distance'].apply(self._clean_orbital_distance)
        # orbital period in days
        df_clean['pl_orbper'] = df['Orbital Period'].apply(self._clean_orbital_period)
        # planet radius in earth radius
        df_clean['pl_rade'] = df['Radius'].apply(self._clear_radius)
        # day length in hours
        df_clean['Day Length [h]'] = df['Day Length'].apply(self._clean_day_length)
        # atmospheric pressure in atm (bar)
        df_clean['Atm. Pressure [atm]'] = df['Atm. Pressure'].apply(self._clean_atm_press)
        # Surface temperature en Kelvin
        df_clean['pl_eqt'] = df['Surface Temp'].apply(self._clean_surf_temp)
        # Surface gravity in g (10 m/s2)
        df_clean['Surface Gravity [g]'] = df['Surface Gravity'].apply(self._clean_gravity)
        # planet mass in earth mass
        df_clean['pl_masse'] = df['Mass'].apply(self._clean_mass)
        # number of satellites
        df_clean['Satellites'] = df['Satellites'].apply(self.clean_satellites)

        df_clean['st_mass'] = df['pl_name'].apply(lambda x: np.nan)
        df_clean['st_rad'] = df['pl_name'].apply(lambda x: np.nan)

        return df_clean


if __name__ == '__main__':
    df = MassEffect().clean_data()
    print(df)
    pass
