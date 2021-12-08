#!/usr/bin/env python

import pandas as pd
import numpy as np
import re
# from planets.scrapper.fandom import Fandom
from planets.scrapper.utils import TableScrapper


RAW_DATA_URL = 'https://promenade.imcce.fr/fr/pages1/19.html'
RAW_DATA = '../raw_data/solarsystem_wiki.csv'


class SolarSys:

    def __init__(self):
        self.raw_data=None
        self._cleaned_data=None

    def scrap_data_to_csv(self, filepath=RAW_DATA):
        """Scrap data from web
        """
        solarsystem = TableScrapper(url=RAW_DATA_URL)
        df = solarsystem.scrap()[0].T.copy()
        df.columns=df.iloc[0,:]
        df = df.iloc[1:,0:]
        # df.rename(columns={'Unnamed: 0':'pl_name'}, inplace=True)
        df.to_csv(filepath)
        return filepath


    def load_raw_data(self, filepath=RAW_DATA) -> pd.DataFrame:
        """
        """
        self.raw_data = pd.read_csv(filepath)
        return self


    def _clean_pl_orbincl(self, text):
        """
        """
        digits = list(re.findall(r"[0-9]+", text))
        if len(digits)==1:
            return int(digits[0])
        return int(digits[0]) + float(digits[1])/1000

    def _clean_pl_orbper(self, text):
        """
        """
        digit = list(re.findall(r"[0-9]+", str(text)))
        if len(digit) == 3:
            jr = float(digit[1]) + int(digit.pop(0))*365.256
            digit[0] = str(jr)
        return float(digit[0]) + float(digit[1])/10**(len(digit[1]))

    def _clean_pl_rev(self, text):
        """
        """
        text=text.replace(',', '.')
        if text.endswith('jr'):
            earth_days = float(text.split()[0])
        if text.endswith('h'):
            earth_days = float(text.split()[0])/24.62
        return earth_days

    def _clean_2pl_rad_deg(self, text):
        if type(text) is not float:
            return float(text[:-1].replace(',', '.'))

    def _clean_pl_rad_assym(self, text):
        """
        """
        text = text.replace('/', ' ').split()
        if len(text) == 2:
            return 1/float(text[-1].replace(',', '.'))
        return 0

    def _clean_pl_mass_sun(self, text):
        text = text.replace('.10-8', '/100000000')
        nums = [ float(n.replace(',', '.')) for n in text.replace('/', ' ').split() ]
        return nums[0]/nums[-1]


    @property
    def cleaned_data(self):
        """Cleaned data
        """

        df = self.raw_data.copy()

        self.columns_raw = list(df.columns)
        self.columns_raw[0]="Nom de l'astre"

        # ===
        df.iloc[:,5] = df.iloc[:,5].apply(self._clean_pl_orbincl)
        df.iloc[:,6] = df.iloc[:,6].apply(lambda s: float(s[:-1].replace(',', '.')) )
        df.iloc[:,7] = df.iloc[:,7].apply(self._clean_pl_orbper)
        df.iloc[:,8] = df.iloc[:,8].apply(self._clean_pl_rev)
        df.iloc[:,9] = df.iloc[:,9].apply(self._clean_2pl_rad_deg)
        df.iloc[:,10] = df.iloc[:,10].apply(lambda x: float(f"0.{str(x)}") if x!=1 else x)
        df.iloc[:,11] = None
        df.iloc[:,12] = df.iloc[:,12].apply(self._clean_pl_rad_assym)
        df.iloc[:,13] = None
        df.iloc[:,14] = df.iloc[:,14].apply(self._clean_pl_mass_sun)
        df.iloc[:,15] = [0.055, 0.815, 1, 0.107, 317.83, 95.16, 14.54, 17.15, 0.0021]
        df.iloc[:,16] = df.iloc[:,16].apply(lambda x: float(f"{str(x)[0]}.{str(x)[1:-5]}")*10**int(str(x)[-2:]))#/59736.1024
        df.iloc[:,18] = df.iloc[:,18].apply(lambda x: float(f"0.{x}") if x!=1 else 1)
        df.iloc[:,19] = None
        df.iloc[:,20] = df.iloc[:,20].apply(lambda x: x/100 if x!=1 else 1)

        df.columns=[ 'pl_name', 'pl_mnum', 'pl_orbsmax', 'pl_orbsmax [km]',
        'pl_orbeccen', 'pl_orbincl', 'pl_orbincl [eq]', 'pl_orbper', 'pl_rev',
        '2pl_rad [deg]', 'pl_rade', '2pl_rad [km]', 'pl_rad_assym', 'pl_vol [earth]', 'pl_masse [sun]', 'pl_masse', 'pl_mass [kg]',
        'pl_masse + moon_mass', 'pl_dens [earth]', 'pl_dens [water]',
        'pl_grav', '2st_rad [apparent]', 'sy_mag [?]']

        df['pl_masse'] = df.iloc[:,16]/df.iloc[2,16]
        df['st_mass'] = 1
        df['st_rad'] = 1
        # ===

        df.loc[:3, 'pl_type'] = 'Terrestrial'.lower()
        df.iloc[4:-1, -1]  = 'Gas Giant'.lower()
        df.iloc[-1, -1]  = 'Dwarf'.lower()

        return df


if __name__ == '__main__':
    pass
