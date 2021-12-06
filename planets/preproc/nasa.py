import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

import nasa_exopl
import wikipedia_pot_livable_pl
import stars_chz
import planets_chz


RAW_DATA_PATH = '../raw_data/nasa_PS_2021.11.23_14.10.10.csv'
COLS_MAPPER_PATH = '../raw_data/nasa_Exoplanet_Archive_Column_Mapping.csv'


class Nasa:

    def __init__(self):

        self.col_dict = self._load_readable_column_names()
        self.coarse_relev_cols = self.get_coarse_relevant_columns()

        self.pl_name_index = None
        self.NOT_RELEV_COLS = {
            'hd_name': 'HD ID',
            'hip_name': 'HIP ID',
            'tic_id': 'TIC ID',
            'gaia_id': 'GAIA ID',
            'disc_refname': 'Discovery Reference',
            'disc_pubdate': 'Discovery Publication Date',
            'disc_locale': 'Discovery Locale',
            'disc_facility': 'Discovery Facility',
            'disc_telescope': 'Discovery Telescope',
            'disc_instrument': 'Discovery Instrument',
            'soltype': 'Solution Type',
            'pl_masse': 'Planet Mass [Earth Mass]',
            'pl_massj': 'Planet Mass [Jupiter Mass]',
            'pl_msinie': 'Planet Mass*sin(i) [Earth Mass]',
            'pl_msinij': 'Planet Mass*sin(i) [Jupiter Mass]',
            'pl_cmasse': 'Planet Mass*sin(i)/sin(i) [Earth Mass]',
            'pl_cmassj': 'Planet Mass*sin(i)/sin(i) [Jupiter Mass]',
            'pl_bmassprov': 'Planet Mass or Mass*sin(i) Provenance',
            'pl_imppar': 'Impact Parameter',
            'pl_ratdor': 'Ratio of Semi-Major Axis to Stellar Radius',
            'pl_occdep': 'Occultation Depth [%]',
            'pl_orbtper': 'Epoch of Periastron [days]',
            'pl_orblper': 'Argument of Periastron [deg]',
            'pl_rvamp': 'Radial Velocity Amplitude [m/s]',
            'rastr': 'RA [sexagesimal]',
            'ra': 'RA [decimal]',
            'glat': 'Galactic Latitude [deg]',
            'glon': 'Galactic Longitude [deg]',
            'elat': 'Ecliptic Latitude [deg]',
            'elon': 'Ecliptic Longitude [deg]',
            'sy_pm': 'Total Proper Motion [mas/yr]',
            'sy_pmra': 'Proper Motion (RA) [mas/yr]',
            'sy_pmdec': 'Proper Motion (Dec) [mas/yr]',
            'sy_plx': 'Parallax [mas]',
            'sy_bmag': 'B (Johnson) Magnitude',
            'sy_vmag': 'V (Johnson) Magnitude',
            'sy_jmag': 'J (2MASS) Magnitude',
            'sy_hmag': 'H (2MASS) Magnitude',
            'sy_kmag': 'Ks (2MASS) Magnitude',
            'sy_umag': 'u (Sloan) Magnitude',
            'sy_gmag': 'g (Sloan) Magnitude',
            'sy_rmag': 'r (Sloan) Magnitude',
            'sy_imag': 'i (Sloan) Magnitude',
            'sy_zmag': 'z (Sloan) Magnitude',
            'sy_w1mag': 'W1 (WISE) Magnitude',
            'sy_w2mag': 'W2 (WISE) Magnitude',
            'sy_w3mag': 'W3 (WISE) Magnitude',
            'sy_w4mag': 'W4 (WISE) Magnitude',
            'sy_gaiamag': 'Gaia Magnitude',
            'sy_icmag': 'I (Cousins) Magnitude',
            'sy_tmag': 'TESS Magnitude',
            'sy_kepmag': 'Kepler Magnitude',
            'rowupdate': 'Date of Last Update',
            'pl_pubdate': 'Planetary Parameter Reference Publication Date',
            'releasedate': 'Release Date',
            'pl_nnotes': 'Number of Notes',
            'st_nphot': 'Number of Photometry Time Series',
            'st_nrvc': 'Number of Radial Velocity Time Series',
            'st_nspec': 'Number of Stellar Spectra Measurements',
            'pl_nespec': 'Number of Emission Spectroscopy Measurements',
            'pl_ntranspec': 'Number of Transmission Spectroscopy Measurements',
            'pl_tsystemref': 'Time Reference Frame and Standard',
            'sy_refname': 'System Parameter Reference',
            'pl_refname': 'Planetary Parameter Reference',
            'pl_letter': 'Planet Letter',
            'st_refname': 'Stellar Parameter Reference',
            'pl_projobliq': 'Projected Obliquity [deg]',
            'pl_trueobliq': 'True Obliquity [deg]',
            # 'st_spectype': 'Spectral Type',
            'st_vsin': 'Stellar Rotational Velocity [km/s]',
            'st_rotp': 'Stellar Rotational Period [days]',
            'st_radv': 'Systemic Radial Velocity [km/s]'
        }

    def _load_raw_data(self):

        df_planets = pd.read_csv(
        RAW_DATA_PATH,
        header=290,
        low_memory=False
        )

        return df_planets

    def get_coarse_relevant_columns(self) -> dict:

        # filter out 'unc', 'err', 'lim' cols
        relev_cols = [
            _ for _ in self.col_dict.keys()
            if 'err' not in str(_)
            and str(_).endswith('lim') is False
            and str(_).endswith('_flag') is False
            and str(_).startswith('disc_') is False
            ]

        # get keys for convert_table
        convert_table = {}
        for key,val in self.col_dict.items():
            if key in relev_cols:
                # !!! keys have spaces and are not always str type !!!
                convert_table[str(key).strip()] = val

        del convert_table['nan']
        del convert_table['pl_orbtper_systemref']

        return convert_table

    def _load_readable_column_names(self) -> dict:

        """
        Gets readable column names from nase column mapping.
        Returns a dict with key, value as db_col_name, readable name. Usable for
        mapping in pandas.
        """

        # get readable columns names for filtering
        columns = pd.read_csv(COLS_MAPPER_PATH)
        columns = columns.iloc[:, [0,1]]
        columns.rename(columns={'Unnamed: 0':'db_name', 'Unnamed: 1':'complete_name'}, inplace=True)
        columns.drop(index=[0,1], inplace=True)
        columns['db_name'] = columns['db_name'].str.strip()
        columns.set_index(keys='db_name', inplace=True)
        columns.dropna(inplace=True)

        return columns.to_dict().get('complete_name')

    def get_readable_column_names(self, cols:list) -> dict:

        dict_readable_cols = {}
        for col in cols:
            if col in self.col_dict.keys():
                dict_readable_cols[col] = self.col_dict[col]

        return dict_readable_cols

    def get_column_blocks(self, prefix_marker='pl_'):

        """
        Gets blocks of data about, Planetary Parameter Reference,
        Stellar Parameter Reference, System Parameter Reference,
        Planetary Parameter Reference Publication Date by splitting the original
        set of columns.
        """
        blocks_list = []
        for col_name in self.relev_cols.keys():
            if prefix_marker in str(col_name).lower():
                blocks_list.append(col_name)

        return blocks_list

    def _aggregate_rows_on_pl_name(self, df):

        """
        Aggregates rows about the same planet. Sets NaN if not values, keeps,
        the only value if single, keeps all values if mutliple.
        """

        return df.groupby('pl_name').agg(self._aggregate_func)

    def _aggregate_func(self, x):

        x = x.dropna()
        if len(x) == 0:
            return np.nan
        elif len(x.unique()) == 1:
            return x.unique()[0]
        else:
            return list(x.unique())

    def _transform_dtype(self, df):

        # list true/false object cols. false cols are transformed with numerical
        # operations
        # true_obj_cols = ['hostname','discoverymethod','st_metratio','decstr']
        true_obj_cols = ['hostname','discoverymethod','st_metratio','decstr', 'st_spectype']
        false_obj_cols = [
            _ for _ in df.select_dtypes(include=object).columns.tolist()
            if _ not in true_obj_cols
        ]

        # for numerical values, compute mean
        for col in false_obj_cols:
            # transform_dtype(df, col)
            df[col] = df[col].apply(np.mean)

        # some true object cols need specific preprocessing
        # st_spectype sometimes has more than 1 value. Needs aggregation;
        df['st_spectype'] = df['st_spectype'].apply(lambda x: x[0] if isinstance(x, list) else x)

        return df

    def load_clean_data(self, filt='coarse'):

        """
        Loads a dataset without obviously redundant columns such as
        ('err', 'lim', 'flag').
        User may use 'self.NOT_RELEV_COLS' for better filtering.
        """

        # load raw data
        df = self._load_raw_data()

        # filter on relevant columns
        if filt == 'coarse':
            df = df[list(self.coarse_relev_cols.keys())]
        elif filt == 'relevant':
            df = df[list(self.coarse_relev_cols.keys())]
            filt_cols = [col for col in df.columns if col not in self.NOT_RELEV_COLS.keys()]
            df = df[filt_cols]

        # merge rows about the same planet
        df = self._aggregate_rows_on_pl_name(df)

        # !!! for security reasons !!!
        if filt == 'relevant':
            # transform dtypes (false object like)
            df = self._transform_dtype(df)

        # join/add planet types
        df_pl_types = nasa_exopl.get_exopl_types()
        df = df.join(df_pl_types)

        # join/add planet livability
        df_pl_pot_liv = wikipedia_pot_livable_pl.get_pot_livable_planets()
        df = df.join(df_pl_pot_liv)

        # join/add star CHZ
        # a st_class must be created to allow merging with table with star chz
        df_st_chz = stars_chz.get_stars_chz()
        df['st_class'] = df['st_spectype'].str[0:2]
        # trick to avoid loosing index with merging
        df.reset_index(inplace=True)
        df = df.merge(df_st_chz, on='st_class', how='left')
        df['pl_orb_is_in_CHZ'] = \
            (df['pl_orbper'] >= df['st_chz_inn_edge (ls)']) \
            & (df['pl_orbper'] <= df['st_chz_out_edge (ls)'])
        df.set_index('pl_name', inplace=True)

        # join/add planet CHZ
        df_pl_chz = planets_chz.get_planets_chz()
        df = df.join(df_pl_chz, how='left')

        # # set False where planet is not livable
        # filt = ['is_pot_livable']

        return df
