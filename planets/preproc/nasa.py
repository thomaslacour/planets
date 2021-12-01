import pandas as pd
import numpy as np

RAW_DATA_PATH = '../raw_data/nasa_PS_2021.11.23_14.10.10.csv'
COLS_MAPPER_PATH = '../raw_data/nasa_Exoplanet_Archive_Column_Mapping.csv'

class Nasa:

    def __init__(self):

        self.col_dict = self.load_readable_column_names()
        self.col_blocks = self.get_column_blocks()
        self.relev_cols = self.get_relevant_columns()
        self.pl_name_index = None

    def load_raw_data(self):

        df_planets = pd.read_csv(
        RAW_DATA_PATH,
        header=290,
        low_memory=False
        )

        return df_planets

    def get_relevant_columns(self) -> dict:

        # filter out 'unc', 'err', 'lim' cols
        relev_cols = [
            _ for _ in self.col_dict.keys() if 'err' not in str(_) and str(_).endswith('lim') is False
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

    def load_readable_column_names(self) -> dict:

        """Gets readable column names from nase column mapping.
        Returns a dict with key, value as db_col_name, readable name"""

        # get readable columns names for filtering
        columns = pd.read_csv(COLS_MAPPER_PATH)
        columns = columns.iloc[:, [0,1]]
        columns.rename(columns={'Unnamed: 0':'db_name', 'Unnamed: 1':'complete_name'}, inplace=True)
        columns.drop(index=[0,1], inplace=True)
        columns['db_name'] = columns['db_name'].str.strip()
        columns.set_index(keys='db_name', inplace=True)
        columns.dropna(inplace=True)

        return columns.to_dict().get('complete_name')

    def get_column_blocks(self):

        """
        Gets blocks of data about, Planetary Parameter Reference,
        Stellar Parameter Reference, System Parameter Reference,
        Planetary Parameter Reference Publication Date by splitting the original
        set of columns.
        """

        # get blocks of datas
        marker = 'parameter reference'
        prev_idx = 0
        blocks_list = {}
        for i, col_name in enumerate(self.col_dict.values()):
            if marker in str(col_name).lower():
                blocks_list[col_name] = list(self.col_dict.values())[prev_idx:i]
                prev_idx = i

        return blocks_list

    def get_column_blocks2(self, prefix_marker='pl_'):

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

    def get_columns_from_blocks(self, block='Planetary Parameter Reference'):

        """
        Gets list of columns from a given block. Mostly used for filtering.
        Blocks are:
        - Planetary Parameter Reference
        - Stellar Parameter Reference
        - System Parameter Reference
        - Planetary Parameter Reference Publication Date

        """

        # get db_col_names from blocks
        corr_keys = []
        for prop in self.col_blocks[block]:
            for key,val in self.col_dict.items():
                if prop == val:
                    if key not in ['pl_orbtper_systemref'] and 'err' not in str(key) and 'lim' not in str(key):
                        corr_keys.append(key)

        return corr_keys

    def aggregate_rows_on_pl_name(self, df):
        return df.groupby('pl_name').agg(self._aggregate_func)

    def _aggregate_func(self, x):

        x = x.dropna()
        if len(x) == 0:
            return np.nan
        elif len(x.unique()) == 1:
            return x.unique()[0]
        else:
            return list(x.unique())

    def load_clean_data(self):

        # load raw data
        df = self.load_raw_data()

        # filter on relevant columns
        df = df[list(self.get_relevant_columns().keys())]

        # merge rows about the same planet
        df = self.aggregate_rows_on_pl_name(df)

        return df
