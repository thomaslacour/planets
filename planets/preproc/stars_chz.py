import pandas as pd


def get_stars_chz():

    """
    Loads and cleans stars specific CHZ from sheet_chz.csv
    Use st_class for joining/merging with main table.
    """
    df_st_hz = pd.read_csv('../raw_data/sheet_chz.csv', header=3, nrows=49)

    df_st_hz.drop(columns=['Unnamed: 0','Unnamed: 1'], inplace=True)
    df_st_hz['Star Class'] = df_st_hz['Star Class'].str.strip()
    cols_mapper = {
        'Star Class':'st_class',
        'Inner Edge (AU)':'st_chz_inn_edge (AU)',
        'Outer Edge (AU)':'st_chz_out_edge (AU)',
        'Luminosity':'st_lum',
        'Width (AU)':'st_chz_width',
        'Inner Edge (ls)':'st_chz_inn_edge (ls)',
        'Outer Edge (ls)':'st_chz_out_edge (ls)'
    }
    df_st_hz.rename(columns=cols_mapper, inplace=True)

    # delete ambiguous coma
    df_st_hz['st_chz_inn_edge (ls)'] = df_st_hz['st_chz_inn_edge (ls)'].str.replace(',', '', regex=False)
    df_st_hz['st_chz_out_edge (ls)'] = df_st_hz['st_chz_out_edge (ls)'].str.replace(',', '', regex=False)
    df_st_hz['st_chz_inn_edge (ls)'] = df_st_hz['st_chz_inn_edge (ls)'].astype('float64')
    df_st_hz['st_chz_out_edge (ls)'] = df_st_hz['st_chz_out_edge (ls)'].astype('float64')

    return df_st_hz
