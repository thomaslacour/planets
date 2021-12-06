import pandas as pd


def get_planets_chz():

    """
    Loads and cleans planets specific CHZ from hzgallery_chz.csv
    Use pl_name for joining/merging with main table.
    """

    df_pl_hz = pd.read_csv('../raw_data/hzgallery_chz.csv')

    cols_mapper = {
        'PLANET':'pl_name',
        'MASS':'massj',
        'RADIUS':'radj',
        'PERIOD':'orbper',
        'ECC':'orb_ecc',
        'OMEGA':'arg_perias',
        'THZC':'pl_orb_prc_in_cCHZ',
        'THZO':'pl_orb_prc_in_cCHZ',
        'TEQA':'pl_eqt_perias_h',
        'TEQB':'pl_eqt_perias_m',
        'TEQC':'pl_eqt_apas_h',
        'TEQD':'pl_eqt_apas_m',
    #     'OHZIN',
    #     'CHZIN',
    #     'CHZOUT',
    #     'OHZOUT'
    }
    df_pl_hz.rename(columns=cols_mapper, inplace=True)
    df_pl_hz['pl_name'] = df_pl_hz['pl_name'].str.strip()
    df_pl_hz.set_index('pl_name', inplace=True)

    drop_cols = [
        'massj', 'radj', 'orbper','orb_ecc','arg_perias','pl_eqt_perias_h',
        'pl_eqt_perias_m','pl_eqt_apas_h','pl_eqt_apas_m'
        ]

    return df_pl_hz.drop(columns=drop_cols)
