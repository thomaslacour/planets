import pandas as pd
import requests
import re


def get_pot_livable_planets():

    # scrap
    url = r'https://fr.wikipedia.org/wiki/Liste_d%27exoplan%C3%A8tes_potentiellement_habitables'
    html = requests.get(url).text

    # parse with pandas
    dfs = pd.read_html(html)
    df = pd.concat(dfs)

    # generate df
    drop_cols = ['Rang','Distance (al)', 'Statut', 'Année dedécouverte']
    mapper = {'Nom':'pl_name', 'pClasse':'pClass', 'hClasse':'hClass'}

    df = df.drop(columns=drop_cols).rename(columns=mapper)

    # manual correction of planet names
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Gliese','GJ', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('(GJ.*[A-Z])([a-z])','\\1 \\2', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('.*(GJ.*)\)','\\1', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Tau Ceti','tau Cet', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Proxima Centauri','Proxima Cen', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Luyten b','GJ 273 b', x))

    df['is_pot_livable'] = True

    df = df.set_index('pl_name')

    return df
