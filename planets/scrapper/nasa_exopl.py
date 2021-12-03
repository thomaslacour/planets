import re
import glob
import pandas as pd



def get_exopl_types():

    rgx = re.compile(r'\d{4}\n\n  \*\s*([^<\*]+)\n    <')

    planet_type_dict = {}
    for pl_type in [
        'gas giant', 'neptune-like', 'super earth', 'terrestrial', 'unkown']:
        for file in glob.glob(
            f'../raw_data/nasa_exoplanet_brut_scrap/*{pl_type}*'):
            with open(file) as f:
                text = f.read()
            planets = rgx.findall(text)
            for planet in planets:
                planet_type_dict[rgx.sub('\1', planet)] = pl_type

    df = pd.DataFrame(
        data=planet_type_dict.values(),
        index=planet_type_dict.keys()
        )

    df.reset_index(inplace=True)
    df.rename(columns={0:'pl_type', 'index':'pl_name'}, inplace=True)

    # manual correction of planet names
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('CI Tauri','CI Tau', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Cancri','Cnc', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Tau Ceti','tau Cet', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Proxima Centauri','Proxima Cen', x))

    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Gamma Librae','gam Lib', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Gamma Cephei','gam Cep', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Gamma','gam', x))

    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Beta Cnc','bet Cnc', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Beta Pictoris','bet Pic', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Beta Ursae Minoris','bet UMi', x))

    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Ursae Minoris','UMi', x))

    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Virginis','Vir', x))

    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Canis Majoris','CMa', x))

    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Herculis','Her', x))

    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('AU Microscopii','AU Mic', x))

    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Ceti','Cet', x))

    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Ursae Majoris','UMa', x))

    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Leonis','Leo', x))

    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Cygni','Cyg', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Scorpii','Sco', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Delphini','Del', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Bootis','Boo', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Arietis','Ari', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Draconis','Dra', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Tauri','Tau', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Eridani','Eri', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Pegasi','Peg', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Lyncis','Lyn', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Aquarii','Aqr', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Piscium','Pis', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Pi Mensae','pi Men', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Lupi','Lup', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Pictoris','Pic', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Tucanae','Tuc', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Andromedae','And', x))

    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('24 Sextantis','24 Sex', x))

    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Epsilon Coronae Borealis','eps CrB', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Epsilon Eridani','eps Eri', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Epsilon Indi','eps Ind', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Epsilon Tauri','eps Tau', x))

    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Epsilon','eps', x))

    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Alpha','alf', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Rho','rho', x))

    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Canun Venaticorum','CVn', x))

    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Coronae Borealis','CrB', x))

    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Kepler-90b','KOI-351 b', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Kepler-90c','KOI-351 c', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Kepler-90d','KOI-351 d', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Kepler-90e','KOI-351 e', x))
    df['pl_name'] = df[['pl_name']].applymap(lambda x: re.sub('Kepler-90f','KOI-351 f', x))

    df.set_index('pl_name', inplace=True)

    return df
