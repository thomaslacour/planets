from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

import joblib

from planets.pipeline.predict import _get_model_from_local, custom_predict, generate_random_planet

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Welcome to Planet U API"}

@app.get("/predict")
def predict(sy_snum, sy_pnum, pl_orbper, pl_rade, pl_bmasse,
    pl_orbeccen, pl_insol, pl_eqt, st_teff, st_rad, st_mass,
    st_logg, n_neighs_shown=0, radius=0):

    X_pred = pd.DataFrame.from_records([
        {
            'sy_snum':sy_snum,
            'sy_pnum':sy_pnum,
            'pl_orbper':pl_orbper,
            'pl_rade':pl_rade,
            'pl_bmasse':pl_bmasse,
            'pl_orbeccen':pl_orbeccen,
            'pl_insol':pl_insol,
            'pl_eqt':pl_eqt,
            'st_teff':st_teff,
            'st_rad':st_rad,
            'st_mass':st_mass,
            'st_logg':st_logg
        }
        ])

    prediction = custom_predict(X_pred, n_neighs_shown, radius)

    return prediction

@app.get("/generate")
def generate(pl_type, reliability='avg', max_iter=1000, sy_snum='null',
    sy_pnum='null', pl_orbper='null', pl_rade='null', pl_bmasse='null',
    pl_orbeccen='null', pl_insol='null', pl_eqt='null', st_teff='null',
    st_rad='null', st_mass='null', st_logg='null'):

    # types conversion for python compatibility
    if sy_snum =='null': sy_snum = None
    if sy_pnum =='null': sy_pnum = None
    if pl_orbper =='null': pl_orbper = None
    if pl_rade =='null': pl_rade = None
    if pl_bmasse =='null': pl_bmasse = None
    if pl_orbeccen =='null': pl_orbeccen = None
    if pl_insol =='null': pl_insol = None
    if pl_eqt =='null': pl_eqt = None
    if st_teff =='null': st_teff = None
    if st_rad =='null': st_rad = None
    if st_mass =='null': st_mass = None
    if st_logg =='null': st_logg = None

    rand_generated_pl = generate_random_planet(pl_type, sy_snum, sy_pnum, pl_orbper, pl_rade, pl_bmasse,
    pl_orbeccen, pl_insol, pl_eqt, st_teff, st_rad, st_mass,
    st_logg, reliability, req_files_path='../planets/pipeline/')

    return rand_generated_pl
