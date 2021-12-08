from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

import joblib

from planets.pipeline.predict import _get_model_from_local, custom_predict

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
