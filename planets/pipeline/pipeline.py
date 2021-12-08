import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.compose import make_column_selector, ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV, train_test_split

from scipy.spatial import distance

from nasa import Nasa



class ModelPipeline:

    def __init__(self):

        self.grid_cv = None
        self.score_acc = None
        self.trained_model = None
        self.X_cols = None


    def build_balanced_train_test_sets(self, kept_cols='default', rebuild=True):

        df = Nasa().load_clean_data(filt='relevant', rebuild=rebuild)
        if kept_cols == 'default':
            kept_cols = [
                'sy_snum', 'sy_pnum', 'pl_orbper', 'pl_rade',
                'pl_bmasse','pl_orbeccen', 'pl_insol', 'pl_eqt', 'st_teff',
                'st_rad', 'st_mass','st_logg', 'pl_type', 'pl_orb_is_in_CHZ'
                ]
        else:
            kept_cols = kept_cols

        df = df[kept_cols]

        # mask to only select rows with target
        mask = (df.pl_type == 'gas giant') | (df.pl_type == 'neptune-like') | \
            (df.pl_type == 'terrestrial') | (df.pl_type == 'super earth')

        # data transformation/balancing -- keep a raw X for readable predictions
        X_raw = df[mask]

        # build a balanced dataset
        X_neptune_like = X_raw[X_raw.pl_type == 'neptune-like']
        X_super_earth = X_raw[X_raw.pl_type == 'super earth']
        X_gas_giant = X_raw[X_raw.pl_type == 'gas giant']
        X_terrestrial = X_raw[X_raw.pl_type == 'terrestrial']

        X_balanced = pd.concat(
            (
                X_neptune_like.sample(600, replace=True),
                X_super_earth.sample(600, replace=True),
                X_gas_giant.sample(600, replace=True),
                X_terrestrial.sample(600, replace=True)
            )
        )

        # index must be reset because of random sampling
        X_balanced.reset_index(inplace=True)
        X_index = X_balanced[['pl_name', 'pl_type']]
        y = X_balanced['pl_type']

        # generate X_train, X_test, etc.
        X_train, X_test, y_train, y_test = train_test_split(X_balanced, y, test_size=0.3)

        X_train.reset_index(drop=True, inplace=True)
        X_index = X_train[['pl_name', 'pl_orb_is_in_CHZ', 'pl_type']]
        X_train.drop(columns=['pl_name', 'pl_orb_is_in_CHZ', 'pl_type'], inplace=True)

        y_train.reset_index(drop=True, inplace=True)

        X_test.drop(columns=['pl_name', 'pl_orb_is_in_CHZ', 'pl_type'], inplace=True)

        # save X_train, X_test for custom and more readable predictions
        joblib.dump(X_train, '../X_train.joblib')
        joblib.dump(X_index, '../X_index.joblib')

        return X_train, X_test, y_train, y_test


    def train_model(self):

        """
        Trains a predefined pipeline with knn-imupter, scalers and knn
        classifier.
        """

        X_train, X_test, y_train, y_test = self.build_balanced_train_test_sets()

        # custom column selector -- allows easier integration of imputer in pipelines
        # returns dtypes indices
        dict_cols = {}
        for i, col in enumerate(X_train.columns):
            if dict_cols.get(str(X_train[col].dtypes)) is not None:
                dict_cols[str(X_train[col].dtypes)].append(i)
            else:
                dict_cols[str(X_train[col].dtypes)] = [i]

        ## temp
        self.X_cols = X_train.columns

        scaler = make_column_transformer(
            (MinMaxScaler(), dict_cols.get('int64')),
            (RobustScaler(), dict_cols.get('float64')),
            remainder='passthrough'
        )

        pipe = make_pipeline(
            (KNNImputer()),
            scaler,
            KNeighborsClassifier()
        )

        grid_params = {
            'knnimputer__n_neighbors': range(2,4),
            'knnimputer__weights': ['uniform', 'distance'],
            'knnimputer__metric': ['nan_euclidean'],

            'kneighborsclassifier__n_neighbors': range(2,4),
            'kneighborsclassifier__weights': ['uniform', 'distance'],
        #     'kneighborsclassifier__metric': ['minkowski'],
        #     'kneighborsclassifier__metric_params': [
        #         {'p':2, 'w':[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5, 0.5]},
        #         {'p':2, 'w':[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5]}
        #     ]
            }

        grid_cv = GridSearchCV(
            pipe, param_grid=grid_params, n_jobs=1, scoring='accuracy', cv=5)

        grid_cv.fit(X_train, y_train)
        self.grid_cv = grid_cv

        model = grid_cv.best_estimator_
        model.fit(X_train, y_train)
        self.score_acc = model.score(X_test, y_test)

        self.trained_model = model


    def save_model(self, path='../model.joblib'):

        """
        Saves model to .joblib format
        """

        if self.trained_model is None:
            return
        joblib.dump(self.trained_model, path)


    def _calculate_dist_norm_factor(self, prc_dist=0.8):

        """
        Computes a distance normalisation factor for more readables classifications.
        Factor is set considering cumulative proportion of distance in all computed distance
        uses by model (distance matrix). By default, this factor is set to a
        threshold of 0.8 and 80% of all computed distance.
        """

        X_train = joblib.load('../X_train.joblib')
        X_train_transf = pd.DataFrame(self.trained_model.steps[1][1].transform(self.trained_model.steps[0][1].transform(X_train)))
        dist_mtx = pd.DataFrame(distance.cdist(X_train_transf , X_train_transf , metric='minkowski'))

        for thresh in range(0, 100, 5):
            if ModelPipeline._get_percent_dist_over(dist_mtx, thresh) > prc_dist:
                dist_norm_fact = thresh
                break

        joblib.dump(dist_norm_fact, '../dist_norm_fact.joblib')

        return dist_norm_fact

    @staticmethod
    def _get_percent_dist_over(dist_mtx, thresh):
        return 1 - ((dist_mtx[dist_mtx < thresh].isna().sum().sum()) / (len(dist_mtx)**2))


    def get_imputed_dataset(self, df):

        if self.trained_model is None:
            return

        df = pd.DataFrame(self.trained_model.steps[0][1].transform(df))

        return df

    def get_imputed_scaled_dataset(self, df):

        if self.trained_model is None:
            return

        df = pd.DataFrame(self.trained_model.steps[1][1].transform(self.trained_model.steps[0][1].transform(df)))

        return df
