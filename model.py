from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer


CONTINUOUS_COLS = [
    'Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6',
    'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5'
    ]

DISCRETE_COLS = [
    'Medical_History_1', 'Medical_History_10', 'Medical_History_15',
    'Medical_History_24', 'Medical_History_32',
    'Medical_Keyword_1', 'Medical_Keyword_2', 'Medical_Keyword_3',
    'Medical_Keyword_4', 'Medical_Keyword_5', 'Medical_Keyword_6',
    'Medical_Keyword_7', 'Medical_Keyword_8', 'Medical_Keyword_9',
    'Medical_Keyword_10', 'Medical_Keyword_11', 'Medical_Keyword_12',
    'Medical_Keyword_13', 'Medical_Keyword_14', 'Medical_Keyword_15',
    'Medical_Keyword_16', 'Medical_Keyword_17', 'Medical_Keyword_18',
    'Medical_Keyword_19', 'Medical_Keyword_20', 'Medical_Keyword_21',
    'Medical_Keyword_22', 'Medical_Keyword_23', 'Medical_Keyword_24',
    'Medical_Keyword_25', 'Medical_Keyword_26', 'Medical_Keyword_27',
    'Medical_Keyword_28', 'Medical_Keyword_29', 'Medical_Keyword_30',
    'Medical_Keyword_31', 'Medical_Keyword_32', 'Medical_Keyword_33',
    'Medical_Keyword_34', 'Medical_Keyword_35', 'Medical_Keyword_36',
    'Medical_Keyword_37', 'Medical_Keyword_38', 'Medical_Keyword_39',
    'Medical_Keyword_40', 'Medical_Keyword_41', 'Medical_Keyword_42',
    'Medical_Keyword_43', 'Medical_Keyword_44', 'Medical_Keyword_45',
    'Medical_Keyword_46', 'Medical_Keyword_47', 'Medical_Keyword_48'
    ]

CAT_COLS = [
    'Product_Info_1', 'Product_Info_3', 'Product_Info_5',
    'Product_Info_6', 'Product_Info_7', 'Employment_Info_2',
    'Employment_Info_3', 'Employment_Info_5', 'InsuredInfo_1', 'InsuredInfo_2',
    'InsuredInfo_3', 'InsuredInfo_4', 'InsuredInfo_5', 'InsuredInfo_6',
    'InsuredInfo_7', 'Insurance_History_1', 'Insurance_History_2',
    'Insurance_History_3', 'Insurance_History_4', 'Insurance_History_7',
    'Insurance_History_8', 'Insurance_History_9', 'Family_Hist_1', 'Medical_History_2',
    'Medical_History_3', 'Medical_History_4', 'Medical_History_5',
    'Medical_History_6', 'Medical_History_7', 'Medical_History_8',
    'Medical_History_9', 'Medical_History_11', 'Medical_History_12',
    'Medical_History_13', 'Medical_History_14', 'Medical_History_16',
    'Medical_History_17', 'Medical_History_18', 'Medical_History_19',
    'Medical_History_20', 'Medical_History_21', 'Medical_History_22',
    'Medical_History_23', 'Medical_History_25', 'Medical_History_26',
    'Medical_History_27', 'Medical_History_28', 'Medical_History_29',
    'Medical_History_30', 'Medical_History_31', 'Medical_History_33',
    'Medical_History_34', 'Medical_History_35', 'Medical_History_36',
    'Medical_History_37', 'Medical_History_38', 'Medical_History_39',
    'Medical_History_40', 'Medical_History_41'
]


RF_PARAMS = {"max_features": None, "min_samples_split": 33, "min_samples_leaf": 2, "n_estimators": 200}


class ContinuousValuesPreprocessing(BaseEstimator, TransformerMixin):
    """Adjusts the goal feature to USD"""

    def __init__(self):
        self.col_avg = {}

    def fit(self, X):
        for col in X.columns:
            v_m = round(X[col].mean(), 6)
            self.col_avg[col] = v_m
        return self

    def transform(self, X):
        for col in X.columns:
            c_a = self.col_avg[col]
            X[col].fillna(c_a, inplace=True)
        print('Successful Continuous Values transformation')
        return X


class DiscreteValuesPreprocessing(BaseEstimator, TransformerMixin):
    """Adjusts the goal feature to USD"""

    def __init__(self):
        self.col_max_ocurr = {}

    def fit(self, X):
        for col in X.columns:
            # print('col: ', col, 'df: ', X[col].head())
            v_c = X[col].value_counts().idxmax()
            self.col_max_ocurr[col] = v_c
        return self

    def transform(self, X):
        for col in X.columns:
            m_o = self.col_max_ocurr[col]
            X[col].fillna(m_o, inplace=True)
        print('Successful Discrete values transformation')
        return X


class InsuranceModel:

    def __init__(self):

        self.model = RandomForestClassifier(**RF_PARAMS)
        self.preprocessor = ColumnTransformer([
            ("continuous_cols", ContinuousValuesPreprocessing(), CONTINUOUS_COLS),
            ("categories_cols", ContinuousValuesPreprocessing(), CAT_COLS),
            ("discrete_cols", DiscreteValuesPreprocessing(), DISCRETE_COLS),
            ("one_hot", OneHotEncoder(sparse=False, handle_unknown="ignore"), ["Product_Info_2"])
        ])
        self.col_avg = {}
        self.col_max_ocurr = {}

    def preprocess_training_data(self, df):

        # df = df.set_index(df.Id)

        y = df.Response

        x = self.preprocessor.fit_transform(df.drop(["Id", "Response"], axis=1))

        return x, y

    def fit(self, X, y):

        self.model.fit(X, y)

    def preprocess_unseen_data(self, df):

        # df = df.set_index(df.Id)

        x = self.preprocessor.transform(df.drop(["Id"], axis=1))

        return x

    def predict(self, X):

        return self.model.predict(X)
