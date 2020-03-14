import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from eli5.sklearn import PermutationImportance
from sklearn.metrics import mean_squared_error
from pdpbox import pdp, get_dataset, info_plots
from sklearn.feature_selection import f_regression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D
from category_encoders.target_encoder import TargetEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from scipy.stats import mode

def rmse(y_real, y_pred):
    return np.sqrt(mean_squared_error(y_real, y_pred))


def compr(list_it, remove_it):
    return [x for x in list_it if x not in remove_it]


def ppl_to_df(ppl, name, X, y, fit=True):
    if fit:
        trans = ppl.fit_transform(X, y)
    else:
        trans = ppl.transform(X)
    ndf = pd.DataFrame(
        trans, columns=ppl[name].get_feature_names(), index=X.index
    ).astype("float")
    return ndf.rename(columns=lambda x: re.findall("\_\_(.*)\Z", x)[0])


def feat_importance(m, df, plot=False, top=10):
    imp_df = pd.DataFrame({"cols": df.columns, "imp": m}).sort_values(
        "imp", ascending=False
    )
    if plot:
        fig, axbo = plt.subplots(1, 1, figsize=(10, 6))
        sns.barplot(x="cols", y="imp", data=imp_df[:top], palette="Blues_d", ax=axbo)
        plt.show()
    return imp_df.set_index("cols")


def clean_split(struct_cleaned, beg, fin, y_name):
    ids = struct_cleaned["Flight_instance_ID"].unique()[beg:fin]
    cleaned = struct_cleaned[struct_cleaned["Flight_instance_ID"].isin(ids)]
    return (cleaned.drop(columns=y_name), cleaned["FF"])


def mix_split(df, sort, target, ids, size):
    sorteddf = df.sort_values(by=sort)
    top = sorteddf[:size]
    X = top.drop(columns=target)
    y = top[target]
    rest = sorteddf[size:]
    return X, y, rest[~rest[ids].isin(X[ids])]


def drop_unvariance(traine):
    z_cols = traine.columns[traine.var() == 0]
    return traine.drop(columns=z_cols)


def smart_subsample(df, ids, sam):
    dfs = df.reset_index()
    grouped = dfs.groupby([ids])["index"]
    reserve = grouped.min().append(grouped.max())
    mask = dfs["index"].isin(reserve.values)
    takesamp = lambda d: d.sample(sam)
    todrop = dfs[~mask].groupby(ids).apply(takesamp)
    return df.drop(todrop["index"].values)


def remove_rows_val(df, val, col):
    return df[df[col] != val].astype("float")


def elapsed(df, beg, fin):
    df["elapsed"] = pd.to_datetime(df.iloc[:, beg:fin]).astype("int") / 1000000000
    return df.sort_values(by=["elapsed"])


def search_by_line(model, params, X, y, sub_pr=False):
    finp = {}
    for grid in params:
        gr = GridSearchCV(model, grid, n_jobs=-1)
        gr.fit(X, y)
        model.set_params(**gr.best_params_)
        if sub_pr:
            print(gr.best_params_)
        finp.update(gr.best_params_)

    print(finp)


def FFpart(X, y, step, colors):
    forplot = X
    forplot["FF"] = y
    forplote = forplot[forplot["PH"] != 0]
    counter = 0
    grouped = forplote.groupby("part")
    types = grouped["PH"].max()
    cons = grouped["FF"].mean()
    fins = pd.DataFrame(columns=["type", "cons", "part"])
    while counter < 1:
        maskc = np.logical_and(types.index >= counter, types.index < counter + step)
        fins = fins.append(
            {
                "type": mode(types[maskc])[0][0],
                "cons": cons[maskc].mean(),
                "part": counter,
            },
            ignore_index=True,
        )
        counter += step
    fins["type"] = fins["type"].replace(colors)
    return fins


class DoNothing(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x=None, y=None):
        self.params = x.columns
        return self

    def transform(self, x=None):
        return x

    def get_feature_names(self):
        return list(self.params)


class FlightPart(BaseEstimator, TransformerMixin):
    def __init__(self, elapsed, ids):
        self.elapsed = elapsed
        self.ids = ids
        pass

    def fit(self, x=None, y=None):
        self.params = x.columns

        return self

    def transform(self, X=None):
        strats = X.groupby(self.ids)[self.elapsed].min()

        finishes = X.groupby(self.ids)[self.elapsed].max()

        X["from_beg"] = X[self.elapsed] - strats[X[self.ids]].values
        X["duration"] = finishes[X[self.ids]].values - strats[X[self.ids]].values
        X["part"] = X["from_beg"] / X["duration"]
        X.drop(columns=self.ids, inplace=True)

        self.params = X.columns
        return X

    def get_feature_names(self):
        return list(self.params)


class MintoMean(BaseEstimator, TransformerMixin):
    def __init__(self, miss):
        self.miss = miss
        self.minv = {}
        self.mean = {}

    def fit(self, x=None, y=None):
        self.params = x.columns
        self.minv = x.apply(lambda cval: cval.min(), axis=0) + self.miss
        self.mean = x.apply(lambda cval: cval[cval > cval.min()].mean(), axis=0)
        return self

    def transform(self, x=None):
        for col in x.columns:
            x[col + "_out"] = x[col] < self.minv[col]
            x[col].where(x[col] > self.minv[col], other=self.mean[col], inplace=True)
        self.params = x.columns
        return x

    def get_feature_names(self):
        return list(self.params)


class ClustTarg(BaseEstimator, TransformerMixin):
    def __init__(self, n_clust):
        self.n_clust = n_clust
        pass

    def fit(self, x=None, y=None):
        self.params = x.columns
        self.kmn_mod = {}
        self.trg_mod = {}
        for col in x.columns:
            tmp = pd.DataFrame([])
            self.kmn_mod[col] = KMeans(n_clusters=self.n_clust[col])
            self.kmn_mod[col].fit(np.reshape(x[col].values, (-1, 1)))

            tmp[col] = self.kmn_mod[col].predict(np.reshape(x[col].values, (-1, 1)))
            self.trg_mod[col] = TargetEncoder()
            self.trg_mod[col].fit(tmp[col].astype("category"), train_y)
        return self

    def transform(self, X=None):
        for col in X.columns:
            X[col] = self.kmn_mod[col].predict(np.reshape(X[col].values, (-1, 1)))
            X[col] = self.trg_mod[col].transform(X[col].astype("category"))
        return X

    def get_feature_names(self):
        return list(self.params)


class CorrCount:
    def __init__(self, dataset, coeff):
        correlation = dataset.corr()
        correlation.dropna(axis=0, how="all", inplace=True)
        correlation.dropna(axis=1, how="all", inplace=True)

        col1 = (
            pd.Series(correlation.index).repeat(correlation.iloc[0].size).reset_index()
        )
        col2 = pd.concat(
            [pd.Series(correlation.index)] * correlation.iloc[0].size
        ).reset_index()
        col3 = pd.Series(np.squeeze(correlation.values.reshape(-1, 1))).reset_index()

        corrtab = pd.DataFrame({"col1": col1[0], "col2": col2[0], "coef": col3[0]})
        corrtab = corrtab[corrtab["col1"] != corrtab["col2"]]

        self.correlation = correlation
        self.highcorrtab = corrtab[
            ((corrtab["coef"] > coeff) | (corrtab["coef"] < coeff * -1))
        ]

    def betterval(self, impor, feat_X):
        def to_remove(feat, imortance):
            if (imortance.loc[feat["col1"]] < imortance.loc[feat["col2"]]).bool():
                return feat["col1"]
            else:
                return False

        rem = self.highcorrtab.apply(lambda x: to_remove(x, impor), axis=1).unique()
        rem = rem[rem != False]
        return feat_X.drop(columns=rem)