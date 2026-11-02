"""Model zoo: sklearn Pipelines (ColumnTransformer + estimator) and the out-of-time split."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

from churn.features import DEMO_COLS, ID_COLS

CAT_COLS = DEMO_COLS
NON_FEATURES = set(ID_COLS) | {"surv_duration", "surv_event"}


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    num = [c for c in df.columns if c not in NON_FEATURES and not c.startswith("churn_") and c not in CAT_COLS]
    return num, list(CAT_COLS)


@dataclass
class Split:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    score: pd.DataFrame  # latest snapshot, no labels yet


def out_of_time_split(snap: pd.DataFrame, label: str, horizon: int, n_valid: int, n_test: int) -> Split:
    """Train on older snapshots, validate/test on the newest labelled ones.

    Purge: a training row's label window (c, c+horizon] must close before the first validation
    cutoff, and validation labels must close before the first test cutoff. Otherwise the model
    trains on outcomes that happen during the evaluation period.
    """
    labelled = snap[snap[label].notna()]
    cuts = sorted(labelled.cutoff_day.unique())
    test_c = cuts[-n_test:]
    valid_c = [c for c in cuts if c + horizon <= test_c[0]][-n_valid:]
    train_c = [c for c in cuts if c + horizon <= valid_c[0]]
    pick = lambda cs: labelled[labelled.cutoff_day.isin(cs)].reset_index(drop=True)  # noqa: E731
    latest = snap.cutoff_day.max()
    return Split(pick(train_c), pick(valid_c), pick(test_c), snap[snap.cutoff_day == latest].reset_index(drop=True))


def _pre(num: list[str], cat: list[str], scale: bool, onehot: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat_enc = (
        OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False)
        if onehot
        else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    )
    return ColumnTransformer(
        [("num", Pipeline(num_steps), num), ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Unknown")), ("enc", cat_enc)]), cat)],
        verbose_feature_names_out=False,
    )


def make_model(name: str, num: list[str], cat: list[str], params: dict | None = None, imbalance: str = "weight",
               pos_weight: float = 1.0, seed: int = 42):
    """imbalance: 'none' | 'weight' (class weights) | 'under' (RandomUnderSampler) | 'smotenc'."""
    params = dict(params or {})
    w = imbalance == "weight"
    if name == "logreg":
        est = LogisticRegression(max_iter=2000, C=params.pop("C", 0.5), class_weight="balanced" if w else None)
        pre = _pre(num, cat, scale=True, onehot=True)
    elif name == "random_forest":
        est = RandomForestClassifier(n_estimators=400, min_samples_leaf=20, n_jobs=-1, random_state=seed,
                                     class_weight="balanced_subsample" if w else None, **params)
        pre = _pre(num, cat, scale=False, onehot=False)
    elif name == "lightgbm":
        base = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=50,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=seed)
        base.update(params)
        est = LGBMClassifier(scale_pos_weight=pos_weight if w else 1.0, **base)
        pre = _pre(num, cat, scale=False, onehot=False)
    elif name == "xgboost":
        base = dict(n_estimators=600, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8,
                    min_child_weight=5, eval_metric="aucpr", random_state=seed, n_jobs=-1)
        base.update(params)
        est = XGBClassifier(scale_pos_weight=pos_weight if w else 1.0, **base)
        pre = _pre(num, cat, scale=False, onehot=False)
    elif name == "catboost":
        base = dict(iterations=800, learning_rate=0.05, depth=6, verbose=0, random_seed=seed)
        base.update(params)
        est = CatBoostClassifier(auto_class_weights="Balanced" if w else None, **base)
        pre = _pre(num, cat, scale=False, onehot=False)
    else:
        raise ValueError(name)

    steps = [("pre", pre)]
    if imbalance == "under":
        steps.append(("sample", RandomUnderSampler(sampling_strategy=0.25, random_state=seed)))
    elif imbalance == "smotenc":
        # categorical columns sit at the end of the ColumnTransformer output (ordinal encoded)
        steps.append(("sample", _LateSMOTENC(n_cat=len(cat), random_state=seed)))
    steps.append(("model", est))
    return ImbPipeline(steps) if imbalance in ("under", "smotenc") else Pipeline(steps)


class _LateSMOTENC(SMOTENC):
    """SMOTENC whose categorical indices are the last `n_cat` columns, known only after preprocessing."""

    def __init__(self, n_cat: int, sampling_strategy=0.25, random_state=None, k_neighbors=5):
        self.n_cat = n_cat
        super().__init__(categorical_features=[0], sampling_strategy=sampling_strategy,
                         random_state=random_state, k_neighbors=k_neighbors)

    def fit_resample(self, X, y, **kw):
        X = np.asarray(X)
        self.categorical_features = list(range(X.shape[1] - self.n_cat, X.shape[1]))
        return super().fit_resample(X, y, **kw)


def positive_weight(y: np.ndarray) -> float:
    y = np.asarray(y)
    return float((1 - y.mean()) / max(y.mean(), 1e-9))
