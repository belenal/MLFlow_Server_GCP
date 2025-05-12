import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, top_feature_names: list[str]) -> None:
        self.top_feature_names = top_feature_names

    def fit(self, _x: pd.DataFrame, _y: pd.Series = None) -> "TopFeatureSelector":
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return x.loc[:, self.top_feature_names]
