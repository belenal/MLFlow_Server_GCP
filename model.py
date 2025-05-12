import joblib
import pandas as pd
import xgboost as xgb
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from feature_selector import TopFeatureSelector


class DelayModel:
    def __init__(
        self,
        categorical_features: list[str],
        top_10_features: list[str],
        threshold: int = 15,
    ) -> None:
        self._model = None  # Model should be saved in this attribute.
        self.categorical_features = categorical_features
        self.top_10_features = top_10_features
        self.threshold = threshold
        self.preprocessor = None
        self.selector = None

    def add_delay(self, df: pd.DataFrame, threshold: int = 15) -> pd.Series:
        fecha_o = pd.to_datetime(df["Fecha-O"], errors="coerce")
        fecha_i = pd.to_datetime(df["Fecha-I"], errors="coerce")
        min_diff = (fecha_o - fecha_i).dt.total_seconds() / 60
        return (min_diff > threshold).astype(int)

    def preprocess(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessor is None:
            categorical_pipeline = Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value="Missing"),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            drop=None, sparse_output=False, handle_unknown="ignore"
                        ),
                    ),
                ]
            )
            self.preprocessor = ColumnTransformer(
                [("cat_pipeline", categorical_pipeline, self.categorical_features)]
            )

            set_config(transform_output="pandas")
            self.preprocessor.fit(features)

        x = self.preprocessor.transform(features)
        feature_names = self.preprocessor.get_feature_names_out()

        if self.selector is None:
            matched_features = [
                name
                for name in feature_names
                if any(top in name for top in self.top_10_features)
            ]
            self.selector = TopFeatureSelector(matched_features)

        # Convert to DataFrame and select features
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x, columns=feature_names)

        x = self.selector.transform(x)
        set_config(transform_output="default")
        return x

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        target = target.squeeze()
        n_y0 = (target == 0).sum()
        n_y1 = (target == 1).sum()
        self.scale_pos_weight = n_y0 / n_y1

        features = self.preprocess(features)

        self._model = xgb.XGBClassifier(
            random_state=1,
            learning_rate=0.01,
            scale_pos_weight=self.scale_pos_weight,
        )
        self._model.fit(features, target)

    def predict(self, features: pd.DataFrame) -> list[int]:
        if self._model is None:
            error_msg = "Model has not been trained."
            raise ValueError(error_msg)
        features = self.preprocess(features)
        return self._model.predict(features).tolist()

    def save_model(self, path: str) -> None:
        if self._model is None:
            error_msg = "Model has not been trained."
            raise ValueError(error_msg)
        joblib.dump(
            {
                "model": self._model,
                "preprocessor": self.preprocessor,
                "selector": self.selector,
            },
            path,
        )

    def load_model(self, path: str) -> None:
        data = joblib.load(path)
        self._model = data["model"]
        self.preprocessor = data["preprocessor"]
        self.selector = data["selector"]
