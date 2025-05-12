import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from onboarding.challenge.config import settings
from onboarding.challenge.model import DelayModel


class DataLoader:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.file_path, low_memory=False)
        # low_memory=False to avoid DtypeWarning for mixed types in columns


if __name__ == "__main__":
    # Load data
    loader = DataLoader("onboarding/challenge/data.csv")
    data = loader.load_data()
    data = shuffle(data, random_state=111)

    # Preprocess
    model = DelayModel(settings["categorical_features"], settings["top_10_features"])
    y = model.add_delay(data, threshold=15)
    y = pd.DataFrame(y, columns=[settings["target_column"]])

    x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.33, random_state=42)

    # Train
    model.fit(x_train, y_train)
    # Predict
    preds = model.predict(x_test)
    # Save model
    model.save_model("onboarding/challenge/delay_model.pkl")

    # Load model
    model.load_model("onboarding/challenge/delay_model.pkl")
    # Predict with loaded model
    preds_loaded = model.predict(x_test)
