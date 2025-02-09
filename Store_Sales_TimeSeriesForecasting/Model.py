from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
import numpy as np

from sklearn.impute import SimpleImputer

class Model:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.imputer = SimpleImputer(strategy='median')  # Use median to fill NaN values

    def train(self, X_train, y_train):
        """
        Train the model on the provided data.
        """
        X_train_imputed = self.imputer.fit_transform(X_train)  # Impute missing values
        self.model.fit(X_train_imputed, y_train)

    def predict(self, X_test):
        """
        Predict the target variable for the test data.
        """
        X_test_imputed = self.imputer.transform(X_test)  # Impute missing values in test data
        return self.model.predict(X_test_imputed)

    def evaluate(self, y_true, y_pred):
        """
        Evaluate the model using Root Mean Squared Logarithmic Error (RMSLE).
        """
        assert len(y_true) == len(y_pred)
        return np.sqrt(np.mean(np.power(np.log1p(y_true) - np.log1p(y_pred), 2)))
