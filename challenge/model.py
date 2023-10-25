import pandas as pd

from typing import Tuple, Union, List

from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import xgboost as xgb

class DelayModel:

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.
        self._dataScale = 0# For scaling data
    
    # Method for feature generation    
    def get_min_diff(self, data):
        fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds())/60
        return min_diff

    # Fixed return type, Union should be have [] not ()
    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        # Generate features
        #Get "Difference in Minutes" feature
        data['min_diff'] = data.apply(self.get_min_diff, axis = 1)
        # Get "Delay" feature
        threshold_in_minutes = 15
        data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
        # Create features
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
            pd.get_dummies(data['MES'], prefix = 'MES')], 
            axis = 1
        )
        # Get data balance
        _, _, y_train, _ = train_test_split(features, data['delay'], test_size = 0.33, random_state = 42)
        n_y0 = len(y_train[y_train == 0])
        n_y1 = len(y_train[y_train == 1])
        self._dataScale = n_y0/n_y1
        # Extract 10 most important features
        top_10_features = [
            "OPERA_Latin American Wings", 
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]
        features = features[top_10_features]
        # Return processed data, return target column if specified
        # Check if column exists, return target column if so
        response = None
        # If no target column is passed (it is None) it won't be returned
        if target_column in data.columns:
            response = (features, data[[target_column]])
        else:
            response = features
        return response

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        # Create train/test split
        x_train, _, y_train, _ = train_test_split(features, target, test_size=0.33, random_state = 42)
        # Create and fit model
        self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight = self._dataScale)
        self._model.fit(x_train, y_train)
        # Save model for future use
        modelData = self._model.get_booster()
        modelData.save_model('delay_model.json')
        return

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        # Load model if it has not been loaded
        if self._model is None:
            self._model = xgb.XGBClassifier()
            self._model.load_model('delay_model.json')
        # Predict using passed features
        predictions = [int(prediction) for prediction in self._model.predict(features)]
        return predictions