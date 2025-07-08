# import pandas as pd
# import numpy as np
# import os
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.optimizers import Adam

# class FinancialPredictionModel:
#     def __init__(self, data_path, sequence_length=12):
#         self.data = pd.read_csv(data_path)
#         self.data['Date'] = pd.to_datetime(self.data['Date'], errors='coerce')  # Convert invalid dates to NaT
#         if self.data['Date'].isna().any():
#             print("Warning: Some rows have invalid dates and will be dropped.")
#             self.data = self.data.dropna(subset=['Date'])  # Drop rows with NaT in the Date column
#         self.scaler = MinMaxScaler()
#         self.sequence_length = sequence_length
#         self.models = {}
#         self.metrics = ['EPS', 'P_E', 'Dividend_Yield']

#     def preprocess_data(self):
#         # Sort by company and date
#         self.data = self.data.sort_values(['Company_Name', 'Date'])
        
#         # Convert metrics to numeric and handle invalid values
#         for metric in self.metrics:
#             self.data[metric] = pd.to_numeric(self.data[metric], errors='coerce')  # Convert invalid values to NaN
        
#         # Drop rows with NaN in any of the metrics
#         self.data = self.data.dropna(subset=self.metrics)
        
#         # Normalize metrics
#         for metric in self.metrics:
#             self.data[metric] = self.scaler.fit_transform(self.data[[metric]])
        
#         return self.data

#     def create_sequences(self, company_data, metric):
#         X, y = [], []
#         data = company_data[metric].values
#         if len(data) < self.sequence_length:
#             # Not enough data to create sequences
#             return np.array(X), np.array(y)
#         for i in range(len(data) - self.sequence_length):
#             X.append(data[i:i + self.sequence_length])
#             y.append(data[i + self.sequence_length])
#         return np.array(X), np.array(y)

#     def build_model(self):
#         model = Sequential([
#             LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 1)),
#             Dropout(0.2),
#             LSTM(50),
#             Dropout(0.2),
#             Dense(25),
#             Dense(1)
#         ])
#         model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
#         return model

#     def train_model(self, epochs=50, batch_size=32):
#         self.preprocess_data()
#         companies = self.data['Company_Name'].unique()
        
#         for company in companies:
#             company_data = self.data[self.data['Company_Name'] == company]
#             self.models[company] = {}
            
#             for metric in self.metrics:
#                 X, y = self.create_sequences(company_data, metric)
#                 if len(X) == 0:
#                     print(f"Skipping company {company} for metric {metric} due to insufficient data.")
#                     continue
                
#                 X = X.reshape((X.shape[0], X.shape[1], 1))
#                 model = self.build_model()
#                 model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
#                 self.models[company][metric] = model

#     def predict(self, company_name, periods=6):
#         if company_name not in self.models:
#             raise ValueError(f"No model trained for {company_name}")
        
#         company_data = self.data[self.data['Company_Name'] == company_name]
#         predictions = {}
        
#         for metric in self.metrics:
#             last_sequence = company_data[metric].values[-self.sequence_length:]
#             pred = []
            
#             for _ in range(periods):
#                 X = last_sequence.reshape((1, self.sequence_length, 1))
#                 next_val = self.models[company_name][metric].predict(X, verbose=0)
#                 pred.append(next_val[0, 0])
#                 last_sequence = np.append(last_sequence[1:], next_val)
            
#             # Inverse transform predictions
#             pred = self.scaler.inverse_transform(np.array(pred).reshape(-1, 1)).flatten()
#             predictions[metric] = pred
        
#         return predictions

#     def save_predictions(self, output_path):
#         predictions = []
#         companies = self.data['Company_Name'].unique()
        
#         for company in companies:
#             try:
#                 pred = self.predict(company)
#                 for i in range(len(pred[self.metrics[0]])):
#                     predictions.append({
#                         'Company_Name': company,
#                         'Period': i + 1,
#                         **{metric: pred[metric][i] for metric in self.metrics}
#                     })
#             except ValueError as e:
#                 print(f"Skipping company {company} due to error: {e}")
#                 continue
        
#         print(f"Total predictions generated: {len(predictions)}")
#         if predictions:
#             pd.DataFrame(predictions).to_csv(output_path, index=False)
#         else:
#             raise ValueError("No predictions generated. Check input data and model configuration.")
                
# if __name__ == "__main__":
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     model = FinancialPredictionModel(os.path.join(script_dir, "../data/time_series_data_cleaned.csv"))
#     model.train_model()
#     model.save_predictions(os.path.join(script_dir, "../data/predicted_metrics.csv"))
#     print(pd.read_csv(os.path.join(script_dir, "../data/predicted_metrics.csv")).head())









# -----------------------------------------------------------------------------------------------------------

# import os
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# import joblib

# class FinancialPredictionModel:
#     def __init__(self, data_path, sequence_length=12, cache_dir="../models/cache"):
#         self.data = pd.read_csv(data_path)
#         self.data['Date'] = pd.to_datetime(self.data['Date'])
#         self.scaler = MinMaxScaler()
#         self.sequence_length = sequence_length
#         self.cache_dir = cache_dir
#         self.models = {}
#         self.metrics = ['EPS', 'P_E', 'Dividend_Yield']
#         os.makedirs(os.path.join(cache_dir, "lstm_models"), exist_ok=True)
#         os.makedirs(os.path.join(cache_dir, "scalers"), exist_ok=True)

#     def preprocess_data(self):
#         # Sort by company and date
#         self.data = self.data.sort_values(['Company_Name', 'Date'])
#         scaler_path = os.path.join(self.cache_dir, "scalers", "prediction_scaler.pkl")
        
#         # Ensure all required metrics are present
#         for metric in self.metrics:
#             if metric not in self.data.columns:
#                 self.data[metric] = 0  # Add missing columns with default value (e.g., 0)
        
#         # Fit or load the scaler
#         if os.path.exists(scaler_path):
#             self.scaler = joblib.load(scaler_path)
#             print("Scaler loaded from cache.")
#         else:
#             self.scaler.fit(self.data[self.metrics])  # Fit the scaler on the entire dataset
#             joblib.dump(self.scaler, scaler_path)
#             print("Scaler fitted and saved to cache.")
        
        
#         # Debug feature names
#         print(f"Feature names during fit: {self.scaler.feature_names_in_}")
#         print(f"Feature names during transform: {list(self.data[self.metrics].columns)}")
        
#         # Normalize the data
#         self.data[self.metrics] = self.scaler.transform(self.data[self.metrics])
#         return self.data
#         # # Sort by company and date
#         # self.data = self.data.sort_values(['Company_Name', 'Date'])
        
#         # # Normalize metrics
#         # for metric in self.metrics:
#         #     self.data[metric] = self.scaler.fit_transform(self.data[[metric]])
        
#         # return self.data

#     def create_sequences(self, company_data, metric):
#         X, y = [], []
#         data = company_data[metric].values
#         for i in range(len(data) - self.sequence_length):
#             X.append(data[i:i + self.sequence_length])
#             y.append(data[i + self.sequence_length])
#         return np.array(X), np.array(y)

#     def build_model(self):
        
#         # Optimized using cache
        
#         model = Sequential([
#             LSTM(32, input_shape=(self.sequence_length, 1)),  # Reduced units
#             Dropout(0.1),  # Reduced dropout
#             Dense(16),  # Simplified dense layer
#             Dense(1)
#         ])
#         model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
#         return model
    
#         # model = Sequential([
#         #     LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 1)),
#         #     Dropout(0.2),
#         #     LSTM(50),
#         #     Dropout(0.2),
#         #     Dense(25),
#         #     Dense(1)
#         # ])
#         # model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
#         # return model

    
#     # Optimized train_model function using cache
    
#     def train_model(self, epochs=5, batch_size=16):
#         self.preprocess_data()
#         companies = self.data['Company_Name'].unique()
        
#         for company in companies:
#             company_data = self.data[self.data['Company_Name'] == company]
#             self.models[company] = {}
            
#             for metric in self.metrics:
#                 model_path = os.path.join(self.cache_dir, "lstm_models", f"{company}_{metric}.keras")
#                 if os.path.exists(model_path):
#                     print(f"Loading cached model for {company} - {metric}")
#                     self.models[company][metric] = load_model(model_path)
#                     continue
                
#                 X, y = self.create_sequences(company_data, metric)
#                 if len(X) == 0:
#                     print(f"Skipping {company} - {metric}: Not enough data to create sequences.")
#                     continue
                
#                 X = X.reshape((X.shape[0], X.shape[1], 1))
#                 model = self.build_model()
#                 print(f"Training model for {company} - {metric}...")
#                 model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
#                 model.save(model_path)
#                 print(f"Model saved for {company} - {metric}")
#                 self.models[company][metric] = model

#     # def train_model(self, epochs=10, batch_size=32):
#     #     self.preprocess_data()
#     #     companies = self.data['Company_Name'].unique()
        
#     #     for company in companies:
#     #         company_data = self.data[self.data['Company_Name'] == company]
#     #         self.models[company] = {}
            
#     #         for metric in self.metrics:
#     #             X, y = self.create_sequences(company_data, metric)
#     #             if len(X) == 0:
#     #                 continue
                
#     #             X = X.reshape((X.shape[0], X.shape[1], 1))
#     #             model = self.build_model()
#     #             model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
#     #             self.models[company][metric] = model

#     def predict(self, company_name, periods=6):
#         if company_name not in self.models:
#             raise ValueError(f"No model trained for {company_name}")
        
#         # Load the scaler if not already loaded
#         scaler_path = os.path.join(self.cache_dir, "scalers", "prediction_scaler.pkl")
#         if not hasattr(self, 'scaler') or self.scaler is None:
#             if os.path.exists(scaler_path):
#                 self.scaler = joblib.load(scaler_path)
#                 print("Scaler loaded in predict method.")
#             else:
#                 raise ValueError("Scaler not found. Ensure it is fitted during preprocessing.")
        
#         company_data = self.data[self.data['Company_Name'] == company_name]
#         predictions = {}
        
#         for metric in self.metrics:
#             last_sequence = company_data[metric].values[-self.sequence_length:]
#             pred = []
            
#             for _ in range(periods):
#                 X = last_sequence.reshape((1, self.sequence_length, 1))
#                 next_val = self.models[company_name][metric].predict(X, verbose=0)
#                 pred.append(next_val[0, 0])
#                 last_sequence = np.append(last_sequence[1:], next_val)
            
#             # Inverse transform predictions
#             pred = self.scaler.inverse_transform(np.array(pred).reshape(-1, 1)).flatten()
#             predictions[metric] = pred
#             print(f"Predictions for {company_name} - {metric}: {pred}")
        
#         return predictions

#     def save_predictions(self, output_path):
#         predictions = []
#         companies = self.data['Company_Name'].unique()
        
#         for company in companies:
#             try:
#                 pred = self.predict(company)
#                 for i in range(len(pred[self.metrics[0]])):
#                     predictions.append({
#                         'Company_Name': company,
#                         'Period': i + 1,
#                         **{metric: pred[metric][i] for metric in self.metrics}
#                     })
#             except ValueError as e:
#                 print(f"Skipping company {company} due to error: {e}")
#                 continue
        
#         print(f"Total predictions generated: {len(predictions)}")
#         if predictions:
#             pd.DataFrame(predictions).to_csv(output_path, index=False)
#             print(f"Predictions saved to {output_path}")
#         else:
#             raise ValueError("No predictions generated. Check input data and model configuration.")

# if __name__ == "__main__":
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     model = FinancialPredictionModel(os.path.join(script_dir, "../data/time_series_data_cleaned.csv"))
#     model.train_model()
#     model.save_predictions(os.path.join(script_dir, "../data/predicted_metrics.csv"))
#     print(pd.read_csv(os.path.join(script_dir, "../data/predicted_metrics.csv")).head())



#----------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
import os
import sys

class FinancialPredictionModel:
    def __init__(self, data_path, sequence_length=6, cache_dir="models/cache"):
        self.data = pd.read_csv(data_path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.scaler = MinMaxScaler()
        self.sequence_length = sequence_length
        self.cache_dir = cache_dir
        self.models = {}
        self.metrics = ['EPS', 'P_E', 'Dividend_Yield']
        os.makedirs(os.path.join(cache_dir, "lstm_models"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "scalers"), exist_ok=True)

    def preprocess_data(self):
        # Sort by company and date
        self.data = self.data.sort_values(['Company_Name', 'Date'])
        scaler_dir = os.path.join(self.cache_dir, "scalers")
        os.makedirs(scaler_dir, exist_ok=True)
        
        # Ensure all required metrics are present
        for metric in self.metrics:
            if metric not in self.data.columns:
                self.data[metric] = 0  # Add missing columns with default value (e.g., 0)
            
            scaler_path = os.path.join(scaler_dir, f"{metric}_scaler.pkl")
            scaler = MinMaxScaler()
            
            # Fit or load the scaler for each metric
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                print(f"Scaler for {metric} loaded from cache.")
            else:
                scaler.fit(self.data[[metric]])  # Fit the scaler on the specific metric
                joblib.dump(scaler, scaler_path)
                print(f"Scaler for {metric} fitted and saved to cache.")
            
            # Normalize the metric
            self.data[metric] = scaler.transform(self.data[[metric]])
        
        return self.data

    def create_sequences(self, company_data, metric):
        X, y = [], []
        data = company_data[metric].values
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)

    def build_model(self):
        model = Sequential([
            LSTM(32, input_shape=(self.sequence_length, 1)),  # Reduced units
            Dropout(0.1),  # Reduced dropout
            Dense(16),  # Simplified dense layer
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def train_model(self, epochs=5, batch_size=16):
        self.preprocess_data()
        companies = self.data['Company_Name'].unique()
        
        for company in companies:
            company_data = self.data[self.data['Company_Name'] == company]
            self.models[company] = {}
            
            for metric in self.metrics:
                model_path = os.path.join(self.cache_dir, "lstm_models", f"{company}_{metric}.keras")
                if os.path.exists(model_path):
                    self.models[company][metric] = load_model(model_path)
                    continue
                
                X, y = self.create_sequences(company_data, metric)
                if len(X) == 0:
                    continue
                
                X = X.reshape((X.shape[0], X.shape[1], 1))
                model = self.build_model()
                model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
                model.save(model_path)
                self.models[company][metric] = model

    def predict(self, company_name, periods=6):
        if company_name not in self.models:
            raise ValueError(f"No model trained for {company_name}")
        
        company_data = self.data[self.data['Company_Name'] == company_name]
        predictions = {}
        
        for metric in self.metrics:
            scaler_path = os.path.join(self.cache_dir, "scalers", f"{metric}_scaler.pkl")
            if not os.path.exists(scaler_path):
                raise ValueError(f"Scaler for {metric} not found. Ensure it is fitted during preprocessing.")
            
            scaler = joblib.load(scaler_path)
            last_sequence = company_data[metric].values[-self.sequence_length:]
            pred = []
            
            for _ in range(periods):
                X = last_sequence.reshape((1, self.sequence_length, 1))
                next_val = self.models[company_name][metric].predict(X, verbose=0)
                pred.append(next_val[0, 0])
                last_sequence = np.append(last_sequence[1:], next_val)
            
            # Inverse transform predictions
            pred = scaler.inverse_transform(np.array(pred).reshape(-1, 1)).flatten()
            predictions[metric] = pred
            print(f"Predictions for {company_name} - {metric}: {pred}")
        
        return predictions

    def save_predictions(self, output_path):
        predictions = []
        companies = self.data['Company_Name'].unique()
        
        for company in companies:
            try:
                pred = self.predict(company)
                for i in range(len(pred[self.metrics[0]])):
                    predictions.append({
                        'Company_Name': company,
                        'Period': i + 1,
                        **{metric: pred[metric][i] for metric in self.metrics}
                    })
            except ValueError as e:
                print(f"Skipping company {company} due to error: {e}")
                continue
        
        print(f"Total predictions generated: {len(predictions)}")
        if predictions:
            pd.DataFrame(predictions).to_csv(output_path, index=False)
            print(f"Predictions saved to {output_path}")
        else:
            raise ValueError("No predictions generated. Check input data and model configuration.")