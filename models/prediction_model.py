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
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class FinancialPredictionModel:
    def __init__(self, data_path, sequence_length=12):
        self.data = pd.read_csv(data_path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.scaler = MinMaxScaler()
        self.sequence_length = sequence_length
        self.models = {}
        self.metrics = ['EPS', 'P_E', 'Dividend_Yield']

    def preprocess_data(self):
        # Sort by company and date
        self.data = self.data.sort_values(['Company_Name', 'Date'])
        
        # Normalize metrics
        for metric in self.metrics:
            self.data[metric] = self.scaler.fit_transform(self.data[[metric]])
        
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
            LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def train_model(self, epochs=10, batch_size=32):
        self.preprocess_data()
        companies = self.data['Company_Name'].unique()
        
        for company in companies:
            company_data = self.data[self.data['Company_Name'] == company]
            self.models[company] = {}
            
            for metric in self.metrics:
                X, y = self.create_sequences(company_data, metric)
                if len(X) == 0:
                    continue
                
                X = X.reshape((X.shape[0], X.shape[1], 1))
                model = self.build_model()
                model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
                self.models[company][metric] = model

    def predict(self, company_name, periods=6):
        if company_name not in self.models:
            raise ValueError(f"No model trained for {company_name}")
        
        company_data = self.data[self.data['Company_Name'] == company_name]
        predictions = {}
        
        for metric in self.metrics:
            last_sequence = company_data[metric].values[-self.sequence_length:]
            pred = []
            
            for _ in range(periods):
                X = last_sequence.reshape((1, self.sequence_length, 1))
                next_val = self.models[company_name][metric].predict(X, verbose=0)
                pred.append(next_val[0, 0])
                last_sequence = np.append(last_sequence[1:], next_val)
            
            # Inverse transform predictions
            pred = self.scaler.inverse_transform(np.array(pred).reshape(-1, 1)).flatten()
            predictions[metric] = pred
        
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
            except ValueError:
                continue
        
        pd.DataFrame(predictions).to_csv(output_path, index=False)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model = FinancialPredictionModel(os.path.join(script_dir, "../data/time_series_data_cleaned.csv"))
    model.train_model()
    model.save_predictions(os.path.join(script_dir, "../data/predicted_metrics.csv"))
    print(pd.read_csv(os.path.join(script_dir, "../data/predicted_metrics.csv")).head())

