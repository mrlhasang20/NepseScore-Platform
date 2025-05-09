import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class CompanyScoringModel:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.scaler = MinMaxScaler()
        self.sector_weights = {
            'Commercial Bank': {'EPS': 0.15, 'P_E': 0.1, 'P_B': 0.1, 'Dividend_Yield': 0.15, 'ROA': 0.15, 'ROE': 0.15, 'Debt_to_Equity': 0.1, 'Price_Momentum': 0.05, 'Volume_Trends': 0.05, 'Ownership_Pattern': 0.05, 'Regulatory_Red_Flags': -0.2},
            'Hydropower': {'EPS': 0.1, 'P_E': 0.1, 'P_B': 0.1, 'Dividend_Yield': 0.1, 'ROA': 0.1, 'ROE': 0.1, 'Debt_to_Equity': 0.15, 'Price_Momentum': 0.1, 'Volume_Trends': 0.05, 'Ownership_Pattern': 0.05, 'Regulatory_Red_Flags': -0.2},
            'Hotel and Tourism': {'EPS': 0.1, 'P_E': 0.15, 'P_B': 0.15, 'Dividend_Yield': 0.1, 'ROA': 0.1, 'ROE': 0.1, 'Debt_to_Equity': 0.1, 'Price_Momentum': 0.1, 'Volume_Trends': 0.05, 'Ownership_Pattern': 0.05, 'Regulatory_Red_Flags': -0.2},
            'Corporate Debentures': {'EPS': 0.05, 'P_E': 0.05, 'P_B': 0.05, 'Dividend_Yield': 0.2, 'ROA': 0.15, 'ROE': 0.15, 'Debt_to_Equity': 0.15, 'Price_Momentum': 0.05, 'Volume_Trends': 0.05, 'Ownership_Pattern': 0.05, 'Regulatory_Red_Flags': -0.2},
            'Finance': {'EPS': 0.15, 'P_E': 0.1, 'P_B': 0.1, 'Dividend_Yield': 0.1, 'ROA': 0.15, 'ROE': 0.15, 'Debt_to_Equity': 0.1, 'Price_Momentum': 0.05, 'Volume_Trends': 0.05, 'Ownership_Pattern': 0.05, 'Regulatory_Red_Flags': -0.2},
            'Government Bonds': {'EPS': 0.05, 'P_E': 0.05, 'P_B': 0.05, 'Dividend_Yield': 0.25, 'ROA': 0.15, 'ROE': 0.15, 'Debt_to_Equity': 0.1, 'Price_Momentum': 0.05, 'Volume_Trends': 0.05, 'Ownership_Pattern': 0.05, 'Regulatory_Red_Flags': -0.2},
            'Investment': {'EPS': 0.1, 'P_E': 0.15, 'P_B': 0.15, 'Dividend_Yield': 0.1, 'ROA': 0.1, 'ROE': 0.1, 'Debt_to_Equity': 0.1, 'Price_Momentum': 0.1, 'Volume_Trends': 0.05, 'Ownership_Pattern': 0.05, 'Regulatory_Red_Flags': -0.2},
            'Life Insurance': {'EPS': 0.15, 'P_E': 0.1, 'P_B': 0.1, 'Dividend_Yield': 0.15, 'ROA': 0.15, 'ROE': 0.15, 'Debt_to_Equity': 0.1, 'Price_Momentum': 0.05, 'Volume_Trends': 0.05, 'Ownership_Pattern': 0.05, 'Regulatory_Red_Flags': -0.2},
            'Trading': {'EPS': 0.1, 'P_E': 0.15, 'P_B': 0.15, 'Dividend_Yield': 0.1, 'ROA': 0.1, 'ROE': 0.1, 'Debt_to_Equity': 0.1, 'Price_Momentum': 0.1, 'Volume_Trends': 0.05, 'Ownership_Pattern': 0.05, 'Regulatory_Red_Flags': -0.2},
            'Manufacturing': {'EPS': 0.1, 'P_E': 0.1, 'P_B': 0.1, 'Dividend_Yield': 0.1, 'ROA': 0.1, 'ROE': 0.1, 'Debt_to_Equity': 0.15, 'Price_Momentum': 0.1, 'Volume_Trends': 0.05, 'Ownership_Pattern': 0.05, 'Regulatory_Red_Flags': -0.2}
        }

    def preprocess_data(self):
        # Handle missing values
        self.data.fillna(self.data.mean(numeric_only=True), inplace=True)
        
        # Normalize numerical features
        numerical_cols = ['EPS', 'P_E', 'P_B', 'Dividend_Yield', 'ROA', 'ROE', 'Debt_to_Equity', 'Price_Momentum', 'Volume_Trends', 'Ownership_Pattern']
        self.data[numerical_cols] = self.scaler.fit_transform(self.data[numerical_cols])
        
        # Convert Regulatory_Red_Flags to binary
        self.data['Regulatory_Red_Flags'] = self.data['Regulatory_Red_Flags'].astype(int)
        
        return self.data

    def calculate_scores(self):
        self.preprocess_data()
        scores = []
        
        for _, row in self.data.iterrows():
            sector = row['Sector']
            weights = self.sector_weights[sector]
            score = 0
            
            # Calculate weighted score
            for metric in weights:
                if metric == 'Regulatory_Red_Flags':
                    score += weights[metric] * row[metric]  # Negative weight for red flags
                else:
                    score += weights[metric] * row[metric]
            
            # Ensure score is between 0 and 100
            score = max(0, min(100, score * 100))
            scores.append(score)
        
        self.data['Score'] = scores
        return self.data

    def save_results(self, output_path):
        self.data.to_csv(output_path, index=False)

if __name__ == "__main__":
    model = CompanyScoringModel("../data/dummy_company_data.csv")
    scored_data = model.calculate_scores()
    model.save_results("../data/scored_company_data.csv")
    print(scored_data[['Company_Name', 'Sector', 'Score']])