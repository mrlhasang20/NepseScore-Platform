import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from models.scoring_model import CompanyScoringModel

class CombinedScoringModel:
    def __init__(self, current_scores_path, predicted_metrics_path, current_weight=0.6, predicted_weight=0.4):
        self.current_scores = pd.read_csv(current_scores_path)
        self.predicted_metrics = pd.read_csv(predicted_metrics_path)
        self.current_weight = current_weight
        self.predicted_weight = predicted_weight
        self.scaler = MinMaxScaler()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.sector_weights = CompanyScoringModel(os.path.join(script_dir, "../data/dummy_company_data.csv")).sector_weights

    def preprocess_predicted_metrics(self):
        # Aggregate predicted metrics (e.g., average over 6 months)
        predicted_avg = self.predicted_metrics.groupby('Company_Name').agg({
            'EPS': 'mean',
            'P_E': 'mean',
            'Dividend_Yield': 'mean'
        }).reset_index()
        
        # Merge with sector information from current scores
        predicted_avg = predicted_avg.merge(
            self.current_scores[['Company_Name', 'Sector']],
            on='Company_Name',
            how='left'
        )
        
        # Normalize numerical metrics
        numerical_cols = ['EPS', 'P_E', 'Dividend_Yield']
        predicted_avg[numerical_cols] = self.scaler.fit_transform(predicted_avg[numerical_cols])
        
        return predicted_avg

    def calculate_predicted_scores(self):
        predicted_data = self.preprocess_predicted_metrics()
        scores = []
        
        for _, row in predicted_data.iterrows():
            sector = row['Sector']
            weights = self.sector_weights.get(sector, self.sector_weights['Commercial Bank'])  # Fallback to default
            score = 0
            
            # Calculate weighted score for predicted metrics
            for metric in ['EPS', 'P_E', 'Dividend_Yield']:
                score += weights[metric] * row[metric]
            
            # Scale score to 0-100
            score = max(0, min(100, score * 100))
            scores.append(score)
        
        predicted_data['Predicted_Score'] = scores
        return predicted_data[['Company_Name', 'Sector', 'Predicted_Score']]

    def combine_scores(self):
        predicted_scores = self.calculate_predicted_scores()
        
        # Merge current and predicted scores
        combined = self.current_scores[['Company_Name', 'Sector', 'Score']].merge(
            predicted_scores[['Company_Name', 'Predicted_Score']],
            on='Company_Name',
            how='left'
        )
        
        # Fill missing predicted scores with current score (if any)
        combined['Predicted_Score'].fillna(combined['Score'], inplace=True)
        
        # Calculate final score
        combined['Final_Score'] = (
            self.current_weight * combined['Score'] +
            self.predicted_weight * combined['Predicted_Score']
        )
        
        return combined

    def save_final_scores(self, output_path):
        final_scores = self.combine_scores()
        final_scores.to_csv(output_path, index=False)
        return final_scores

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    model = CombinedScoringModel(
        current_scores_path=os.path.join(script_dir, "../data/scored_company_data_cleaned.csv"),
        predicted_metrics_path=os.path.join(script_dir, "../data/predicted_metrics.csv")
    )
    final_scores = model.save_final_scores(os.path.join(script_dir, "../data/final_scores.csv"))
    print(final_scores[['Company_Name', 'Sector', 'Score', 'Predicted_Score', 'Final_Score']])