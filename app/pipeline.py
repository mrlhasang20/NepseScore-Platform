import os
import sys

# Adding the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import subprocess
import pandas as pd
from models.scoring_model import CompanyScoringModel
from models.prediction_model import FinancialPredictionModel
from models.combined_scoring import CombinedScoringModel


def run_scoring_model(data_path, output_path):
    print("Running scoring model...")
    try:
        model = CompanyScoringModel(data_path)
        scored_data = model.calculate_scores()
        model.save_results(output_path)
        print(f"Scoring completed. Results saved to {output_path}")
    except Exception as e:
        print(f"Error in scoring model: {e}")
        raise

def run_prediction_model(data_path, output_path):
    print("Running prediction model...")
    try:
        model = FinancialPredictionModel(data_path)
        model.train_model(epochs=10)  # Reduced epochs for faster execution
        model.save_predictions(output_path)
        print(f"Predictions completed. Results saved to {output_path}")
    except Exception as e:
        print(f"Error in prediction model: {e}")
        raise

def run_combined_scoring(current_scores_path, predicted_metrics_path, output_path):
    print("Running combined scoring model...")
    try:
        model = CombinedScoringModel(current_scores_path, predicted_metrics_path)
        final_scores = model.save_final_scores(output_path)
        print(f"Combined scoring completed. Results saved to {output_path}")
    except Exception as e:
        print(f"Error in combined scoring: {e}")
        raise
    

def launch_streamlit_app():
    print("Launching Streamlit app...")
    try:
        subprocess.run(["streamlit", "run", "app/main.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error launching Streamlit: {e}")
        raise

def main():
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define paths relative to the script's location
    scoring_data_path = os.path.join(script_dir, "../data/dummy_company_data.csv")
    scoring_output_path = os.path.join(script_dir, "../data/scored_company_data.csv")
    prediction_data_path = os.path.join(script_dir, "../data/time_series_data_cleaned.csv")
    prediction_output_path = os.path.join(script_dir, "../data/predicted_metrics.csv")
    final_scores_path = os.path.join(script_dir, "../data/final_scores.csv")


    # Verify input files exist
    if not os.path.exists(scoring_data_path):
        raise FileNotFoundError(f"Scoring data not found at {scoring_data_path}")
    if not os.path.exists(prediction_data_path):
        raise FileNotFoundError(f"Prediction data not found at {prediction_data_path}")
    
    # Run scoring model
    run_scoring_model(scoring_data_path, scoring_output_path)
    
    # Run prediction model
    run_prediction_model(prediction_data_path, prediction_output_path)
    
    # Run combined scoring
    run_combined_scoring(scoring_output_path, prediction_output_path, final_scores_path)
    
    # Launch Streamlit app
    launch_streamlit_app()

if __name__ == "__main__":
    main()