# NepseScore Platform

NepseScore Platform is an AI-powered analytics and prediction system for evaluating and ranking companies on the Nepal Stock Exchange (NEPSE). **This project currently operates on dummy (simulated) data for demonstration and research purposes.** Built for data scientists, analysts, and investors, it leverages state-of-the-art machine learning (ML) and deep learning (DL) models—particularly LSTM neural networks—for robust, data-driven scoring, forecasting, and interactive visualization. The platform efficiently processes simulated financial time series data, performing end-to-end feature engineering, normalization, and weighted scoring.

## Key AI & Data Science Features

- **AI-Powered Scoring Model:**  
  Calculates company scores using sector-specific, ML-calculated weights on normalized features (EPS, P/E, P/B, Dividend Yield, ROA, ROE, Debt-to-Equity, etc.), with automated handling of missing values and regulatory red flags.
- **Deep Learning Prediction Model:**  
  Employs LSTM neural networks (TensorFlow/Keras) for multivariate financial forecasting (EPS, P/E, Dividend Yield) per company. Each LSTM is trained for up to 10 epochs (configurable), using a 12-month sliding window for robust temporal prediction.
- **Combined Scoring (Ensemble):**  
  Merges current scores and predicted metrics using tunable ensemble weights to deliver a comprehensive, forward-looking score for each company.
- **Data Preprocessing:**  
  Automates handling of missing data, normalization (via MinMaxScaler), and caching for efficient, reproducible processing across simulated records.
- **Automated Pipeline Execution:**  
  Orchestrates the entire AI/ML workflow end-to-end: from company scoring, through deep learning model training and prediction, to final combined scoring and dashboard visualization.
- **Interactive ML Dashboard:**  
  Visualizes model outputs and predictions in real-time with Streamlit and Plotly, allowing dynamic filtering by sector, company, and score.

## Technical Overview

### Data Volume & Training

- **Training Data:**  
  - Operates on dummy (simulated) datasets with company-month records for demonstration.
  - Cleans, normalizes and prepares features for both ML and DL models.
- **Model Training:**  
  - LSTM models are trained per company/metric for 10 epochs (adjustable).
  - Sequence length (time steps) is set to 12 (annual cycles).
  - Batch size of 32 (configurable).
  - Models skip companies with insufficient data to ensure stability.

## Features (At a Glance)

- **Scoring Model:** Calculates scores for companies based on current (simulated) financial metrics.
- **Prediction Model:** Uses LSTM models to predict future (simulated) financial metrics for companies.
- **Combined Scoring:** Merges current scores and predicted metrics to generate final scores.
- **Data Preprocessing:** Handles missing data, normalization, and caching for efficient processing.
- **Pipeline Execution:** Automates the entire workflow from scoring to prediction and final scoring.

## Directory Structure

```
.
├── app/
│   ├── main.py          # Interactive Streamlit dashboard
│   └── pipeline.py      # Runs full AI/ML pipeline and launches dashboard
├── data/
│   ├── dummy_company_data.csv         # Example input data for scoring (simulated)
│   ├── scored_company_data.csv        # AI-calculated current scores
│   ├── predicted_metrics.csv          # DL-predicted metrics
│   ├── final_scores.csv               # Combined final scores
│   ├── time_series_data.csv           # Raw historical input (simulated)
│   └── time_series_data_cleaned.csv   # Cleaned time series data
├── models/
│   ├── scoring_model.py               # ML scoring module
│   ├── prediction_model.py            # LSTM DL prediction module
│   └── combined_scoring.py            # Ensemble scoring module
├── notebooks/
│   ├── data_exploration.ipynb         # EDA and feature engineering
│   ├── prediction_exploration.ipynb   # Model training/evaluation
│   └── *.png                          # Visualizations
├── requirements.txt                   # Python dependencies
└── README.md
```

## Getting Started

### Requirements

Install all dependencies (including ML/DL libraries):
```bash
pip install -r requirements.txt
```

### Usage

1. Place the provided dummy/simulated input data in the `data/` directory.
2. Run the complete AI/ML pipeline (scoring, training, prediction, ensemble, dashboard):
   ```bash
   python app/pipeline.py
   ```
3. Or launch the dashboard directly (if output files already exist):
   ```bash
   streamlit run app/main.py
   ```

## Example Visualizations

- AI-generated correlation heatmaps, feature distributions, and LSTM prediction plots are available in the `notebooks/` directory.

## Dependencies

```
streamlit
pandas
plotly
scikit-learn
numpy
tensorflow
```

## License

MIT License.

---

**Note:** This project currently operates on dummy data for research and educational purposes only. For real-world or production use, integration with authentic NEPSE datasets and further validation of AI/ML models are required.
