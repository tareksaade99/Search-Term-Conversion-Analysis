# Conversion Prediction with Attention-Based LSTM Model

## Project Overview
This project implements a deep learning model to predict the number of conversions (`AllConversions`) for search terms and associated numerical features. The model combines LSTM networks with attention mechanisms to analyze both textual (search terms) and numerical data. Attention scores are used to interpret feature and token importance, while predictions are visualized and compared with actual values.

## Dataset
- Source: CSV file (`data-big.csv`) containing columns like `SearchTerm`, `Date`, `Impressions`, `Interactions`, `CTR`, and `AllConversions`.
- Preprocessing:
  - Dates were converted to day, month, and day-of-week features.
  - CTR normalized from percentages to decimals.
  - Search terms tokenized and padded to fixed sequence lengths.
  - Numerical features scaled using MinMaxScaler.
  - Zero-conversion rows were downsampled to handle class imbalance.

## Model Architecture
- **Search Term Input:**
  - Embedding layer for tokenized search terms.
  - LSTM layer with attention mechanism.
- **Numerical Features Input:**
  - LSTM layer for numerical sequences.
  - Separate attention mechanism.
- **Fusion:**
  - Outputs of search term and numerical LSTMs concatenated.
  - Fully connected layer to predict `AllConversions`.

- **Loss & Optimization:**  
  - Loss: Mean Squared Error (MSE)  
  - Optimizer: Adam  

## Training
- Epochs: 1000  
- Batch size: 32  
- Both inputs (search term and numerical features) used simultaneously.  
- Training history tracked for loss visualization.

## Attention Analysis
- Average attention weights computed per sample for search terms and numerical features.
- Visualization:
  - Attention vs actual conversions over time.
  - Importance distribution for numerical features.
  - Top words and bigrams identified based on attention scores and average conversions.

## Key Insights
- **Feature Importance:**
  - Search terms contribute significantly (~X%) to model predictions.
  - Numerical features (impressions, interactions, CTR, date info) also play a crucial role.
- **Top Predictive Keywords & Bigrams:**
  - Single-word and bigram analysis highlights terms with the highest average conversions.
- **Interpretability:** Attention mechanisms allow identifying which words or features drive predictions.

## Visualizations
- Training loss over epochs.
- Actual vs predicted conversions by date.
- Shifted attention vs conversions over time.
- Average attention per token and numerical feature.
- Top 10 words and bigrams by attention or average conversions.

## Usage
1. Load the dataset CSV.
2. Run preprocessing and tokenization.
3. Train the model with both search term and numerical features.
4. Extract attention scores for interpretability.
5. Visualize results and analyze top contributing features.

## Dependencies
- Python 3.x
- pandas, numpy, matplotlib
- TensorFlow / Keras
- scikit-learn

