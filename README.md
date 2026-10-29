# LSTM Stock Price Prediction

A time-series forecasting project that explores Long Short-Term Memory (LSTM) networks for predicting NVIDIA (NVDA) closing prices. It compares custom and built-in LSTM implementations in PyTorch and TensorFlow/Keras and investigates the effect of moving-average smoothing on prediction performance.

## Overview

The project uses historical stock-market data to train recurrent neural networks on sequences of past market observations. Model predictions are evaluated against actual closing prices, with experiments examining training behavior, prediction errors, and the effects of smoothing.

## Experiments

### 1. Custom LSTM with PyTorch
- Implementation of a multi-layer LSTM using `torch.nn.LSTM`
- Sliding-window time-series preparation
- Model training with the Adam optimizer and mean squared error loss
- Training-loss visualization

### 2. TensorFlow/Keras Comparison
- Built-in LSTM layers with a dense output layer
- Comparison with the custom PyTorch implementation
- Evaluation using Root Mean Squared Error (RMSE)
- Visualization of predictions from both frameworks

### 3. Moving-Average Smoothing
- Application of a rolling mean to the closing-price series
- Training an LSTM using the smoothed target
- Comparison of predicted prices with actual, unsmoothed closing prices

## Methodology

1. Load and inspect historical NVDA stock data.
2. Normalize numerical features using MinMaxScaler.
3. Create input sequences using a sliding window of 60 time steps.
4. Train LSTM models to predict the closing price.
5. Transform predictions back to the original price scale.
6. Evaluate prediction errors using RMSE.
7. Investigate how moving-average smoothing affects forecasting results.

## Technologies

- Python
- PyTorch
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Dataset

The notebook expects an `NVDA.csv` file containing historical NVIDIA stock data, including a date column (`<DATE>`) and a closing-price column (`<CLOSE>`).

The dataset is not included in this repository and must be obtained separately.

## Results

The notebook reports training and test RMSE and visualizes predictions against actual closing prices. The custom PyTorch and TensorFlow/Keras implementations produced similar results in the current experiments.

The denoising experiment investigates whether smoothing the target series affects forecasting performance. Results should be interpreted using the same unsmoothed test prices for both the original and smoothed-target models.

