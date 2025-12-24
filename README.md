# Hysteresis_Research

This repository contains reproducible machine learning pipelines for classifying
magnetic hysteresis curve shapes using:

- Long Short-Term Memory (LSTM) networks
- One-dimensional Convolutional Neural Networks (1-D CNN)
- An optimized 1-D CNN variant

All models are compatible with **Python ≥ 3.10** and are intended to accompany
the associated research manuscript.

---

## Repository Structure

```text
.
├── src_lstm/                 # LSTM model pipeline
├── src_cnn/                  # 1-D CNN pipeline
├── src_cnn_optimized/        # Optimized 1-D CNN pipeline
├── data/                     # Extracted dataset (see Data Availability)
├── results_lstm/             # Generated automatically
├── results_cnn/              # Generated automatically
├── README.md
├── requirements.txt
