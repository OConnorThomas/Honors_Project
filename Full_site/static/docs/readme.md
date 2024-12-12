# Financial Analysis Readme

Train an ml model to predict the 1-year performance of a stock based on financial statement ratios.

- Aggregate SEC filings (company data)
- Calculate financial ratios
- Visualize correlations in data
- Train and test a NN and ML

## Dataset provenance

https://www.kaggle.com/datasets/finnhub/reported-financials

Financials as Reported 2010-2020 - SEC Filings
Financial data parsed from 10-Q, 10-Q/A, 10-K, 10-K/A SEC filings from 2010.

Includes data 2009 Q4 - 2022 Q3

Downloaded from repo November 2023.
Dataset looks unchanged as of 12/5/2024

## Neural Network Architecture

Using keras libraries in python:

model = Sequential([
Input(shape=(len(features),)),
Dense(512, activation='relu'),
Dropout(0.3),
Dense(256, activation='relu'),
Dropout(0.3),
Dense(128, activation='relu'),
Dense(64, activation='relu'),
Dense(1, activation='linear')
])

using standard scaler

## APIs

yfinance (yahoo finance http query for python)s
