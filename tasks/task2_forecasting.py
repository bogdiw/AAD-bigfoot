# -*- coding: utf-8 -*-
"""
Task 2 - Time Series Forecasting: Raportări per lună
Owner: Membru 2

Obiectiv: Predicția numărului de raportări Bigfoot pentru următoarele 12 luni.
Modele utilizate: ARIMA, Prophet, Linear Regression.

Input:
  data/reports.csv (sau data/reports_augmented.csv)
Output:
  output/forecasting/*.png (Descompunere, ACF/PACF, Comparări modele)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
import warnings

warnings.filterwarnings('ignore')

# Setup directoare
OUTPUT_DIR = 'output/forecasting'
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")

print("=" * 70)
print("TASK 2: TIME SERIES FORECASTING - BIGFOOT REPORTS")
print("=" * 70)

# ============================================================================
# 2.1 Pregătirea Datelor
# ============================================================================
print("\n2.1 PREGĂTIREA DATELOR")

# Încercăm să folosim datele augmentate dacă există, altfel cele originale
data_path = 'data/reports_augmented.csv' if os.path.exists('data/reports_augmented.csv') else 'data/reports.csv'
df_raw = pd.read_csv(data_path)

# Curățare și parsare date (Submitted Date)
df_raw['date'] = pd.to_datetime(df_raw['Submitted Date'], errors='coerce')
df_raw = df_raw.dropna(subset=['date'])

# Filtrare interval 1990 - 2019 conform cerinței
df_filtered = df_raw[(df_raw['date'].dt.year >= 1990) & (df_raw['date'].dt.year <= 2019)].copy()

# Agregare pe lună
ts = df_filtered.resample('ME', on='date').size().to_frame(name='reports')

# Asigurăm continuitatea seriei (completăm lunile lipsă cu 0 dacă există)
all_months = pd.date_range(start='1990-01-01', end='2019-12-31', freq='ME')
ts = ts.reindex(all_months).fillna(0)

print(f"Serie temporală creată: {len(ts)} luni (Ian 1990 - Dec 2019)")

# ============================================================================
# 2.2 Descompunere și Analiză (EDA)
# ============================================================================
print("\n2.2 DESCOMPUNERE ȘI ANALIZĂ ACF/PACF")

# Descompunere: Trend + Sezonalitate + Reziduu
decomposition = seasonal_decompose(ts['reports'], model='additive', period=12)
fig = decomposition.plot()
fig.set_size_inches(12, 8)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_decomposition.png')
plt.close()

# Grafice ACF și PACF pentru ARIMA
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
plot_acf(ts['reports'], lags=40, ax=ax1)
plot_pacf(ts['reports'], lags=40, ax=ax2)
plt.savefig(f'{OUTPUT_DIR}/02_acf_pacf.png')
plt.close()

# ============================================================================
# 2.3 Train/Test Split (Ultimele 24 luni = test)
# ============================================================================
train = ts.iloc[:-24]
test = ts.iloc[-24:]
print(f"Split: Train = {len(train)} luni, Test = {len(test)} luni")

# ============================================================================
# 2.4 Modelare
# ============================================================================
results = {}

# --- Model 1: SARIMA (Sezonal ARIMA) ---
print("\nAntrenare Model 1: SARIMA...")
# Parametrii (p,d,q) x (P,D,Q,s) aleși pe baza ACF/PACF și experimentare
model_sarima = SARIMAX(train['reports'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_fit = model_sarima.fit(disp=False)
results['ARIMA'] = sarima_fit.get_forecast(steps=24).predicted_mean

# --- Model 2: Prophet ---
print("Antrenare Model 2: Prophet...")
prophet_df = train.reset_index().rename(columns={'index': 'ds', 'reports': 'y'})
m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
m.fit(prophet_df)
future = m.make_future_dataframe(periods=24, freq='ME')
forecast_prophet = m.predict(future)
results['Prophet'] = forecast_prophet.iloc[-24:]['yhat'].values

# --- Model 3: Linear Regression (cu elemente ciclice) ---
print("Antrenare Model 3: Linear Regression...")
def create_features(df_ts):
    df_feat = df_ts.copy().reset_index()
    df_feat['month'] = df_feat['index'].dt.month
    df_feat['year'] = df_feat['index'].dt.year
    df_feat['time_idx'] = np.arange(len(df_feat))
    # Caracteristici ciclice pentru lună
    df_feat['sin_month'] = np.sin(2 * np.pi * df_feat['month'] / 12)
    df_feat['cos_month'] = np.cos(2 * np.pi * df_feat['month'] / 12)
    return df_feat[['time_idx', 'sin_month', 'cos_month']]

X_train = create_features(train)
y_train = train['reports'].values
X_test = create_features(test)
X_test['time_idx'] += len(train) # Continuăm indexul de timp

lr = LinearRegression()
lr.fit(X_train, y_train)
results['LinearRegression'] = lr.predict(X_test)

# ============================================================================
# 2.5 Evaluare și Comparare
# ============================================================================
print("\n2.5 EVALUARE METRICI (RMSE, MAE)")

metrics_rows = []
plt.figure(figsize=(14, 7))
plt.plot(ts.index[-48:], ts['reports'].iloc[-48:], label='Actual (Last 4 years)', color='black', linewidth=2)

for name, pred in results.items():
    rmse = np.sqrt(mean_squared_error(test['reports'], pred))
    mae = mean_absolute_error(test['reports'], pred)
    metrics_rows.append({'Model': name, 'RMSE': round(rmse, 2), 'MAE': round(mae, 2)})
    plt.plot(test.index, pred, label=f'Pred: {name} (RMSE: {round(rmse, 2)})', linestyle='--')

plt.title('Comparare Modele: Predicții vs Realitate (Ultimele 24 luni)')
plt.legend()
plt.savefig(f'{OUTPUT_DIR}/03_forecast_compare.png')
plt.close()

metrics_df = pd.DataFrame(metrics_rows).sort_values('RMSE')
print("\nRezultate finale:")
print(metrics_df.to_string(index=False))

# ============================================================================
# 2.6 Forecast Viitor (Următoarele 12 luni)
# ============================================================================
print("\n2.6 GENERARE FORECAST VIITOR (2020)")

# Folosim Prophet pentru forecastul final (de obicei cel mai robust la sezonalitate)
m_final = Prophet(yearly_seasonality=True)
m_final.fit(ts.reset_index().rename(columns={'index': 'ds', 'reports': 'y'}))
future_12 = m_final.make_future_dataframe(periods=12, freq='ME')
forecast_final = m_final.predict(future_12)

fig_final = m_final.plot(forecast_final)
plt.title('Bigfoot Reports Forecast: 12 Luni Viitoare (Prophet)')
plt.savefig(f'{OUTPUT_DIR}/04_final_forecast_12m.png')
plt.close()

print(f"\nTask 2 complet. Graficele au fost salvate în '{OUTPUT_DIR}/'.")