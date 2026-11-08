#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 15:33:42 2023

@author: uwe.mueller
"""

import numpy as np
import pandas as pd
import random
import pywt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def calculate_wavelet_features(
    df: pd.DataFrame,
    window: int = 128,
    wavelet: str = "db4",
    level: int = 4
) -> pd.DataFrame:
    """
    Berechnet kausale Wavelet-Features aus EURUSD-Schlusskursen.
    Garantiert ohne Look-Ahead Bias für das LSTM-Modell.
    """
    if window < 1:
        raise ValueError("window muss mindestens 1 sein.")
    if level < 1:
        raise ValueError("level muss mindestens 1 sein.")

    close = df['Close']

    # Logarithmische Renditen berechnen
    returns = np.log(close / close.shift(1))
    return_values = returns.to_numpy()

    # Maximale sinnvolle Zerlegungstiefe prüfen
    w = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(window, w.dec_len)

    if level > max_level:
        raise ValueError(
            f"Zerlegungstiefe {level} ist zu hoch. "
            f"Maximal zulässig für window={window}: {max_level}"
        )

    # Spaltennamen explizit definieren (von Level bis 1)
    columns = [f"we{window}_D{i}" for i in range(level, 0, -1)]
    columns += [f"we{window}_A{level}", f"we{window}_total"]

    # Ergebnis-DataFrame vorbereiten
    result = pd.DataFrame(
        np.nan,
        index=close.index,
        columns=columns,
        dtype=float
    )

    # Die Schleife startet erst, wenn genügend valide Renditen vorhanden sind
    # Da return_values[0] = NaN ist, starten wir ab window + 1
    start_idx = window + 1

    for t in range(start_idx, len(close)):
        # Ausschließlich die letzten 'window' Renditen bis Zeitpunkt t
        x = return_values[t - window + 1:t + 1]

        if not np.isfinite(x).all():
            continue

        # Wichtig: mode="periodization" verhindert unsaubere Randeffekte
        # und sorgt für mathematisch stabile Längen der Koeffizientenvektoren
        coeffs = pywt.wavedec(
            x,
            wavelet=wavelet,
            mode="periodization",
            level=level
        )

        # coeffs Struktur: [cA_level, cD_level, cD_level-1, ..., cD_1]
        energies = np.array([np.sum(c ** 2) for c in coeffs])
        total_energy = energies.sum()

        if total_energy <= 0:
            continue

        # Explizite und sichere Trennung:
        approximation_energy = energies[0] / total_energy
        detail_energies = energies[1:] / total_energy  # Enthält [D_level, ..., D1]

        # Rohe Gesamtenergie des aktuellen Fensters (Volatilitätsindikator)
        total_energy_raw = np.sum(x ** 2)

        # Array exakt in der Reihenfolge der Columns zusammenbauen:
        # [D_level, ..., D1, A_level, total]
        values = np.concatenate([
            detail_energies,
            [approximation_energy, total_energy_raw]
        ])

        result.iloc[t, :] = values
    #
    # logarithmische Transformation der total-Komponente
    result[f"we{window}_total"] = -np.log(result[f"we{window}_total"])

    # Zusammenführen mit dem ursprünglichen DataFrame
    return pd.concat([df, result], axis=1)



def calculate_optimized_wavelet_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet ein schlankes, gematchtes Feature-Set aus EURUSD 5-Minuten-Kursen.
    Verhindert Daten-Redundanz und schützt das LSTM vor Overfitting.
    """
    close = df['Close'].to_numpy()
    returns = np.log(df['Close'] / df['Close'].shift(1)).to_numpy()

    # Vorbereitung des Ziel-DataFrames
    feature_dict = {}

    # Schleife über alle Zeitschritte (wir starten ab Index 258 wegen des größten Fensters)
    start_idx = 258
    length = len(df)

    # Arrays für die Rohdaten-Sammlung initialisieren
    # (Wir befüllen Dicts mit Listen und konvertieren am Ende für maximale Performance)
    target_columns = [
        'we32_D1', 'we32_total',
        'we64_D2', 'we64_total',
        'we128_D3', 'we128_D4', 'we128_total',
        'we256_A4', 'we256_total'
    ]
    for col in target_columns:
        feature_dict[col] = np.full(length, np.nan)

    for t in range(start_idx, length):

        # --- WINDOW 32 (Level 1 reicht hier völlig aus für D1) ---
        x32 = returns[t - 32 + 1: t + 1]
        v32 = df['Volume'].to_numpy()[t - 32 + 1: t + 1]
        mean_v32 = np.mean(v32)
        std_v32 = np.std(v32)
        if std_v32 > 0:
            # Wie viele Standardabweichungen weicht das aktuelle Volumen vom 32er-Schnitt ab?
            df.loc[df.index[t], 'v32_zscore'] = (v32[-1] - mean_v32) / std_v32
        else:
            df.loc[df.index[t], 'v32_zscore'] = 0.0

        if np.isfinite(x32).all():
            c32 = pywt.wavedec(x32, 'db4', mode='periodization', level=1)
            e32 = np.array([np.sum(c ** 2) for c in c32])
            sum_e32 = e32.sum()
            if sum_e32 > 0:
                feature_dict['we32_D1'][t] = e32[1] / sum_e32  # cD1 ist bei Level 1 am Index 1
                feature_dict['we32_total'][t] = np.sum(x32 ** 2)

        # --- WINDOW 64 (Level 2 für D2) ---
        x64 = returns[t - 64 + 1: t + 1]
        v64 = df['Volume'].to_numpy()[t - 64 + 1: t + 1]
        mean_v64 = np.mean(v64)
        std_v64 = np.std(v64)
        if std_v64 > 0:
            # Wie viele Standardabweichungen weicht das aktuelle Volumen vom 64er-Schnitt ab?
            df.loc[df.index[t], 'v64_zscore'] = (v64[-1] - mean_v64) / std_v64
        else:
            df.loc[df.index[t], 'v64_zscore'] = 0.0

        if np.isfinite(x64).all():
            c64 = pywt.wavedec(x64, 'db4', mode='periodization', level=2)
            e64 = np.array([np.sum(c ** 2) for c in c64])
            sum_e64 = e64.sum()
            if sum_e64 > 0:
                feature_dict['we64_D2'][t] = e64[1] / sum_e64  # cD2 ist bei Level 2 am Index 1
                feature_dict['we64_total'][t] = np.sum(x64 ** 2)

        # --- WINDOW 128 (Level 4 für D3 und D4) ---
        x128 = returns[t - 128 + 1: t + 1]
        if np.isfinite(x128).all():
            c128 = pywt.wavedec(x128, 'db4', mode='periodization', level=4)
            e128 = np.array([np.sum(c ** 2) for c in c128])
            sum_e128 = e128.sum()
            if sum_e128 > 0:
                feature_dict['we128_D3'][t] = e128[2] / sum_e128  # cD3 ist bei Level 4 an Index 2
                feature_dict['we128_D4'][t] = e128[1] / sum_e128  # cD4 ist bei Level 4 an Index 1
                feature_dict['we128_total'][t] = np.sum(x128 ** 2)

        # --- WINDOW 256 (Level 4 für A4) ---
        x256 = returns[t - 256 + 1: t + 1]
        if np.isfinite(x256).all():
            c256 = pywt.wavedec(x256, 'db4', mode='periodization', level=4)
            e256 = np.array([np.sum(c ** 2) for c in c256])
            sum_e256 = e256.sum()
            if sum_e256 > 0:
                feature_dict['we256_A4'][t] = e256[0] / sum_e256  # cA4 ist immer an Index 0
                feature_dict['we256_total'][t] = np.sum(x256 ** 2)

    # In DataFrame gießen
    features_df = pd.DataFrame(feature_dict, index=df.index)
    return pd.concat([df, features_df], axis=1).dropna()



def scale_total_energies(df):
    total_cols = ['we32_total', 'we64_total', 'we128_total', 'we256_total']

    # 1. Logarithmieren, um die winzigen Werte linear zu strecken
    for col in total_cols:
        df[col] = -np.log(df[col])

    # 2. MinMaxScaler anwenden (Train fitten -> Test transformieren)
    for col in total_cols:
        scaler = MinMaxScaler(feature_range=(0, 1))
        df[col] = scaler.fit_transform(df[[col]])

    return df



path = '/home/uwe/Hope/data/raw/'
df = pd.read_csv(path + 'EURUSD/EURUSD_M5_20220102_20230218.csv')
df.rename({'Gmt time': 'ttime'}, inplace=True, axis=1)
df.reset_index(drop=True, inplace=True)
df['ttime'] = pd.to_datetime(df['ttime'], utc=True,  format='mixed', dayfirst=True)
df = df.sort_values(by='ttime').reset_index(drop=True)
df.set_index(pd.DatetimeIndex(df.ttime, name='time'), inplace=True)
df.drop('ttime', inplace=True, axis=1)

"""
df = calculate_wavelet_features(df,
                                window=32,
                                wavelet="db4",
                                level=2)

df = calculate_wavelet_features(df,
                                window=64,
                                wavelet="db4",
                                level=3)

df = calculate_wavelet_features(df,
                                window=128,
                                wavelet="db4",
                                level=4)

df = calculate_wavelet_features(df,
                                window=256,
                                wavelet="db4",
                                level=4)
"""


df = calculate_optimized_wavelet_features(df)
df = scale_total_energies(df)
print(df.head(20))
df.to_csv(path + 'wavelets.csv')

