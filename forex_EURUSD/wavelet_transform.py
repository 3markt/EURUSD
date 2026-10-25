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


def calculate_wavelet_features(
    df: pd.DataFrame,
    window: int = 128,
    wavelet: str = "db4",
    level: int = 4
) -> pd.DataFrame:
    """
    Berechnet kausale Wavelet-Features aus EURUSD-Schlusskursen.

    Parameter
    ---------
    close:
        Zeitreihe der EURUSD-Schlusskurse auf 5-Minuten-Basis.
    window:
        Anzahl vergangener Renditen im rollierenden Fenster.
    wavelet:
        Verwendete Wavelet-Familie.
    level:
        Zerlegungstiefe.

    Rückgabe
    --------
    DataFrame mit fünf normalisierten Wavelet-Energie-Features
    und einem Feature für die Gesamtenergie.

    Es werden ausschließlich Daten bis zum jeweiligen
    Zeitpunkt t verwendet.
    """

    if window < 1:
        raise ValueError("window muss mindestens 1 sein.")

    if level < 1:
        raise ValueError("level muss mindestens 1 sein.")

    close = df['Close']

    # Logarithmische Renditen aus den Schlusskursen
    returns = np.log(close / close.shift(1))

    # Maximale sinnvolle Zerlegungstiefe prüfen
    w = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(window, w.dec_len)

    if level > max_level:
        raise ValueError(
            f"Zerlegungstiefe {level} ist zu hoch. "
            f"Maximal zulässig für window={window}: {max_level}"
        )

    # Namen der Features
    columns = [
        f"wavelet_energy_D{i}"
        for i in range(level, 0, -1)
    ]

    columns += [
        f"wavelet_energy_A{level}",
        "wavelet_total_energy"
    ]

    result = pd.DataFrame(
        np.nan,
        index=close.index,
        columns=columns,
        dtype=float
    )

    # Berechnung für jedes 5-Minuten-Intervall
    return_values = returns.to_numpy()

    for t in range(window, len(close)):

        # Ausschließlich die letzten window Renditen
        # bis einschließlich Zeitpunkt t
        x = return_values[t - window + 1:t + 1]

        if not np.isfinite(x).all():
            continue

        # Diskrete Wavelet-Transformation
        coeffs = pywt.wavedec(
            x,
            wavelet=wavelet,
            mode="symmetric",
            level=level
        )

        # PyWavelets liefert:
        # [cA_level, cD_level, ..., cD2, cD1]

        energies = np.array([
            np.sum(c ** 2)
            for c in coeffs
        ])

        total_energy = energies.sum()

        if total_energy <= 0:
            continue

        # Relative Energie der Detailkomponenten
        # Reihenfolge: D_level, ..., D1
        detail_energies = energies[1:] / total_energy

        # Relative Energie der Approximation
        approximation_energy = energies[0] / total_energy

        # Gesamtenergie der Renditen im Fenster
        total_energy_raw = np.sum(x ** 2)

        values = np.concatenate([
            detail_energies,
            [approximation_energy, total_energy_raw]
        ])

        result.iloc[t, :] = values

    df = pd.concat([df, result], axis=1)

    return df



path = '/home/uwe/Hope/data/raw/'
df = pd.read_csv(path + 'EURUSD_Candlestick_5_m_BID_19.02.2023-08.04.2023.csv')
df = df.sort_index()

df = calculate_wavelet_features(df,
                                window=128,
                                wavelet="db4",
                                level=4)
print(df.head(20))
df.to_csv(path + 'wavelets.csv')

