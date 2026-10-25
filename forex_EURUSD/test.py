#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""
import numpy as np
import pandas as pd
from os import listdir, makedirs
from os.path import isfile, join, isdir
import datetime
import pywt

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
            print('in wavelet: x is not finit', x)
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
            print('in wavelet: total_energy <= 0', total_energy)
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


cpath = '/home/uwe/Hope/data/raw/EURUSD/'
ppath = '/home/uwe/Hope/data/processed/EURUSD/'

cfiles = [f for f in listdir(cpath) if isfile(join(cpath, f))]
cfiles.sort()

df = pd.DataFrame()
for c in cfiles:
    dfi = pd.read_csv(cpath + c, dayfirst=True, parse_dates=[0])
    dfi.rename({'Gmt time': 'ttime'}, inplace=True, axis=1)
    dfi = dfi[['ttime', 'Open', 'High', 'Low', 'Close', ]]
    df = pd.concat([df, dfi])
#
# re-initialisieren
df.reset_index(drop=True, inplace=True)
#
# Sicherstellen, dass das Datetime-Format aktiv ist
df['ttime'] = pd.to_datetime(df['ttime'])
#
# und vorsichtshalber nochmals sortieren
df = df.sort_values(by='ttime').reset_index(drop=True)
#
# Addiere 5 Minuten zur Zeit, um das Label auf die rechte Seite zu bekommen
df['ttime'] = df['ttime'] + pd.Timedelta(minutes=5)
#
# und jetzt der Index auf datetime
df.set_index(pd.DatetimeIndex(df.ttime, name='time'), inplace=True)
df.drop('ttime', inplace=True, axis=1)

print(len(df))
# Löschen der Daten für Freitag ab 22:00 Uhr, Samstag komplett und Sonntage bis 21:00 Uhr
#df[df.index.weekday == 6] = df[(df.index.weekday == 6) & (df.index.time > datetime.time(21,0))]
df = df[~(((df.index.weekday == 4) & (df.index.hour > 21))  \
          | (df.index.weekday == 5) \
          | ((df.index.weekday == 6) & (df.index.hour < 21)) \
          )]
# Korrektur der Winterzeiten (Sonntage ab 22 Uhr)
df = df[~(df.index.month.isin([1, 2, 3, 11, 12]) & \
          (df.index.weekday == 6) & (df.index.hour < 22))]
# Korrektur der Sommerzeiten (Freitage bis 21 Uhr)
df = df[~(df.index.month.isin([4, 5, 6, 7, 8, 9, 10]) & \
          (df.index.weekday == 4) & (df.index.hour > 20))]
# ... und jetzt verschieben wir die Daten vom Sonntag auf den Freitag
df.index = df.index.where(df.index.weekday != 6, df.index - pd.DateOffset(days=2))
print(len(df))
df = df[~df.index.duplicated(keep='first')]
#
# Auffüllen von fehlenden Werte: aufgrund vom Wechsel von Winter-
# auf Sommerzeit (es fehlt dann 1 Stunde), oder aufgrund von System-
# ausfällen
#df = df.asfreq('5min', method='ffill')
#
# Löschen der Samstage und Sonntage
df = df[(df.index.weekday != 5)]
df = df[(df.index.weekday != 6)]
# Weihnachten, Sylvester und Neujahr müssen weg
df = df[(df.index.month != 12) | (df.index.day != 24)]
df = df[(df.index.month != 12) | (df.index.day != 25)]
df = df[(df.index.month != 12) | (df.index.day != 31)]
df = df[(df.index.month != 1) | (df.index.day != 1)]
print(len(df))
df = df.sort_index()
df = df.loc['2005-01-02 00:00:00':'2024-12-31 23:55:00']

#df = calculate_wavelet_features(df, window=128, wavelet="db4", level=4)

dff = df.groupby(df.index.date).count()
print('Anzahl Tage insgesamt:', len(dff))
dff = dff[dff['Close'] < 288]
print(len(dff))

dff.to_csv(ppath + 'df_tage.csv')

df_check =  df.loc['2025-01-13':'2025-01-17']
df_check.to_csv(ppath + 'df_check.csv')
