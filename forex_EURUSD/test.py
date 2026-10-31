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
import ta
import pywt



def createTechInd(self, df):
    df['scaledClose'] = df['Close']

    df['WL'] = 10000 * (df['Close'] - df['Close'].shift(1)) / df['Close']

    df['zeitIndex'] = df.index.hour + df.index.minute / 60

    df['round_lot'] = 10000 * (df['Close'] - df['Close'].round(decimals=2))

    BB = ta.volatility.BollingerBands(df['Close'],
                                      window=10,
                                      window_dev=2)
    df['bb_h'] = (BB.bollinger_hband() - df['Close']) / df['Close']
    df['bb_l'] = (BB.bollinger_lband() - df['Close']) / df['Close']

    MACD = ta.trend.MACD(df['Close'],
                         window_slow=26,
                         window_fast=12,
                         window_sign=9)
    df['macd'] = MACD.macd()
    df['macd_d'] = MACD.macd_diff()

    df['rsi'] = ta.momentum.RSIIndicator(df['Close'], 14).rsi()
    df = df.dropna()

    return df



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
            print('in wavelet: total_energy <= 0', x)
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




def cleanEcoDate(row):
    #
    # apply-aufruf mit axis=0
    # Entfernen der Klammern im Datum
    #
    row = row.split('(')[0]

    return row



def cleanEcoPct(row):
    #
    # apply-aufruf mit axis=0
    # Entfernen der %-Zeichen und Umwandeln nach float
    #
    row = row.replace('.', '')
    row = float(row.strip('%KMB').replace(',', '.'))

    return row



def missingEcoData(row):
    #
    # apply-Aufruf mit axis=1
    # Fehlende Werte im Forecast werden mit dem Actual befüllt
    #
    if pd.isna(row['Forecast'] or row['Forecast'].strip() == ''):
        row['Forecast'] = row['Actual']

    return row



def readAndCleanEcoFeatureFile(path, fn):
    #
    # Liest und bereinigt die Eco Feature Date
    #
    df = pd.read_csv(path + fn,
                     dtype={'Actual': str,
                            'Forecast': str,
                            'Previous': str})
    df.rename({'Release Date': 'Date'}, inplace=True, axis=1)
    #
    # Bereinigen der Daten und Auffüllen fehlender Werte
    df['Date'] = df['Date'].apply(cleanEcoDate)
    if ('Forecast' in df.columns):
        df = df.apply(missingEcoData, axis=1)
        df['Forecast'] = df['Forecast'].apply(cleanEcoPct)
    if ('Previous' in df.columns):
        df['Previous'] = df['Previous'].apply(cleanEcoPct)
    df['Actual'] = df['Actual'].apply(cleanEcoPct)
    #
    # Zusammenfassen von Date und Time
    df['ttime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'].str.strip(), dayfirst=True)
    #
    # Zeit-Index Festlegen und sortieren
    df.set_index(pd.DatetimeIndex(df.ttime, name='time'), inplace=True)
    df.drop(['ttime'], axis=1, inplace=True)
    df = df.sort_index()
    #
    # Umwandeln von Europe/Berlin auf utc - 1 Stunde abziehen
    df.index = df.index.tz_localize('Europe/Berlin', ambiguous='infer').tz_convert('UTC')

    return df



def processNormalEcoFeature(path, df_all, ecoFeatureName, fn, delta):
    #
    # Verarbeitung eines normalen Eco-Features und Anhängen an den DataFrame
    #
    df = readAndCleanEcoFeatureFile(path, fn)
    #
    # Benamung des Eco Features
    df.rename({'Actual': ecoFeatureName}, inplace=True, axis=1)
    #
    # Löschen der ueberfluessigen Spalten
    for s in ['Date', 'Time', 'time', 'Forecast', 'Previous']:
        if (s in df.columns):
            df.drop(s, axis=1, inplace=True)
    #
    # Verarbeitung der Deltas
    dd = [int(i) for i in delta.split(',')]
    if (dd[0] > 0):
        for d in dd:
            delta_name = 'delta' + str(d) + '_' + ecoFeatureName
            df[delta_name] = ((df[ecoFeatureName] \
                               - df[ecoFeatureName].shift(d)).div(df[ecoFeatureName].replace(0, np.nan))).round(3)
            df[delta_name] = df[delta_name].fillna(0)

            df.dropna(inplace=True)
    #
    # Verknüpfen der eco features mit df_all
    df_all = pd.merge_ordered(df_all.reset_index(),
                              df.reset_index(),
                              on='time',
                              how='left',
                              fill_method='ffill').set_index('time')

    return df_all



def processSpreadFeature(path, df_all, ecoFeatureName, fn, delta):
    #
    # Verarbeitung eines Spread Features - spread zwischen zwei unterschiedlichen Eco Feature
    #
    fn1 = fn.split(',')[0].strip()
    fn2 = fn.split(',')[1].strip()
    #
    # Einlesen der beiden Dateien
    df1 = readAndCleanEcoFeatureFile(path, fn1, )
    for s in ['Date', 'Time', 'time', 'Forecast', 'Previous']:
        if (s in df1.columns):
            df1.drop(s, axis=1, inplace=True)

    df2 = readAndCleanEcoFeatureFile(path, fn2)
    for s in ['Date', 'Time', 'time', 'Forecast', 'Previous']:
        if (s in df2.columns):
            df2.drop(s, axis = 1, inplace = True)

    #
    # da nicht alle Daten den gleichen Zeitstempel haben, müssen wir die beiden dataframes mit merge order verbinden
    df = pd.merge_ordered(df1, df2, on='time', how='outer', fill_method='ffill', suffixes=('_1', '_2'))
    #
    # Index auf datetime
    df.set_index(pd.DatetimeIndex(df.time), inplace=True)
    #
    # berechnen des spread
    df[ecoFeatureName] = (df['Actual_1'] - df['Actual_2']).round(3)
    #
    # Bereinigen des Ergebnisses
    df.drop(['time', 'Actual_1', 'Actual_2'], axis=1, inplace=True)
    df.dropna(inplace=True)
    #
    # Verarbeitung der Deltas
    dd = [int(i) for i in delta.split(',')]
    if (dd[0] > 0):
        for d in dd:
            delta_name = 'delta' + str(d) + '_' + ecoFeatureName
            df[delta_name] = ((df[ecoFeatureName] \
                               - df[ecoFeatureName].shift(d)).div(df[ecoFeatureName].replace(0, np.nan))).round(3)
            df[delta_name] = df[delta_name].fillna(0)
            df.dropna(inplace=True)
    #
    # Verknüpfen der eco features mit df_all
    df_all = pd.merge_ordered(df_all.reset_index(),
                              df.reset_index(),
                              on='time',
                              how='left',
                              fill_method='ffill').set_index('time')

    return df_all



def processSchockFeature(path, df_all, ecoSchockName, fn, schockMeanStd):
    #
    # Verarbeitung eines Schock-Features
    #
    fn = fn.strip()
    df = readAndCleanEcoFeatureFile(path, fn)
    #
    # Berechne den Schock und löschen aller unnötigen Spalten
    df['schock'] = ((df['Actual'] - df['Forecast'])).round(3)
    for s in ['Date', 'Time', 'Forecast', 'Previous', 'Actual']:
        if (s in df.columns):
            df.drop(s, axis=1, inplace=True)
    #
    # calculate Mean and STD for standardization
    s_mean = df['schock'].mean()
    s_std = df['schock'].std()
    if (s_std == 0):
        s_std = 1
    df['schock'] = (df['schock'] - s_mean)/s_std
    schockMeanStd.append([fn, s_mean, s_std])
    #
    # Verknüpfen der eco features mit df_all
    df_all = pd.merge_ordered(df_all.reset_index(),
                              df.reset_index(),
                              on='time',
                              how='left').set_index('time')


    #
    # Addieren des Schocks - erst prüfen, ob es den ecoSchock bereits gibt
    if (ecoSchockName not in df_all.columns):
        df_all[ecoSchockName] = 0
    #
    # nach dem merge_ordered haben wir etliche nan-werte, die in den schocks mit 0 befüült werden müssen
    for c in df_all.columns:
        if (c.split('_')[0] == 'schock'):
            df_all[c] = df_all[c].fillna(0)
        else:
            df_all[c] = df_all[c].ffill()

    # für alle restlichen eco-features werden die Lücken mit ffill befüllt



    df_all[ecoSchockName] = (df_all[ecoSchockName] + df_all['schock']).round(3)
    df_all.drop(['schock'], axis=1, inplace=True)

    return df_all



cpath = '/home/uwe/Hope/data/raw/EURUSD/'
ppath = '/home/uwe/Hope/data/processed/EURUSD/'

cfiles = [f for f in listdir(cpath) if isfile(join(cpath, f))]
cfiles.sort()
print(cfiles)
df = pd.DataFrame()
for c in cfiles:
    if (c == 'EURUSD_M5_20230219_20260731.csv'):
        dfirst = False
    else:
        dfirst = True
    dfi = pd.read_csv(cpath + c, parse_dates=[0], dayfirst=dfirst)
    dfi.rename({'Gmt time': 'ttime'}, inplace=True, axis=1)
    dfi = dfi[['ttime', 'Open', 'High', 'Low', 'Close', ]]
    df = pd.concat([df, dfi])
#
# re-initialisieren
df.reset_index(drop=True, inplace=True)
#
# Sicherstellen, dass das Datetime-Format aktiv ist
df['ttime'] = pd.to_datetime(df['ttime'], utc=True,  format='mixed', dayfirst=True)
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

print('Anzahl Rohdaten', len(df))
#
# Löschen der Daten für Freitag ab 22:00 Uhr, Samstag komplett und Sonntage bis 21:00 Uhr
df = df[~((df.index.weekday == 4) & (df.index.hour > 21))]
#
# ... und Sonntag bis 21:00 Uhr
df = df[~((df.index.weekday == 6) & (df.index.hour < 21))]
#
# Korrektur der Winterzeiten (Sonntage ab 22 Uhr - ein kleiner Fehler bleibt)
df = df[~(df.index.month.isin([1, 2, 3, 11, 12]) & (df.index.weekday == 6) & (df.index.hour < 22))]
#
# Korrektur der Sommerzeiten (Freitage bis 21 Uhr - ein kleiner Fehler bleibt)
df = df[~(df.index.month.isin([4, 5, 6, 7, 8, 9, 10]) & (df.index.weekday == 4) & (df.index.hour > 20))]
#
# ... und jetzt verschieben wir die Daten vom Sonntag auf den Freitag
df.index = df.index.where(df.index.weekday != 6, df.index - pd.DateOffset(days=2))

print('Anzahl Daten nach Löschen von Samstag und Transferieren von Sonntag auf Freitag', len(df))
#
# Löschen aller Duplikate, insbesondere beim Übergang von Sommerzeit auf Winterzeit
df = df[~df.index.duplicated(keep='first')]
#
# Auffüllen von fehlenden Werte: aufgrund vom Wechsel von Winter-
# auf Sommerzeit (es fehlt dann 1 Stunde), oder aufgrund von System-
# ausfällen
df = df.sort_index()
df = df.asfreq('5min', method='ffill')
#
# Löschen der Samstage und Sonntage komplett, falls noch etwas übrig ist
print('vor Löschen Sonntag', len(df))
df = df[~(df.index.weekday == 5)]
df = df[(df.index.weekday != 6)]
print('nach Löschen Sonntag', len(df))
# Weihnachten, Sylvester und Neujahr müssen komplett weg, da die Daten i.d.R. nicht vollständig sind
df = df[(df.index.month != 12) | (df.index.day != 24)]
df = df[(df.index.month != 12) | (df.index.day != 25)]
df = df[(df.index.month != 12) | (df.index.day != 31)]
df = df[(df.index.month != 1) | (df.index.day != 1)]
print('Anzahl Tage nach Löschen von Feiertagen', len(df))

#df = calculate_wavelet_features(df, window=128, wavelet="db4", level=4)
#
# Prüfen der Anzahl Tage != 288
df_tag = df.resample("D").size().to_frame(name="anzahl")
df_tag["wochentag"] = df_tag.index.day_name(locale="de_DE.utf8")
print('Anzahl Tage insgesamt:', len(df_tag))
df_tag = df_tag[df_tag['anzahl'] < 288]
print('Anzahl Tage mit weniger als 288 5-min Intervalle', len(df_tag))

df_tag.to_csv(ppath + 'df_tage.csv')

# Prüfen wie viele gleiche Werte pro Tag
df_gleich = pd.DataFrame(index=df.index, columns=["wert", 'IstGleich'])
df_gleich['wert'] = df['Close'] - df['Close'].shift(1)
df_gleich['IstGleich'] = np.where(df_gleich['wert'] == 0, 1, 0)
df_gleich = df_gleich[['IstGleich']].resample('D').sum()
df_gleich["wochentag"] = df_gleich.index.day_name(locale="de_DE.utf8")
print(df_gleich.head())
df_gleich.to_csv(ppath + 'df_gleich.csv')







path = '/home/uwe/Hope/data/processed/EURUSD/kalender/neu/'
fn = 'kalender-features.csv'

df_eco_feature = pd.read_csv(path + fn,
                             dtype={'Feature': str,
                                    'Datei': str,
                                    'Delta': str})

ecoFeatures = df_eco_feature.values.tolist()
print(ecoFeatures)
schockMeanStd = []
for ef in ecoFeatures:
    print(ef)
    if (ef[0].split('_')[0] == 'spread'):
        df = processSpreadFeature(path, df, ef[0].strip(), ef[1].strip(), ef[2].strip())
    elif (ef[0].split('_')[0] == 'schock'):
        df = processSchockFeature(path, df, ef[0].strip(), ef[1].strip(), schockMeanStd)
    else:
        df = processNormalEcoFeature(path, df, ef[0].strip(), ef[1].strip(), ef[2].strip())

start_date = datetime.date(2005, 1, 1)
end_date = datetime.date(2026, 6, 30)

pd.DataFrame(schockMeanStd, columns=['schock', 'mean', 'std']).to_csv(path + 'mean-std-schock.csv')
df = df[(df.index.date >= start_date) & (df.index.date <= end_date)]
df.to_csv(ppath + 'df_all.csv')
