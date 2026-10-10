#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:30:10 2021

@author: uwe.mueller
"""

import yfinance as yf
import os


def download_brent_1h_data():
    # Ticker-Symbol für Brent Crude Oil Futures auf Yahoo Finance
    ticker_symbol = "BZ=F"

    print(f"Lade 1-Stunden-Daten für {ticker_symbol} herunter...")

    # Daten abrufen
    # Intervall '1h' zieht stündliche Kerzen.
    # Zeitraum '60d' (maximal möglich bei 1h sind 730d)
    brent_data = yf.download(tickers=ticker_symbol, period="730d", interval="1h")

    if brent_data.empty:
        print("Fehler: Es konnten keine Daten abgerufen werden.")
        return

    # Index (Datum/Uhrzeit) bereinigen, damit er als normale Spalte in der CSV landet
    brent_data = brent_data.reset_index()

    # Dateiname definieren
    csv_filename = "brent_crude_1h_historical.csv"

    # Als CSV speichern
    path = '/home/uwe/Hope/data/raw/'
    brent_data.to_csv(path+csv_filename, index=False)

    print(f"Erfolgreich gespeichert unter: {os.path.abspath(csv_filename)}")
    print("\nErste Zeilen der Daten:")
    print(brent_data.head())


if __name__ == "__main__":
    download_brent_1h_data()
