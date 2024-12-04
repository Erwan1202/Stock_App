import pandas as pd
from prophet import Prophet
import yfinance as yf
import datetime as dt

# Définir la période de téléchargement des données (7 derniers jours)
start_date = dt.datetime.now() - dt.timedelta(days=7)
end_date = dt.datetime.now()

# Fonction pour charger les données
def load_data():
    # Télécharger les données de Bitcoin
    data = yf.download('BTC-USD', start=start_date, end=end_date)
    
    # Réinitialiser l'index et renommer les colonnes pour Prophet
    data.reset_index(inplace=True)
    data = data[['Date', 'Close']]
    data.columns = ['ds', 'y']  # Prophet attend les colonnes 'ds' et 'y'
    
    print(data.head())
    return data

# Charger les données
data = load_data()

# Créer et entraîner le modèle Prophet
model = Prophet(daily_seasonality=True)
model.fit(data)

# Faire des prédictions (ajusté pour la fréquence journalière)
future = model.make_future_dataframe(data, periods=7)  # Prédictions pour 7 jours
forecast = model.predict(future)

# Afficher les résultats
model.plot(forecast)
model.plot_components(forecast)
