import pandas as pd
from prophet import Prophet

# Example data
data = pd.DataFrame({
    'ds': ['2023-01-01', '2023-01-02', '2023-01-03'],
    'y': [10, 20, 30]
})

# Convert 'ds' column to datetime
data['ds'] = pd.to_datetime(data['ds'])

# Fit model
model = Prophet()
model.fit(data)

# Predict
print(model.predict(data))
