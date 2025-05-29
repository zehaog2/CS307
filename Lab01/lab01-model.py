import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump


weather_train = pd.read_csv(
    "https://cs307.org/lab-01/data/weather-train.csv",
    index_col="date",
    parse_dates=True
)
weather_vtrain = pd.read_csv(
    "https://cs307.org/lab-01/data/weather-vtrain.csv",
    index_col="date",
    parse_dates=True
) # try different models on different validtion data, find best models, then use that model on the validation-train data
weather_validation = pd.read_csv(
    "https://cs307.org/lab-01/data/weather-validation.csv",
    index_col="date",
    parse_dates=True
)
# data from 2016-2022
X_train = weather_train[["year", "day_of_year"]]
y_train = weather_train["temperature_2m_min"] 
#data from 2016-2020
X_vtrain = weather_vtrain[["year", "day_of_year"]]
y_vtrain = weather_vtrain["temperature_2m_min"]
# data from 2021-2022
X_validation = weather_validation[["year", "day_of_year"]]
y_validation = weather_validation["temperature_2m_min"]
def plot_validation_curve(X_vtrain, y_vtrain, X_validation, y_validation): # plotting RMSE after KNN trained on validation-train data
    neighbor_range = range(1, 100)  # Neighbor range adjustments
    validation_rmse = []
    for n in neighbor_range:
        model = KNeighborsRegressor(n_neighbors=n)
        model.fit(X_vtrain, y_vtrain)
        y_validation_pred = model.predict(X_validation)
        rmse = np.sqrt(mean_squared_error(y_validation, y_validation_pred))
        validation_rmse.append(rmse)


    plt.figure(figsize=(8, 5))
    plt.plot(neighbor_range, validation_rmse, label='Validation RMSE')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('RMSE')
    plt.title('KNN Validation Curve')
    plt.legend()
    plt.show()
plot_validation_curve(X_vtrain, y_vtrain, X_validation, y_validation)


knn_model = KNeighborsRegressor(n_neighbors=24)

knn_model.fit(X_vtrain, y_vtrain) # fit model on validation train data
y_predict_knn = knn_model.predict(X_validation) # predict our model on validation data
rmse_knn = np.sqrt(mean_squared_error(y_validation, y_predict_knn))
knn_model.fit(X_train, y_train) # fit model on train data

print("RMSE of KNN model:", rmse_knn)