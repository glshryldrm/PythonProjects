import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#%%

# CSV dosyasını yükle
train_data = pd.read_csv('Training.csv')
test_data = pd.read_csv('Testing.csv')

# Eksik verileri kontrol et (kontrol amaçlı)
print("Eksik değerler:\n", train_data.isnull().sum())
print("Eksik değerler:\n", test_data.isnull().sum())

X_train = train_data.iloc[:, :-1]  # Tüm satırlar, son sütun hariç tüm sütunlar (features)
y_train = train_data.iloc[:, -1]   # Son sütun (target)

X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]
#%%
scaler_standard = StandardScaler()
X_train_standard = scaler_standard.fit_transform(X_train)
X_test_standard = scaler_standard.transform(X_test)

#%%

# Karar Ağacı Regresyonu modeli oluşturma ve eğitme
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Performans metrikleri
mse = mean_squared_error(y_test, y_pred_dt)
r2 = r2_score(y_test, y_pred_dt)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Gerçek ve tahmin edilen değerlerin karşılaştırılması
plt.scatter(y_test, y_pred_dt)
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.title('Gerçek vs Tahmin Edilen Değerler')
plt.show()
#%%

# Rastgele Orman Regresyonu modeli oluşturma ve eğitme
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)


# Performans metrikleri
mse = mean_squared_error(y_test, y_pred_rf)
r2 = r2_score(y_test, y_pred_rf)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Gerçek ve tahmin edilen değerlerin karşılaştırılması
plt.scatter(y_test, y_pred_rf)
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.title('Gerçek vs Tahmin Edilen Değerler')
plt.show()