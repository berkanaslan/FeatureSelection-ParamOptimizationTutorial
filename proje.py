# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 09:16:46 2020

@author: Berkan
"""

#Veri setini uygulamaya dahil ettim.
import pandas as pd
fileName = 'grup1.xlsx'
colNames=['id','cycle','setting1','setting2','setting3','s1','s2','s3'
          's4','s5','s6','s7','s8','s9','s10','s11','s12',
          's13','s14','s15','s16','s17','s18','s19','s20','s21','ttf']
data = pd.read_excel(fileName, name=colNames)
peek = data.head(5)
print(peek)

#Veri setinin özniteliklerinin değişken türünü görüntüledim.
types = data.dtypes
print(types)

#Veri seti özniteliklerinin ortalama, standart sapma gibi istatistiklerini görüntüledim.
from pandas import set_option
set_option('display.width', 125)
set_option('precision',3)
description = data.describe()
print(description)

#Veri setinde eksik veri olup olmadığını kontrol ettim.
data.isnull().sum()

#Verinin histogramına baktım. Adı: Figure_1.png
import matplotlib.pyplot as plt
data.hist(figsize=(30,30))
plt.show()

#Veri setinin scatter matrixini görüntüledim. Adı: Figure_2.png
from pandas.plotting import scatter_matrix
scatter_matrix(data,figsize=(30,40))
plt.show()

#Veri setinin korelasyonunu görüntüledim. Adı: Figure_3.png
import seaborn as sns
import matplotlib.pyplot as plt
j = data.corr()
f, ax = plt.subplots(figsize=(19,19))
sns.heatmap(j, annot=True, linewidths=.5, ax=ax)

#Düşük varyans gösteren öznitelikleri çıkardım ve öznitelik detaylarını yazdırdım.
dropList = ['setting3', 's5', 's10', 's16', 's18', 's19']
data = data.drop(dropList, axis=1)
print(data.info())

#Yeni özniteliklerle birlikte dataFrame oluşturdum.
newColNames=['id','cycle','setting1','setting2','s1','s2','s3','s4','s6','s7',
             's8','s9','s11','s12','s13','s14','s15','s17','s20','s21','ttf']
veri = pd.DataFrame(data)
veri.columns = newColNames
print(veri.head())

#Çıktı değişkenini farklı bir değişkene atadım.
veri = data.values
X = veri[:,0:20]
Y = veri[:,20]
print(X.shape)
print(Y.shape)

#Eğitim - Test parçalaması yaptım.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.3, random_state = 0)

#Normalizasyon işlemi
from sklearn.preprocessing import MinMaxScaler
minMaxScaler = MinMaxScaler()
minMaxScaler.fit(x_train)
x_train_norm = minMaxScaler.transform(x_train)
x_test_norm = minMaxScaler.transform(x_test)


# Modelleme işlemleri

# Gerekli kütüphaneler
import numpy
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

# KNN Modellemesi: Ham
print('\n') 
print('1.1. KNeighborsRegressor:')
print('-')

knn_model_ham =  KNeighborsRegressor()
knn_model_ham_fit = knn_model_ham.fit(x_train_norm, y_train)

tahmin_train = knn_model_ham_fit.predict(x_train_norm)
tahmin_test = knn_model_ham_fit.predict(x_test_norm)

print("KNeighborsRegressor: Test Sonucu:")
print('KNN R² (Eğitim): %.4f' % r2_score(y_train, tahmin_train))
print('KNN MSE (Eğitim): %.4f' % mean_squared_error(y_train, tahmin_train))
print('KNN R² (Test): %.4f' % r2_score(y_test, tahmin_test))
print('KNN MSE (Test): %.4f' % mean_squared_error(y_test, tahmin_test))

# Gradient Boosting  Modellemesi: Ham
print('\n') 
print('1.2. GradientBoostingRegressor:')
print('-')

gbr_model_ham =  GradientBoostingRegressor()
gbr_model_ham_fit = gbr_model_ham.fit(x_train_norm, y_train)

tahmin_train_gbr = gbr_model_ham_fit.predict(x_train_norm)
tahmin_test_gbr = gbr_model_ham_fit.predict(x_test_norm)

print("GradientBoostingRegressor: Test Sonucu:")
print('GradientBoosting R² (Eğitim): %.4f' % r2_score(y_train, tahmin_train_gbr))
print('GradientBoosting MSE (Eğitim): %.4f' % mean_squared_error(y_train, tahmin_train_gbr))
print('GradientBoosting R² (Test): %.4f' % r2_score(y_test, tahmin_test_gbr))
print('GradientBoosting MSE (Test): %.4f' % mean_squared_error(y_test, tahmin_test_gbr))

# Random Forest Modellemesi: Ham
print('\n') 
print('1.3. RandomForestRegressor:')
print('-')

rfg_model_ham =  RandomForestRegressor()
rfg_model_ham_fit = rfg_model_ham.fit(x_train_norm, y_train)

tahmin_train_rfg = rfg_model_ham_fit.predict(x_train_norm)
tahmin_test_rfg = rfg_model_ham_fit.predict(x_test_norm)

print("RandomForestRegressor: Test Sonucu:")
print('RandomForestRegressor R² (Eğitim): %.4f' % r2_score(y_train, tahmin_train_rfg))
print('RandomForestRegressor MSE (Eğitim): %.4f' % mean_squared_error(y_train, tahmin_train_rfg))
print('RandomForestRegressor R² (Test): %.4f' % r2_score(y_test, tahmin_test_rfg))
print('RandomForestRegressor MSE (Test): %.4f' % mean_squared_error(y_test, tahmin_test_rfg))

print('\n')
print('-')
print("2. adım: Öznitelik seçme ile modelleme işlemleri:")
print('-')

#SelectKBest: Öznitelik seçimi:
select_feature = SelectKBest(chi2, k=15).fit(x_train_norm, y_train)

# SelectKBest ile gelen değerleri değişkenlere atadım.
x_train_k = select_feature.transform(x_train_norm)
x_test_k = select_feature.transform(x_test_norm)

#Feature Selection sonrası modelleme işlemleri: 1. KNeighborsRegressor
print('\n') 
print('2.1. KNeighborsRegressor: Feature Selection')
print('-')

knn_model =  KNeighborsRegressor()
knn_model_fit = knn_model.fit(x_train_k, y_train)

tahmin_train = knn_model.predict(x_train_k)
tahmin_test = knn_model.predict(x_test_k)

print("KNeighborsRegressor: Feature Selection: Test sonucu:")
print('KNN R² (Eğitim): %.4f' % r2_score(y_train, tahmin_train))
print('KNN MSE (Eğitim): %.4f' % mean_squared_error(y_train, tahmin_train))
print('KNN R² (Test): %.4f' % r2_score(y_test, tahmin_test))
print('KNN MSE (Test): %.4f' % mean_squared_error(y_test, tahmin_test))

#Feature Selection sonrası modelleme işlemleri: 2. GradientBoosting
print('\n') 
print('2.2. GradientBoostingRegressor: Feature Selection')
print('-')

gbr_model =  GradientBoostingRegressor()
gbr_model_fit = gbr_model.fit(x_train_k, y_train)

tahmin_train = gbr_model.predict(x_train_k)
tahmin_test = gbr_model.predict(x_test_k)

print("GradientBoostingRegressor: Feature Selection: Test Sonucu:")
print('Gradient Boosting R² (Eğitim): %.4f' % r2_score(y_train, tahmin_train))
print('Gradient Boosting MSE (Eğitim): %.4f' % mean_squared_error(y_train, tahmin_train))
print('Gradient Boosting R² (Test): %.4f' % r2_score(y_test, tahmin_test))
print('Gradient Boosting MSE (Test): %.4f' % mean_squared_error(y_test, tahmin_test))

#Feature Selection sonrası modelleme işlemleri: 3. RandomForestRegressor
print('\n') 
print('2.3. RandomForestRegressor:')
print('-')

rfg_model =  RandomForestRegressor()
rfg_model_fit = rfg_model.fit(x_train_k, y_train)

tahmin_train_rfg = rfg_model.predict(x_train_k)
tahmin_test_rfg = rfg_model.predict(x_test_k)

print("RandomForestRegressor: Feature Selection: Test Sonucu:")
print('RandomForestRegressor R² (Eğitim): %.4f' % r2_score(y_train, tahmin_train_rfg))
print('RandomForestRegressor MSE (Eğitim): %.4f' % mean_squared_error(y_train, tahmin_train_rfg))
print('RandomForestRegressor R² (Test): %.4f' % r2_score(y_test, tahmin_test_rfg))
print('RandomForestRegressor MSE (Test): %.4f' % mean_squared_error(y_test, tahmin_test_rfg))

#Parametre optimizasyonu ile modelleme işlemleri: 1. KNeighborsRegressor
print('\n')
print('3.1. KNeighborsRegressor: Parametre Optimizasyonu')
print('-')

model = KNeighborsRegressor()

# Parametre alternetifleri:
k_values = numpy.array([1,3,5,7,9,11,13,15,17,19,21])

param_grid = dict(n_neighbors=k_values)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=None)
grid_result = grid.fit(x_train_k, y_train)

parameter = grid_result.best_params_
print("En iyi KNN %f ile %s" % (grid_result.best_score_, grid_result.best_params_))
print('\n') 

print('Diğer sonuçlar:')
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) : %r" % (mean, stdev, param))

model = KNeighborsRegressor(**parameter)
model.fit(x_train_k, y_train)
tahmin_train = model.predict(x_train_k)
tahmin_test = model.predict(x_test_k)

print('\n')
print('KNeighborsRegressor: Parametre Optimizasyonu: Test Sonucu:')
print('KNN R² (Eğitim): %.4f' % r2_score(y_train, tahmin_train))
print('KNN MSE (Eğitim): %.4f' % mean_squared_error(y_train, tahmin_train))
print('KNN R² (Test): %.4f' % r2_score(y_test, tahmin_test))
print('KNN MSE (Test): %.4f' % mean_squared_error(y_test, tahmin_test))

#Parametre optimizasyonu ile modelleme işlemleri: 2. GradientBoostingRegressor
print('\n') 
print('3.2. GradientBoostingRegressor: Parametre Optimizasyonu')
print('-')

model = GradientBoostingRegressor()

#Parametre alternetifleri:
param_grid = {'learning_rate': [0.01,0.02,0.03],
                  'n_estimators' : [100,500,1000],
                  'max_depth'    : [4,6,8] 
                 }

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=None)
grid_result = grid.fit(x_train_k, y_train)

parameter = grid_result.best_params_
print("En iyi GradientBoosting %f ile %s" % (grid_result.best_score_, grid_result.best_params_))
print('\n') 

print('Diğer sonuçlar:')
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) : %r" % (mean, stdev, param))

model = GradientBoostingRegressor(**parameter)
model.fit(x_train_k, y_train)
tahmin_train = model.predict(x_train_k)
tahmin_test = model.predict(x_test_k)

print('\n')
print("GradientBoostingRegressor: Parametre Optimizasyonu: Test Sonucu:")
print('Gradient Boosting R² (Eğitim): %.4f' % r2_score(y_train, tahmin_train))
print('Gradient Boosting MSE (Eğitim): %.4f' % mean_squared_error(y_train, tahmin_train))
print('Gradient Boosting R² (Test): %.4f' % r2_score(y_test, tahmin_test))
print('Gradient Boosting MSE (Test): %.4f' % mean_squared_error(y_test, tahmin_test))

print('\n')
#Parametre optimizasyonu ile modelleme işlemleri: 3. RandomForestRegressor
print('3.3. RandomForestRegressor: Parametre Optimizasyonu')
print('-')
print('\n')

model = RandomForestRegressor()

#Parametre alternetifleri:
param_grid = {'n_estimators': [100,200,300,400],
              'max_depth': [5,10,15,20]
                 }

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=None)
grid_result = grid.fit(x_train_k, y_train)

parameter = grid_result.best_params_
print("En iyi RandomForestRegressor %f ile %s" % (grid_result.best_score_, grid_result.best_params_))
print('\n') 
print('Diğer sonuçlar:')
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) : %r" % (mean, stdev, param))

model = RandomForestRegressor(**parameter)
model.fit(x_train_k, y_train)

tahmin_train = model.predict(x_train_k)
tahmin_test = model.predict(x_test_k)

print('\n') 
print('RandomForestRegressor: Parametre Optimizasyonu: Test Sonucu:')
print('RandomForestRegressor R² (Eğitim): %.4f' % r2_score(y_train, tahmin_train))
print('RandomForestRegressor MSE (Eğitim): %.4f' % mean_squared_error(y_train, tahmin_train))
print('RandomForestRegressor R² (Test): %.4f' % r2_score(y_test, tahmin_test))
print('RandomForestRegressor MSE (Test): %.4f' % mean_squared_error(y_test, tahmin_test))

print("\n \n")
print("Berkan ASLAN")
print("20174703014")

