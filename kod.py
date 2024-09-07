# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 19:22:10 2023

@author: AdemTrks
"""
EtYPMDVXsYe7
import matplotlib.pyplot as plt
import numpy as np # bilimsel hesaplamalar için kullanılan bir kütüphanedir. NumPy, çok boyutlu diziler ve matrisler üzerinde yüksek performanslı matematiksel işlemler yapma imkanı sağlar. Ayrıca, rastgele sayı üretimi, lineer cebir, Fourier dönüşümleri gibi birçok matematiksel ve istatistiksel fonksiyonu içerir.
import pandas as pd # Pandas, Python programlama dilinde veri manipülasyonu ve analizi için kullanılan güçlü bir kütüphanedir. Pandas, veri çerçeveleri (dataframes) ve seriler (series) adını verdiği veri yapıları üzerine kuruludur.
import seaborn as sns # Seaborn, Python programlama dilinde veri görselleştirmesi için kullanılan bir yüksek seviyeli bir kütüphanedir. Seaborn, Matplotlib kütüphanesine dayanır ve daha yüksek düzeyde bir arayüz sunarak çekici ve bilgilendirici istatistiksel grafikler oluşturmayı kolaylaştırır.
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold

veri_setim=pd.read_csv("C:\\Users\\AdemTrks\\OneDrive\\Masaüstü\\dataset.csv") #veri kümesini tanıtma

# Tüm boş hücreleri en sık kullanılan değer ile doldurma
for column in veri_setim.columns:
    most_common_value = veri_setim[column].mode()[0] #en sık kullanılan değerleri sırası ile yazdırır
    veri_setim[column].fillna(most_common_value, inplace=True)

veri_setim['Dependents'] = veri_setim['Dependents'].replace('3+', 4) #bakması gereken kişi sayısı
veri_setim['Dependents'].value_counts()


veri_setim.replace({"Loan_Status":{'N':0 ,'Y':1}},inplace=True) # Burda tabloda kredi durumu yes ve no ile yazılmıştı replace komutu ile yes olanları 1 ve no olanları 0 ile değiştiriyoruz
veri_setim.replace({'Married':{'No':0,'Yes':1}, #evlilik durumu : 1 ise evli , 0 ise evli değil
                      'Gender':{'Male':1,'Female':0}, #cinsiyet : 1 ise erkek , 0 ise kadın
                      'Education':{'Graduate':1,'Not Graduate':0}, #Eğitim : mezun ise 1 , mezun değil ise 0
                      'Self_Employed':{'Yes':1,'No':0}, #Serbest Meslek :evet ise 1 , hayır ise 0
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2}},inplace=True) #emlak alanı hakkında: kırsal ise 0 ,yarı kentsel ise 1 ve kentsel ise 2

veri_setim['Loan_Status'].value_counts()  #kredi alanların sayısı
veri_setim['Loan_Status'].value_counts(normalize = True) #kredi alanların oranları

X=veri_setim.drop(columns=['Loan_ID','Loan_Status'],axis=1) 
Y=veri_setim['Loan_Status']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=42)
print(X.shape,X_train.shape,X_test.shape)

# SVM modelini oluşturma
svm_model = SVC(kernel='sigmoid', C=1, gamma='auto') 

# K-Fold (Çapraz Doğrulama) Cross-Validation için ayarlar
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# K-Fold Cross-Validation ile modelin değerlendirilmesi
accuracies = cross_val_score(svm_model, X, Y, cv=kfold, scoring='accuracy')

# Her bir katman için doğruluk değerlerini yazdırma
for i, accuracy in enumerate(accuracies, 1):
    print(f"Değer {i} Doğruluk Puanı: {accuracy}")

# Tüm katmanların ortalamasını yazdırma
print(f"\nOrtalama Doğruluk Puanı: {np.mean(accuracies)}\n")

# Tam modeli eğitme
svm_model.fit(X_train, Y_train)

# Test seti üzerinde tahmin yapma
y_pred = svm_model.predict(X_test)

# Modelin performansını değerlendirme
accuracy = accuracy_score(Y_test, y_pred)
conf_matrix = confusion_matrix(Y_test, y_pred)

print(f"Model Doğruluk Puanı: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
