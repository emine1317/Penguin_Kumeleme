import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Veri setini yükleme ve inceleme
df = pd.read_csv("C:/Users/Casper/Desktop/Makine_Ogrenmesi/penguins/penguins.csv")
# Eksik değerleri düşürme
df = df.dropna()
df = df[(df["flipper_length_mm"] > 0) & (df["flipper_length_mm"] < 4000)]

# ANFIS modeli için giriş ve çıkışları tanımlama
inputs = df[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']]
target = df['sex']

# Bulanık mantık değişkenlerini tanımlama
culmen_length = ctrl.Antecedent(np.arange(30, 60, 1), 'culmen_length')
culmen_depth = ctrl.Antecedent(np.arange(15, 25, 1), 'culmen_depth')
flipper_length = ctrl.Antecedent(np.arange(170, 220, 1), 'flipper_length')
body_mass = ctrl.Antecedent(np.arange(3000, 5000, 100), 'body_mass')
sex = ctrl.Consequent(np.arange(0, 3, 1), 'sex')

# Giriş ve çıkış değişkenleri için evreni tanımlama
culmen_length['short'] = fuzz.trimf(culmen_length.universe, [30, 30, 45])
culmen_length['medium'] = fuzz.trimf(culmen_length.universe, [35, 45, 55])
culmen_length['long'] = fuzz.trimf(culmen_length.universe, [45, 60, 60])

culmen_depth['shallow'] = fuzz.trimf(culmen_depth.universe, [15, 15, 20])
culmen_depth['medium'] = fuzz.trimf(culmen_depth.universe, [17, 20, 23])
culmen_depth['deep'] = fuzz.trimf(culmen_depth.universe, [20, 25, 25])

flipper_length['short'] = fuzz.trimf(flipper_length.universe, [170, 170, 190])
flipper_length['medium'] = fuzz.trimf(flipper_length.universe, [180, 190, 200])
flipper_length['long'] = fuzz.trimf(flipper_length.universe, [190, 220, 220])

body_mass['light'] = fuzz.trimf(body_mass.universe, [3000, 3000, 3500])
body_mass['medium'] = fuzz.trimf(body_mass.universe, [3250, 3750, 4250])
body_mass['heavy'] = fuzz.trimf(body_mass.universe, [4000, 5000, 5000])

sex['female'] = fuzz.trimf(sex.universe, [0, 0, 1])
sex['male'] = fuzz.trimf(sex.universe, [1, 2, 2])

# Kuralları tanımlama
rule1 = ctrl.Rule(culmen_length['short'] | culmen_depth['shallow'] | flipper_length['short'] | body_mass['light'], sex['female'])
rule2 = ctrl.Rule(culmen_length['long'] | culmen_depth['deep'] | flipper_length['long'] | body_mass['heavy'], sex['male'])

# Bulanık mantık sistemi oluşturma
system = ctrl.ControlSystem([rule1, rule2])
anfis = ctrl.ControlSystemSimulation(system)

# Gerçek etiketler ve model tahminlerini toplu olarak saklamak için listeler oluşturma
predictions = []
true_labels = []

# Modeli eğitme
for index, row in inputs.iterrows():
    anfis.input['culmen_length'] = row['culmen_length_mm']
    anfis.input['culmen_depth'] = row['culmen_depth_mm']
    anfis.input['flipper_length'] = row['flipper_length_mm']
    anfis.input['body_mass'] = row['body_mass_g']
    anfis.compute()

    # Çıktıların bulanık küme değerlerini alma
    sex_female = fuzz.interp_membership(sex.universe, sex['female'].mf, anfis.output['sex'])
    sex_male = fuzz.interp_membership(sex.universe, sex['male'].mf, anfis.output['sex'])

    # Tahmin edilen sınıfı belirleme
    predicted_class = 1 if sex_male > sex_female else 0

    # Gerçek etiketlerle karşılaştırma yapma
    true_class = 1 if target[index] == 'male' else 0

    # Modelin tahminini 'predictions' listesine ekleme
    predictions.append(predicted_class)

    # Gerçek etiketi 'true_labels' listesine ekleme
    true_labels.append(true_class)

    # Sonucu yazdırma
    print("Predicted sex:", "male" if predicted_class == 1 else "female", "Actual sex:", target[index])

# Doğruluğu hesaplama
accuracy = accuracy_score(true_labels, predictions)
print("Accuracy:", accuracy)

#----------------------------------------------------------------------------------------
'''
# Grafik oluşturma
plt.figure(figsize=(8, 5))
plt.bar(['Accuracy'], [accuracy], color=['blue'])
plt.ylim(0, 1)  # Y eksenini 0 ile 1 arasında sınırla
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.show()

# Tahmin edilen cinsiyetlerin dağılımını hesaplama
predicted_counts = np.bincount(predictions)
labels = ['Female', 'Male']

# Pasta grafiği oluşturma
plt.figure(figsize=(8, 5))
plt.pie(predicted_counts, labels=labels, autopct='%1.1f%%', colors=['lightcoral', 'lightskyblue'])
plt.title('Predicted Gender Distribution')
plt.axis('equal')  # Dairesel pasta grafiği yapmak için
plt.show()

# Gerçek cinsiyetlerin dağılımını hesaplama
true_counts = np.bincount(true_labels)

# Tahmin edilen ve gerçek cinsiyetlerin dağılımını içeren bir DataFrame oluşturma
gender_distribution = pd.DataFrame({'Predicted': predicted_counts, 'True': true_counts}, index=labels)

# Gruplanmış bar grafiği oluşturma
gender_distribution.plot(kind='bar', figsize=(10, 6), color=['lightcoral', 'lightskyblue'])
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(rotation=0)  # x eksenindeki etiketlerin döndürülmesini engelleme
plt.legend(title='Labels')
plt.show()
'''
# Giriş değişkenlerinin üyelik fonksiyonlarını görselleştirme
culmen_length.view()
culmen_depth.view()
flipper_length.view()
body_mass.view()

# Çıkış değişkeninin üyelik fonksiyonlarını görselleştirme
sex.view()
