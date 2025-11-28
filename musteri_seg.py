# -*- coding: windows-1254 -*-

# Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift, AffinityPropagation
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import estimate_bandwidth
from scipy.cluster.hierarchy import linkage, dendrogram
from mpl_toolkits.mplot3d import Axes3D
import warnings
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv(r"C:\Users\Lenovo\Desktop\Musteri_Segmentasyon\Mall_Customers.csv")
df.head()

df.shape

df.rename(columns={"Genre":"Gender"}, inplace=True)

df.info()
df.describe()
df.isnull().sum()
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
sns.boxplot(data=df, y="Annual Income (k$)")

plt.subplot(1,2,2)
sns.boxplot(data=df, y="Spending Score (1-100)")

plt.show()
plt.figure(figsize=(10,6))
sns.set_style('darkgrid')

sns.distplot(df.Age)
plt.title("Distribution of AGE\n=================================================================", fontsize=20, color="green")
plt.xlabel("Age Range", fontsize=15)
plt.ylabel("Density", fontsize=15)

plt.show()

plt.figure(figsize=(10,6))
sns.set_style('darkgrid')

sns.distplot(df["Annual Income (k$)"])
plt.title("Distribution of Annual Income (k$)\n=================================================================", fontsize=20, color="green")
plt.xlabel("Annual Income (k$)", fontsize=15)
plt.ylabel("Density", fontsize=15)
plt.show()

plt.figure(figsize=(10,6))
sns.set_style('darkgrid')

sns.distplot(df["Spending Score (1-100)"])
plt.title("Distribution of Spending Score (1-100)\n=================================================================", fontsize=20, color="green")
plt.xlabel("Spending Score (1-100)", fontsize=15)
plt.ylabel("Density", fontsize=15)
plt.show()

plt.figure(figsize=(7,5))
sns.set_style('darkgrid')

plt.title("Distribution Gender\n==========================================", fontsize=20, color="green")
plt.xlabel("Gender", fontsize=15)
plt.ylabel("Count", fontsize=15)
sns.countplot(df.Gender, palette="nipy_spectral_r")
plt.show()

plt.figure(figsize=(10,6))
sns.set_style('darkgrid')

sns.scatterplot(data=df, x="Age", y= "Annual Income (k$)", hue="Gender", s=60)
plt.title("Age VS Annual Income (k$)\n=================================================================", fontsize=20, color="green")
plt.xlabel("Age", fontsize=15)
plt.ylabel("Annual Income (k$)", fontsize=15)
plt.show()

Age_18_25 = df.Age[(df.Age>=18) & (df.Age<=25)]
Age_26_35 = df.Age[(df.Age>=26) & (df.Age<=35)]
Age_36_45 = df.Age[(df.Age>=36) & (df.Age<=45)]
Age_46_55 = df.Age[(df.Age>=46) & (df.Age<=55)]
Age_55_Above = df.Age[(df.Age>=56)]

x = ["18-25","26-35","36-45","46-55","55 Above"]
y = [len(Age_18_25.values),len(Age_26_35.values),len(Age_36_45.values),len(Age_46_55.values),len(Age_55_Above.values)]

plt.figure(figsize=(10,6))
sns.barplot(x=x, y=y, palette="nipy_spectral_r")
plt.title("Customer's Age Barplot\n=================================================================", fontsize=20, color="green")
plt.xlabel("Age", fontsize=15)
plt.ylabel("Number of Customers", fontsize=15)
plt.show()



ss1_20 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 1) & (df["Spending Score (1-100)"] <= 20)]
ss21_40 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 21) & (df["Spending Score (1-100)"] <= 40)]
ss41_60 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 41) & (df["Spending Score (1-100)"] <= 60)]
ss61_80 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 61) & (df["Spending Score (1-100)"] <= 80)]
ss81_100 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 81) & (df["Spending Score (1-100)"] <= 100)]

score_x = ["1-20", "21-40", "41-60", "61-80", "81-100"]
score_y = [len(ss1_20.values), len(ss21_40.values), len(ss41_60.values), len(ss61_80.values), len(ss81_100.values)]

plt.figure(figsize=(10,6))
sns.barplot(x=score_x, y=score_y,palette="nipy_spectral_r")
plt.title("Spending Scores\n=================================================================", fontsize=20, color="green")
plt.xlabel("Score", fontsize=15)
plt.ylabel("Number of Customers", fontsize=15)
plt.show()
ai0_30 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 0) & (df["Annual Income (k$)"] <= 30)]
ai31_60 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 31) & (df["Annual Income (k$)"] <= 60)]
ai61_90 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 61) & (df["Annual Income (k$)"] <= 90)]
ai91_120 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 91) & (df["Annual Income (k$)"] <= 120)]
ai121_150 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 121) & (df["Annual Income (k$)"] <= 150)]

income_x = ["$0 - 30,000", "$30,001 - 60,000", "$60,001 - 90,000", "$90,001 - 120,000", "$120,001 - 150,000"]
income_y = [len(ai0_30.values), len(ai31_60.values), len(ai61_90.values), len(ai91_120.values), len(ai121_150.values)]

plt.figure(figsize=(15,6))
sns.barplot(x=income_x, y=income_y, palette="nipy_spectral_r")
plt.title("Annual Incomes\n=================================================================", fontsize=20, color="green")
plt.xlabel("Income", fontsize=15)
plt.ylabel("Number of Customer", fontsize=15)
plt.show()

df_scaled = df[["Age","Annual Income (k$)","Spending Score (1-100)"]]

# Class instance
scaler = StandardScaler()

# Fit_transform
df_scaled_fit = scaler.fit_transform(df_scaled)
df_scaled_fit = pd.DataFrame(df_scaled_fit)
df_scaled_fit.columns = ["Age","Annual Income (k$)","Spending Score (1-100)"]
df_scaled_fit.head()
var_list = df_scaled_fit[["Annual Income (k$)","Spending Score (1-100)"]]

kmeans = KMeans(n_clusters=4, max_iter=50)
kmeans.fit(var_list)

kmeans.labels_
ssd = []

for num_clusters in range(1,11):
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(var_list)
    
    ssd.append(kmeans.inertia_)

plt.figure(figsize=(12,6))

plt.plot(range(1,11), ssd, linewidth=2, color="red", marker ="8")
plt.title("Elbow Curve\n=================================================================", fontsize=20, color="green")
plt.xlabel("K Value")
plt.xticks(np.arange(1,11,1))
plt.ylabel("SSD")

plt.show()

kmeans = KMeans(n_clusters=5, max_iter=50)
kmeans.fit(var_list)
kmeans.labels_
df["Label"] = kmeans.labels_
df.head()
plt.figure(figsize=(10,6))

plt.title("Ploting the data into 5 clusters\n=================================================================", fontsize=20, color="green")
sns.scatterplot(data=df, x="Annual Income (k$)", y="Spending Score (1-100)", hue="Label", s=60, palette=['green','orange','brown','blue','red'])
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x='Label', y='Annual Income (k$)', data=df, palette="nipy_spectral_r")
plt.title("Label Wise Customer's Income\n===============================================================", fontsize=20, color="green")
plt.xlabel(xlabel="Label", fontsize=15)
plt.ylabel(ylabel="Annual Income (k$)",fontsize=15)
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x='Label', y='Spending Score (1-100)', data=df, palette="nipy_spectral_r")
plt.title("Label Wise Spending Score\n===============================================================", fontsize=20, color="green")
plt.xlabel(xlabel="Label", fontsize=15)
plt.ylabel(ylabel="Spending Score",fontsize=15)
plt.show()

# Getting the CustomerId for each group

cust1 = df[df.Label==0]
print("The number of customers in 1st group = ", len(cust1))
print("The Customer Id are - ", cust1.CustomerID.values)
print("============================================================================================\n")

cust2 = df[df.Label==1]
print("The number of customers in 2nd group = ", len(cust2))
print("The Customer Id are - ", cust2.CustomerID.values)
print("============================================================================================\n")

cust3 = df[df.Label==2]
print("The number of customers in 3rd group = ", len(cust3))
print("The Customer Id are - ", cust3.CustomerID.values)
print("============================================================================================\n")

cust4 = df[df.Label==3]
print("The number of customers in 4th group = ", len(cust4))
print("The Customer Id are - ", cust4.CustomerID.values)
print("============================================================================================\n")

cust5 = df[df.Label==4]
print("The number of customers in 5th group = ", len(cust5))
print("The Customer Id are - ", cust5.CustomerID.values)
print("============================================================================================\n")


df.head()
var_list_1 = df_scaled_fit[["Age","Annual Income (k$)","Spending Score (1-100)"]]
var_list_1.head()
kmeans1 = KMeans(n_clusters=5, max_iter=50)
kmeans1.fit(var_list_1)

kmeans1.labels_
df["Label"] = kmeans1.labels_
df.head()
ssd = []

for num_clusters in range(1,11):
    kmeans1 = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans1.fit(var_list_1)
    
    ssd.append(kmeans1.inertia_)
# Elbow curve

plt.figure(figsize=(12,6))

plt.plot(range(1,11), ssd, linewidth=2, color="red", marker ="8")
plt.title("Elbow Curve\n=================================================================", fontsize=20, color="green")
plt.xlabel("K Value")
plt.xticks(np.arange(1,11,1))
plt.ylabel("SSD")

plt.show()
from mpl_toolkits.mplot3d import Axes3D
#3D Plot as we did the clustering on the basis of 3 input features

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.Age[df.Label == 0], df["Annual Income (k$)"][df.Label == 0], df["Spending Score (1-100)"][df.Label == 0], c='purple', s=60)
ax.scatter(df.Age[df.Label == 1], df["Annual Income (k$)"][df.Label == 1], df["Spending Score (1-100)"][df.Label == 1], c='red', s=60)
ax.scatter(df.Age[df.Label == 2], df["Annual Income (k$)"][df.Label == 2], df["Spending Score (1-100)"][df.Label == 2], c='blue', s=60)
ax.scatter(df.Age[df.Label == 3], df["Annual Income (k$)"][df.Label == 3], df["Spending Score (1-100)"][df.Label == 3], c='green', s=60)
ax.scatter(df.Age[df.Label == 4], df["Annual Income (k$)"][df.Label == 4], df["Spending Score (1-100)"][df.Label == 4], c='yellow', s=60)
ax.view_init(35, 185)
plt.title("3D view of the data distribution\n=================================================================", fontsize=20, color="green")
plt.xlabel("Age", fontsize=15)
plt.ylabel("Annual Income (k$)", fontsize=15)
ax.set_zlabel('Spending Score (1-100)', fontsize=15)
plt.show()

cust1 = df[df.Label==0]
print("The number of customers in 1st group = ", len(cust1))
print("The Customer Id are - ", cust1.CustomerID.values)
print("============================================================================================\n")

cust2 = df[df.Label==1]
print("The number of customers in 2nd group = ", len(cust2))
print("The Customer Id are - ", cust2.CustomerID.values)
print("============================================================================================\n")

cust3 = df[df.Label==2]
print("The number of customers in 3rd group = ", len(cust3))
print("The Customer Id are - ", cust3.CustomerID.values)
print("============================================================================================\n")

cust4 = df[df.Label==3]
print("The number of customers in 4th group = ", len(cust4))
print("The Customer Id are - ", cust4.CustomerID.values)
print("============================================================================================\n")

cust5 = df[df.Label==4]
print("The number of customers in 5th group = ", len(cust5))
print("The Customer Id are - ", cust5.CustomerID.values)
print("============================================================================================\n")


# ==================================================================
# 2. VERI YUKLEME VE ON ISLEME (EDA ONCESI TEMIZLIK)
# ==================================================================

# r"" kullanimi ile dosya yolu hatasi onlenmistir.
df = pd.read_csv(r"C:\Users\Lenovo\Desktop\Musteri_Segmentasyon\Mall_Customers.csv")

# Sütun isimlerini duzeltme
df.rename(columns={"Genre":"Gender"}, inplace=True)

# Aykiri Deger Duzeltmesi (Annual Income uzerinden)
Q1 = df['Annual Income (k$)'].quantile(0.25)
Q3 = df['Annual Income (k$)'].quantile(0.75)
IQR = Q3 - Q1
ust_sinir = Q3 + 1.5 * IQR
df.loc[df['Annual Income (k$)'] > ust_sinir, 'Annual Income (k$)'] = ust_sinir

print("--- Veri Yuklendi, Gender Duzeltildi, Aykiri Degerler Temizlendi ---")


# ==================================================================
# 3. VERI KESFI (EDA) VE GORSELLESTIRMELER (MEVCUT KODUNUZ)
# ==================================================================
# NOT: Bu bolumdeki tum plt.title metinleri encoding hatasi vermemek icin duzeltilmistir.

# (Korelasyon Isı Haritasi)
df_corr = df.drop(columns=['CustomerID'], errors='ignore')
sns.heatmap(df_corr.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Degiskenler Arasi Korelasyon Isi Haritasi')
plt.show()

# (Yas Dagilimi, Gelir Dagilimi, Cinsiyet Sayimi vb. grafikler buraya dahildir)
# ... Tum eski EDA grafik kodlariniz buradadir. ...


# ==================================================================
# 4. VERI HAZIRLIGI (OLCEKLENDIRME)
# ==================================================================

# Sadece sayisal ve kumeleme icin kullanilacak sutunlar (Gender disarida tutuldu)
df_scaled = df[["Age","Annual Income (k$)","Spending Score (1-100)"]] 
scaler = StandardScaler()
df_scaled_fit = scaler.fit_transform(df_scaled)

# Kumeleme icin sadece Gelir ve Skor kullanilacak
X_cluster = pd.DataFrame(df_scaled_fit, columns=["Age","Annual Income (k$)","Spending Score (1-100)"])[["Annual Income (k$)","Spending Score (1-100)"]]

print("--- Veri Olceklendirildi (Scaling) ---")


# ==================================================================
# 5. K-MEANS VE OPTIMUM K BULMA (ELBOW METHOD)
# ==================================================================

ssd = []
for num_clusters in range(1,11):
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50, n_init=10, random_state=42)
    kmeans.fit(X_cluster)
    ssd.append(kmeans.inertia_)

plt.figure(figsize=(12,6))
plt.plot(range(1,11), ssd, linewidth=2, color="red", marker ="8")
plt.title("Elbow Curve (Optimum K Bulma)")
plt.xlabel("K Value (Kume Sayisi)")
plt.xticks(np.arange(1,11,1))
plt.ylabel("SSD (Hata)")
plt.show()

# Elbow curve sonucuna gore K=5 secilir
kmeans_final = KMeans(n_clusters=5, max_iter=50, random_state=42, n_init=10)
df["Label"] = kmeans_final.fit_predict(X_cluster)

plt.title("5 Kumeye Ayrilmis Musteri Dagilimi")
sns.scatterplot(data=df, x="Annual Income (k$)", y="Spending Score (1-100)", hue="Label", s=60)
plt.show()


# ==================================================================
# 6. TUM KUMELEME ALGORITMALARININ IKI METRIKLE KARSILASTIRILMASI (3 ALGORITMA)
# ==================================================================

skorlar_sil = {} 
skorlar_db = {}
# Mean Shift'i kaldirdik
modeller_kume = ['K-Means (K=5)', 'Hiyerarşik (K=5)', 'Affinity Prop.'] 

# 1. K-MEANS (K=5)
# KMeans skorlari, önceki kodda hesaplanmis olan 'kmeans_final' modelinden alinir.
skorlar_sil['K-Means (K=5)'] = silhouette_score(X_cluster, kmeans_final.labels_)
skorlar_db['K-Means (K=5)'] = davies_bouldin_score(X_cluster, kmeans_final.labels_)


# 2. HIYERARSIK KUMELEME (K=5)
hc_model = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
hc_labels = hc_model.fit_predict(X_cluster)
skorlar_sil['Hiyerarşik (K=5)'] = silhouette_score(X_cluster, hc_labels)
skorlar_db['Hiyerarşik (K=5)'] = davies_bouldin_score(X_cluster, hc_labels)


# 3. AFFINITY PROPAGATION (K sayisini otomatik bulur)
ap_model = AffinityPropagation(damping=0.9, random_state=42, max_iter=200, convergence_iter=15)
ap_labels = ap_model.fit_predict(X_cluster)
if len(np.unique(ap_labels)) > 1:
    skorlar_sil['Affinity Prop.'] = silhouette_score(X_cluster, ap_labels)
    skorlar_db['Affinity Prop.'] = davies_bouldin_score(X_cluster, ap_labels)
else:
    # Eger tek bir kume bulursa, skorlari sifir kabul et
    skorlar_sil['Affinity Prop.'] = 0.0
    skorlar_db['Affinity Prop.'] = 100.0


# GORSEL KARŞILAŞTIRMA GRAFİKLERİ
skorlar_sil_values = [skorlar_sil[m] for m in modeller_kume]
skorlar_db_values = [skorlar_db[m] for m in modeller_kume]

plt.figure(figsize=(14, 6)) # Grafik boyutunu 3 algoritmaya gore kuculttuk

# 1. SILHOUETTE SKORU (Büyük Daha İyi)
plt.subplot(1, 2, 1)
sns.barplot(x=modeller_kume, y=skorlar_sil_values, palette='viridis')
plt.title('Silhouette Skor Karsilastirmasi (Buyuk Daha Iyi)')
plt.ylabel('Silhouette Skoru')

# 2. DAVIES-BOULDIN SKORU (Küçük Daha İyi)
plt.subplot(1, 2, 2)
sns.barplot(x=modeller_kume, y=skorlar_db_values, palette='plasma')
plt.title('Davies-Bouldin Skor Karsilastirmasi (Kucuk Daha Iyi)')
plt.ylabel('Davies-Bouldin Skoru')

plt.tight_layout()
plt.savefig('Kumeleme_Iki_Metrik_Karsilastirma.png')
plt.show()

print("\n*** TUM KUMELEME ALGORITMALARI KARSILASTIRILDI. ***")
print(f"\nSilhouette Skorlari: {skorlar_sil}")
print(f"Davies-Bouldin Skorlari: {skorlar_db} icermektedir.")