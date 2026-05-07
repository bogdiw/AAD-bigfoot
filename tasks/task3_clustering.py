# -*- coding: utf-8 -*-
"""
Task 3 - Clustering: Descoperirea arhetipurilor de raportari
Owner: Membru 3

Identifica grupuri naturale in raportarile Bigfoot folosind features mixte:
State (frequency encoding), Season, Month, Class (one-hot) si Text (TF-IDF).
Compara KMeans, Agglomerative Clustering si DBSCAN.

Input:
  data/reports.csv (folosim raportarile originale curatate)

Output:
  output/clustering/*.png (grafice PCA, Silhouette, Dendrograma, profil clustere)
"""

# ============================================================================
# Setup initial
# ============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as shc

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

RANDOM_STATE = 42
OUTPUT_DIR = 'output/clustering'

sns.set_theme(style='whitegrid')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("TASK 3: CLUSTERING - DESCOPERIREA ARHETIPURILOR")
print("=" * 70)

# ============================================================================
# 3.1 Pregatirea Datelor si Feature Engineering
# ============================================================================
print("\n3.1 PREGATIREA DATELOR SI FEATURE ENGINEERING")

df_raw = pd.read_csv('data/reports.csv')

# Filtram doar rapoartele reale (excludem Media Articles)
df = df_raw[df_raw['Report Type'] == 'Report'].copy()

# Curatare text (combina Headline cu Observed)
df['text'] = df['Headline'].fillna('') + ' ' + df['Observed'].fillna('')

# Pastram doar randurile care au completate campurile esentiale pentru clustering
cols_to_check = ['State', 'Season', 'Month', 'Class', 'text']
df = df.dropna(subset=['State', 'Season', 'Month', 'Class']).copy()

# Convertim Month intr-o valoare numerica (1-12)
month_numeric = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}
df['Month_Num'] = df['Month'].map(month_numeric)

print(f"Date valabile pentru clustering: {len(df)} randuri.")

# --- Feature Engineering ---

# 1. State: Frequency Encoding
state_freq = df['State'].value_counts(normalize=True)
df['State_Freq'] = df['State'].map(state_freq)

# 2. Season & Class: One-Hot Encoding
season_dummies = pd.get_dummies(df['Season'], prefix='Season')
class_dummies = pd.get_dummies(df['Class'], prefix='Class')

# 3. Text: TF-IDF (pastram doar un numar mic de features pt a nu domina modelul)
tfidf = TfidfVectorizer(stop_words='english', max_features=50)
text_features = tfidf.fit_transform(df['text']).toarray()
text_cols = [f'tfidf_{w}' for w in tfidf.get_feature_names_out()]
df_text = pd.DataFrame(text_features, columns=text_cols, index=df.index)

# ADĂUGAT: Atașăm coloanele TF-IDF în df-ul principal pentru a le putea folosi la profilare
df = pd.concat([df, df_text], axis=1)

# Combinam toate feature-urile
X_raw = pd.concat([
    df[['State_Freq', 'Month_Num']],
    season_dummies,
    class_dummies,
    df_text
], axis=1)

# Scalare (StandardScaler) obligatorie pentru clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Reducere dimensionalitate (PCA) pentru vizualizare in 2D
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca_2d = pca.fit_transform(X_scaled)
df['pca_x'] = X_pca_2d[:, 0]
df['pca_y'] = X_pca_2d[:, 1]

print(f"Varianta explicata de primele 2 componente PCA: {pca.explained_variance_ratio_.sum() * 100:.1f}%")

# ============================================================================
# 3.2 Antrenarea Algoritmilor de Clustering
# ============================================================================
print("\n3.2 EVALUARE SI GASIRE NUMAR OPTIM DE CLUSTERE")

# --- KMeans: Metoda Elbow si Silhouette ---
k_range = range(2, 9)
inertia_values = []
silhouette_values = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    km_labels = km.fit_predict(X_scaled)
    inertia_values.append(km.inertia_)
    silhouette_values.append(silhouette_score(X_scaled, km_labels))

# Grafic Elbow & Silhouette
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(k_range, inertia_values, marker='o', color='tab:blue', label='Inertia')
ax1.set_xlabel('Numar de clustere (K)')
ax1.set_ylabel('Inertia (SSW)', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.plot(k_range, silhouette_values, marker='s', color='tab:red', label='Silhouette Score')
ax2.set_ylabel('Silhouette Score', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.title('KMeans: Elbow Method & Silhouette Score')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_elbow_silhouette.png', dpi=150)
plt.close()

# Alegem K=4 pentru analizele urmatoare (presupunand 4 arhetipuri mari)
K_OPTIM = 4
print(f"-> Selectam K={K_OPTIM} pentru modelarea finala.")

# 1. KMeans
kmeans = KMeans(n_clusters=K_OPTIM, random_state=RANDOM_STATE, n_init=10)
df['Cluster_KMeans'] = kmeans.fit_predict(X_scaled)

# 2. Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=K_OPTIM, metric='euclidean', linkage='ward')
df['Cluster_Agglo'] = agglo.fit_predict(X_scaled)

# 3. DBSCAN (densitate)
dbscan = DBSCAN(eps=4.5, min_samples=10)
df['Cluster_DBSCAN'] = dbscan.fit_predict(X_scaled)

# ============================================================================
# 3.3 Metrici si Vizualizari
# ============================================================================
print("\n3.3 COMPARATIE ALGORITMI")

metrics = []
for model_name, col in [('KMeans', 'Cluster_KMeans'), ('Agglomerative', 'Cluster_Agglo')]:
    sil = silhouette_score(X_scaled, df[col])
    db_idx = davies_bouldin_score(X_scaled, df[col])
    metrics.append({'Model': model_name, 'Silhouette': sil, 'Davies-Bouldin': db_idx})

dbscan_core = df[df['Cluster_DBSCAN'] != -1]
if len(dbscan_core['Cluster_DBSCAN'].unique()) > 1:
    sil_db = silhouette_score(X_scaled[df['Cluster_DBSCAN'] != -1], dbscan_core['Cluster_DBSCAN'])
    db_idx_db = davies_bouldin_score(X_scaled[df['Cluster_DBSCAN'] != -1], dbscan_core['Cluster_DBSCAN'])
    metrics.append({'Model': 'DBSCAN (excl. noise)', 'Silhouette': sil_db, 'Davies-Bouldin': db_idx_db})

metrics_df = pd.DataFrame(metrics)
print(metrics_df.to_string(index=False))

# --- Vizualizare PCA pentru cei 3 algoritmi ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
models_to_plot = [('KMeans', 'Cluster_KMeans'), ('Agglomerative', 'Cluster_Agglo'), ('DBSCAN', 'Cluster_DBSCAN')]

for ax, (title, col) in zip(axes, models_to_plot):
    sns.scatterplot(
        x='pca_x', y='pca_y', hue=col, data=df, palette='tab10',
        ax=ax, s=40, alpha=0.7, legend='full'
    )
    ax.set_title(f'PCA 2D - {title}')
    ax.set_xlabel('Componenta Principala 1')
    ax.set_ylabel('Componenta Principala 2')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_pca_clusters.png', dpi=150)
plt.close()

# --- Dendrograma pentru Agglomerative Clustering ---
print("\nGenerare Dendrograma (Ward linkage)...")
plt.figure(figsize=(12, 6))
plt.title("Dendrograma Ierarhica (Agglomerative Clustering)")
dend = shc.dendrogram(
    shc.linkage(X_scaled, method='ward'),
    truncate_mode='level', p=5,
    show_leaf_counts=True
)
plt.xlabel("Numar de puncte in nod")
plt.ylabel("Distanta Euclidiana")
plt.axhline(y=100, color='r', linestyle='--')
plt.savefig(f'{OUTPUT_DIR}/03_dendrograma.png', dpi=150)
plt.close()

# ============================================================================
# 3.4 Profilarea Clusterelor (Descoperirea Arhetipurilor)
# ============================================================================
print("\n3.4 PROFILAREA CLUSTERELOR (KMeans)")

cluster_profiles = []

for c in range(K_OPTIM):
    subset = df[df['Cluster_KMeans'] == c]

    top_season = subset['Season'].mode()[0]
    top_class = subset['Class'].mode()[0]
    top_state = subset['State'].mode()[0]

    tfidf_means = subset[text_cols].mean().sort_values(ascending=False)
    top_words = [w.replace('tfidf_', '') for w in tfidf_means.index[:3]]

    cluster_profiles.append({
        'Cluster': f'Cluster {c}',
        'Count': len(subset),
        'Top Season': top_season,
        'Top Class': top_class,
        'Top State': top_state,
        'Keywords': ", ".join(top_words)
    })

profile_df = pd.DataFrame(cluster_profiles)
print("\nArhetipuri Descoperite:")
print("-" * 70)
print(profile_df.to_string(index=False))
print("-" * 70)

# --- MAPĂRI PENTRU LEGENDE DESCRIPTIVE ---

# 1. Mapăm Class A/B/C către grupul real
class_mapping = {
    'Class A': 'Contact Vizual (Class A)',
    'Class B': 'Dovezi Indirecte / Sunete (Class B)',
    'Class C': 'Informații Second-Hand (Class C)'
}
df['Grup_Clasa'] = df['Class'].map(class_mapping)

# 2. Mapăm numerele de clustere (0,1,2,3) către arhetipurile identificate din README
cluster_mapping = {
    0: '0: Camping / Vară',
    1: '1: Contact Vizual pe Drum',
    2: '2: Vânătoare / Toamnă',
    3: '3: Iarnă / Urme'
}
df['Arhetip_Cluster'] = df['Cluster_KMeans'].map(cluster_mapping)

# Vizualizare distributie clase per cluster cu legende explicite
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Ordinea pe axa X pentru a fi consecventă
cluster_order = [cluster_mapping[i] for i in range(4)]

sns.countplot(data=df, x='Arhetip_Cluster', hue='Season', ax=ax[0], palette='Set2', order=cluster_order)
ax[0].set_title('Distribuția Sezoanelor în Arhetipurile Descoperite', fontsize=12, fontweight='bold')
ax[0].set_xlabel('Arhetipuri')
ax[0].set_ylabel('Număr de Raportări')
ax[0].legend(title='Sezonul Anului')

sns.countplot(data=df, x='Arhetip_Cluster', hue='Grup_Clasa', ax=ax[1], palette='Set1', order=cluster_order)
ax[1].set_title('Distribuția Tipului de Raportare în Arhetipuri', fontsize=12, fontweight='bold')
ax[1].set_xlabel('Arhetipuri')
ax[1].set_ylabel('Număr de Raportări')
ax[1].legend(title='Grup (Clasificare BFRO)')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_cluster_profiles.png', dpi=150)
plt.close()

print(f"\nGraficele au fost salvate cu succes in '{OUTPUT_DIR}/'.")
print("Task 3 a fost finalizat.")