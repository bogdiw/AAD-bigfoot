# -*- coding: utf-8 -*-
"""
Task 1 - Clasificare Class A vs Class B vs Class C din text
Owner: FRATIMAN Bogdan-Gabriel

Antreneaza un model care prezice Class (A/B/C) pe baza textului raportului
(Headline + Observed). Aplica modelul pe Media Articles (care nu au Class)
pentru a completa coloana lipsa.

Tip ML: clasificare multi-class (Class A vs B vs C, sever imbalansat).
Pastram Class C pentru ca Media Articles sunt prin definitie secondhand
reports, deci se potrivesc cu definitia BFRO pentru Class C.

Input:
  data/reports.csv (5467 randuri originale)

Output:
  data/reports_augmented.csv (5376 randuri = 4925 Reports + 451 Media cu predict)
  output/classification/*.png (grafice EDA + model evaluation + feature importance)
"""

# ============================================================================
# Setup initial
# ============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

RANDOM_STATE = 42
OUTPUT_DIR = 'output/classification'

sns.set_theme(style='whitegrid')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# 2.1 Pregatirea datelor
# ============================================================================

print("2.1 PREGATIREA DATELOR")

df = pd.read_csv('data/reports.csv')
print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")

# Clean Year (same logic as checkpoint1)
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df_raw = pd.read_csv('data/reports.csv')
mask = df['Year'].isna()
extracted = df_raw.loc[mask.index[mask], 'Year'].astype(str).str.extract(r'(\d{4})')[0]
df.loc[mask, 'Year'] = pd.to_numeric(extracted, errors='coerce')
df.loc[(df['Year'] < 1800) | (df['Year'] > 2025), 'Year'] = np.nan
df = df[(df['Year'].isna()) | (df['Year'] < 2020)].copy()
df['Year'] = df['Year'].astype('Int64')

# Split Reports vs Media Articles
df_reports = df[df['Report Type'] == 'Report'].copy()
df_articles = df[df['Report Type'] == 'Media Article'].copy()
print(f"\nReports (au Class):  {len(df_reports)}")
print(f"Media Articles:      {len(df_articles)}")

# Build unified 'text' column:
#   Reports -> Headline + Observed
#   Media   -> Headline + Observed.1
df_reports['text'] = (
    df_reports['Headline'].fillna('') + ' ' +
    df_reports['Observed'].fillna('')
)
df_articles['text'] = (
    df_articles['Headline'].fillna('') + ' ' +
    df_articles['Observed.1'].fillna('')
)

print(f"\nLungime medie text:")
print(f"  Reports: {df_reports['text'].str.len().mean():.0f} chars")
print(f"  Articles:   {df_articles['text'].str.len().mean():.0f} chars")

print("\nClass distribution (Reports):")
print(df_reports['Class'].value_counts())

# Pastram toate cele 3 clase. Class C = secondhand reports = sever nebalansat (Class C ~ 0.6%)
# -> compensam cu class_weight='balanced' si folosim F1 la evaluare.

feature_cols = ['text', 'Year']
X = df_reports[feature_cols]
y = df_reports['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain: {len(X_train)} samples")
print(f"Test:  {len(X_test)} samples")
print(f"Train Class A/B/C: {(y_train == 'Class A').sum()} / {(y_train == 'Class B').sum()} / {(y_train == 'Class C').sum()}")
print(f"Test  Class A/B/C: {(y_test == 'Class A').sum()} / {(y_test == 'Class B').sum()} / {(y_test == 'Class C').sum()}")


# ============================================================================
# EDA pe text si Class
# ============================================================================

print("EDA - ANALIZA EXPLORATORIE")

df_reports['text_length'] = df_reports['text'].str.len()
df_articles['text_length'] = df_articles['text'].str.len()

print(f"\nLungime text per Class (median):")
for c in ['Class A', 'Class B', 'Class C']:
    med = df_reports[df_reports['Class'] == c]['text_length'].median()
    print(f"  {c}: {med:.0f} chars")

# --- Chart 1: Class distribution + text length boxplot ---
class_labels = ['Class A', 'Class B', 'Class C']
class_colors = ['#2ecc71', '#3498db', '#e74c3c']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

class_counts = df_reports['Class'].value_counts().reindex(class_labels)
axes[0].bar(class_counts.index, class_counts.values, color=class_colors, edgecolor='black')
axes[0].set_ylabel('Number of reports')
axes[0].set_title('Class distribution (severely imbalanced)')
for i, v in enumerate(class_counts.values):
    axes[0].text(i, v + 30, str(v), ha='center', fontsize=11, fontweight='bold')

data_box = [df_reports[df_reports['Class'] == c]['text_length'] for c in class_labels]
bp = axes[1].boxplot(data_box, tick_labels=class_labels, patch_artist=True, showfliers=False)
for patch, color in zip(bp['boxes'], class_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1].set_ylabel('Text length (characters)')
axes[1].set_title('Text length per Class (no outliers)')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_class_and_length.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Chart 2: Top words per Class ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, cls in enumerate(class_labels):
    sub_text = df_reports[df_reports['Class'] == cls]['text']
    cv = CountVectorizer(stop_words='english', max_features=20, min_df=3, ngram_range=(1, 1))
    cv.fit(sub_text)
    word_counts = cv.transform(sub_text).sum(axis=0).A1
    words = cv.get_feature_names_out()
    order = np.argsort(word_counts)[::-1][:15]
    axes[i].barh(range(15), word_counts[order][::-1], color=class_colors[i], edgecolor='black')
    axes[i].set_yticks(range(15))
    axes[i].set_yticklabels(words[order][::-1])
    axes[i].set_title(f'Top 15 words - {cls}')
    axes[i].set_xlabel('Frequency')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_top_words_per_class.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Chart 3: Text length Reports vs Media Articles ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df_reports['text_length'].clip(upper=5000), bins=50, alpha=0.6,
        label=f'Reports (n={len(df_reports)})', color='steelblue', edgecolor='black')
ax.hist(df_articles['text_length'].clip(upper=5000), bins=50, alpha=0.6,
        label=f'Media Articles (n={len(df_articles)})', color='orange', edgecolor='black')
ax.set_xlabel('Text length (characters, clipped at 5000)')
ax.set_ylabel('Number of reports')
ax.set_title('Text length distribution: Reports vs Media Articles')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_text_length_reports_vs_media.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================================
# 2.2 Implementarea modelelor
# ============================================================================

print("2.2 IMPLEMENTAREA MODELELOR")

# Preprocessor: TF-IDF on text + StandardScaler on Year
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=5,
            sublinear_tf=True
        ), 'text'),
        ('year', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), ['Year']),
    ]
)

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=20, class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5,
        random_state=RANDOM_STATE
    ),
}

# GradientBoosting nu suporta class_weight, folosim sample_weight pentru Class C
sample_weights_map = {'Class A': 1.0, 'Class B': 1.0, 'Class C': 80.0}
sample_weights = y_train.map(sample_weights_map).values

trained = {}
print("\nAntrenare modele...")
for name, model in models.items():
    print(f"  -> {name}")
    pipe = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
    if name == 'Gradient Boosting':
        pipe.fit(X_train, y_train, classifier__sample_weight=sample_weights)
    else:
        pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)
    trained[name] = {'pipe': pipe, 'y_pred': y_pred, 'y_proba': y_proba}

# ============================================================================
# 2.3 Evaluarea si compararea modelelor
# ============================================================================

print("2.3 EVALUAREA MODELELOR")

results_rows = []

print("\n--- Classification reports ---\n")
for name, info in trained.items():
    y_pred = info['y_pred']
    print(f"### {name}")
    print(classification_report(y_test, y_pred, labels=class_labels, zero_division=0))

    results_rows.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1 (macro)': f1_score(y_test, y_pred, average='macro', labels=class_labels, zero_division=0),
        'F1 (weighted)': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'Precision (macro)': precision_score(y_test, y_pred, average='macro', labels=class_labels, zero_division=0),
        'Recall (macro)': recall_score(y_test, y_pred, average='macro', labels=class_labels, zero_division=0),
    })

results_df = pd.DataFrame(results_rows).sort_values('F1 (macro)', ascending=False)
print("\n--- Tabel comparativ (sortat dupa F1 macro) ---")
print(results_df.to_string(index=False))

# --- Chart 4: 3 confusion matrices side-by-side ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, info) in zip(axes, trained.items()):
    cm = confusion_matrix(y_test, info['y_pred'], labels=class_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_labels, yticklabels=class_labels,
                cbar=False, linewidths=0.5)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(name)
plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Chart 5: Bar chart model comparison ---
fig, ax = plt.subplots(figsize=(12, 6))
metrics_to_plot = ['Accuracy', 'F1 (macro)', 'F1 (weighted)', 'Precision (macro)', 'Recall (macro)']
x_pos = np.arange(len(results_df))
width = 0.16
metric_colors = sns.color_palette('Set2', len(metrics_to_plot))
for i, metric in enumerate(metrics_to_plot):
    ax.bar(x_pos + i * width, results_df[metric].values, width, label=metric, color=metric_colors[i])
ax.set_xticks(x_pos + width * 2)
ax.set_xticklabels(results_df['Model'].values)
ax.set_ylabel('Score')
ax.set_title('Model comparison - metrics (multi-class A/B/C)')
ax.set_ylim(0, 1.0)
ax.legend(loc='lower right', ncol=2)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

best_name = results_df.iloc[0]['Model']
print(f"\n>>> Best model (F1 macro): {best_name}")


# ============================================================================
# 2.4 Interpretare - feature importance si top words
# ============================================================================

print("2.4 INTERPRETARE SI CONCLUZII")

# Random Forest feature importance
rf_pipe = trained['Random Forest']['pipe']
rf_model = rf_pipe.named_steps['classifier']
feature_names = rf_pipe.named_steps['preprocessor'].get_feature_names_out()
importances = rf_model.feature_importances_
top_n = 20
idx = np.argsort(importances)[-top_n:]

# --- Chart 6: RF top features ---
fig, ax = plt.subplots(figsize=(10, 8))
clean_names = [n.replace('text__', '').replace('year__', '[Year] ') for n in feature_names[idx]]
ax.barh(range(top_n), importances[idx], color='steelblue', edgecolor='black')
ax.set_yticks(range(top_n))
ax.set_yticklabels(clean_names)
ax.set_xlabel('Importance')
ax.set_title(f'Random Forest - Top {top_n} Features')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_rf_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# Logistic Regression coefficients per class
logreg_pipe = trained['Logistic Regression']['pipe']
logreg_model = logreg_pipe.named_steps['classifier']
coefs = logreg_model.coef_
classes_lr = logreg_model.classes_

# --- Chart 7: LogReg top words per class ---
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
for ax, cls, coef, color in zip(axes, classes_lr, coefs, class_colors):
    top_idx = np.argsort(coef)[-15:]
    names = [feature_names[i].replace('text__', '').replace('year__', '[Year] ')
             for i in top_idx]
    ax.barh(range(15), coef[top_idx], color=color, edgecolor='black')
    ax.set_yticks(range(15))
    ax.set_yticklabels(names)
    ax.set_xlabel('Coefficient')
    ax.set_title(f'Top 15 words for {cls}')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_logreg_top_words_per_class.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================================
# Aplicare pe Media Articles + salvare augmented dataset
# ============================================================================

print("\n" + "=" * 60)
print("APLICARE PE MEDIA ARTICLES")
print("=" * 60)

X_media = df_articles[feature_cols]

print(f"\nDistributia predictiilor per model (pe {len(df_articles)} Media Articles):")
print(f"{'Model':<22} {'Class A':>10} {'Class B':>10} {'Class C':>10}")
print("-" * 55)
media_predictions = {}
for name, info in trained.items():
    preds = info['pipe'].predict(X_media)
    media_predictions[name] = preds
    a = (preds == 'Class A').sum()
    b = (preds == 'Class B').sum()
    c = (preds == 'Class C').sum()
    print(f"{name:<22} {a:>10} {b:>10} {c:>10}")

# Folosim Logistic Regression
# (singurul care prezice si Class C, ne valideaza ipoteza)
final_model_name = 'Logistic Regression'
final_pipe = trained[final_model_name]['pipe']
df_articles['Class'] = final_pipe.predict(X_media)
df_articles['Class_confidence'] = final_pipe.predict_proba(X_media).max(axis=1)
df_articles['Class_source'] = 'predicted'

print(f"\n>>> Folosim {final_model_name} ")
print(f"\nDistributia finala in Media Articles:")
print(df_articles['Class'].value_counts())
print(f"\nConfidence statistics:")
print(df_articles['Class_confidence'].describe())

# Marcam Reports cu Class_source = 'original'
df_reports['Class_source'] = 'original'
df_reports['Class_confidence'] = 1.0

# Combinam si salvam
common_cols = ['Id', 'Class', 'Class_source', 'Class_confidence',
               'Year', 'Submitted Date', 'Headline', 'text']
augmented = pd.concat([
    df_reports[common_cols],
    df_articles[common_cols]
], ignore_index=True)

augmented.to_csv('data/reports_augmented.csv', index=False)
print(f"\n>>> Salvat in data/reports_augmented.csv")
print(f"    Total: {len(augmented)} randuri")
print(f"    Original (Reports): {(augmented['Class_source'] == 'original').sum()}")
print(f"    Predicted (Media):  {(augmented['Class_source'] == 'predicted').sum()}")

# --- Chart 8: Distribution of predictions on Media Articles per model ---
fig, ax = plt.subplots(figsize=(10, 6))
pred_summary = pd.DataFrame({
    name: pd.Series(preds).value_counts().reindex(class_labels).fillna(0)
    for name, preds in media_predictions.items()
})
pred_summary.T.plot(kind='bar', stacked=False, ax=ax,
                   color=class_colors, edgecolor='black')
ax.set_xlabel('Model')
ax.set_ylabel('Number of Media Articles')
ax.set_title('Prediction distribution on 451 Media Articles')
ax.legend(title='Class')
ax.set_xticklabels(pred_summary.columns, rotation=0)
for c in ax.containers:
    ax.bar_label(c, fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_media_predictions.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# Impact asupra rezultatelor din Checkpoint 1 (before vs after augmentare)
# ============================================================================

print("IMPACT ASUPRA CHECKPOINT 1")

# Distributia Class: inainte (Reports only) vs dupa (augmented = Reports + Media)
before_counts = df_reports['Class'].value_counts().reindex(class_labels).fillna(0).astype(int)
after_counts = augmented['Class'].value_counts().reindex(class_labels).fillna(0).astype(int)

print(f"\nClass distribution:")
print(f"{'Class':<10} {'Before (Reports)':>20} {'After (Augmented)':>20} {'Diff':>10}")
print("-" * 65)
for c in class_labels:
    b = before_counts[c]
    a = after_counts[c]
    print(f"{c:<10} {b:>20} {a:>20} {a-b:>+10}")
print(f"{'Total':<10} {before_counts.sum():>20} {after_counts.sum():>20} {after_counts.sum()-before_counts.sum():>+10}")

# --- Chart 9: Pie chart side-by-side before vs after ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
explode = [0.05] * 3

axes[0].pie(before_counts, labels=class_labels, autopct='%1.1f%%',
            colors=class_colors, explode=explode, startangle=90,
            textprops={'fontsize': 11})
axes[0].set_title(f'Before (Checkpoint 1)\nReports only - {before_counts.sum()} rows',
                  fontsize=12, fontweight='bold')

axes[1].pie(after_counts, labels=class_labels, autopct='%1.1f%%',
            colors=class_colors, explode=explode, startangle=90,
            textprops={'fontsize': 11})
axes[1].set_title(f'After (Checkpoint 2 - Task 1)\nAugmented - {after_counts.sum()} rows',
                  fontsize=12, fontweight='bold')

plt.suptitle('Class distribution: before vs after augmentation with Media Articles',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/09_class_before_after.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Chart 10: Stacked bar showing source contribution per class ---
fig, ax = plt.subplots(figsize=(10, 6))
source_breakdown = pd.crosstab(augmented['Class'], augmented['Class_source']).reindex(class_labels)
source_breakdown = source_breakdown[['original', 'predicted']]
source_breakdown.plot(kind='bar', stacked=True, ax=ax,
                      color=['#4a4a4a', '#f39c12'], edgecolor='black')
ax.set_xlabel('Class')
ax.set_ylabel('Number of reports')
ax.set_title('Source contribution (original Reports vs predicted Media Articles)')
ax.set_xticklabels(class_labels, rotation=0)
ax.legend(title='Source', labels=['Original (Reports)', 'Predicted (Media)'])
for c in ax.containers:
    ax.bar_label(c, fontsize=9, label_type='center')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/10_class_source_contribution.png', dpi=150, bbox_inches='tight')
plt.show()

# Comparatie cu pie chart-ul original din Checkpoint 1
n_added_C = after_counts['Class C'] - before_counts['Class C']
n_added_total = after_counts.sum() - before_counts.sum()
print(f"\nIn urma augmentarii:")
print(f"  - Numarul de randuri a crescut cu {n_added_total} randuri (Media Articles cu coloana Class prezisa)")
print(f"  - Class C a crescut: {before_counts['Class C']} -> {after_counts['Class C']} (+{n_added_C})")
print(f"  - Procentul pentru Class C: {before_counts['Class C']/before_counts.sum()*100:.1f}% -> {after_counts['Class C']/after_counts.sum()*100:.1f}%")
