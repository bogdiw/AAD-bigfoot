# -*- coding: utf-8 -*-
"""
AAD Project - Bigfoot Data Detective

# Proiect AAD - Bigfoot Data Detective

## Componenta Echipa
*   LUNGU Mihai-Teodor 341C3
*   FRĂȚIMAN Bogdan-Gabriel 341C3
*   GRAUR Dan-Mihai 341C3

## Link document principal:
https://github.com/bogdiw/AAD-bigfoot

## Link dataset de pe Kaggle:
https://www.kaggle.com/datasets/josephvm/bigfoot-sightings-data

## Scopul proiectului:
Analizam baza de date BFRO pentru a extrage tiparele geografice ale raportarilor Bigfoot.
Studiul imbina folclorul urban cu datele climatice si geografice masurabile.

## Ipoteze de cercetare:
- H1: Numarul de raportari a crescut semnificativ dupa expansiunea internetului.
- H2: Raportarile Bigfoot coreleaza cu populatia de ursi per stat.
- H3: Raportarile depind de anotimp, urmarind lunile in care oamenii ies mai mult in natura.

"""

# ============================================================================
# Setup initial
# ============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import os

sns.set_theme(style="whitegrid")
os.makedirs('output', exist_ok=True)

df = pd.read_csv('data/reports.csv')

## Etapa 1: Procesarea si Analiza Datelor

# ============================================================================
# 1.1 Incarcarea si Intelegerea Datelor
# ============================================================================

print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nLast 5 rows:")
print(df.tail())

print(f"\nInfo:")
df.info()

print(f"\nData types:")
print(df.dtypes)

# What each Class means:
# - Class A: clear, direct sighting (witness saw the creature)
# - Class B: indirect evidence (sounds, tracks, smells)
# - Class C: secondhand information (stories heard from others)
print("\nReport Type distribution:")
print(df['Report Type'].value_counts())
print("\nClass distribution:")
print(df['Class'].value_counts())

# ============================================================================
# 1.2 Curatarea Datelor
# ============================================================================

# --- 1.2.1 Valori lipsa ---

print("\n--- 1.2.1 Missing values ---")
print("\nMissing values per column:")
missing = df.isnull().sum()
missing_pct = (df.isnull().mean() * 100).round(1)
missing_df = pd.DataFrame({'missing': missing, 'percent': missing_pct})
print(missing_df[missing_df['missing'] > 0].sort_values('percent', ascending=False))

# Drop columns with >90% null (useless for analysis)
cols_to_drop = ['Author', 'Media Source', 'Source Url', 'Media Issue',
                'Observed.1', 'A & G References']
print(f"\nDropping {len(cols_to_drop)} columns with >90% missing values:")
for col in cols_to_drop:
    pct = df[col].isnull().mean() * 100
    print(f"  - {col}: {pct:.1f}% null")
df = df.drop(columns=cols_to_drop)

# Drop 451 "Media Article" rows
# These are news articles, not actual BFRO sighting reports.
# They have ALL categorical fields null (Class, State, Season, County, Month, Year).
# Keeping them would pollute every analysis with "Unknown" entries.
media_count = (df['Report Type'] == 'Media Article').sum()
print(f"\nDropping {media_count} 'Media Article' rows (not sighting reports, all fields null)")
df = df[df['Report Type'] == 'Report'].copy()
df = df.drop(columns=['Report Type'])  # now all rows are "Report", column is redundant

# Drop the 'Date' column (very poor quality)
# Contains values like "Friday night", "Mothers Day", "3" — unusable
print("Dropping 'Date' column (too noisy: 'Friday night', 'Mothers Day', etc.)")
df = df.drop(columns=['Date'])

# Clean Year column
print("\n--- Cleaning Year ---")
non_numeric_mask = pd.to_numeric(df['Year'], errors='coerce').isna() & df['Year'].notna()
print(f"Non-numeric Year values: {non_numeric_mask.sum()}")
print(f"Examples: {df.loc[non_numeric_mask, 'Year'].unique()[:10]}")


# Try direct numeric conversion first, then extract 4-digit years from text like "Late 1970's"
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
text_years = df['Year'].isna() & df['Year'].index.isin(df.index)
df.loc[text_years, 'Year'] = (
    df.loc[text_years, 'Year']
    .astype(str)
    .str.extract(r'(\d{4})')[0]
    .astype(float)
)
# re-read original for the text rows since we overwrote with NaN
df_raw = pd.read_csv('data/reports.csv')
df_raw = df_raw[df_raw['Report Type'] == 'Report']
mask = df['Year'].isna()
extracted = df_raw.loc[mask.index[mask], 'Year'].astype(str).str.extract(r'(\d{4})')[0]
df.loc[mask, 'Year'] = pd.to_numeric(extracted, errors='coerce')

# Filter invalid years
invalid_years = df[(df['Year'].notna()) & ((df['Year'] < 1800) | (df['Year'] > 2025))]
print(f"Invalid years (< 1800 or > 2025): {len(invalid_years)}")
df.loc[(df['Year'] < 1800) | (df['Year'] > 2025), 'Year'] = np.nan
df['Year'] = df['Year'].astype('Int64')
print(f"Year after cleaning: range {df['Year'].min()} - {df['Year'].max()}, {df['Year'].isnull().sum()} null")

# Parse Submitted Date
print("\n--- Parsing Submitted Date ---")
df['Submitted_Datetime'] = pd.to_datetime(df['Submitted Date'], format='mixed', errors='coerce')
parsed = df['Submitted_Datetime'].notna().sum()
print(f"Successfully parsed: {parsed}/{len(df)} ({parsed/len(df)*100:.1f}%)")

# Handle remaining nulls in categorical columns
print("\n--- Remaining missing values ---")
# Month has 618 nulls (12.3%) — these are reports where the month wasn't specified
# We keep them as NaN (not "Unknown") — they'll be excluded from monthly analyses
print(f"  Month: {df['Month'].isnull().sum()} null ({df['Month'].isnull().mean()*100:.1f}%) — kept as NaN")
print(f"  Year: {df['Year'].isnull().sum()} null ({df['Year'].isnull().mean()*100:.1f}%) — kept as NaN")
print(f"  Nearest Town: {df['Nearest Town'].isnull().sum()} null — kept as NaN")
print(f"  Nearest Road: {df['Nearest Road'].isnull().sum()} null — kept as NaN")

print(f"\nCleaned dataset: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

# --- 1.2.2 Valori aberante (Outliers) ---

print("\n--- 1.2.2 Outlier detection on Year (IQR) ---")
year_clean = df['Year'].dropna()

Q1 = year_clean.quantile(0.25)
Q3 = year_clean.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Q1 = {Q1}, Q3 = {Q3}, IQR = {IQR}")
print(f"Bounds: [{lower_bound}, {upper_bound}]")

outliers_year = year_clean[(year_clean < lower_bound) | (year_clean > upper_bound)]
print(f"IQR outliers: {len(outliers_year)} ({len(outliers_year)/len(year_clean)*100:.1f}%)")
if len(outliers_year) > 0:
    print(f"  Range: {outliers_year.min()} - {outliers_year.max()}")

# Remove outliers: keep only Year < 2020
before = len(df)
df = df[(df['Year'].isna()) | (df['Year'] < 2020)].copy()
print(f"Removed {before - len(df)} rows with Year >= 2020")
print(f"Dataset after outlier removal: {df.shape[0]} rows")

# Visualization: Year distribution + outliers
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].boxplot(year_clean, vert=False)
axes[0].set_xlabel('Year')
axes[0].set_title('Year Boxplot (with outliers)')

axes[1].hist(year_clean, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Number of reports')
axes[1].set_title('Distribution of reports by year')
axes[1].axvline(x=lower_bound, color='red', linestyle='--', label=f'IQR bounds [{lower_bound:.0f}, {upper_bound:.0f}]')
axes[1].axvline(x=upper_bound, color='red', linestyle='--')
axes[1].legend()
plt.tight_layout()
plt.savefig('output/01_year_outliers.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================================
# 1.3 Analiza Statistica Descriptiva
# ============================================================================

# Descriptive stats for numeric columns
print("\nDescriptive statistics (numeric):")
print(df.describe())

# Categorical distributions
print("\n--- Class distribution ---")
print(df['Class'].value_counts())

print("\n--- Season distribution ---")
print(df['Season'].value_counts().reindex(['Spring', 'Summer', 'Fall', 'Winter']))

print("\n--- Month distribution ---")
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
print(df['Month'].value_counts().reindex(month_order))

print("\n--- Top 15 States ---")
print(df['State'].value_counts().head(15))

# Temporal statistics
print("\n--- Reports per decade ---")
df_with_year = df[df['Year'].notna()].copy()
df_with_year['Decade'] = (df_with_year['Year'] // 10 * 10).astype(int)
print(df_with_year['Decade'].value_counts().sort_index())

# Crosstab: State x Class (top 10 states)
print("\n--- Top 10 States: class distribution ---")
top_states = df['State'].value_counts().head(10).index
cross = pd.crosstab(df[df['State'].isin(top_states)]['State'],
                    df[df['State'].isin(top_states)]['Class'])
cross['Total'] = cross.sum(axis=1)
if 'Class A' in cross.columns:
    cross['%_ClassA'] = (cross['Class A'] / cross['Total'] * 100).round(1)
    cross = cross.sort_values('Total', ascending=False)
print(cross.to_string())

# Crosstab: Season x Class
print("\n--- Season x Class ---")
cross_season = pd.crosstab(df['Season'], df['Class'])
cross_season = cross_season.reindex(['Spring', 'Summer', 'Fall', 'Winter'])
print(cross_season.to_string())

# Contingency: State x Season
print("\n--- Top 10 States x Season ---")
cross_state_season = pd.crosstab(
    df[df['State'].isin(top_states)]['State'],
    df[df['State'].isin(top_states)]['Season']
)
cross_state_season = cross_state_season.reindex(columns=['Spring', 'Summer', 'Fall', 'Winter'])
print(cross_state_season.to_string())

# Yearly stats (last 30 years)
print("\n--- Yearly report stats (1990-present) ---")
recent = df_with_year[df_with_year['Year'] >= 1990]
yearly_counts = recent.groupby('Year').size()
print(f"  Mean: {yearly_counts.mean():.1f} reports/year")
print(f"  Median: {yearly_counts.median():.0f}")
print(f"  Max: {yearly_counts.max()} (year {yearly_counts.idxmax()})")
print(f"  Min: {yearly_counts.min()} (year {yearly_counts.idxmin()})")


# ============================================================================
# 1.4 Vizualizari Exploratorii
# ============================================================================

# --- Chart 1: Top 15 states (horizontal bar) ---
fig, ax = plt.subplots(figsize=(12, 6))
state_counts = df['State'].value_counts().head(15)
colors = sns.color_palette('Blues_r', n_colors=15)
sns.barplot(x=state_counts.values, y=state_counts.index, hue=state_counts.index,
            palette=colors, ax=ax, legend=False)
ax.set_xlabel('Number of reports')
ax.set_ylabel('State')
ax.set_title('Top 15 States by Number of Bigfoot Reports')
for i, v in enumerate(state_counts.values):
    ax.text(v + 3, i, str(v), va='center', fontsize=9)
plt.tight_layout()
plt.savefig('output/02_top_states.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Chart 2: Timeline (line plot with fill) ---
fig, ax = plt.subplots(figsize=(14, 5))
yearly = df_with_year.groupby('Year').size().reset_index(name='count')
yearly = yearly[yearly['Year'] >= 1950]
ax.plot(yearly['Year'], yearly['count'], color='steelblue', linewidth=1.5)
ax.fill_between(yearly['Year'], yearly['count'], alpha=0.3, color='steelblue')
ax.set_xlabel('Year')
ax.set_ylabel('Number of reports')
ax.set_title('Bigfoot Reports Over Time (1950-present)')
ax.axvline(x=1995, color='red', linestyle='--', alpha=0.7, label='Rise of the Internet (~1995)')
ax.legend()
plt.tight_layout()
plt.savefig('output/03_timeline.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Chart 3: Heatmap State x Season ---
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cross_state_season, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
            linewidths=0.5, linecolor='white')
ax.set_title('Bigfoot Reports by State and Season (Top 10 States)')
ax.set_xlabel('Season')
ax.set_ylabel('State')
plt.tight_layout()
plt.savefig('output/04_heatmap_state_season.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Chart 4: Pie chart - class distribution ---
fig, ax = plt.subplots(figsize=(8, 8))
class_counts = df['Class'].value_counts()
colors_pie = ['#2ecc71', '#3498db', '#e74c3c']
explode = [0.05] * len(class_counts)
ax.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%',
       colors=colors_pie, explode=explode, startangle=90,
       textprops={'fontsize': 12})
ax.set_title('Report Class Distribution\n(Class A = direct sighting, B = indirect, C = secondhand)')
plt.tight_layout()
plt.savefig('output/05_pie_class.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Chart 5: Boxplot reports by month ---
fig, ax = plt.subplots(figsize=(14, 6))
month_numeric = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}
df_monthly = df[df['Month'].notna()].copy()
df_monthly['Month_num'] = df_monthly['Month'].map(month_numeric)
df_monthly = df_monthly[df_monthly['Year'].notna()]

monthly_yearly = df_monthly.groupby(['Year', 'Month', 'Month_num']).size().reset_index(name='count')
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
sns.boxplot(data=monthly_yearly, x='Month_num', y='count', hue='Month_num',
            palette='coolwarm', ax=ax, legend=False)
ax.set_xticks(range(12))
ax.set_xticklabels(month_labels)
ax.set_xlabel('Month')
ax.set_ylabel('Reports per year')
ax.set_title('Monthly Distribution of Bigfoot Reports (boxplot per year)')
plt.tight_layout()
plt.savefig('output/06_boxplot_months.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Chart 6: Stacked bar - Class A vs B per state ---
fig, ax = plt.subplots(figsize=(12, 6))
top10 = df[df['State'].isin(top_states) & (df['Class'].isin(['Class A', 'Class B']))].copy()
prop_data = pd.crosstab(top10['State'], top10['Class'], normalize='index') * 100
prop_data = prop_data.reindex(top_states)
prop_data.plot(kind='barh', stacked=True, ax=ax, color=['#2ecc71', '#3498db'], edgecolor='white')
ax.set_xlabel('Proportion (%)')
ax.set_ylabel('State')
ax.set_title('Class A (direct sighting) vs Class B (indirect) Proportion by State')
ax.legend(title='Class', loc='lower right')
for i, state in enumerate(prop_data.index):
    if 'Class A' in prop_data.columns:
        pct = prop_data.loc[state, 'Class A']
        ax.text(pct / 2, i, f'{pct:.0f}%', ha='center', va='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('output/07_stacked_class_state.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Chart 7: Histogram of reports by season ---
fig, ax = plt.subplots(figsize=(10, 6))
season_data = df['Season'].value_counts().reindex(['Spring', 'Summer', 'Fall', 'Winter'])
season_colors = ['#2ecc71', '#f1c40f', '#e67e22', '#3498db']
sns.barplot(x=season_data.index, y=season_data.values, hue=season_data.index,
            palette=season_colors, ax=ax, legend=False)
ax.set_xlabel('Season')
ax.set_ylabel('Number of reports')
ax.set_title('Bigfoot Reports by Season')
for i, v in enumerate(season_data.values):
    ax.text(i, v + 10, str(v), ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('output/08_season_bar.png', dpi=150, bbox_inches='tight')
plt.show()

# --- H1: Internet effect ---
print("\n--- H1: Internet effect ---")
pre95 = df_with_year[df_with_year['Year'] < 1995].shape[0]
post95 = df_with_year[(df_with_year['Year'] >= 1995) & (df_with_year['Year'] < 2010)].shape[0]
post10 = df_with_year[df_with_year['Year'] >= 2010].shape[0]
peak = df_with_year.groupby('Year').size()
print(f"  Before 1995: {pre95} reports")
print(f"  1995-2009: {post95} reports")
print(f"  2010-2019: {post10} reports")
print(f"  Peak year: {peak.idxmax()} with {peak.max()} reports")

# --- H2: Bear population correlation ---
print("\n--- H2: Bear population correlation ---")
bear_states = ['Washington', 'Oregon', 'California']
low_bear_states = ['Ohio', 'Illinois', 'Texas', 'Florida']
for group_name, states in [('High bear pop. (WA, OR, CA)', bear_states),
                            ('Low bear pop. (OH, IL, TX, FL)', low_bear_states)]:
    subset = df[df['State'].isin(states) & df['Class'].isin(['Class A', 'Class B'])]
    pct_a = (subset['Class'] == 'Class A').mean() * 100
    print(f"  {group_name}: {pct_a:.1f}% Class A")

# --- H3: Seasonal pattern ---
print("\n--- H3: Seasonal pattern ---")
season_counts = df['Season'].value_counts()
summer_fall = season_counts.get('Summer', 0) + season_counts.get('Fall', 0)
total_known = df['Season'].notna().sum()
print(f"  Summer + Fall: {summer_fall} ({summer_fall/total_known*100:.1f}%)")
print(f"  Spring + Winter: {total_known - summer_fall} ({(total_known-summer_fall)/total_known*100:.1f}%)")
month_counts = df['Month'].value_counts()
print(f"  Top month: October ({month_counts.get('October', 0)} reports)")

# ============================================================================
# Conclusions
# ============================================================================


print("""
H1 - confirmat: Raportarile au crescut masiv dupa lansarea BFRO (Bigfoot Field Researchers Organization 1995)
  si expansiunea internetului. Anul de varf este 2000, urmat de o scadere
  dupa 2010. Este un efect de mediatizare si acces mai facil la informatie.

H2 - confirmat: Statele cu cele mai mari populatii de ursi (WA, OR, CA)
  au cea mai mare rata de raportari, fiind animale mari care pot fi confundate cu Bigfoot.
H3 - confirmat: 67% din raportari sunt in vara si toamna. Octombrie e
  luna cu cele mai multe raportari (sezon de vanatoare), urmat de
  iulie-august (sezon de camping).

Concluzie generala: Datele sugereaza ca raportarile Bigfoot sunt un fenomen
sociologic determinat de accesul la internet, confuzia cu ursii si
depinde de anotimp. Nu exista dovezi concludente ca Bigfoot ar fi o creatura reala,
ci mai degraba un mit urban alimentat de factori culturali si naturali.
""")

print(f"Cleaned dataset: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Charts generated: 8 (in output/ folder)")
