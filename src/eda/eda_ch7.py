# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter

# Set plot theme and figures/log directory path
sns.set_theme(style="whitegrid")
FIG_DIR = "reports/figures/ch7"
LOG_PATH = "reports/ch7_eda_summary.md"

cols = pd.read_csv("data/raw/train_transaction.csv", nrows=0).columns
dtype_map = {c: 'float32' for c in cols if c.startswith('V')}  # V-cols: float32 is plenty
dtype_map['isFraud'] = 'int8'
dtype_map['TransactionID'] = 'int32'

# Read data section and column formatting
tx = pd.read_csv("data/raw/train_transaction.csv")
idn = pd.read_csv("data/raw/train_identity.csv")
tx.columns = tx.columns.str.lower()
idn.columns = idn.columns.str.lower()

# Function for logging steps
log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)

# %% - Integrity checks
log("## Integrity Checks")
log(f"train_transaction shape: {tx.shape}")
log(f"train_identity shape: {idn.shape}")

# Duplicate key check 
log(f"Duplicate transactionid in tx: {tx['transactionid'].duplicated().sum()}")
log(f"Duplicate transactionid in idn: {idn['transactionid'].duplicated().sum()}")

# Label sanity
log(f"isfraud unique values: {tx['isfraud'].unique()}")

# Amount sanity
log(f"transactionamt min/max: {tx['transactionamt'].min()} / {tx['transactionamt'].max()}")
log(f"transactionamt <= 0 count: {(tx['transactionamt'] <= 0).sum()}")

# Identity coverage (re-confirm Step 8 number here for the log file)
coverage = idn['transactionid'].nunique() / tx['transactionid'].nunique()
log(f"Identity coverage: {coverage:.2%}")

# Cardinality of key categoricals
for col in ["productcd", "card4", "card6"]:
    log(f"{col} value counts:\n{tx[col].value_counts(dropna=False)}")

# %% Class balance (Figure 7.1)
fraud_counts = tx['isfraud'].value_counts(normalize=True) * 100
log(f"Class balance: {fraud_counts.to_dict()}")

fig, ax = plt.subplots(figsize=(5, 4))
sns.countplot(x='isfraud', data=tx, ax=ax)
ax.set_xticklabels(['Legitimate', 'Fraud'])

### ax.set_title("Figure 7.1 — Class Distribution")
ax.set_xlabel("")
ax.set_ylabel("Count")
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

ax.set_ylim(0, 700000)
for p in ax.patches:
    ax.annotate(
        f'{int(p.get_height()):,}',
        (p.get_x() + p.get_width() / 2, p.get_height()),
        ha='center',
        va='bottom',
        xytext=(0, 9),
        textcoords='offset points'
    )

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_7_1_class_balance.png", dpi=150)
plt.close()
# %% Transaction amount distribution (Figure 7.2)
fig, ax = plt.subplots(figsize=(9, 4.5))

# Using KDE plot with log scaling for a cleaner, more insightful plot
sns.kdeplot(
    data=tx,
    x='transactionamt',
    hue='isfraud',
    log_scale=True,
    common_norm=False,  # Evaluating each class density independently
    palette={0: '#1f77b4', 1: '#d62728'},
    linewidth=2,
    ax=ax
)

# Explicitly set tick positions and custom dollar labels
ticks = [1, 10, 100, 1000, 10000]
ax.set_xticks(ticks)
ax.set_xticklabels(['$1', '$10', '$100', '$1,000', '$10,000'])

ax.set_xlabel("Transaction Amount (USD, log scale)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_ylim(0, 2.0) 

# Rename legend labels cleanly
handles, _ = ax.get_legend().legend_handles if hasattr(ax.get_legend(), 'legend_handles') else (ax.get_legend().legendHandles, None)
ax.legend(handles=ax.get_legend().legend_handles, labels=['Legitimate (0)', 'Fraud (1)'], title=None, frameon=True)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_7_2_amount_distribution.png", dpi=300)
plt.close()

log(tx.groupby('isfraud')['transactionamt'].describe().to_string())
# %% Temporal structure (Figure 7.3)
tx['day'] = tx['transactiondt'] / (24 * 60 * 60)
log(f"Dataset spans approximately {tx['day'].max():.0f} days")

# ax.set_title("Figure 7.3 — Temporal Structura of Transaction Volume and Fraud Rate")
df_daily = tx.groupby(tx['day'].astype(int))['isfraud'].agg(
    ['count', 'mean']).reset_index()
df_daily['fraud_pct'] = df_daily['mean'] * 100
df_daily['count_7d'] = df_daily['count'].rolling(7, min_periods=1).mean()
df_daily['fraud_pct_7d'] = df_daily['fraud_pct'].rolling(7, min_periods=1).mean()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

# Top Plot: Transaction Volume
ax1.plot(
    df_daily['day'],
    df_daily['count'],
    alpha=0.3,
    color='tab:blue',
    label='Daily Transaction Count',
)
ax1.plot(
    df_daily['day'],
    df_daily['count_7d'],
    color='tab:blue',
    lw=2,
    label='7-Day Average',
)
ax1.set_ylabel('Transaction Volume')
ax1.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax1.legend(loc='upper right')
ax1.grid(True, linestyle='--', alpha=0.5)

# Bottom Plot: Fraud Rate
ax2.plot(
    df_daily['day'],
    df_daily['fraud_pct'],
    alpha=0.3,
    color='tab:red',
    label='Daily Fraud Rate',
)
ax2.plot(
    df_daily['day'],
    df_daily['fraud_pct_7d'],
    color='tab:red',
    lw=2,
    label='7-Day Average',
)
ax2.set_ylabel('Fraud Rate (%)')
ax2.set_xlabel('Time (Days)')
ax2.legend(loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.5)

# plt.suptitle('Figure 7.3: Temporal Structure of Transactions and Fraud Rate')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_7_3_temporal_structure.png", dpi=150)
plt.close()
# %% Categorical fields and fraud rate (Tables 7.1–7.2)
for col in ["productcd", "card4", "card6"]:
    tbl = tx.groupby(col)['isfraud'].agg(['count', 'mean']).sort_values('mean', ascending=False)
    tbl.columns = ['transaction_count', 'fraud_rate']
    log(f"### Fraud rate by {col}\n{tbl.to_string()}")
    tbl.to_csv(f"reports/tables/table_7_{col}.csv")

# %% Missingness analysis (Figure 7.4)
missing_pct = tx.isnull().mean().sort_values(ascending=False) * 100

# Group by column family for a readable summary rather than 394 individual bars
def col_group(c):
    if c.startswith('v'): return 'V-columns (anonymized)'
    if c.startswith('c') and c[1:].isdigit(): return 'C-columns'
    if c.startswith('d') and c[1:].isdigit(): return 'D-columns'
    if c.startswith('m') and c[1:].isdigit(): return 'M-columns'
    return 'Named/business fields'

missing_by_group = missing_pct.groupby(col_group).mean().sort_values(ascending=False)
log(f"Mean missingness by column group:\n{missing_by_group.to_string()}")

fig, ax = plt.subplots(figsize=(7, 4))
missing_by_group.plot(kind='barh', ax=ax)
#### ax.set_title("Figure 7.4 — Mean Missingness by Column Group")
ax.set_xlabel("% Missing")
ax.set_xlim(0, 60) 

ax.grid(axis='x', linestyle='--', alpha=0.5)
ax.grid(axis='y', visible=False)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_7_4_missingness.png", dpi=150)
plt.close()

# Identity table missingness separately, since it's a different join population
idn_missing = idn.isnull().mean().sort_values(ascending=False) * 100
log(f"Top 10 most-missing identity columns:\n{idn_missing.head(10).to_string()}")
# %% Correlation analysis on named fields (Figure 7.5)

# ax.set_title("Figure 7.5 — Correlation Matrix, Named Numeric Fields")
# Cleaner feature display names
feature_labels = [
    'Transaction Amount', 'Card 1', 'Card 2', 'Card 3', 'Card 5',
    'Address 1', 'Address 2', 'Distance 1', 'Distance 2', 'Is Fraud']
# Calculate correlation matrix
named_numeric = ['transactionamt', 'card1', 'card2', 'card3', 'card5',
                 'addr1', 'addr2', 'dist1', 'dist2', 'isfraud']
corr = tx[named_numeric].corr()

sns.set_theme(style="white")
fig, ax = plt.subplots(figsize=(8, 6.5))
# Generate heat map
sns.heatmap(
    corr, 
    annot=True, 
    fmt=".2f", 
    cmap="coolwarm", 
    vmin=-1, vmax=1, center=0,
    square=True, 
    linewidths=0.5, 
    cbar_kws={"shrink": 0.8},
    xticklabels=feature_labels,
    yticklabels=feature_labels,
    ax=ax
)
# Rotate x-axis labels 30 degrees and align correctly to the right
plt.xticks(rotation=30, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)

# Remove in-figure title
ax.set_title("")

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig_7_5_correlation.png", dpi=300)
plt.close()
# %% Write Summary
with open(LOG_PATH, "w") as f:
    f.write("\n\n".join(log_lines))
print(f"Summary written to {LOG_PATH}")