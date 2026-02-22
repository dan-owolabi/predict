import pandas as pd
df = pd.read_csv('understat_xg.csv')
print(f"Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head().to_string())
print(f"\nColumn types:")
print(df.dtypes)
print(f"\nNull counts:")
print(df.isnull().sum())
print(f"\nBasic stats:")
for c in df.columns:
    if df[c].dtype in ['float64', 'int64']:
        print(f"  {c}: min={df[c].min():.2f}, max={df[c].max():.2f}, mean={df[c].mean():.2f}")
if 'date' in [c.lower() for c in df.columns]:
    date_col = [c for c in df.columns if c.lower() == 'date'][0]
    print(f"\nDate range: {df[date_col].min()} to {df[date_col].max()}")
if 'team' in [c.lower() for c in df.columns]:
    team_col = [c for c in df.columns if c.lower() == 'team'][0]
    print(f"\nTeams: {sorted(df[team_col].unique())}")
# Check for home/away or team structure
print(f"\nSample of unique values per column:")
for c in df.columns:
    if df[c].dtype == 'object':
        u = df[c].unique()
        print(f"  {c}: {len(u)} unique, samples: {list(u[:10])}")
