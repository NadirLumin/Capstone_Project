import pandas as pd

df_syn = pd.read_csv('./data/cleaned_synonyms_data.csv')
df_syn["key"] = list(zip(df_syn["Outdated Term"], df_syn["POS"]))
print("All mapping keys:", df_syn["key"].tolist())
