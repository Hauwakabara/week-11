import pandas as pd
import plotly.express as px

# Load real Gapminder data
df = px.data.gapminder()

# Pick 8 countries x 12 years = 96 rows
countries = ["Nigeria", "Kenya", "Brazil", "China", "India", "United States", "France", "Australia"]
df = df[df["country"].isin(countries)].copy()
df = df.sort_values(["country", "year"]).reset_index(drop=True)
print("Rows:", len(df))

# Create exactly 17 instructional blanks
# 5x continent, 6x lifeExp, 6x gdpPercap = 17
import numpy as np
np.random.seed(11)

# Pick rows to blank
continent_idx = df.sample(5, random_state=1).index
lifeExp_idx = df.sample(6, random_state=2).index
gdp_idx = df.sample(6, random_state=3).index

df.loc[continent_idx, "continent"] = None
df.loc[lifeExp_idx, "lifeExp"] = None
df.loc[gdp_idx, "gdpPercap"] = None

print("Total missing:", int(df.isna().sum().sum()))
print(df.isna().sum())

df.to_csv("gapminder_instructional_missing.csv", index=False)
print("Saved gapminder_instructional_missing.csv")