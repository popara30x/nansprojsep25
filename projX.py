#biblioteke
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

#spisak godina i drzava
years = list(range(2013, 2023))          # 2013-2022
years_predict = list(range(2023, 2033))  # 2023-2032
states = [
    "Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut","Delaware",
    "District of Columbia","Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa",
    "Kansas","Kentucky","Louisiana","Maine","Maryland","Massachusetts","Michigan","Minnesota",
    "Mississippi","Missouri","Montana","Nebraska","Nevada","New Hampshire","New Jersey","New Mexico",
    "New York","North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania",
    "Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah","Vermont",
    "Virginia","Washington","West Virginia","Wisconsin","Wyoming",
]

#numericke kolone
numeric_cols = ["GasTax","PerCapita","Population","VehicleAmount",
                "VehiclesPerCapita","IncomeTax","OilPrice","Inflation","YearNum"]

#ucitavanje CSV-a
df_oil = pd.read_csv("annual average crude oil price, 2013-2022 (gallons).csv")
df_gas_price = pd.read_csv("gasoline prices, 2013-2022.csv")
df_inflation = pd.read_csv("inflation, 2013-2022.csv")
df_gas_tax = pd.read_csv("gasoline tax rates, 2013-2022, dollars.csv")
df_per_capita = pd.read_csv("per capita personal income, 2013-2022.csv")
df_pop = pd.read_csv("population, 2013-2022.csv")
df_tot_veh = pd.read_csv("amount of vehicles per state, 2013-2022.csv")

#preuredjivanje/"topljenje"
tax_long = df_gas_tax.melt(id_vars="State", var_name="Year", value_name="GasTax")
per_capita_long = df_per_capita.melt(id_vars="State", var_name="Year", value_name="PerCapita")
pop_long = df_pop.melt(id_vars="State", var_name="Year", value_name="Population")
tot_veh_long = df_tot_veh.melt(id_vars="State", var_name="Year", value_name="VehicleAmount")

df_gas_price["Year"] = df_gas_price["Year"].astype(int)
for df in [tax_long, per_capita_long, pop_long, tot_veh_long]:
    df["Year"] = df["Year"].astype(int)

#spajanje datasetova
df_merge = (
    tax_long
    .merge(per_capita_long, on=["State","Year"])
    .merge(pop_long, on=["State","Year"])
    .merge(tot_veh_long, on=["State","Year"])
    .merge(df_gas_price, on="Year")
    .merge(df_oil, on="Year")
    .merge(df_inflation, on="Year")
)

#izvedena svojstva
df_merge["StateGasPrice"] = df_merge["GasPrice"] + df_merge["GasTax"]
df_merge["VehiclesPerCapita"] = df_merge["VehicleAmount"] / df_merge["Population"]
df_merge["IncomeTax"] = df_merge["GasTax"] / df_merge["PerCapita"]

# brojanje godina (0 = 2013, 1 = 2014 itd.)
df_merge["YearNum"] = df_merge["Year"] - df_merge["Year"].min()

#backtesting
bt_results = []
window = 3  # br godina po iteraciji

# najraniji testovi krecu za 2016. godinu
for start_year in range(2016, 2023 - window + 1): 
    test_years = list(range(start_year, start_year + window))

    bt_train_df = df_merge[df_merge['Year'] < start_year].copy()
    bt_test_df  = df_merge[df_merge['Year'].isin(test_years)].copy()

    if bt_train_df.empty or bt_test_df.empty:
        continue

    # popunjavanje nedostajucih num. vrednosti
    bt_train_df[numeric_cols] = bt_train_df[numeric_cols].fillna(0)
    bt_test_df[numeric_cols]  = bt_test_df[numeric_cols].fillna(0)

    # skaliranje
    bt_scaler = StandardScaler()
    x_train_num = pd.DataFrame(bt_scaler.fit_transform(bt_train_df[numeric_cols]),
                               columns=numeric_cols, index=bt_train_df.index)
    x_test_num  = pd.DataFrame(bt_scaler.transform(bt_test_df[numeric_cols]),
                               columns=numeric_cols, index=bt_test_df.index)

    # one-hot kodiranje
    x_train_state = pd.get_dummies(bt_train_df['State'], drop_first=True)
    x_test_state  = pd.get_dummies(bt_test_df['State'], drop_first=True)
    x_test_state  = x_test_state.reindex(columns=x_train_state.columns, fill_value=0)

    # kombinovanje numerickih svojstava i onih na nivou drzava
    x_train_combined = pd.concat([x_train_num, x_train_state], axis=1)
    x_test_combined  = pd.concat([x_test_num,  x_test_state],  axis=1)

    y_train_bt = bt_train_df['StateGasPrice']
    y_test_bt  = bt_test_df['StateGasPrice']

    model_bt = LinearRegression()
    model_bt.fit(x_train_combined, y_train_bt)
    y_pred_bt = model_bt.predict(x_test_combined)

    rmse = mean_squared_error(y_test_bt, y_pred_bt, squared=False)
    r2   = r2_score(y_test_bt, y_pred_bt)

    bt_results.append({
        "TestYears": f"{start_year}-{start_year + window - 1}",
        "RMSE": rmse,
        "R2": r2
    })

bt_results_df = pd.DataFrame(bt_results)
print(bt_results_df)

scaler = StandardScaler()

#treniranje poslednjeg modela
x_full_num = pd.DataFrame(scaler.fit_transform(df_merge[numeric_cols]), columns=numeric_cols)
x_full_state = pd.get_dummies(df_merge['State'], drop_first=True)
x_full_combined = pd.concat([x_full_num, x_full_state], axis=1)
y_full = df_merge['StateGasPrice']

model = LinearRegression()
model.fit(x_full_combined, y_full)


#proslost (2013-2022)
#procenjene stope rasta
growth_rates = {
    "GasTax": 0.015,
    "PerCapita": 0.03,
    "Population": 0.0075,
    "VehicleAmount": 0.012,
    "OilPrice": 0.025,
    "Inflation": 0.02
}

df_future = pd.DataFrame([(s, y) for s in states for y in years_predict],
                         columns=["State", "Year"])

# za bazu se stavljaju
df_latest = df_merge[df_merge["Year"] == 2022][
    ["State", "GasTax", "PerCapita", "Population", "VehicleAmount", "OilPrice", "Inflation"]
].copy()
df_future = df_future.merge(df_latest, on="State", how="left")

for col, rate in growth_rates.items():
    df_future[col] = df_future[col] * ((1 + rate) ** (df_future["Year"] - 2022))

# izvedena svojstva
df_future["VehiclesPerCapita"] = df_future["VehicleAmount"] / df_future["Population"]
df_future["IncomeTax"] = df_future["GasTax"] / df_future["PerCapita"]

# normalizacija godina
df_future["YearNum"] = (df_future["Year"] - df_merge["Year"].min()) / (df_future["Year"].max() - df_merge["Year"].min())

# -----------------------------
# treniranje konacnog modela
# -----------------------------
# numericke + drzave
x_hist_num = pd.DataFrame(scaler.fit_transform(df_merge[numeric_cols]), columns=numeric_cols)
x_hist_state = pd.get_dummies(df_merge['State'], drop_first=True)
x_hist_combined = pd.concat([x_hist_num, x_hist_state], axis=1)

y_hist = df_merge['StateGasPrice']

# Train model
final_model = LinearRegression()
final_model.fit(x_hist_combined, y_hist)
# -----------------------------
# projekcije za buducnost
# -----------------------------

#procenjene stope rasta
growth_rates = {
    "GasTax": 0.015,         # 1.5% per year
    "PerCapita": 0.03,       # 3% per year
    "Population": 0.0075,    # 0.75% per year
    "VehicleAmount": 0.012,  # 1.2% per year
    "OilPrice": 0.025,       # 2.5% per year
    "Inflation": 0.02        # 2% per year
}

# pravljenje buduceg df
df_future = pd.DataFrame(
    [(s, y) for s in states for y in years_predict],
    columns=["State", "Year"]
)

# vrednosti iz 2022 kao osnova
df_latest = df_merge[df_merge["Year"] == 2022][
    ["State", "GasTax", "PerCapita", "Population", "VehicleAmount", "OilPrice", "Inflation"]
].copy()

df_future = df_future.merge(df_latest, on="State", how="left")

# rast
for col, rate in growth_rates.items():
    df_future[col] = df_future[col] * ((1 + rate) ** (df_future["Year"] - 2022))

# izvedena svojstva
df_future["VehiclesPerCapita"] = df_future["VehicleAmount"] / df_future["Population"]

# iste kolone kao na treningu?
df_future["IncomeTax"] = df_future["GasTax"] / df_future["PerCapita"]

# normalizacija godina
df_future["YearNum"] = (df_future["Year"] - 2022) / (2032 - 2022)

# --- glatki prelaz za 2-3 god ---
blend_years = [2023, 2024]
blend_factor = 0.5

for yr in blend_years:
    idx = df_future["Year"] == yr
    for col in ["GasTax", "PerCapita", "Population", "VehicleAmount", "OilPrice", "Inflation"]:
        base_val = df_latest.set_index("State")[col]
        projected_val = df_future.loc[idx, col]
        df_future.loc[idx, col] = (1 - blend_factor) * base_val.values + blend_factor * projected_val.values

# pripremanje num. svoj. za predvidjanje
numeric_cols_future = [
    "GasTax", "PerCapita", "Population", "VehicleAmount",
    "VehiclesPerCapita", "IncomeTax", "OilPrice", "Inflation", "YearNum"
]

x_future_num = pd.DataFrame(
    scaler.transform(df_future[numeric_cols_future]),
    columns=numeric_cols_future
)

# one-hot encode
x_future_state = pd.get_dummies(df_future["State"], drop_first=True)
x_future_state = x_future_state.reindex(columns=x_train_state.columns, fill_value=0)

# kombinacija numerickih i drzavnih svojstava
x_future_combined = pd.concat([x_future_num, x_future_state], axis=1)

# predvidjanja cena goriva
df_future["PredictedStateGasPrice"] = model.predict(x_future_combined)

# preview
print(pd.concat([df_future.head(10), df_future.tail(10)], ignore_index=True))


#vizuelizacija
def plot_state_trends(df_hist):
    plt.figure(figsize=(14,7))
    sb.lineplot(data=df_hist, x='Year', y='StateGasPrice', hue='State', marker='o')
    plt.title("prosla cena goriva (1 gal/3,78 L) po drzavama")
    plt.ylabel("dolara")
    plt.xlabel("godina")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.tight_layout()
    plt.show()

def plot_combined(df_hist, df_pred):
    df_pred_renamed = df_pred.rename(columns={'PredictedStateGasPrice':'StateGasPrice'})
    df_combined = pd.concat([df_hist[['Year','State','StateGasPrice']], 
                             df_pred_renamed[['Year','State','StateGasPrice']]])
    plt.figure(figsize=(14,7))
    sb.lineplot(data=df_combined, x='Year', y='StateGasPrice', hue='State', marker='o')
    plt.title("prosle i buduce cene goriva (1 gal/3,78 L)")
    plt.ylabel("dolara")
    plt.xlabel("godina")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.tight_layout()
    plt.show()

def plot_average(df_combined):
    df_avg = df_combined.groupby('Year')['StateGasPrice'].mean().reset_index()
    plt.figure(figsize=(10,5))
    plt.plot(df_avg['Year'], df_avg['StateGasPrice'], marker='o')
    plt.title("prosecna cena goriva (1 gal/3,78 L) po drzavama")
    plt.ylabel("dolara")
    plt.xlabel("godina")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_heatmap(df_future):
    df_pivot = df_future.pivot(index='State', columns='Year', values='PredictedStateGasPrice')
    plt.figure(figsize=(14,12))
    sb.heatmap(df_pivot, cmap='YlOrRd', linewidths=0.5, annot=True, fmt=".2f")
    plt.title("predvidjane cene goriva")
    plt.ylabel("drzava")
    plt.xlabel("godina")
    plt.tight_layout()
    plt.show()

# Plots
plot_state_trends(df_merge)
plot_combined(df_merge, df_future)
plot_average(pd.concat([
    df_merge[['Year','State','StateGasPrice']], 
    df_future.rename(columns={'PredictedStateGasPrice':'StateGasPrice'})[['Year','State','StateGasPrice']]
]))
plot_heatmap(df_future)
