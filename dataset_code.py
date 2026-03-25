

import numpy as np
import pandas as pd

np.random.seed(42)

TOTAL_ROWS = 10000

CLASS_COUNTS = {
    0: 3800,
    1: 2400,
    2: 1600,
    3: 1300,
    4: 900
}

DWELLINGS = ["Apartment", "Independent House", "Villa", "Commercial"]

def random_dates(n):
    start = pd.Timestamp("2022-01-01")
    end   = pd.Timestamp("2023-12-31")
    days = (end - start).days
    dates = start + pd.to_timedelta(np.random.randint(0, days+1, n), unit="D")
    return dates

def seasonal_temp_humidity(months):
    temp = np.zeros(len(months))
    hum  = np.zeros(len(months))

    # Summer: Mar–May
    summer = months.isin([3,4,5])
    temp[summer] = np.random.uniform(28, 38, summer.sum())
    hum[summer]  = np.random.uniform(55, 80, summer.sum())

    # Monsoon: Jun–Sep
    monsoon = months.isin([6,7,8,9])
    temp[monsoon] = np.random.uniform(24, 32, monsoon.sum())
    hum[monsoon]  = np.random.uniform(75, 95, monsoon.sum())

    # Winter: Oct–Feb
    winter = ~(summer | monsoon)
    temp[winter] = np.random.uniform(22, 30, winter.sum())
    hum[winter]  = np.random.uniform(45, 70, winter.sum())

    return temp, hum

def compute_outage(months):
    outage = np.zeros(len(months))

    monsoon = months.isin([6,7,8,9])

    for i in range(len(months)):
        if monsoon[i]:
            if np.random.rand() < 0.50:
                outage[i] = np.random.uniform(0.5, 8.0)
        else:
            if np.random.rand() < 0.25:
                outage[i] = np.random.uniform(0.5, 8.0)

    return outage

def make_peak_offpeak(actual_vals):
    peak = []
    off  = []
    for a in actual_vals:
        combined_ratio = np.random.uniform(0.80, 0.95)
        peak_share     = np.random.uniform(0.58, 0.72)
        p = a * combined_ratio * peak_share
        o = a * combined_ratio * (1 - peak_share)
        peak.append(f"{p:.2f} kWh")
        off.append(f"{o:.2f} kWh")
    return peak, off

def gen_meter_ids(n):
    nums = np.random.randint(0, 10**8, n)
    return [f"IN-KL-ELC-{x:08d}" for x in nums]

def generate_class(n, cls):

    df = pd.DataFrame()


    df["Meter_ID"] = gen_meter_ids(n)
    df["Dwelling_Type"] = np.random.choice(DWELLINGS, n)
    df["Num_Occupants"] = np.random.randint(1, 9, n)
    df["House_Area_sqft"] = np.random.randint(400, 5001, n)

    # Connected Load correlated with area
    load = (
        1.5
        + (df["House_Area_sqft"] - 400) / 4600 * 13.5
        + np.random.normal(0, 1.2, n)
    )
    df["Connected_Load_kW"] = np.clip(load, 1.5, 15.0)

    df["Meter_Age_Years"] = np.random.randint(1, 20, n)

    # -----------------------
    # Dates & weather
    # -----------------------
    dates = random_dates(n)
    months = dates.month
    df["Date"] = dates.strftime("%d-%m-%Y")

    temp, hum = seasonal_temp_humidity(months)
    df["Temperature_C"] = temp
    df["Humidity_pct"] = hum

    df["Grid_Outage_Hours"] = compute_outage(months)

    # Appliance score
    df["Appliance_Score"] = np.random.randint(1, 26, n)


    if cls == 0:
        df["Anomaly_Type"] = 0
        df["Meter_Bypass_Signal"] = 0
        df["Has_Solar_Panel"] = (np.random.rand(n) < 0.32).astype(int)

        df["Voltage_V"] = np.clip(np.random.normal(230, 5, n), 210, 250)

        pf = np.random.uniform(0.78, 0.99, n)
        old = df["Meter_Age_Years"] > 14
        pf[old] -= np.random.uniform(0.03, 0.08, old.sum())
        df["Power_Factor"] = np.clip(pf, 0.65, 0.99)

        exp = np.random.uniform(5, 60, n)
        solar_neg = (df["Has_Solar_Panel"] == 1) & (np.random.rand(n) < 0.35)
        exp[solar_neg] *= -1

        actual = exp.clip(1, None) * np.random.uniform(1.2, 2.5, n)

        # seasonal multiplier
        summer = months.isin([3,4,5])
        monsoon = months.isin([6,7,8,9])
        winter = ~(summer | monsoon)

        actual[summer] *= 1.15
        actual[monsoon] *= 0.95
        actual[winter] *= 0.90

        # AC effect
        hot = df["Temperature_C"] > 32
        actual[hot] *= np.random.uniform(1.10, 1.15, hot.sum())

    elif cls == 1:
        df["Anomaly_Type"] = 1

        bypass = np.zeros(n, dtype=int)
        bypass[:2040] = 1
        np.random.shuffle(bypass)
        df["Meter_Bypass_Signal"] = bypass

        df["Has_Solar_Panel"] = (np.random.rand(n) < 0.25).astype(int)
        df["Voltage_V"] = np.clip(np.random.normal(226, 7, n), 200, 252)
        df["Power_Factor"] = np.random.uniform(0.68, 0.85, n)

        exp = np.random.uniform(5, 45, n)
        actual = df["Connected_Load_kW"] * 24 * np.random.uniform(0.30, 0.70, n)

    elif cls == 2:
        df["Anomaly_Type"] = 2
        df["Meter_Bypass_Signal"] = 0
        df["Has_Solar_Panel"] = (np.random.rand(n) < 0.32).astype(int)

        df["Meter_Age_Years"] = np.random.triangular(6, 14, 19, n).astype(int)
        df["Voltage_V"] = np.clip(np.random.normal(230, 12, n), 195, 260)
        df["Power_Factor"] = np.random.uniform(0.65, 0.88, n)

        exp = np.random.uniform(10, 55, n)
        actual = np.clip(exp * np.random.uniform(1.5, 3.0, n), 20, 100)

    elif cls == 3:
        df["Anomaly_Type"] = 3
        df["Meter_Bypass_Signal"] = 0
        df["Has_Solar_Panel"] = 1

        df["Voltage_V"] = np.clip(np.random.normal(230, 6, n), 205, 252)
        df["Power_Factor"] = np.random.uniform(0.78, 0.95, n)

        exp = np.random.uniform(5, 30, n)
        neg_mask = np.random.rand(n) < 0.85
        exp[neg_mask] *= -1

        actual = np.random.uniform(5, 70, n)

    else:  # cls == 4
        df["Anomaly_Type"] = 4
        df["Meter_Bypass_Signal"] = 0
        df["Has_Solar_Panel"] = (np.random.rand(n) < 0.20).astype(int)

        df["Connected_Load_kW"] = np.random.uniform(8.0, 15.0, n)
        df["Voltage_V"] = np.clip(np.random.normal(228, 8, n), 200, 255)
        df["Power_Factor"] = np.random.uniform(0.65, 0.78, n)

        exp = np.random.uniform(20, 60, n)
        actual = np.clip(exp * np.random.uniform(3.0, 6.0, n), 80, 200)

    # ============================================================
    # kWh STRINGS
    # ============================================================

    df["Expected_Energy_kWh"] = [f"{x:.2f} kWh" for x in exp]
    df["Actual_Energy_kWh"]   = [f"{x:.2f} kWh" for x in actual]

    peak, off = make_peak_offpeak(actual)
    df["Peak_Hour_Usage_kWh"] = peak
    df["Off_Peak_Usage_kWh"]  = off

    return df




frames = []
for cls, cnt in CLASS_COUNTS.items():
    frames.append(generate_class(cnt, cls))

df = pd.concat(frames, ignore_index=True)


df["Temperature_C"] += np.random.normal(0, 0.3, len(df))
df["Humidity_pct"]  += np.random.normal(0, 0.5, len(df))
df["Voltage_V"]     += np.random.normal(0, 0.8, len(df))
df["Power_Factor"]  += np.random.normal(0, 0.01, len(df))

# Clip
df["Temperature_C"] = df["Temperature_C"].clip(22.0, 38.0)
df["Humidity_pct"]  = df["Humidity_pct"].clip(45.0, 95.0)
df["Voltage_V"]     = df["Voltage_V"].clip(195.0, 260.0)
df["Power_Factor"]  = df["Power_Factor"].clip(0.65, 0.99)

# ============================================================
# ROUNDING (MANDATORY STEP)
# ============================================================

float_cols = [
    "Connected_Load_kW",
    "Temperature_C",
    "Humidity_pct",
    "Voltage_V",
    "Power_Factor",
    "Grid_Outage_Hours"
]
df[float_cols] = df[float_cols].round(2)



type0_idx = df[df["Anomaly_Type"] == 0].index
zero_peak_idx = np.random.choice(type0_idx, 12, replace=False)
zero_off_idx  = np.random.choice(list(set(type0_idx) - set(zero_peak_idx)), 8, replace=False)

df.loc[zero_peak_idx, "Peak_Hour_Usage_kWh"] = "0.0 kWh"
df.loc[zero_off_idx,  "Off_Peak_Usage_kWh"]  = "0.0 kWh"



nan_pool = np.random.choice(np.arange(10000), size=438, replace=False)

df.loc[nan_pool[0:15],   "Temperature_C"]      = np.nan
df.loc[nan_pool[15:30],  "Humidity_pct"]       = np.nan
df.loc[nan_pool[30:40],  "Power_Factor"]       = np.nan
df.loc[nan_pool[40:50],  "Appliance_Score"]    = np.nan
df.loc[nan_pool[50:303], "Expected_Energy_kWh"] = np.nan
df.loc[nan_pool[303:438],"Actual_Energy_kWh"]   = np.nan



dup_idx = np.random.choice(df.index, 35, replace=False)
df_final = pd.concat([df, df.loc[dup_idx]], ignore_index=True)



df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)


print("=== VALIDATION SUMMARY ===")
print(f"Total rows: {len(df_final)}")
print(f"Total columns: {len(df_final.columns)}")
print(f"Anomaly_Type counts:\n{df_final['Anomaly_Type'].value_counts().sort_index()}")

print(f"Bypass rate for Type-1: {(df_final[df_final.Anomaly_Type==1].Meter_Bypass_Signal==1).mean():.3f}")
print(f"Solar=1 for Type-3: {df_final[df_final.Anomaly_Type==3].Has_Solar_Panel.mean():.3f}")

def parse_kwh(s):
    try:
        return float(str(s).replace(" kWh",""))
    except:
        return np.nan

solar_rows = df_final[(df_final.Has_Solar_Panel==1) & df_final.Expected_Energy_kWh.notna()]
neg_exp = solar_rows.Expected_Energy_kWh.apply(lambda x: parse_kwh(x) < 0)
print(f"Solar rows with negative Expected: {neg_exp.mean():.3f}")

print(f"Temperature_C max: {df_final.Temperature_C.max():.2f}")

print("\nNaN counts per column:")
print(df_final.isnull().sum()[df_final.isnull().sum()>0])

nan_per_row = df_final.isnull().sum(axis=1)
print(f"Max NaNs in any row: {nan_per_row.max()}")
print(f"Rows with >1 NaN: {(nan_per_row>1).sum()}")

print(f"Duplicate rows: {df_final.duplicated().sum()}")

actual_n  = df_final.Actual_Energy_kWh.apply(parse_kwh)
peak_n    = df_final.Peak_Hour_Usage_kWh.apply(parse_kwh)
offpeak_n = df_final.Off_Peak_Usage_kWh.apply(parse_kwh)

violations = ((peak_n + offpeak_n) > actual_n).sum()
print(f"Peak+OffPeak > Actual violations: {violations}")

print("\nMean Actual_Energy_kWh per class:")
df_final["_act_num"] = actual_n
print(df_final.groupby("Anomaly_Type")["_act_num"].mean().round(2))

print(f"\nGrid outage rate: {(df_final.Grid_Outage_Hours>0).mean():.3f}")

corr_df = df_final[['House_Area_sqft','Connected_Load_kW','Meter_Age_Years','Power_Factor']].copy()
corr_df['act'] = actual_n
corr_df['temp'] = df_final.Temperature_C

print(f"Corr House_Area <-> Connected_Load: {corr_df.House_Area_sqft.corr(corr_df.Connected_Load_kW):.3f}")
print(f"Corr Meter_Age  <-> Power_Factor:   {corr_df.Meter_Age_Years.corr(corr_df.Power_Factor):.3f}")
print(f"Corr Temp       <-> Actual_Energy:  {corr_df.temp.corr(corr_df.act):.3f}")

for col in ['Expected_Energy_kWh','Actual_Energy_kWh','Peak_Hour_Usage_kWh','Off_Peak_Usage_kWh']:
    print(f"dtype of {col}: {df_final[col].dtype}")

print(f"\nConnected_Load_kW sample: {df_final.Connected_Load_kW.head(3).tolist()}")
print(f"Voltage_V sample: {df_final.Voltage_V.head(3).tolist()}")
print(f"Power_Factor sample: {df_final.Power_Factor.head(3).tolist()}")

print("\n=== END VALIDATION ===")


df_final.drop(columns=["_act_num"], inplace=True, errors="ignore")
df_final.to_csv("kerala_smart_meter_v6.csv", index=False)
print("\nSaved: kerala_smart_meter_v6.csv")
