import pandas as pd

def load_data():
    df = pd.read_csv("data.csv")

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # ---------- COUNTRY ----------
    if "country" in df.columns:
        df["Country"] = df["country"]
    elif "area" in df.columns:
        df["Country"] = df["area"]
    else:
        df["Country"] = "Unknown"

    # ---------- CROP ----------
    if "crop" in df.columns:
        df["Crop"] = df["crop"]
    elif "item" in df.columns:
        df["Crop"] = df["item"]
    else:
        df["Crop"] = "General"

    # ---------- YEAR ----------
    if "year" in df.columns:
        df["Year"] = df["year"]
    elif "y" in df.columns:
        df["Year"] = df["y"]
    else:
        df["Year"] = range(len(df))

    # ---------- YIELD ----------
    if "yield" in df.columns:
        df["Yield"] = df["yield"]
    elif "yld" in df.columns:
        df["Yield"] = df["yld"]
    elif "value" in df.columns:
        df["Yield"] = df["value"]
    else:
        df["Yield"] = 0

    # ---------- PRODUCTION ----------
    if "production" in df.columns:
        df["Production"] = df["production"]
    elif "prod" in df.columns:
        df["Production"] = df["prod"]
    else:
        df["Production"] = df["Yield"]

    # ---------- RAINFALL ----------
    if "rainfall" in df.columns:
        df["Rainfall"] = df["rainfall"]
    elif "rain" in df.columns:
        df["Rainfall"] = df["rain"]
    else:
        df["Rainfall"] = 0

    # ---------- TEMPERATURE ----------
    if "temperature" in df.columns:
        df["Temperature"] = df["temperature"]
    elif "temp" in df.columns:
        df["Temperature"] = df["temp"]
    else:
        df["Temperature"] = 0

    # ---------- FOOD SECURITY INDEX ----------
    if "fsi" in df.columns:
        df["FSI"] = df["fsi"]
    else:
        df["FSI"] = df["Yield"]  # fallback

    return df