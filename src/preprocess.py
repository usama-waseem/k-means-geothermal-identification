
"""
 
Turns the raw AASG workbook into the paper's 196-sample analysis table with a
binary IM/EQ label attached.
 
Scope stops there: no scaling, no log transform, no modelling. The paper
derives its log10 step *from* the exploratory data analysis, so applying it
here would assume the conclusion before any plot has been drawn.
 
Verified against the paper:
    416 rows in the sheet
    412 after dropping blank template rows
    301 with a valid maturity code
    196 after the completeness filter   (paper: 196)
    124 IM / 72 EQ                      (paper: 100+24 IM, 58+14 EQ)
 
Usage
-----
    from preprocess import build_dataset
    data = build_dataset()
"""
 
from pathlib import Path
 
import numpy as np
import pandas as pd
 
 
# ===========================================================================
# CONFIGURATION
# ===========================================================================
 
SHEET = "ThermalSpringGeothermometry"
 
# This file lives in src/, so the repo root is one level up.
# In the notebook we found ROOT with Path.cwd(), but that depends on where you
# launched from. __file__ is the module's own path, so this is correct no
# matter which directory you run from.
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_WORKBOOK = RAW_DIR / "CO_Hot-Spring-Geothermometry-Template_2012-2-16.xlsx"
 
# The 15 columns the paper uses (Table 1), mapped from the workbook's names
# to short names. One dict drives both the selection and the rename.
COLUMN_MAP = {
    "Name": "Name",
    "FeatureType": "Type",
    "LatDegree": "Latitude",
    "LongDegree": "Longitude",
    "Temperature": "Temperature",
    "Well Depth": "Depth",
    "Conductivity": "Cond",
    "pH": "pH",
    "TDS": "TDS",
    "Ca": "Ca",
    "Mg": "Mg",
    "Na": "Na",
    "K": "K",
    "SiO2": "SiO2",
    "Giggenbach Maturity Classification": "Equilibrium",
}
 
NUMERIC_COLS = ["Latitude", "Longitude", "Temperature", "Depth", "Cond", "pH",
                "TDS", "Ca", "Mg", "Na", "K", "SiO2"]
 
# All five valid maturity codes. Rows with anything else (blank, unparseable)
# are dropped: 412 -> 301.
EQ_LEVELS = {"IM", "FE", "PE", "PE/IM", "FE/PE"}
 
# The four codes that count as equilibrium when collapsing to a binary target.
#
# JUDGEMENT CALL: "PE/IM" covers 20 samples the compilers could not confidently
# place. Including them gives 72 EQ, matching the paper's confusion matrices.
# Excluding them gives 51. So the authors included them -- but roughly a
# quarter of the positive class has genuinely uncertain ground truth.
EQ_CODES = {"PE", "FE", "PE/IM", "FE/PE"}
 
# The completeness filter. The paper never defines "complete geochemical
# information", so it was reverse-engineered against the target count:
#     5 chemistry columns only   -> 286
#     + T, Cond, pH, TDS         -> 200
#     + a valid maturity code    -> 196   <- matches the paper
#
# Depth is deliberately absent: 161 of the 196 are surface-sampled springs
# with no depth, so requiring it would gut the data set.
REQUIRED_COLS = ["Temperature", "Cond", "pH", "TDS", "Ca", "Mg", "Na", "K", "SiO2"]
 
 
# ===========================================================================
# STEP 1 - LOAD
# ===========================================================================
 
def load_raw(path=None):
    """
    Read the sheet, keep the 15 columns of interest, fix dtypes.
 
    Returns a 301-row frame: every row has a valid maturity code, but
    measurements may still be missing.
    """
    path = Path(path) if path is not None else DEFAULT_WORKBOOK
 
    if not path.exists():
        raise FileNotFoundError(
            f"workbook not found: {path}\nplace the AASG .xlsx in {RAW_DIR}"
        )
 
    # skiprows=[1] drops the units row ("mg/l", "Field", ...). Without it every
    # measurement column is typed object, and `Mg < 0.028` would compare
    # strings -- a wrong answer with no error. It must be a LIST: skiprows=1
    # would skip the header instead.
    raw = pd.read_excel(path, sheet_name=SHEET, skiprows=[1])
 
    # The sheet is a template and ships with blank rows past the real data.
    raw = raw[raw["Name"].notna()].copy()
 
    # Catch a schema change in a future version of the workbook.
    missing = set(COLUMN_MAP) - set(raw.columns)
    if missing:
        raise KeyError(f"expected columns absent from sheet: {sorted(missing)}")
 
    # Note this assigns to a NEW name rather than back to `raw`. In the
    # notebook, `raw = raw.loc[...]` fails on a second run because the old
    # column names are gone. Returning a new frame avoids that entirely.
    df = raw.loc[:, list(COLUMN_MAP)].rename(columns=COLUMN_MAP)
 
    # Maturity codes carry trailing spaces, so "PE " and "PE" are distinct
    # strings and would filter out separately.
    df["Equilibrium"] = df["Equilibrium"].astype(str).str.strip()
    df = df[df["Equilibrium"].isin(EQ_LEVELS)].copy()
 
    # A few columns still hold stray text after the units row is gone.
    # errors="coerce" turns those into NaN so the completeness filter catches
    # them. This is not cosmetic: it moved 17 Depth values and 1 pH value out
    # of "present" and into "missing", where they belong.
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
 
    return df.reset_index(drop=True)
 
 
# ===========================================================================
# STEP 2 - FILTER AND LABEL
# ===========================================================================
 
def clean(df, fill_depth=0.0):
    """
    Apply the completeness filter and build the binary target.
 
    301 rows in, 196 out.
    """
    # .notna() gives a boolean frame; .all(axis=1) collapses each row to one
    # value, True only when every required column is present.
    mask = df[REQUIRED_COLS].notna().all(axis=1)
    data = df[mask].copy().reset_index(drop=True)
 
    # Missingness here is informative: NaN means "surface spring", not
    # "unmeasured". Zero encodes that. Filling with the mean -- the usual
    # reflex -- would give 161 surface springs a fictional ~330 m depth.
    data["Depth"] = data["Depth"].fillna(fill_depth)
 
    # The target variable. Worth being clear about what it is: the model will
    # learn to predict what the AASG compilers wrote in a spreadsheet column,
    # not geochemistry from first principles. Their consistency is the ceiling
    # on any accuracy we can honestly claim.
    data["label"] = np.where(data["Equilibrium"].isin(EQ_CODES), "EQ", "IM")
 
    return data
 
 
# ===========================================================================
# ASSEMBLY
# ===========================================================================
 
def build_dataset(path=None):
    """Raw workbook -> 196-row analysis table, in original measured units."""
    return clean(load_raw(path))
 
 
def verify(data):
    """Check the output against the counts reported in the paper."""
    counts = data["label"].value_counts()
    types = data["Type"].value_counts()
    checks = {
        "n = 196": len(data) == 196,
        "IM = 124": counts.get("IM") == 124,
        "EQ = 72": counts.get("EQ") == 72,
        "Spring = 153": types.get("Thermal Spring") == 153,
    }
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    return all(checks.values())
 
 
def save_processed(data, out_dir=PROCESSED_DIR):
    """Write the clean table so later stages need not re-parse Excel."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "samples_196.csv"
    data.to_csv(target, index=False)
    print(f"wrote {target}")
 
 
# Runs only when the file is executed directly (`python src/preprocess.py`),
# not when another module imports it. That is what lets one file serve as both
# a library and a runnable script.
if __name__ == "__main__":
    import sys
 
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WORKBOOK
    data = build_dataset(src)
 
    print(f"samples: {len(data)}")
    print(f"labels : {dict(data['label'].value_counts())}")
    print(f"types  : {dict(data['Type'].value_counts())}")
    print()
    verify(data)
    print()
    save_processed(data)