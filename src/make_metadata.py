"""
Build metadata.csv for BreaKHis: one row per image, with a frozen patient-level split.

Usage (in Colab, after extracting the dataset to /content):
    !python src/make_metadata.py --data_root /content/BreaKHis_v1 --out "{DRIVE_ROOT}/data/metadata.csv"

File name format:  SOB_B_TA-14-4659-40-001.png
                   SOB _ <B|M> _ <subtype>-<year>-<slide id>-<magnification>-<image number>.png
Patient ID = <year>-<slide id>, e.g. 14-4659
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SUBTYPES = {
    # code: (full name, class index, benign/malignant)
    "A": ("adenosis", 0, "B"),
    "F": ("fibroadenoma", 1, "B"),
    "PT": ("phyllodes_tumor", 2, "B"),
    "TA": ("tubular_adenoma", 3, "B"),
    "DC": ("ductal_carcinoma", 4, "M"),
    "LC": ("lobular_carcinoma", 5, "M"),
    "MC": ("mucinous_carcinoma", 6, "M"),
    "PC": ("papillary_carcinoma", 7, "M"),
}
MAGNIFICATIONS = [40, 100, 200, 400]


def parse_filename(name: str) -> dict:
    """SOB_B_TA-14-4659-40-001.png -> tumor class, subtype, patient, magnification."""
    stem = name.rsplit(".", 1)[0]
    _, tumor_class, rest = stem.split("_", 2)
    parts = rest.split("-")
    return {
        "tumor_class": tumor_class,
        "subtype": parts[0],
        "patient_id": "-".join(parts[1:-2]),
        "magnification": int(parts[-2]),
    }


def build_metadata(data_root: Path) -> pd.DataFrame:
    paths = sorted(data_root.rglob("*.png"))
    if not paths:
        sys.exit(f"No .png files found under {data_root}. Did you extract the dataset?")

    rows = []
    for p in paths:
        info = parse_filename(p.name)
        name, idx, expected_class = SUBTYPES[info["subtype"]]
        if info["tumor_class"] != expected_class:
            print(f"WARNING: class/subtype mismatch in {p.name}")
        rows.append({
            "path": str(p.relative_to(data_root)),
            "filename": p.name,
            "patient_id": info["patient_id"],
            "tumor_class": info["tumor_class"],
            "binary_label": 0 if info["tumor_class"] == "B" else 1,
            "subtype": info["subtype"],
            "subtype_name": name,
            "subtype_label": idx,
            "magnification": info["magnification"],
        })
    return pd.DataFrame(rows)


def assign_split(meta: pd.DataFrame, val_frac: float, test_frac: float, seed: int) -> pd.Series:
    """Split by patient, stratified by subtype, with at least one patient per subtype in val and test.

    The split is decided per patient, so every image of a patient (all magnifications)
    lands in the same split. This prevents leakage in the cross-magnification test.
    """
    # Each patient is stratified by its most frequent subtype
    # (handles the rare case of a patient ID appearing under two subtypes).
    patient_subtype = meta.groupby("patient_id")["subtype"].agg(lambda s: s.mode().iloc[0])

    rng = np.random.default_rng(seed)
    assignment = {}
    for subtype in sorted(patient_subtype.unique()):
        patients = np.array(sorted(patient_subtype[patient_subtype == subtype].index))
        patients = rng.permutation(patients)
        n = len(patients)
        n_test = max(1, round(test_frac * n))
        n_val = max(1, round(val_frac * n))
        if n_test + n_val >= n:
            print(f"WARNING: subtype {subtype} has only {n} patients; train gets {n - n_test - n_val}")
        for p in patients[:n_test]:
            assignment[p] = "test"
        for p in patients[n_test:n_test + n_val]:
            assignment[p] = "val"
        for p in patients[n_test + n_val:]:
            assignment[p] = "train"
    return meta["patient_id"].map(assignment)


def run_checks(meta: pd.DataFrame) -> None:
    print("\n=== Sanity checks ===")
    print(f"Images:   {len(meta)}   (expected 7,909)")
    print(f"Patients: {meta['patient_id'].nunique()}   (expected 82)")
    print("\nImages per magnification (expected 40X 1995, 100X 2081, 200X 2013, 400X 1820):")
    print(meta["magnification"].value_counts().sort_index().to_string())

    multi = meta.groupby("patient_id")["subtype"].nunique()
    multi = multi[multi > 1]
    if len(multi):
        print(f"\nWARNING: {len(multi)} patient ID(s) appear under more than one subtype: {list(multi.index)}")
        print("They are kept in a single split (no leakage). Mention this in the report.")

    mags = meta.groupby("patient_id")["magnification"].nunique()
    incomplete = mags[mags < 4]
    if len(incomplete):
        print(f"\nNOTE: {len(incomplete)} patient(s) missing some magnifications: {list(incomplete.index)}")

    leak = meta.groupby("patient_id")["split"].nunique()
    assert (leak == 1).all(), "LEAKAGE: a patient appears in more than one split"
    print("\nNo patient appears in more than one split. ✓")


def print_split_tables(meta: pd.DataFrame) -> None:
    print("\n=== Patients per subtype and split ===")
    print(pd.crosstab(meta.drop_duplicates("patient_id")["subtype_name"],
                      meta.drop_duplicates("patient_id")["split"], margins=True).to_string())
    print("\n=== Images per subtype and split ===")
    print(pd.crosstab(meta["subtype_name"], meta["split"], margins=True).to_string())
    print("\n=== Images per magnification and split ===")
    print(pd.crosstab(meta["magnification"], meta["split"], margins=True).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", required=True, help="Folder where BreaKHis_v1 was extracted")
    parser.add_argument("--out", required=True, help="Where to save metadata.csv (use the shared Drive folder)")
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacing an existing metadata.csv (the split must stay frozen!)")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists() and not args.overwrite:
        sys.exit(f"{out} already exists. The split is frozen; use --overwrite only if the team agreed.")

    meta = build_metadata(Path(args.data_root))
    meta["split"] = assign_split(meta, args.val_frac, args.test_frac, args.seed)

    run_checks(meta)
    print_split_tables(meta)

    out.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out, index=False)
    print(f"\nSaved {len(meta)} rows to {out}")


if __name__ == "__main__":
    main()
