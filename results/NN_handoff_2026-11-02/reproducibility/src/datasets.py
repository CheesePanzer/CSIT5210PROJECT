import glob
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file

def compute_imbalance_ratio(y: np.ndarray):
    #Calculate the Imbalance Ratio (IR).
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max() / counts.min())

def get_imbalance_group(ir: float):
    #Classify the IR.
    if ir < 3.0:
        return "low"
    elif ir < 9.0:
        return "medium"
    return "high"

def parse_keel_dat(filepath: str):
    #Read a KEEL .dat file and returning the features, target, and categorical columns.

    attributes = []
    categorical_cols = []
    data_lines = []
    in_data = False

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            lower = line.lower()
            if lower.startswith("@attribute"):
                # Divide the line into parts: @attribute, name, type
                parts = line.split(None, 2)
                attr_name = parts[1].strip("'\"")
                attr_type = parts[2] if len(parts) > 2 else ""
                attributes.append((attr_name, attr_type))
            elif lower.startswith("@data"):
                in_data = True
            elif in_data:
                #Reads Data lines, splitting by commas and stripping whitespace
                data_lines.append([x.strip() for x in line.split(",")])
                
    col_names = [a[0] for a in attributes]
    df = pd.DataFrame(data_lines, columns=col_names)
    df.replace(["?", "<null>", "null", "NA"], np.nan, inplace=True)

    # Last attribute is assumed to be the target variable
    target_col = col_names[-1]
    feature_attrs = attributes[:-1]

    for col_name, attr_type in feature_attrs:
        if "{" in attr_type:
            # Categorical variable (nominal)
            categorical_cols.append(col_name)
            df[col_name] = df[col_name].astype(str)
        else:
            # Numerical variable (real or integer)
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
        
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(str)
    return X, y, categorical_cols


def load_dataset(dataset_id: str, data_dir: str = "data/raw"):
    """
   Loads a dataset from the specified directory, automatically detecting its format (KEEL, UCI, or LIBSVM) and extracting features, target labels, and metadata.
    """
    dataset_id = dataset_id.upper()
    folder = Path(data_dir) / dataset_id
    
    if not folder.exists():
        raise FileNotFoundError(f"folder not found: {folder}")

    predefined_folds = None
    categorical_cols = []
    
    # Detect if the dataset is in KEEL (.dat), UCI (.csv), or LIBSVM (.scale/.original) format
    dat_files = glob.glob(str(folder / "*.dat"))
    csv_files = glob.glob(str(folder / "*.csv"))
    libsvm_files = glob.glob(str(folder / "*.scale")) + glob.glob(str(folder / "*.original"))

    if dat_files:
        source = "KEEL"
        # Check for the presence of 5 predefined folds in KEEL datasets
        tst_files = sorted(glob.glob(str(folder / "*-5-*tst.dat")))
        tra_files = sorted(glob.glob(str(folder / "*-5-*tra.dat")))

        if len(tst_files) == 5 and len(tra_files) == 5:
            test_dfs_X, test_dfs_y = [], []
            fold_test_indices = []
            current_idx = 0

            for tst_f in tst_files:
                # Parse each test fold and collect the data
                X_part, y_part, cat_cols = parse_keel_dat(tst_f)
                categorical_cols = cat_cols
                n_part = len(X_part)
                fold_test_indices.append(np.arange(current_idx, current_idx + n_part))
                current_idx += n_part
                test_dfs_X.append(X_part)
                test_dfs_y.append(y_part)

            X = pd.concat(test_dfs_X, ignore_index=True)
            y_raw = pd.concat(test_dfs_y, ignore_index=True)

            # Create predefined folds as a list of (train_indices, test_indices) tuples
            all_indices = np.arange(len(X))
            predefined_folds = []
            for test_idx in fold_test_indices:
                train_idx = np.setdiff1d(all_indices, test_idx)
                predefined_folds.append((train_idx, test_idx))
        else:
            # Fallback if the expected 5 folds are not found
            raise ValueError(f"the dataset {dataset_id} does not have the 5 official KEEL folds expected!")

    elif csv_files:
        source = "UCI"
        df = pd.read_csv(csv_files[0])
        # Remove the index column
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        # Remove rows with missing values and reset the index
        df = df.dropna().reset_index(drop=True)
            
            
        # Find the target column (assumes 'Category' as in HCV) and the rest features
        target_col = "Category" 
        y_raw = df[target_col].astype(str)
        X = df.drop(columns=[target_col])
        categorical_cols = ["Sex"] 

    elif libsvm_files:
        source = "LIBSVM"
        #Read the LIBSVM file using load_svmlight_file
        X_sparse, y_raw = load_svmlight_file(libsvm_files[0])
        # Convert the sparse matrix to a dense DataFrame with feature names f_0, f_1, ..., f_n
        X = pd.DataFrame(X_sparse.toarray(), columns=[f"f_{i}" for i in range(X_sparse.shape[1])])
        y_raw = pd.Series(y_raw).astype(str)
        categorical_cols = []
    
    else:
        raise FileNotFoundError(f"Nenhum formato reconhecido (.dat, .csv, .scale, .original) em {folder}")

    # Modify the target variable to integer labels and compute the imbalance ratio
    unique_labels, y_int = np.unique(y_raw, return_inverse=True)
    ir_computed = compute_imbalance_ratio(y_int)

    # dictionary to hold metadata about the dataset
    metadata = {
        "id": dataset_id,
        "source": source,
        "n_samples": len(X),
        "n_features_raw": X.shape[1],
        "n_classes": len(unique_labels),
        "class_labels": unique_labels.tolist(),
        "class_counts": np.bincount(y_int).tolist(),
        "ir_computed": round(ir_computed, 2),
        "imbalance_group": get_imbalance_group(ir_computed),
        "categorical_cols": categorical_cols,
        "has_predefined_folds": predefined_folds is not None,
        "predefined_folds": predefined_folds,
    }

    return X, y_int, metadata





