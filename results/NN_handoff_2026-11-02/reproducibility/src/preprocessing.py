from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class FoldPreprocessor:

    def __init__(self, categorical_cols: Optional[List[str]] = None):
        #Save the list of categorical columns and initialize the transformer to None, before the fit() function. The transformer will be set up during the fit process.
        self.categorical_cols = categorical_cols or []
        self.transformer: Optional[ColumnTransformer] = None

    def fit(self, X_train: pd.DataFrame):
        # Sort all columns into categorical and numerical columns
        all_cols = list(X_train.columns)
        cat_cols = [c for c in self.categorical_cols if c in all_cols]
        num_cols = [c for c in all_cols if c not in cat_cols]

        transformers = []

        if num_cols:
            num_pipeline = Pipeline([
                # Pipeline for numerical columns:
                # Impute missing values in numerical columns with the median
                ("imputer", SimpleImputer(strategy="median")),
                # Scale numerical features to the range [0, 1] using MinMaxScaler.
                ("scaler", MinMaxScaler(feature_range=(0, 1))),
            ])    
            transformers.append(("num", num_pipeline, num_cols))

        if cat_cols:
            cat_pipeline = Pipeline([
                 # Pipeline for categorical columns:
                # Impute missing values in categorical columns with the most frequent value
                ("imputer", SimpleImputer(strategy="most_frequent")),
                # One-Hot Encode categorical features
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
            transformers.append(("cat", cat_pipeline, cat_cols))

        self.transformer = ColumnTransformer(transformers=transformers, remainder="drop")
        self.transformer.fit(X_train)
        return self

    def transform(self, X: pd.DataFrame):
        if self.transformer is None:
            raise RuntimeError("FoldPreprocessor must be fitted before calling transform.")
        X_trans = self.transformer.transform(X)
        # Return the transformed data as a NumPy array of type float32 for consistency and compatibility
        return np.asarray(X_trans, dtype=np.float32)

    def fit_transform(self, X_train: pd.DataFrame):
        return self.fit(X_train).transform(X_train)


def preprocess_partition(
    X: pd.DataFrame,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    categorical_cols: Optional[List[str]] = None
):
    # Preprocesses the training and evaluation partitions of the dataset using the FoldPreprocessor class. It fits the preprocessor on the training data and transforms both the training and evaluation data accordingly.
    preprocessor = FoldPreprocessor(categorical_cols=categorical_cols)
    X_train_proc = preprocessor.fit_transform(X.iloc[train_idx])
    X_eval_proc = preprocessor.transform(X.iloc[eval_idx])
    return X_train_proc, X_eval_proc