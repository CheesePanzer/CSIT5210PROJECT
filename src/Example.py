from datasets import load_dataset
from split import get_outer_folds, get_inner_subsets
from preprocessing import preprocess_partition

# 1. load dataset and get outer folds, Change "CAR" to any other dataset ID available in your data/raw folder
X, y, meta = load_dataset("CAR")
outer_folds = get_outer_folds(X, y, metadata=meta)

# 2. Iterate through outer folds, generate inner subsets, and preprocess data for training and evaluation
for fold_idx, (train_val_idx, test_idx) in enumerate(outer_folds):
    # Divide the training/validation set into inner subsets for hyperparameter tuning or cross-validation
    inner_subsets = get_inner_subsets(X, train_val_idx, y)

    # Example (For NN use only inner_subsets[0]; for SVM iterate through the 5 inner_subsets):
    inner_tr_rel, inner_val_rel = inner_subsets[0]
    inner_tr_idx = train_val_idx[inner_tr_rel]
    inner_val_idx = train_val_idx[inner_val_rel]

    # Preprocess the inner training and validation subsets using the preprocess_partition function
    X_inner_tr, X_inner_val = preprocess_partition(X, inner_tr_idx, inner_val_idx, meta["categorical_cols"])
    y_inner_tr, y_inner_val = y[inner_tr_idx], y[inner_val_idx]

    #Hiperparameter tuning or model training can be performed here using X_inner_tr, y_inner_tr for training and X_inner_val, y_inner_val for validation.
    
    X_train, X_test = preprocess_partition(X, train_val_idx, test_idx, meta["categorical_cols"])
    y_train, y_test = y[train_val_idx], y[test_idx]
    
    