import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold, train_test_split, StratifiedKFold


def get_cv(df, cfg):
    df['fold'] = 0
    fold_type = cfg.fold_type
    if fold_type == 'kfold':
        kf = KFold(n_splits=cfg.n_fold, shuffle=True, random_state=71)
        for i, (tr_idx, val_index) in enumerate(kf.split(df)):
            df.loc[val_index, 'fold'] = int(i)
    elif fold_type == 'skfold':
        # num_bins = int(np.floor(1 + np.log2(len(df))))
        # bins = pd.cut(df[cfg.target_col], bins=num_bins, labels=False)
        Fold = StratifiedKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)
        for n, (train_index, val_index) in enumerate(Fold.split(df, df[cfg.target_col])):
            df.loc[val_index, 'fold'] = int(n)
    return df
