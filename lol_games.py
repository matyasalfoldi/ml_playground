import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

def dec_tree(X, y, cv):
    tree_acc = 0
    for train_idx, valid_idx in cv.split(X, y):
        tree = DecisionTreeClassifier(random_state=1, max_depth=7)
        tree.fit(X[train_idx], y[train_idx])
        y_pred = tree.predict(X[valid_idx])
        tree_acc += np.mean(y_pred == y[valid_idx])*100

    tree_acc /= 10
    print(f'{tree_acc=}')

def rand_forest(X, y, cv):
    forest_acc = 0
    
    for train_idx, valid_idx in cv.split(X, y):
        forest = RandomForestClassifier(n_estimators=1000, random_state=1)
        forest.fit(X[train_idx], y[train_idx])
        y_pred = forest.predict(X[valid_idx])
        forest_acc += np.mean(y_pred == y[valid_idx])*100

    forest_acc /= 10
    print(f'{forest_acc=}')

def log_reg(X, y, cv):
    log_acc = 0
    for train_idx, valid_idx in cv.split(X, y):
        log = LogisticRegression(random_state=1)
        log.fit(X[train_idx], y[train_idx])
        y_pred = log.predict(X[valid_idx])
        log_acc += np.mean(y_pred == y[valid_idx])*100

    log_acc /= 10
    print(f'{log_acc=}')


def main(args):
    df = pd.read_csv(args.input)
    df["result"] = df["result"].map({"win":1, "lose":0})

    X = df.values[:,0:-1]
    y = np.ravel(df.values[:,-1:])

    cv = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

    dec_tree(X, y, cv)

    rand_forest(X, y, cv)

    log_reg(X, y, cv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Input csv')
    args = parser.parse_args()
    main(args)
