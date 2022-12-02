from joblib import dump
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier

enc = OneHotEncoder(handle_unknown="ignore", sparse=False)


def read_data(
    path="./results.csv",
):
    df = pd.read_csv(path)
    return df


def relabel(df):
    if df["home_score"] == df["away_score"]:
        return 0
    elif df["home_score"] > df["away_score"]:
        return 1
    else:
        return -1


def get_obj_cols_in_df(df):
    obj_cols = df.columns[df.dtypes == "object"].to_list()
    return obj_cols


def filter_relevant_data(data, label_function):
    df = data.drop(["city", "country", "neutral"], axis=1)
    df = df[df.date > "2015-01-01"]
    df = df.drop(["tournament", "date"], axis=1)
    df["Heimsieg"] = df.apply(label_function, axis=1)
    df.drop(["home_score", "away_score"], axis=1, inplace=True)
    return df


def preprocess_df(df, obj_cols, enc):
    encoded_columns = pd.DataFrame(enc.fit_transform(df[obj_cols]))
    numerical_data = df.drop(obj_cols, axis=1)
    numerical_data = numerical_data.reset_index().drop("index", axis=1)
    encoded_columns = encoded_columns.reset_index().drop("index", axis=1)
    preprocessed_df = pd.concat([numerical_data, encoded_columns], axis=1)
    return preprocessed_df


df = read_data("./results.csv")
df = filter_relevant_data(df, label_function=relabel)

obj_cols = get_obj_cols_in_df(df)
trainings_df = preprocess_df(df, obj_cols=obj_cols, enc=enc)

X = trainings_df.drop("Heimsieg", axis=1)
y = trainings_df["Heimsieg"]

# grid = {"max_depth": [50, 100, 200], "n_estimators": [100, 200, 400, 600]}
# rf_random = RandomizedSearchCV(
#     estimator=RandomForestClassifier(),
#     param_distributions=grid,
#     random_state=42,
#     n_jobs=-1,
#     verbose=3,
# )
# rf_random.fit(X, y)
# print(rf_random.best_params_)
#
# dump(rf_random, "model.job")

light_gbm_grid = {
    'num_leaves': [31, 127],
    'reg_alpha': [0.1, 0.5],
    'min_data_in_leaf': [30, 50, 100, 300, 400],
    'lambda_l1': [0, 1, 1.5],
    'lambda_l2': [0, 1]
    }

light_gbm = RandomizedSearchCV(
    estimator=LGBMClassifier(),
    param_distributions=light_gbm_grid,
    random_state=42,
    n_jobs=-1,
    verbose=3,
)
light_gbm.fit(X, y)
print(light_gbm.best_params_)

dump(light_gbm, "light_gbm.job")