import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from table import MatchPredictor, TableResolver


def preprocess_df(df, obj_cols, enc):
    encoded_columns = pd.DataFrame(enc.fit_transform(df[obj_cols]))
    numerical_data = df.drop(obj_cols, axis=1)
    numerical_data = numerical_data.reset_index().drop("index", axis=1)
    encoded_columns = encoded_columns.reset_index().drop("index", axis=1)
    preprocessed_df = pd.concat([numerical_data, encoded_columns], axis=1)
    return preprocessed_df


def train_model_from_df(prediction_df, train_size, use_confusion_matrix=False):
    y_data = prediction_df["Heimsieg"]
    X_data = prediction_df.drop("Heimsieg", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, train_size=train_size, test_size=1 - train_size, random_state=0
    )
    clf = RandomForestClassifier(n_estimators=250, random_state=0)
    clf.fit(X_train, y_train)
    if use_confusion_matrix:
        y_pred = clf.predict(X_test)
        print(confusion_matrix(y_pred, y_test))
    return clf


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


def filter_relevant_data(data, teams, label_function):
    df = data.drop(["city", "country", "neutral"], axis=1)
    df = df[df.date > "2018-01-01"]
    df = df.drop(["tournament", "date"], axis=1)
    df = df[(df["home_team"].isin(teams)) | (df["away_team"].isin(teams))]
    df["Heimsieg"] = df.apply(label_function, axis=1)
    df.drop(["home_score", "away_score"], axis=1, inplace=True)
    return df


def get_obj_cols_in_df(df):
    obj_cols = df.columns[df.dtypes == "object"].to_list()
    return obj_cols


def solve_matches(
    df,
    enc,
    matches: list[list[str]],
    tableclass=None,
    settattrhelper=None,
    message=None,
    sleep=4,
    groupstage=False,
):
    print(message)
    if settattrhelper is None:
        settattrhelper = ["a", "b", "c", "d", "e", "f", "g", "h"]

    for index, match in enumerate(matches):
        predictor = MatchPredictor(match)
        prediction_df = predictor.create_new_prediction_df(
            overall_df=df, obj_cols=get_obj_cols_in_df(df=df), enc=enc
        )
        prediction_df = predictor.predict_matches(prediction_df)
        resolver = TableResolver(match, prediction_df=prediction_df)
        resolver.get_results()
        time.sleep(sleep)
        if tableclass:
            setattr(
                tableclass, settattrhelper[index], resolver.get_winner(group=groupstage)
            )
