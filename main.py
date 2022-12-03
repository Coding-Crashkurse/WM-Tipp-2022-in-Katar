import joblib
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import time

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

clf = joblib.load("./light_gbm.job")


def create_new_prediction_df(df, kommende_spiele, obj_cols, enc):
    new_df = pd.concat([df, kommende_spiele], ignore_index=True)
    encoded_columns = pd.DataFrame(enc.fit_transform(new_df[obj_cols]))
    numerical_data = df.drop(obj_cols, axis=1)
    numerical_data = numerical_data.reset_index().drop("index", axis=1)
    encoded_columns = encoded_columns.reset_index().drop("index", axis=1)
    preprocessed_df = pd.concat([numerical_data, encoded_columns], axis=1)
    return preprocessed_df


def predict_winner(home, away):
    kommende_spiele_dict = {
        "home_team": [home, away],
        "away_team": [away, home],
        "Heimsieg": [None, None],
    }
    kommende_spiele = pd.DataFrame(data=kommende_spiele_dict)
    prediction_df = create_new_prediction_df(
        df, kommende_spiele=kommende_spiele, obj_cols=obj_cols, enc=enc
    )
    X_data = prediction_df.drop("Heimsieg", axis=1)
    X_test = X_data.tail(2)
    y_pred = clf.predict_proba(X_test)
    # print(y_pred)
    home_added = (y_pred[0][0] + y_pred[1][0]) / 2
    away_added = (y_pred[0][2] + y_pred[1][2]) / 2
    winner = home if home_added > away_added else away
    print(
        f"{home} - {round(home_added * 100, 2)}% | {away} - {round(away_added * 100, 2)}% - Sieger: {winner}"
    )
    time.sleep(5)
    return winner


viertelfinale1 = [None, None]
viertelfinale2 = [None, None]
viertelfinale3 = [None, None]
viertelfinale4 = [None, None]

halbfinale1 = [None, None]
halbfinale2 = [None, None]

finale = [None, None]

print("---- Achtelfinals ----")
viertelfinale1[0] = predict_winner("Netherlands", "United States")
viertelfinale1[1] = predict_winner("Argentina", "Australia")
viertelfinale2[0] = predict_winner("France", "Poland")
viertelfinale2[1] = predict_winner("England", "Senegal")
viertelfinale3[0] = predict_winner("Japan", "Croatia")
viertelfinale3[1] = predict_winner("Brazil", "South Korea")
viertelfinale4[0] = predict_winner("Morocco", "Spain")
viertelfinale4[1] = predict_winner("Portugal", "Switzerland")

print("---- Viertelfinals ----")
halbfinale1[0] = predict_winner(viertelfinale1[0], viertelfinale1[1])
halbfinale1[1] = predict_winner(viertelfinale2[0], viertelfinale2[1])
halbfinale2[0] = predict_winner(viertelfinale3[0], viertelfinale3[1])
halbfinale2[1] = predict_winner(viertelfinale4[0], viertelfinale4[1])

print("---- Halbfinals ----")
finale[0] = predict_winner(halbfinale1[0], halbfinale1[1])
finale[1] = predict_winner(halbfinale2[0], halbfinale2[1])

print("---- Finale ----")
sieger = predict_winner(finale[0], finale[1])
print("Weltmeister ist:", sieger)
