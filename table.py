import itertools

import pandas as pd
from prettytable import PrettyTable
from sklearn.ensemble import RandomForestClassifier


class MatchPredictor:
    def __init__(self, teams: list):
        self.matches = list(itertools.combinations(teams.copy(), 2))
        self.kommende_spiele = self._upcoming_matches()

    def _upcoming_matches(self):
        home_teams = [team[0] for team in self.matches]
        away_teams = [team[1] for team in self.matches]

        kommende_spiele_dict = {
            "home_team": home_teams,
            "away_team": away_teams,
        }
        kommende_spiele = pd.DataFrame(data=kommende_spiele_dict)
        return kommende_spiele

    def create_new_prediction_df(self, overall_df, obj_cols, enc):
        gesamt_df = pd.concat([overall_df, self.kommende_spiele]).reset_index()
        gesamt_df.drop("index", axis=1, inplace=True)
        encoded_columns = pd.DataFrame(enc.fit_transform(gesamt_df[obj_cols]))
        numerical_data = gesamt_df.drop(obj_cols, axis=1)
        numerical_data = numerical_data.reset_index().drop("index", axis=1)
        encoded_columns = encoded_columns.reset_index().drop("index", axis=1)
        prediction_df = pd.merge(
            numerical_data, encoded_columns, left_index=True, right_index=True
        )
        return prediction_df

    def predict_matches(self, prediction_df):
        y_data = prediction_df["Heimsieg"]
        X_data = prediction_df.drop("Heimsieg", axis=1)
        X_train = X_data.head(len(prediction_df) - len(self.matches))
        X_test = X_data.tail(len(self.matches))
        y_train = y_data.head(len(prediction_df) - len(self.matches))

        clf = RandomForestClassifier(n_estimators=500, max_depth=100, bootstrap=True, random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        result = pd.merge(
            self.kommende_spiele,
            pd.Series(y_pred, name="pred"),
            left_index=True,
            right_index=True,
        )
        return result


class TableResolver:
    def __init__(self, teams: list[str], prediction_df):
        self._teams = teams
        self._table = {y: 0 for y in self._teams.copy()}
        self._predition_df = prediction_df
        self._resolve_df()

    def _resolve_df(self):
        for index, row in self._predition_df.iterrows():
            if row["pred"] == 1:
                self._table[row["home_team"]] += 3
            elif row["pred"] == -1:
                self._table[row["away_team"]] += 3
            else:
                self._table[row["home_team"]] += 1
                self._table[row["away_team"]] += 1
        self._table = {
            k: v
            for k, v in sorted(
                self._table.items(), key=lambda item: item[1], reverse=True
            )
        }

    def get_results(self):
        t = PrettyTable(["Team", "Punkte"])
        for key, value in self._table.items():
            t.add_row([key, value])
        print(t)

    def get_winner(self, group: bool):
        if group:
            return list(self._table.keys())[0:2]
        return list(self._table.keys())[0]


class AchtelFinals:
    def __init__(self):
        self.a = None
        self.b = None
        self.c = None
        self.d = None
        self.e = None
        self.f = None
        self.g = None
        self.h = None

        self.first = []
        self.second = []
        self.third = []
        self.fourth = []
        self.fifth = []
        self.sixth = []
        self.seventh = []
        self.eigth = []

    def _new_matches(self):
        self.first.extend([self.a[0], self.b[1]])
        self.second.extend([self.c[0], self.d[1]])
        self.third.extend([self.d[0], self.c[1]])
        self.fourth.extend([self.b[0], self.a[1]])
        self.fifth.extend([self.e[0], self.f[1]])
        self.sixth.extend([self.g[0], self.h[1]])
        self.seventh.extend([self.f[0], self.e[1]])
        self.eigth.extend([self.h[0], self.g[1]])

    def get_all_matches(self):
        self._new_matches()
        return [
            self.first,
            self.second,
            self.third,
            self.fourth,
            self.fifth,
            self.sixth,
            self.seventh,
            self.eigth,
        ]


class ViertelFinals:
    def __init__(self):
        self.a = None
        self.b = None
        self.c = None
        self.d = None
        self.e = None
        self.f = None
        self.g = None
        self.h = None

        self.first = []
        self.second = []
        self.third = []
        self.fourth = []

    def _new_matches(self):
        self.first.extend([self.a, self.b])
        self.second.extend([self.c, self.d])
        self.third.extend([self.e, self.f])
        self.fourth.extend([self.g, self.h])

    def get_all_matches(self):
        self._new_matches()
        return [self.first, self.second, self.third, self.fourth]


class Semifinals:
    def __init__(self):
        self.a = None
        self.b = None
        self.c = None
        self.d = None

        self.first = []
        self.second = []

    def _new_matches(self):
        self.first.extend([self.a, self.b])
        self.second.extend([self.c, self.d])

    def get_all_matches(self):
        self._new_matches()
        return [self.first, self.second]


class Final:
    def __init__(self):
        self.a = None
        self.b = None

        self.first = []

    def _new_matches(self):
        self.first.extend([self.a, self.b])

    def get_all_matches(self):
        self._new_matches()
        return [self.first]
