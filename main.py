import time

from sklearn.preprocessing import OneHotEncoder

from helper import (
    filter_relevant_data,
    get_obj_cols_in_df,
    preprocess_df,
    read_data,
    relabel,
    solve_matches,
    train_model_from_df,
)
from table import AchtelFinals, Final, Semifinals, ViertelFinals

enc = OneHotEncoder(handle_unknown="ignore", sparse=False)

teams = [
    "Germany",
    "Denmark",
    "France",
    "Belgium",
    "Croatia",
    "Spain",
    "Serbia",
    "Switzerland",
    "England",
    "Netherlands",
    "Portugal",
    "Poland",
    "Wales",
    "Marokko",
    "Kamerun",
    "Tunisia",
    "Senegal",
    "Ghana",
    "Qatar",
    "Japan",
    "South Korea",
    "Saudi Arabia",
    "Iran",
    "Australia",
    "Brasil",
    "Argentina",
    "Uruguay",
    "Ecuador",
    "Canada",
    "Mexiko",
    "USA",
    "Costa Rica",
]

af = AchtelFinals()
vf = ViertelFinals()
sf = Semifinals()
finals = Final()

df = read_data()
df = filter_relevant_data(data=df, teams=teams, label_function=relabel)
prediction_df = preprocess_df(df, obj_cols=get_obj_cols_in_df(df=df), enc=enc)


train_model_from_df(
    prediction_df=prediction_df, train_size=0.8, use_confusion_matrix=True
)
time.sleep(20)
clf = train_model_from_df(
    prediction_df=prediction_df, train_size=0.99, use_confusion_matrix=False
)

# Gruppe A
gruppe_a = ["Ecuador", "Qatar", "Netherlands", "Senegal"]
# Gruppe B
gruppe_b = ["England", "Iran", "USA", "Wales"]
# Gruppe C
gruppe_c = ["Argentina", "Mexiko", "Poland", "Saudi Arabia"]
# Gruppe D
gruppe_d = ["Australia", "Denmark", "France", "Tunisia"]
# Gruppe E
gruppe_e = ["Germany", "Costa Rica", "Japan", "Spain"]
# Gruppe F
gruppe_f = ["Belgium", "Canada", "Croatia", "Marokko"]
# Gruppe G
gruppe_g = ["Brasil", "Kamerun", "Switzerland", "Serbia"]
# Gruppe H
gruppe_h = ["Ghana", "Portugal", "South Korea", "Uruguay"]

vorrunde = [
    gruppe_a,
    gruppe_b,
    gruppe_c,
    gruppe_d,
    gruppe_e,
    gruppe_f,
    gruppe_g,
    gruppe_h,
]


solve_matches(
    df=df,
    enc=enc,
    matches=vorrunde,
    tableclass=af,
    groupstage=True,
    message="---- VORRUNDE ----",
)
achtelfinals = af.get_all_matches()
print(achtelfinals)
time.sleep(20)

solve_matches(
    df=df,
    enc=enc,
    matches=achtelfinals,
    tableclass=vf,
    message="---- ACHTELFINALS ----",
)
viertelfinals = vf.get_all_matches()
print(viertelfinals)
time.sleep(15)

solve_matches(
    df=df,
    enc=enc,
    matches=viertelfinals,
    tableclass=sf,
    message="---- VIERTELFINALS ----",
)
semifinals = sf.get_all_matches()
print(semifinals)
time.sleep(15)

solve_matches(
    df=df,
    enc=enc,
    matches=semifinals,
    tableclass=finals,
    message="---- SEMIFINALS ----",
)
final = finals.get_all_matches()
print(final)
time.sleep(15)

solve_matches(df=df, enc=enc, matches=final, message="---- FINALS ----")
