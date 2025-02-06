import json
import streamlit as st
import pandas as pd  # type: ignore
from pycaret.clustering import load_model, predict_model  # type: ignore
import plotly.express as px  # type: ignore
import seaborn as sns
import matplotlib.pyplot as plt

# Stałe
MODEL_NAME = 'welcome_survey_clustering_pipeline_v2'
DATA = 'welcome_survey_simple_v2.csv'
CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v2.json'

# Cache dla wydajności
@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())

@st.cache_data
def get_all_participants():
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(get_model(), data=all_df)
    return df_with_clusters

# Sidebar - wybór użytkownika
with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
    
    age = st.selectbox("Wiek", ["-- Wybierz --", "<18", "18-24", "25-34", "35-44", "45-54", "55-64", ">=65", "unknown"])
    edu_level = st.selectbox("Wykształcenie", ["-- Wybierz --", 'Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ["-- Wybierz --", 'Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ["-- Wybierz --", 'Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ["-- Wybierz --", 'Mężczyzna', 'Kobieta'])

# Notatnik w sidebarze
with st.sidebar:
    st.header("📝 Notatnik")
    st.text_area("Wpisz notatkę (notatki nie zostaną zapisane po odświeżeniu)", height=200, max_chars=1000)

# Sprawdzamy, czy użytkownik wybrał wszystkie opcje
if all(option != "-- Wybierz --" for option in [age, edu_level, fav_animals, fav_place, gender]):
    person_df = pd.DataFrame([
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender,
        }
    ])

    model = get_model()
    all_df = get_all_participants()
    cluster_names_and_descriptions = get_cluster_names_and_descriptions()

    predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
    predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

    # Usuwamy centrowanie i rozszerzamy wyniki na pełną szerokość
    st.header(f"🎯 Najbliżej Ci do grupy: {predicted_cluster_data['name']}")
    st.markdown(predicted_cluster_data['description'])
    same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
    st.metric("Liczba twoich znajomych", len(same_cluster_df))

    st.header("🏆 Najczęściej wybierane opcje w Twojej grupie")

    # Mapowanie nazw kolumn
    column_names = {
        "fav_animals": "Ulubione zwierze",
        "fav_place": "Ulubione miejsce",
        "edu_level": "Poziom wykształcenia"
    }

    for col in column_names.keys():
        most_common = same_cluster_df[col].value_counts().idxmax()
        st.write(f"🔹 Najczęściej wybierane **{column_names[col]}**: **{most_common}**")

    st.header("📊 Rozkłady w Twojej grupie")

    fig = px.histogram(same_cluster_df.sort_values("age"), x="age", title="Rozkład wieku w grupie")
    st.plotly_chart(fig, use_container_width=True)  # Używamy pełnej szerokości

    fig = px.histogram(same_cluster_df, x="edu_level", title="Rozkład wykształcenia w grupie")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(same_cluster_df, x="fav_animals", title="Rozkład ulubionych zwierząt w grupie")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(same_cluster_df, x="fav_place", title="Rozkład ulubionych miejsc w grupie")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(same_cluster_df, x="gender", title="Rozkład płci w grupie")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.pie(same_cluster_df, names="gender", title="Proporcje procentowe płci")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("⏳ Wybierz wszystkie opcje w panelu bocznym, aby zobaczyć wyniki.")
