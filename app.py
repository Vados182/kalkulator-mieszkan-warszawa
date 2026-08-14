import os
import base64
import pandas as pd
import streamlit as st
from pycaret.regression import load_model, predict_model
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv
import instructor
from openai import OpenAI
from langfuse import observe

# Wczytanie środowiska (.env)
load_dotenv()

# --- 1. KONFIGURACJA STRONY STREAMLIT ---
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

possible_names = ["tło.webp"]
found_image = None

for name in possible_names:
    if os.path.exists(name):
        found_image = name
        break

css_bg = ""
if found_image:
    encoded_bg = get_base64_image(found_image)
    ext = found_image.split('.')[-1].lower()
    if ext == 'jpg': ext = 'jpeg'
    css_bg = f"data:image/{ext};base64,{encoded_bg}"

# --- 3. STYLIZACJA CSS ---
st.markdown(
    f"""
    <style>
    /* Resetujemy czarne tło Streamlita na całkowicie przezroczyste */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
        background: transparent !important;
    }}

    /* Nakładamy zdjęcie na sam spód strony */
    html {{
        background-image: url('{css_bg}') !important;
        background-size: cover !important;
        background-position: center !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
    }}

    /* Główna karta na środku - lekko półprzezroczysta dla widoczności zdjęcia */
    .stMainBlockContainer {{
        background-color: rgba(255, 255, 255, 0.50) !important;
        backdrop-filter: blur(8px) !important;
        padding: 2.5rem !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
        margin-top: 2rem !important;
        margin-bottom: 2rem !important;
    }}

    /* Wyraźny czarny tekst na karcie */
    .stMainBlockContainer h1, 
    .stMainBlockContainer h2, 
    .stMainBlockContainer h3, 
    .stMainBlockContainer p, 
    .stMainBlockContainer span, 
    .stMainBlockContainer label,
    .stMainBlockContainer div {{
        color: #0F172A !important;
        font-weight: 600 !important;
    }}

    /* Etykiety zakładek */
    button[data-baseweb="tab"] span {{
        color: #0F172A !important;
        font-weight: 700 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- 3. WCZYTANIE MODELU ML (ZACACHOWANE) ---
@st.cache_resource
def get_ml_model():
    return load_model("real_estate_model")

model = get_ml_model()

# --- 4. MODEL PYDANTIC I STRUKTURA NLP ---
class PropertyExtractor(BaseModel):
    dzielnica: Optional[str] = Field(default=None, description="Dzielnica Warszawy, np. Mokotów, Wola, Śródmieście")
    metraz: Optional[float] = Field(default=None, description="Powierzchnia w m2")
    pokoje: Optional[int] = Field(default=None, description="Liczba pokoi")
    pietro: Optional[int] = Field(default=2, description="Piętro, domyślnie 2 jeśli brak")

@observe(name="extract_property_data")
def extract_data_from_text(text: str) -> PropertyExtractor:
    client = instructor.from_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
    return client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=PropertyExtractor,
        messages=[
            {
                "role": "system",
                "content": "Wyciągnij parametry mieszkania w Warszawie z tekstu: dzielnicę, metraż (m²), liczbę pokoi i piętro."
            },
            {"role": "user", "content": text}
        ]
    )

# --- 5. FUNKCJA POMOCNICZA DO WYCENY ---
def make_prediction(dzielnica: str, metraz: float, pokoje: int, pietro: int):
    df = pd.DataFrame([{
        'Dzielnica': dzielnica,
        'Metraż': metraz,
        'Pokoje': pokoje,
        'Piętro': pietro
    }])
    prediction = predict_model(model, data=df)
    cena = prediction['prediction_label'].iloc[0]
    cena_m2 = cena / metraz
    return cena, cena_m2

DZIELNICE = ['Mokotów', 'Wola', 'Ursynów', 'Praga-Południe', 'Bielany', 'Śródmieście', 
             'Targówek', 'Bemowo', 'Białołęka', 'Ochota', 'Wawer', 'Żoliborz', 
             'Ursus', 'Włochy', 'Praga-Północ', 'Wesoła', 'Rembertów', 'Wilanów', 'Inna']

# --- 6. INTERFEJS UŻYTKOWNIKA (STREAMLIT) ---
st.title("Kalkulator Cen Mieszkań w Warszawie")
st.write("Aplikacja szacuje wartość nieruchomości na podstawie modelu Machine Learning.")

tab1, tab2 = st.tabs(["🎛️ Szybki Formularz", "🤖 Analiza Tekstu (NLP)"])

# ==========================================
# ZAKŁADKA 1: FORMULARZ (SUWAKI)
# ==========================================
with tab1:
    st.subheader("Wprowadź parametry ręcznie")
    
    col1, col2 = st.columns(2)
    with col1:
        dzielnica_input = st.selectbox("Dzielnica", DZIELNICE, index=0)
        metraz_input = st.number_input("Metraż (m²)", min_value=15.0, max_value=200.0, value=50.0, step=0.5)
    with col2:
        pokoje_input = st.slider("Liczba pokoi", min_value=1, max_value=6, value=2)
        pietro_input = st.number_input("Piętro", min_value=0, max_value=30, value=2)

    if st.button("Calculate", type="primary", key="btn_form"):
        cena, cena_m2 = make_prediction(dzielnica_input, metraz_input, pokoje_input, pietro_input)
        
        st.success("✅ Szacunkowa wycena gotowa!")
        c1, c2 = st.columns(2)
        c1.metric("Szacowana Cena", f"{cena:,.2f} PLN")
        c2.metric("Cena za m²", f"{cena_m2:,.2f} PLN/m²")

# ==========================================
# ZAKŁADKA 2: NLP (TEKST OGŁOSZENIA)
# ==========================================
with tab2:
    st.subheader("Wklej opis ogłoszenia lub wpisz frazę")
    text_input = st.text_area(
        "Opis ogłoszenia", 
        placeholder="np. Sprzedam bezpośrednio mieszkanie na Woli przy metrze, 45 metrów, 2 pokoje, 3 piętro...",
        height=120
    )

    if st.button("Przeanalizuj i Wyceniaj", type="primary", key="btn_nlp"):
        if not text_input.strip():
            st.warning("⚠️ Wpisz lub wklej tekst ogłoszenia.")
        else:
            with st.spinner("🤖 AI analizuje tekst ogłoszenia..."):
                extracted = extract_data_from_text(text_input)
            
            # --- WALIDACJA BRAKUJĄCYCH DANYCH ---
            missing_fields = []
            if not extracted.dzielnica or extracted.dzielnica not in DZIELNICE:
                missing_fields.append("Dzielnica")
            if not extracted.metraz:
                missing_fields.append("Metraż (m²)")
            if not extracted.pokoje:
                missing_fields.append("Liczba pokoi")

            if missing_fields:
                st.error(f"❌ **Nie udało się wycenć mieszkania!** Brakuje następujących kluczowych danych w tekście: **{', '.join(missing_fields)}**.")
                st.info("💡 Spróbuj dopisać brakujące informacje do tekstu i spróbuj ponownie.")
            else:
                # Wyświetlenie odczytanych danych z tekstu
                st.write("🔍 **Odczytane dane z tekstu:**")
                st.json({
                    "Dzielnica": extracted.dzielnica,
                    "Metraż": f"{extracted.metraz} m²",
                    "Pokoje": extracted.pokoje,
                    "Piętro": extracted.pietro
                })
                
                cena, cena_m2 = make_prediction(
                    extracted.dzielnica, 
                    extracted.metraz, 
                    extracted.pokoje, 
                    extracted.pietro
                )
                
                st.success("✅ Wycena wygenerowana pomyślnie!")
                c1, c2 = st.columns(2)
                c1.metric("Szacowana Cena", f"{cena:,.2f} PLN")
                c2.metric("Cena za m²", f"{cena_m2:,.2f} PLN/m²")