import pandas as pd
from pycaret.regression import load_model, predict_model

# 1. Wczytujemy model RAZ (na poziomie globalnym)
PIPELINE = load_model("real_estate_model")

def wycen_mieszkanie(dzielnica: str, metraz: float, pokoje: int, pietro: int = 2):
    # 2. Tworzymy ramkę danych
    dane = pd.DataFrame([{
        'Dzielnica': dzielnica,
        'Metraż': metraz,
        'Pokoje': pokoje,
        'Piętro': pietro
    }])
    
    # 3. Fast predykcja ze zgromadzonego w pamięci RAM modelu
    prediction = predict_model(PIPELINE, data=dane)
    szacowana_cena = prediction['prediction_label'].iloc[0]
    cena_za_m2 = szacowana_cena / metraz
    
    print(f"\n🏠 --- SZACOWANA WYCENA NIERUCHOMOŚCI ---")
    print(f"📍 Lokalizacja: Warszawa, {dzielnica}")
    print(f"📐 Parametry:   {metraz} m² | {pokoje} pok. | {pietro}. piętro")
    print(f"💰 Szacowana cena całkowita: {szacowana_cena:,.2f} PLN")
    print(f"📊 Średnia cena za m²:      {cena_za_m2:,.2f} PLN/m²")
    print("-" * 42)

if __name__ == "__main__":
    wycen_mieszkanie("Mokotów", 50.0, 2, 3)
    wycen_mieszkanie("Wola", 35.0, 1)
    wycen_mieszkanie("Ursynów", 80.0, 4, 4)