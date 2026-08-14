import os
import asyncio
import re
import pandas as pd
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from pycaret.regression import setup, compare_models, finalize_model, save_model
from dotenv import load_dotenv
from langfuse import observe, Langfuse

load_dotenv()
langfuse_client = Langfuse()

BASE_URL = "https://www.otodom.pl/pl/oferty/sprzedaz/mieszkanie/warszawa"
MAX_PAGES = 30  # 30 stron = ok. 1000 ofert

def clean_number(text: str) -> float | None:
    if not text:
        return None
    cleaned = re.sub(r'[^\d,.]', '', text).replace(',', '.').strip()
    try:
        return float(cleaned)
    except ValueError:
        return None

async def scrape_otodom_page(page, page_num: int) -> list[dict]:
    url = f"{BASE_URL}?page={page_num}"
    print(f"🔍 Pobieranie strony {page_num} z {MAX_PAGES}...")
    
    try:
        response = await page.goto(url, wait_until="domcontentloaded", timeout=25000)
        if response and response.status in [403, 429]:
            print(f"⚠️ Blokada antyscrapingowa (Status {response.status})")
            return []
            
        await page.wait_for_selector('article', timeout=10000)
    except Exception:
        return []

    content = await page.content()
    soup = BeautifulSoup(content, 'html.parser')
    offers = []
    
    articles = soup.find_all('article')
    
    for article in articles:
        try:
            text = article.get_text(separator=" | ").strip()
            
            # Szukanie ceny
            price_match = re.search(r'([\d\s]+)\xa0?zł', text) or re.search(r'([\d\s]{5,})\s?zł', text)
            if not price_match:
                continue
            price = clean_number(price_match.group(1))
            
            # Szukanie metrażu
            area_match = re.search(r'([\d\s,.]+)\s?m²', text)
            area = clean_number(area_match.group(1)) if area_match else None
            
            # Szukanie liczby pokoi
            rooms_match = re.search(r'(\d+)\s?(pokoje|pokój|pokoi)', text, re.IGNORECASE)
            rooms = int(rooms_match.group(1)) if rooms_match else None

            # Dzielnica z tekstu
            dzielnice = ['Mokotów', 'Wola', 'Ursynów', 'Praga-Południe', 'Bielany', 'Śródmieście', 
                         'Targówek', 'Bemowo', 'Białołęka', 'Ochota', 'Wawer', 'Żoliborz', 
                         'Ursus', 'Włochy', 'Praga-Północ', 'Wesoła', 'Rembertów', 'Wilanów']
            
            found_dzielnica = "Inna"
            for d in dzielnice:
                if d.lower() in text.lower():
                    found_dzielnica = d
                    break

            if price and area and rooms:
                offers.append({
                    "Dzielnica": found_dzielnica,
                    "Metraż": area,
                    "Pokoje": rooms,
                    "Piętro": 2, # Wartość domyślna jeśli brak na liście
                    "Cena": price
                })
        except Exception:
            continue
            
    return offers

async def run_scraper() -> pd.DataFrame:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            locale="pl-PL"
        )
        page = await context.new_page()
        
        all_offers = []
        for p_num in range(1, MAX_PAGES + 1):
            offers = await scrape_otodom_page(page, p_num)
            all_offers.extend(offers)
            await asyncio.sleep(0.5)
            
        await browser.close()
        return pd.DataFrame(all_offers)

def train_model(df: pd.DataFrame):
    print("\n🚀 Inicjalizacja uczenia maszynowego w PyCaret...")
    
    # Usuwanie duplikatów i braków
    df_train = df.drop_duplicates().dropna(subset=['Cena', 'Metraż', 'Pokoje']).copy()
    
    # Filtrowanie skrajności
    df_train = df_train[(df_train['Cena'] >= 250000) & (df_train['Cena'] <= 3500000)]
    df_train = df_train[(df_train['Metraż'] >= 18) & (df_train['Metraż'] <= 180)]
    
    df_train = df_train.reset_index(drop=True)
    print(f"📊 Liczba czystych rekordów do nauki: {len(df_train)}")

    exp = setup(
        data=df_train,
        target='Cena',
        categorical_features=['Dzielnica'],
        numeric_features=['Metraż', 'Pokoje'],
        preprocess=True,
        normalize=True,
        transformation=True,
        session_id=42,
        verbose=False
    )
    
    print("🤖 Wybór najlepszego algorytmu regresyjnego...")
    best_model = compare_models(sort='RMSE', n_select=1)
    
    print("🔒 Finalizacja i zapis modelu...")
    final_pipeline = finalize_model(best_model)
    save_model(final_pipeline, "real_estate_model")
    print("✅ Model zapisany pomyślnie jako 'real_estate_model.pkl'.")

def main():
    print("=== START SZYBKIEGO PIPELINE'U TRENINGOWEGO ===")
    csv_file = "dane_warszawa.csv"
    
    if os.path.exists(csv_file):
        os.remove(csv_file) # Automatyczne czyszczenie dla świeżego przebiegu
        
    print("🌐 Pobieranie hurtowej ilości ofert ze strony Otodom...")
    df_raw = asyncio.run(run_scraper())
    
    if df_raw.empty:
        print("❌ Brak danych ze scrapingu.")
        return
        
    df_raw.to_csv(csv_file, index=False, encoding="utf-8-sig")
    print(f"💾 Zapisano {len(df_raw)} ofert do '{csv_file}'.")

    train_model(df_raw)

if __name__ == "__main__":
    main()