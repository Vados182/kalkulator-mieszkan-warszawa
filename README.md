# 🏠 Kalkulator Cen Mieszkań w Warszawie (ML & NLP)

Aplikacja webowa do szacowania wartości nieruchomości w Warszawie z wykorzystaniem modeli Machine Learning oraz zaawansowanej analizy tekstu (NLP).

🔗 **Live Demo:** [Otwórz aplikację Streamlit](https://kalkulator-mieszkan-warszawa-mwvogqtaov9nyf7h6fsmaj.streamlit.app)

## 🚀 Funkcjonalności

- **🎛️ Szybki Formularz:** Ręczny dobór parametrów mieszkania (dzielnica, metraż, liczba pokoi, piętro) i błyskawiczna wycena.
- **🤖 Analiza Tekstu (NLP):** Wklejenie surowego opisu ogłoszenia – LLM (`gpt-4o-mini` z `instructor`) automatycznie wyciąga kluczowe parametry do wyceny.
- **🎨 Zcustomizowany UI:** Dedykowany styl CSS z obsługą motywu i responsywnym układem.
- **📊 Observability:** Integracja z narzędziem Langfuse do monitorowania wywołań LLM.

## 🛠️ Technologie

- **Python 3.11**
- **Streamlit** – Interfejs użytkownika & CSS Customization
- **PyCaret** – Modelowanie predykcyjne (Regresja)
- **OpenAI API & Instructor** – Strukturyzowana ekstrakcja danych z tekstu (Pydantic)
- **Langfuse** – Tracing i monitoring zapytań AI

