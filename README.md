# Sistema di Analisi Documentale e Generazione Checklist

Questa applicazione Flask consente di analizzare documenti PDF mediante l'estrazione di testo e immagini, l'elaborazione tramite un modello LLM e un semplice classificatore PyTorch, e la generazione automatica di checklist strutturate.  
L'interfaccia web permette di consultare, approvare o rifiutare le checklist generate, con aree dedicate sia agli utenti che agli amministratori.

## Caratteristiche principali

- **Caricamento e analisi dei PDF:**  
  Utilizza PyMuPDF e PyPDFLoader per estrarre testo e immagini dai documenti.
  
- **Elaborazione del testo con LLM:**  
  Suddivide il testo in chunk e invia ogni parte a un LLM tramite OllamaLLM per ricevere un output in formato JSON.
  
- **Generazione checklist automatica:**  
  Converte l'output del LLM in un set strutturato di checklist, applicando regole per associare procedure specifiche.
  
- **Analisi delle immagini:**  
  Esegue OCR sulle immagini per filtrare quelle rilevanti e usa un modello PyTorch per predizioni preliminari.
  
- **Dashboard e interfaccia web:**  
  Area utente e admin panel realizzate con Flask per la gestione e visualizzazione dei dati.
  
- **Esportazione e Integrazione:**  
  Possibilità di esportare le checklist in JSON e di inviare i dati via FTP.

## Requisiti

- Python 3.8+  
- Flask  
- SQLite (incluso in Python standard library)  
- PyMuPDF (fitz), PyPDFLoader, pytesseract  
- Pillow, NumPy  
- PyTorch  
- Langchain e langchain_community  
- Requests  
- Altri moduli Python elencati nel file `requirements.txt` (consigliato)

## Installazione

1. **Clona il repository:**

   ```bash
   git clone https://github.com/tuo-username/nome-repo.git
   cd nome-repo
