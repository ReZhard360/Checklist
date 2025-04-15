import os
import re
import io
import json
import csv
import sqlite3
import logging
from datetime import datetime
from ftplib import FTP

from flask import Flask, redirect, flash, send_from_directory, render_template_string, request
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import fitz  # PyMuPDF per estrazione immagini
import pytesseract  # OCR

# Import per analisi del PDF e LLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaLLM

logging.basicConfig(level=logging.INFO)

# Nomi dei file PDF
DEFAULT_PDF = "Relazione_procedure.pdf"  # Manuale generale
COMPONENT_PDFS = [
    "D840MIT-Rev.02_20171030.pdf",
    "Martino-Atti-1Sessione-LAcqua-n.-1-2-2002.pdf"
]

# Parametri per l'analisi
CHUNK_SIZE = 1500
OVERLAP_SIZE = 300
TEMPERATURE = 0.1

UPLOAD_FOLDER = 'uploads'
DATABASE = 'checklist.db'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'your_secret_key'  # Sostituisci con una chiave sicura

# Cache in memoria per risultati recenti (evita rianalisi se non forzato)
in_memory_cache = {}


def load_cached_analysis_memory(file_key):
    return in_memory_cache.get(file_key)


def save_analysis_memory(file_key, analysis_text, checklist, image_analysis):
    in_memory_cache[file_key] = (analysis_text, checklist, image_analysis)


# Inizializza il LLM (modello Mistral tramite Ollama)
llm = OllamaLLM(
    model="mistral",
    base_url="http://localhost:11434",
    temperature=TEMPERATURE
)


# Modello PyTorch semplice per l'analisi delle immagini
class SimpleNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(128 * 128 * 3, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


# Inizializza il database (crea la tabella se non esiste)
def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS checklists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            generated_at TEXT,
            analysis_text TEXT,
            checklist_json TEXT,
            image_analysis TEXT
        )
    ''')
    conn.commit()
    conn.close()


init_db()


# Funzione per estrarre immagini da un PDF, con verifica dell'integrità
def extract_images_from_pdf(file_path):
    images = []
    try:
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            for img in doc.get_page_images(page_num):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                try:
                    img_obj = Image.open(io.BytesIO(image_bytes))
                    img_obj.verify()  # Verifica l'integrità
                    img_obj = Image.open(io.BytesIO(image_bytes))  # Riapri l'immagine
                    images.append(img_obj)
                except Exception as e:
                    logging.error(
                        f"Impossibile identificare l'immagine in {file_path} (page {page_num}, xref {xref}): {e}")
        logging.info(f"Estrazione immagini completata per {file_path}.")
        return images
    except Exception as e:
        logging.error(f"Errore durante l'estrazione delle immagini da {file_path}: {e}")
        return []


# Filtra le immagini utili tramite OCR (parole chiave in italiano)
def filter_useful_images(images):
    useful_images = []
    keywords = ["disegno", "planta", "schema", "componente", "elenco"]
    for img in images:
        try:
            ocr_text = pytesseract.image_to_string(img, lang="ita")
            if any(keyword in ocr_text.lower() for keyword in keywords):
                useful_images.append({"image": img, "ocr_text": ocr_text})
        except Exception as e:
            logging.error(f"Errore nell'OCR dell'immagine: {e}")
    return useful_images


def preprocess_images(images):
    processed_images = []
    for item in images:
        img = item["image"].resize((128, 128))
        img_array = np.array(img.convert('RGB')) / 255.0
        processed_images.append(img_array)
    return np.array(processed_images)


def analyze_images(images):
    processed = preprocess_images(images)
    input_tensor = torch.tensor(processed, dtype=torch.float32)
    model = SimpleNN(num_classes=5)
    with torch.no_grad():
        outputs = model(input_tensor)
        predictions = torch.argmax(outputs, dim=1).numpy()
    return predictions


def load_document(file_path: str):
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text = " ".join([doc.page_content for doc in docs]).strip()
        logging.info(f"Documento {file_path} caricato correttamente.")
        return text
    except Exception as e:
        logging.error(f"Errore nel caricamento del documento {file_path}: {e}")
        return None


# Analizza il testo combinato dei manuali con un prompt estremamente severo
def analyze_text_combined(text, chunk_size, overlap_size, temperature):
    llm.temperature = temperature
    prompt_template = (
        "Sei un esperto di manutenzione industriale e sicurezza operativa, noto per il tuo rigore e precisione. "
        "Analizza con estrema severità il seguente testo, che unisce le informazioni di più manuali, per identificare e classificare "
        "in modo dettagliato tutti i problemi che possono compromettere il funzionamento di un sistema idraulico e dei suoi componenti. "
        "Per ciascun problema, fornisci:\n"
        " - Un titolo mirato e specifico (es. 'Interruzione alimentazione elettrica', 'Manutenzione Motore', ecc.)\n"
        " - Una descrizione estremamente dettagliata del problema, evidenziando il componente interessato, l'impatto operativo e i rischi associati\n"
        " - Un elenco rigoroso delle possibili cause\n"
        " - Una lista completa di procedure di manutenzione che descriva chiaramente COME procedere, includendo tutti i passaggi operativi e di sicurezza obbligatori\n\n"
        "Il formato di output deve essere esattamente il seguente:\n\n"
        "Problema 1:\n"
        "- Titolo: [Titolo specifico del problema]\n"
        "- Descrizione: [Descrizione estremamente dettagliata del problema]\n"
        "- Cause: [Elenco rigoroso delle possibili cause]\n"
        "- Procedure di Manutenzione: [Lista completa dei passaggi operativi e di sicurezza]\n"
        "...\n"
        "Problema N:\n"
        "- Titolo: ...\n"
        "- Descrizione: ...\n"
        "- Cause: ...\n"
        "- Procedure di Manutenzione: ...\n\n"
        "Testo:\n{0}"
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap_size)
    split_docs = text_splitter.split_text(text)
    analyses = []
    for idx, doc in enumerate(split_docs):
        prompt = prompt_template.format(doc)
        try:
            logging.info(f"Invio chunk {idx + 1} al LLM...")
            analysis = llm.invoke(prompt)
            analyses.append(analysis)
        except Exception as e:
            logging.error(f"Errore nell'invocazione del LLM: {e}")
    full_analysis = "\n\n".join(analyses)
    return full_analysis, analyses


# Applica regole smart per assegnare il tipo di intervento e integra procedure standard
def apply_rules(checklist):
    rules = {
        "motore": ("Manutenzione Motore",
                   "Verificare il funzionamento del motore, controllare collegamenti e sensori; staccare la corrente prima dell'intervento; comunicare alla sala di controllo; procedere seguendo le istruzioni di sicurezza."),
        "alimentazione": ("Controllo Alimentazione",
                          "Verificare l'alimentazione elettrica; in caso di anomalie, staccare la corrente ed attivare il backup; informare la sala di controllo; procedere con cautela."),
        "pressione": ("Regolazione Pressione",
                      "Controllare la pressione del circuito; se anomala, ispezionare valvole e raccordi e regolare in sicurezza; seguire le procedure operative indicate."),
        "perdita": ("Riparazione Perdite",
                    "Individuare la fonte della perdita; bloccare il flusso; staccare la corrente se necessario; informare la sala di controllo; procedere immediatamente per evitare danni.")
    }
    for key, value in checklist.items():
        lower_title = key.lower()
        lower_description = value['description'].lower()
        tipo_intervento = "Intervento Generico"
        for rule_keyword, (tipo, rule_text) in rules.items():
            if rule_keyword in lower_title or rule_keyword in lower_description:
                tipo_intervento = tipo
                if rule_text not in value['procedures']:
                    value['procedures'] += "\n" + rule_text
        value['tipo_intervento'] = tipo_intervento
        value['associated_images'] = []
    return checklist


def generate_checklist(analyses):
    problems = {}
    pattern = re.compile(
        r'Problema\s*(\d+):\s*'
        r'- Titolo:\s*(.*?)\n'
        r'- Descrizione:\s*(.*?)\n'
        r'- Cause:\s*(.*?)\n'
        r'- Procedure di Manutenzione:\s*(.*?)(?=Problema\s*\d+:|$)',
        re.DOTALL
    )
    for analysis in analyses:
        matches = pattern.findall(analysis)
        for match in matches:
            _, title, description, causes, procedures = match
            problems[title.strip()] = {
                'description': description.strip(),
                'causes': causes.strip(),
                'procedures': procedures.strip()
            }
    problems = apply_rules(problems)
    return problems


# Associa le immagini utili alle voci della checklist in base all'OCR
def associate_images_to_checklist(checklist, filtered_images):
    for item in filtered_images:
        ocr_text = item["ocr_text"].lower()
        keywords = ["disegno", "planta", "schema", "componente", "elenco"]
        for title, details in checklist.items():
            text_to_check = (title + " " + details["description"]).lower()
            if any(kw in ocr_text for kw in keywords) and any(kw in text_to_check for kw in keywords):
                details["associated_images"].append(item["image_filename"])
    return checklist


def save_checklist_to_db(file_key, analysis_text, checklist, image_analysis):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    generated_at = datetime.now().isoformat()
    checklist_json = json.dumps(checklist, ensure_ascii=False)
    image_analysis_json = json.dumps(image_analysis, ensure_ascii=False)
    c.execute(
        "INSERT INTO checklists (file_name, generated_at, analysis_text, checklist_json, image_analysis) VALUES (?, ?, ?, ?, ?)",
        (file_key, generated_at, analysis_text, checklist_json, image_analysis_json))
    conn.commit()
    conn.close()


def load_cached_analysis(file_key):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute(
        "SELECT analysis_text, checklist_json, image_analysis FROM checklists WHERE file_name = ? ORDER BY id DESC LIMIT 1",
        (file_key,))
    row = c.fetchone()
    conn.close()
    if row:
        return row[0], json.loads(row[1]), json.loads(row[2])
    else:
        return None


# Rotta per esportare le checklist in formato JSON strutturato
def export_checklists_to_structured_json(json_filename="checklists_structured.json"):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("SELECT file_name, generated_at, checklist_json FROM checklists ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    checklist_list = []
    for row in rows:
        checklist_obj = {
            "file_name": row[0],
            "generated_at": row[1],
            "checklist": []
        }
        checklist_data = json.loads(row[2])
        for title, details in checklist_data.items():
            causes = details.get("causes", "").split("\n") if isinstance(details.get("causes", ""),
                                                                         str) else details.get("causes", [])
            procedures = details.get("procedures", "").split("\n") if isinstance(details.get("procedures", ""),
                                                                                 str) else details.get("procedures", [])
            checklist_obj["checklist"].append({
                "title": title,
                "description": details.get("description", ""),
                "causes": [cause.strip() for cause in causes if cause.strip()],
                "procedures": [proc.strip() for proc in procedures if proc.strip()],
                "tipo_intervento": details.get("tipo_intervento", ""),
                "associated_images": details.get("associated_images", [])
            })
        checklist_list.append(checklist_obj)

    output = {
        "generated_at": datetime.now().isoformat(),
        "source_files": COMPONENT_PDFS + [DEFAULT_PDF],
        "checklists": checklist_list
    }

    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    logging.info(f"Esportazione strutturata JSON completata: {json_filename}")
    return send_from_directory(os.getcwd(), json_filename, as_attachment=True)


@app.route('/export_json_structured')
def export_json_structured():
    return export_checklists_to_structured_json()


# Rotta per caricare il file JSON via FTP
@app.route('/upload_ftp')
def upload_ftp():
    json_filename = "checklists_structured.json"
    ftp_host = "ftp.water4.altervista.org"
    ftp_port = 21
    ftp_username = "hespanel@water4"
    ftp_password = "saoihj$q1"
    # Usa un percorso relativo (l'utente ha accesso solo alla sua root)
    remote_directory = "procedures"
    remote_filename = "checklists_structured.json"

    # Esporta il JSON strutturato (lo crea se non esiste)
    export_checklists_to_structured_json(json_filename)

    try:
        ftp = FTP(timeout=30)
        ftp.encoding = "latin-1"
        ftp.connect(ftp_host, ftp_port)
        ftp.login(ftp_username, ftp_password)
        try:
            ftp.cwd(remote_directory)
        except Exception as e:
            logging.info(f"La directory '{remote_directory}' non esiste. Creazione in corso...")
            ftp.mkd(remote_directory)
            ftp.cwd(remote_directory)
        with open(json_filename, "rb") as f:
            ftp.storbinary(f"STOR {remote_filename}", f)
        ftp.quit()
        return f"File {json_filename} caricato via FTP nella cartella {remote_directory} come {remote_filename}."
    except Exception as e:
        return f"Errore durante l'upload FTP: {e}"


# Template per la home con overlay di caricamento
index_template = """
<!doctype html>
<html lang="it">
  <head>
    <meta charset="utf-8">
    <title>Analisi Manuale Idraulico</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
      #loadingOverlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%%;
        height: 100%%;
        background: rgba(0, 0, 0, 0.5);
        z-index: 9999;
        display: none;
        align-items: center;
        justify-content: center;
        color: #fff;
        font-size: 1.5rem;
      }
    </style>
  </head>
  <body>
    <div id="loadingOverlay">Caricamento in corso, attendere...</div>
    <div class="container my-5">
      <h1 class="mb-4">Analisi dei Manuali</h1>
      <p>Il sistema unirà le informazioni del manuale generale <strong>{{ default_pdf }}</strong> e dei seguenti manuali di componente:</p>
      <ul>
        {% for file in component_pdfs %}
          <li><strong>{{ file }}</strong></li>
        {% endfor %}
      </ul>
      <p>Clicca per eseguire un'analisi unificata, estremamente professionale e severa.<br>
         Se non viene richiesto il "force", verrà caricata la checklist già salvata.</p>
      <a href="/process?file=all" class="btn btn-primary" id="analyzeBtn">Avvia Analisi Unificata</a>
      <br><br>
      <a href="/checklists" class="btn btn-secondary">Visualizza Checklist Salvate (dinamiche)</a>
      <br><br>
      <a href="/export_json_structured" class="btn btn-info">Esporta Checklist in JSON</a>
      <br><br>
      <a href="/upload_ftp" class="btn btn-warning">Carica JSON via FTP</a>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
      document.getElementById("analyzeBtn").addEventListener("click", function(){
          document.getElementById("loadingOverlay").style.display = "flex";
      });
    </script>
  </body>
</html>
"""

# Template per i risultati con modale per dettagli
result_template = """
<!doctype html>
<html lang="it">
  <head>
    <meta charset="utf-8">
    <title>Risultato Checklist</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
      pre { background-color: #f8f9fa; padding: 15px; border-radius: 5px; }
    </style>
  </head>
  <body>
    <div class="container my-5">
      <h1 class="mb-4">Checklist Generata: {{ file_key }}</h1>
      {% if checklist %}
        <div class="row">
          {% for title, details in checklist.items() %}
            <div class="col-md-6">
              <div class="card mb-3">
                <div class="card-body">
                  <h5 class="card-title">{{ title }}</h5>
                  <h6 class="card-subtitle mb-2 text-muted">Tipo Intervento: {{ details.tipo_intervento }}</h6>
                  <p class="card-text">{{ details.description[:100] ~ ('...' if details.description|length > 100 else '') }}</p>
                  {% if details.associated_images %}
                    <p class="mb-2"><strong>Immagini associate:</strong></p>
                    <div>
                      {% for img in details.associated_images %}
                        <img src="/uploads/{{ img }}" style="max-height: 100px; margin-right: 5px;" alt="Immagine associata">
                      {% endfor %}
                    </div>
                  {% endif %}
                  <button type="button" class="btn btn-primary view-details-btn" 
                          data-bs-toggle="modal" data-bs-target="#problemModal"
                          data-title="{{ title }}"
                          data-description="{{ details.description }}"
                          data-causes="{{ details.causes }}"
                          data-procedures="{{ details.procedures|replace('\n', '|||') }}">
                    Visualizza Dettagli
                  </button>
                </div>
              </div>
            </div>
          {% endfor %}
        </div>
      {% else %}
        <p class="text-warning">Nessun problema identificato.</p>
      {% endif %}

      {% if image_analysis %}
      <h2 class="mt-5">Immagini Utili per l'Intervento</h2>
      <div class="row">
        {% for idx, result in image_analysis.items() %}
          <div class="col-md-4">
            <div class="card mb-3">
              <img src="/uploads/{{ result.image_filename }}" class="card-img-top" alt="Immagine {{ idx }}">
              <div class="card-body">
                <h6 class="card-subtitle mb-2 text-muted">Classe Predetta: {{ result.prediction }}</h6>
              </div>
            </div>
          </div>
        {% endfor %}
      </div>
      {% endif %}

      <a href="/" class="btn btn-secondary mt-3">Torna alla Home</a>
    </div>

    <!-- Modal per il dettaglio del problema -->
    <div class="modal fade" id="problemModal" tabindex="-1" aria-labelledby="problemModalLabel" aria-hidden="true">
      <div class="modal-dialog modal-lg">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title" id="problemModalLabel">Dettagli del Problema</h5>
            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Chiudi"></button>
          </div>
          <div class="modal-body">
            <h5 id="modalTitle"></h5>
            <p><strong>Descrizione:</strong> <span id="modalDescription"></span></p>
            <p><strong>Cause:</strong> <span id="modalCauses"></span></p>
            <p><strong>Procedure di Manutenzione:</strong></p>
            <div id="modalProcedures" class="list-group"></div>
          </div>
          <div class="modal-footer">
            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Chiudi</button>
          </div>
        </div>
      </div>
    </div>

    <script>
      document.addEventListener('DOMContentLoaded', function() {
        var problemModal = document.getElementById('problemModal');
        problemModal.addEventListener('show.bs.modal', function (event) {
          var button = event.relatedTarget;
          var title = button.getAttribute('data-title');
          var description = button.getAttribute('data-description');
          var causes = button.getAttribute('data-causes');
          var procedures_raw = button.getAttribute('data-procedures');
          var procedures = procedures_raw.split('|||').filter(function(step) {
              return step.trim() !== "";
          });
          document.getElementById('modalTitle').innerText = title;
          document.getElementById('modalDescription').innerText = description;
          document.getElementById('modalCauses').innerText = causes;
          var modalProceduresDiv = document.getElementById('modalProcedures');
          modalProceduresDiv.innerHTML = "";
          procedures.forEach(function(step, index) {
            var div = document.createElement('div');
            div.className = "form-check mb-2";
            var checkbox = document.createElement('input');
            checkbox.className = "form-check-input";
            checkbox.type = "checkbox";
            checkbox.id = "procedureCheck" + index;
            var label = document.createElement('label');
            label.className = "form-check-label";
            label.htmlFor = "procedureCheck" + index;
            label.innerText = step.trim();
            div.appendChild(checkbox);
            div.appendChild(label);
            modalProceduresDiv.appendChild(div);
          });
        });
      });
    </script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
  </body>
</html>
"""

# Template per visualizzare le checklist salvate (accordion)
checklists_template = """
<!doctype html>
<html lang="it">
  <head>
    <meta charset="utf-8">
    <title>Checklist Salvate</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
      pre { background-color: #f8f9fa; padding: 15px; border-radius: 5px; }
    </style>
  </head>
  <body>
    <div class="container my-5">
      <h1 class="mb-4">Elenco Checklist Salvate</h1>
      {% if checklists %}
        <div class="accordion" id="checklistAccordion">
          {% for item in checklists %}
            <div class="accordion-item">
              <h2 class="accordion-header" id="heading{{ item.id }}">
                <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapse{{ item.id }}" aria-expanded="false" aria-controls="collapse{{ item.id }}">
                  Record ID: {{ item.id }} - File: {{ item.file_name }} - Generata il: {{ item.generated_at }}
                </button>
              </h2>
              <div id="collapse{{ item.id }}" class="accordion-collapse collapse" aria-labelledby="heading{{ item.id }}" data-bs-parent="#checklistAccordion">
                <div class="accordion-body">
                  <pre>{{ item.checklist | tojson(indent=2) }}</pre>
                </div>
              </div>
            </div>
          {% endfor %}
        </div>
      {% else %}
        <p class="text-warning">Nessuna checklist salvata.</p>
      {% endif %}
      <a href="/" class="btn btn-secondary mt-3">Torna alla Home</a>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
  </body>
</html>
"""


# Rotta per la home
@app.route('/')
def index():
    return render_template_string(index_template, default_pdf=DEFAULT_PDF, component_pdfs=COMPONENT_PDFS)


# Rotta per processare i file; se file=all, unisce i manuali
@app.route('/process')
def process():
    file_param = request.args.get("file", DEFAULT_PDF)
    force_update = request.args.get("force", "false").lower() == "true"

    if file_param == "all":
        files_to_process = [DEFAULT_PDF] + COMPONENT_PDFS
        file_key = "all"
    else:
        files_to_process = [file_param]
        file_key = file_param

    for f in files_to_process:
        if not os.path.exists(f):
            flash(f"Il file {f} non esiste nella directory.")
            return redirect("/")

    # Verifica cache (in memoria o DB)
    cached = load_cached_analysis_memory(file_key)
    if not cached:
        cached = load_cached_analysis(file_key)
        if cached:
            save_analysis_memory(file_key, *cached)
    if cached and not force_update:
        analysis_text, checklist, image_analysis = cached
        logging.info("Risultato caricato dalla cache.")
        return render_template_string(result_template, analysis_text=analysis_text, checklist=checklist,
                                      image_analysis=image_analysis, file_key=file_key)

    combined_text = ""
    combined_images = []
    for f in files_to_process:
        text = load_document(f)
        if text:
            combined_text += text + "\n"
        images = extract_images_from_pdf(f)
        if images:
            combined_images.extend(images)

    if not combined_text:
        flash("Errore nel caricamento dei documenti.")
        return redirect("/")

    analysis_text, analyses = analyze_text_combined(combined_text, CHUNK_SIZE, OVERLAP_SIZE, TEMPERATURE)
    checklist = generate_checklist(analyses)

    # Filtra immagini utili tramite OCR
    filtered_images = filter_useful_images(combined_images)
    useful_classes = [2, 3, 4]  # Ipotesi: solo queste classi sono operative
    image_analysis = {}
    if filtered_images:
        predictions = analyze_images(filtered_images)
        for idx, (item, pred) in enumerate(zip(filtered_images, predictions)):
            if int(pred) in useful_classes:
                img_filename = f"{file_key}_img_{idx}.png"
                img_path = os.path.join(UPLOAD_FOLDER, img_filename)
                item["image"].save(img_path)
                item["image_filename"] = img_filename
                image_analysis[idx] = {"image_filename": img_filename, "prediction": int(pred),
                                       "ocr_text": item["ocr_text"]}

    # Associa le immagini alle voci della checklist
    checklist = associate_images_to_checklist(checklist, filtered_images)

    save_checklist_to_db(file_key, analysis_text, checklist, image_analysis)
    save_analysis_memory(file_key, analysis_text, checklist, image_analysis)

    return render_template_string(result_template, analysis_text=analysis_text, checklist=checklist,
                                  image_analysis=image_analysis, file_key=file_key)


# Rotta per visualizzare le checklist salvate (accordion)
@app.route('/checklists')
def view_checklists():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("SELECT id, file_name, generated_at, checklist_json FROM checklists ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    checklists = []
    for row in rows:
        checklists.append({
            'id': row[0],
            'file_name': row[1],
            'generated_at': row[2],
            'checklist': json.loads(row[3])
        })
    return render_template_string(checklists_template, checklists=checklists)


# Rotta per servire i file caricati (immagini)
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
