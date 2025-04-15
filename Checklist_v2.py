import os
import re
import io
import json
import csv
import sqlite3
import logging
from datetime import datetime
from ftplib import FTP
import requests  # IMPORT AGGIUNTO PER LE RICHIESTE HTTP

from flask import Flask, redirect, flash, send_from_directory, render_template_string, request, url_for
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

# Configurazione logging
logging.basicConfig(level=logging.INFO)

# Variabili globali e configurazione
UPLOAD_FOLDER = 'uploads'
DATABASE = 'checklist.db'
DEFAULT_PDF = "Relazione_procedure.pdf"
COMPONENT_PDFS = [
    "D840MIT-Rev.02_20171030.pdf",
    "Martino-Atti-1Sessione-LAcqua-n.-1-2-2002.pdf",
    "08_manuale_di_manutenzione.pdf",
    "MANUALE_degli_impianti_di_acquedotto-1.pdf"
]
CHUNK_SIZE = 1500
OVERLAP_SIZE = 300
TEMPERATURE = 0.2

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'your_secret_key'  # Sostituire con una chiave sicura

# Cache in memoria per evitare rianalisi
in_memory_cache = {}

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

# ---------------------------
# INIZIALIZZAZIONE DATABASE E MIGRAZIONI
# ---------------------------
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
            image_analysis TEXT,
            status TEXT DEFAULT 'da_revisionare'
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT NOT NULL,
            ruolo TEXT,
            email TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS zones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS devices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT NOT NULL,
            zone_id INTEGER,
            FOREIGN KEY (zone_id) REFERENCES zones(id)
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS alarms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id INTEGER,
            checklist_id INTEGER,
            descrizione TEXT,
            stato TEXT DEFAULT 'attivo',
            FOREIGN KEY (device_id) REFERENCES devices(id),
            FOREIGN KEY (checklist_id) REFERENCES checklists(id)
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_alarms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            alarm_id INTEGER,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (alarm_id) REFERENCES alarms(id)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def migrate_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("PRAGMA table_info(checklists)")
    columns = [info[1] for info in c.fetchall()]
    if 'status' not in columns:
        logging.info("La colonna 'status' non esiste nella tabella checklists. Aggiungiamola ora.")
        c.execute("ALTER TABLE checklists ADD COLUMN status TEXT DEFAULT 'da_revisionare'")
        conn.commit()
    conn.close()

migrate_db()

# ---------------------------
# FUNZIONI DI CACHING
# ---------------------------
def load_cached_analysis_memory(file_key):
    return in_memory_cache.get(file_key)

def save_analysis_memory(file_key, analysis_text, checklist, image_analysis):
    in_memory_cache[file_key] = (analysis_text, checklist, image_analysis)

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

# ---------------------------
# TEMPLATE PER LA HOME
# ---------------------------
index_template = """
<!doctype html>
<html lang="it">
  <head>
    <meta charset="utf-8">
    <title>Dashboard Analisi Manuali</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
      body {
        background: linear-gradient(135deg, #74ABE2, #5563DE);
        color: #fff;
      }
      .card {
        background-color: rgba(255, 255, 255, 0.9);
        color: #333;
      }
      #hero {
        padding: 60px 0;
        text-align: center;
      }
      #hero h1 {
        font-size: 3rem;
        font-weight: bold;
      }
      #hero p {
        font-size: 1.25rem;
      }
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
    <div class="container">
      <section id="hero" class="my-5">
        <h1>Benvenuto nel Sistema di Analisi Manuali</h1>
        <p>Unisci e analizza i tuoi manuali in modo intelligente e rapido</p>
      </section>
      <div class="row">
        <div class="col-md-6 mb-4">
          <div class="card shadow-sm">
            <div class="card-body">
              <h5 class="card-title">Analisi Manuali</h5>
              <p class="card-text">Unisci il manuale generale <strong>{{ default_pdf }}</strong> con i manuali di componente:</p>
              <ul>
                {% for file in component_pdfs %}
                  <li>{{ file }}</li>
                {% endfor %}
              </ul>
              <a href="/process?file=all" class="btn btn-primary" id="analyzeBtn">Avvia Analisi Unificata</a>
              <a href="/process?file=all&force=true" class="btn btn-warning">Rianalizza Documenti</a>
            </div>
          </div>
        </div>
        <div class="col-md-6 mb-4">
          <div class="card shadow-sm">
            <div class="card-body">
              <h5 class="card-title">Gestione Checklist e Revisione</h5>
              <p class="card-text">Visualizza le checklist generate, revisiona e gestisci approvazioni o rifiuti.</p>
              <a href="/checklists" class="btn btn-secondary">Visualizza Checklist</a>
              <a href="/review_checklists" class="btn btn-danger">Revisiona Checklist (per modulo)</a>
            </div>
          </div>
        </div>
      </div>
      <div class="row">
        <div class="col-md-6 mb-4">
          <div class="card shadow-sm">
            <div class="card-body">
              <h5 class="card-title">Esporta e Carica Dati</h5>
              <p class="card-text">Esporta le checklist in JSON o carica i dati su server FTP.</p>
              <a href="/export_json_structured" class="btn btn-info">Esporta JSON</a>
              <a href="/upload_ftp" class="btn btn-dark">Carica via FTP</a>
            </div>
          </div>
        </div>
        <div class="col-md-6 mb-4">
          <div class="card shadow-sm">
            <div class="card-body">
              <h5 class="card-title">Admin Panel</h5>
              <p class="card-text">Gestisci utenti, dispositivi, zone, allarmi e assegnazioni.</p>
              <a href="/admin" class="btn btn-success">Accedi all'Admin Panel</a>
            </div>
          </div>
        </div>
      </div>
      <div class="row">
        <div class="col-md-12 mb-4">
          <div class="card shadow-sm">
            <div class="card-body">
              <h5 class="card-title">Area Utente</h5>
              <p class="card-text">Accedi per visualizzare i tuoi allarmi assegnati e le relative checklist.</p>
              <a href="/user" class="btn btn-primary">Accedi all'Area Utente</a>
            </div>
          </div>
        </div>
      </div>
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

# ---------------------------
# TEMPLATE PER LA PAGINA UTENTE
# ---------------------------
user_panel_template = """
<!doctype html>
<html lang="it">
  <head>
    <meta charset="utf-8">
    <title>Area Utente</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  </head>
  <body>
    <div class="container my-5">
      <h1>Area Utente</h1>
      <p>Benvenuto!</p>
      <h2>Allarmi Assegnati</h2>
      {% if alarms %}
        <ul>
        {% for alarm in alarms %}
          <li>
            <strong>ID:</strong> {{ alarm.id }} -
            <strong>Dispositivo:</strong> {{ alarm.device_nome }} -
            <strong>Descrizione:</strong> {{ alarm.descrizione }}
            {% if checklists.get(alarm.id) %}
              <br><strong>Checklist:</strong>
              <pre>{{ checklists.get(alarm.id)|tojson(indent=2) }}</pre>
            {% endif %}
          </li>
        {% endfor %}
        </ul>
      {% else %}
        <p>Nessun allarme assegnato.</p>
      {% endif %}
      <a href="/" class="btn btn-primary mt-3">Torna alla Home</a>
    </div>
  </body>
</html>
"""

# ---------------------------
# TEMPLATE PER IL RESULTATO DELLA CHECKLIST CON INTERFACCIA INTERATTIVA
# (rimane invariato)
# ---------------------------
result_template = """
<!doctype html>
<html lang="it">
  <head>
    <meta charset="utf-8">
    <title>Risultato Checklist</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
      pre { background-color: #f8f9fa; padding: 15px; border-radius: 5px; }
      .filter-container { margin-bottom: 20px; }
      .completed {
        text-decoration: line-through;
        color: gray;
      }
    </style>
  </head>
  <body>
    <div class="container my-5">
      <h1 class="mb-4">Checklist Generata: {{ file_key }}</h1>
      <div class="filter-container">
        <label for="cardFilter" class="form-label">Filtra per Argomento:</label>
        <select id="cardFilter" class="form-select" style="width: 300px;">
          <option value="all">Tutti</option>
        </select>
      </div>
      {% if checklist %}
        <div class="row" id="cardsContainer">
          {% for title, details in checklist.items() %}
            <div class="col-md-6 mb-3 checklist-item" data-topic="{{ details.tipo_intervento | lower }}">
              <div class="card">
                <div class="card-body">
                  <h5 class="card-title">{{ title }}</h5>
                  <h6 class="card-subtitle mb-2 text-muted">Tipo Intervento: {{ details.tipo_intervento }}</h6>
                  <p class="card-text">
                    {{ details.description[:100] }}{{ '...' if details.description|length > 100 else '' }}
                  </p>
                  {% if details.associated_images %}
                    <p><strong>Immagini:</strong></p>
                    <div>
                      {% for img in details.associated_images %}
                        <img src="/uploads/{{ img }}" style="max-height: 100px; margin-right: 5px;" alt="Immagine">
                      {% endfor %}
                    </div>
                  {% endif %}
                  <button type="button"
                          class="btn btn-primary"
                          data-bs-toggle="modal"
                          data-bs-target="#problemModal"
                          data-title="{{ title }}"
                          data-description="{{ details.description }}"
                          data-causes="{{ details.causes }}"
                          data-procedures="{{ details.procedures|replace('\n', '|||') }}">
                    Dettagli
                  </button>
                </div>
              </div>
            </div>
          {% endfor %}
        </div>
      {% else %}
        <p class="text-warning">Nessun problema identificato.</p>
      {% endif %}
      <a href="/" class="btn btn-secondary mt-3">Torna alla Home</a>
    </div>

    <!-- Modale per visualizzare i dettagli -->
    <div class="modal fade" id="problemModal" tabindex="-1" aria-labelledby="problemModalLabel" aria-hidden="true">
      <div class="modal-dialog modal-lg">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title" id="problemModalLabel">Dettagli del Problema</h5>
            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
          </div>
          <div class="modal-body">
            <h5 id="modalTitle"></h5>
            <p><strong>Descrizione:</strong> <span id="modalDescription"></span></p>
            <p><strong>Cause:</strong> <span id="modalCauses"></span></p>
            <p><strong>Procedure:</strong></p>
            <div id="modalProcedures" class="list-group"></div>
          </div>
          <div class="modal-footer">
            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Chiudi</button>
          </div>
        </div>
      </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
      const items = document.querySelectorAll(".checklist-item");
      const cardFilter = document.getElementById("cardFilter");
      const topics = new Set();
      items.forEach(item => {
          topics.add(item.getAttribute("data-topic"));
      });
      topics.forEach(topic => {
          const option = document.createElement("option");
          option.value = topic;
          option.textContent = topic.charAt(0).toUpperCase() + topic.slice(1);
          cardFilter.appendChild(option);
      });
      cardFilter.addEventListener("change", function(){
          const selected = this.value;
          items.forEach(item => {
              if(selected === "all"){
                  item.style.display = "";
              } else {
                  item.style.display = item.getAttribute("data-topic") === selected ? "" : "none";
              }
          });
      });

      var problemModal = document.getElementById('problemModal')
      problemModal.addEventListener('show.bs.modal', function (event) {
        var button = event.relatedTarget;
        var title = button.getAttribute('data-title');
        var description = button.getAttribute('data-description');
        var causes = button.getAttribute('data-causes');
        var procedures_raw = button.getAttribute('data-procedures');
        var procedures = procedures_raw.split('|||').filter(function(p){ return p.trim() !== ''; });

        document.getElementById('modalTitle').innerText = title;
        document.getElementById('modalDescription').innerText = description;
        document.getElementById('modalCauses').innerText = causes;

        var modalProceduresDiv = document.getElementById('modalProcedures');
        modalProceduresDiv.innerHTML = '';

        procedures.forEach(function(proc, index) {
          var containerDiv = document.createElement('div');
          containerDiv.className = 'list-group-item form-check';

          var checkbox = document.createElement('input');
          checkbox.type = 'checkbox';
          checkbox.className = 'form-check-input me-2';
          checkbox.id = 'proc-' + index;

          var label = document.createElement('label');
          label.className = 'form-check-label';
          label.setAttribute('for', 'proc-' + index);
          label.innerText = proc;

          checkbox.addEventListener('change', function() {
            if (this.checked) {
              label.classList.add('completed');
            } else {
              label.classList.remove('completed');
            }
          });

          containerDiv.appendChild(checkbox);
          containerDiv.appendChild(label);
          modalProceduresDiv.appendChild(containerDiv);
        });
      });
    </script>
  </body>
</html>
"""

# ---------------------------
# TEMPLATE PER LA VISUALIZZAZIONE DELLE CHECKLIST (AREA ADMIN/UTENTE) – con checkbox interattive
# ---------------------------
checklists_template = """
<!doctype html>
<html lang="it">
  <head>
    <meta charset="utf-8">
    <title>Checklist Generate</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
      .filter-container { margin-bottom: 20px; }
      .completed {
        text-decoration: line-through;
        color: gray;
      }
    </style>
  </head>
  <body>
    <div class="container my-5">
      <h1>Elenco Checklist Generate</h1>
      <div class="filter-container">
        <label for="cardTopicFilter" class="form-label">Filtra per Argomento:</label>
        <select id="cardTopicFilter" class="form-select" style="width: 300px;">
          <option value="all">Tutti</option>
        </select>
      </div>
      {% if checklists %}
      <div class="row" id="checklistCards">
        {% for item in checklists %}
          {% for title, details in item.checklist.items() %}
            <div class="col-md-6 mb-3 checklist-card" data-topic="{{ details.tipo_intervento | lower }}">
              <div class="card">
                <div class="card-body">
                  <h5 class="card-title">{{ title }}</h5>
                  <h6 class="card-subtitle mb-2 text-muted">Tipo Intervento: {{ details.tipo_intervento }}</h6>
                  <p class="card-text">
                    {{ details.description[:100] }}{{ '...' if details.description|length > 100 else '' }}
                  </p>
                  {% if details.associated_images %}
                    <p><strong>Immagini:</strong></p>
                    <div>
                      {% for img in details.associated_images %}
                        <img src="/uploads/{{ img }}" style="max-height: 100px; margin-right: 5px;" alt="Immagine">
                      {% endfor %}
                    </div>
                  {% endif %}
                  <button type="button"
                          class="btn btn-primary"
                          data-bs-toggle="modal"
                          data-bs-target="#problemModal"
                          data-title="{{ title }}"
                          data-description="{{ details.description }}"
                          data-causes="{{ details.causes }}"
                          data-procedures="{{ details.procedures|replace('\n', '|||') }}">
                    Dettagli
                  </button>
                </div>
              </div>
            </div>
          {% endfor %}
        {% endfor %}
      </div>
      {% else %}
      <p>Nessuna checklist generata.</p>
      {% endif %}
      <a href="{{ url_for('admin_index') }}" class="btn btn-secondary">Torna al Admin Panel</a>
    </div>

    <!-- Modale per i dettagli -->
    <div class="modal fade" id="problemModal" tabindex="-1" aria-labelledby="problemModalLabel" aria-hidden="true">
      <div class="modal-dialog modal-lg">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title" id="problemModalLabel">Dettagli del Problema</h5>
            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
          </div>
          <div class="modal-body">
            <h5 id="modalTitle"></h5>
            <p><strong>Descrizione:</strong> <span id="modalDescription"></span></p>
            <p><strong>Cause:</strong> <span id="modalCauses"></span></p>
            <p><strong>Procedure:</strong></p>
            <div id="modalProcedures" class="list-group"></div>
          </div>
          <div class="modal-footer">
            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Chiudi</button>
          </div>
        </div>
      </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
      const rows = document.querySelectorAll("#checklistCards .checklist-card");
      const cardTopicFilter = document.getElementById("cardTopicFilter");
      const topics = new Set();
      rows.forEach(row => {
          topics.add(row.getAttribute("data-topic"));
      });
      topics.forEach(topic => {
          const option = document.createElement("option");
          option.value = topic;
          option.textContent = topic.charAt(0).toUpperCase() + topic.slice(1);
          cardTopicFilter.appendChild(option);
      });
      cardTopicFilter.addEventListener("change", function(){
          const selected = this.value;
          rows.forEach(row => {
              if(selected === "all"){
                  row.style.display = "";
              } else {
                  row.style.display = row.getAttribute("data-topic") === selected ? "" : "none";
              }
          });
      });

      var problemModal = document.getElementById('problemModal')
      problemModal.addEventListener('show.bs.modal', function (event) {
        var button = event.relatedTarget;
        var title = button.getAttribute('data-title');
        var description = button.getAttribute('data-description');
        var causes = button.getAttribute('data-causes');
        var procedures_raw = button.getAttribute('data-procedures');
        var procedures = procedures_raw.split('|||').filter(function(p){ return p.trim() !== ''; });

        document.getElementById('modalTitle').innerText = title;
        document.getElementById('modalDescription').innerText = description;
        document.getElementById('modalCauses').innerText = causes;

        var modalProceduresDiv = document.getElementById('modalProcedures');
        modalProceduresDiv.innerHTML = '';

        procedures.forEach(function(proc, index) {
          var containerDiv = document.createElement('div');
          containerDiv.className = 'list-group-item form-check';

          var checkbox = document.createElement('input');
          checkbox.type = 'checkbox';
          checkbox.className = 'form-check-input me-2';
          checkbox.id = 'proc-' + index;

          var label = document.createElement('label');
          label.className = 'form-check-label';
          label.setAttribute('for', 'proc-' + index);
          label.innerText = proc;

          checkbox.addEventListener('change', function() {
            if (this.checked) {
              label.classList.add('completed');
            } else {
              label.classList.remove('completed');
            }
          });

          containerDiv.appendChild(checkbox);
          containerDiv.appendChild(label);
          modalProceduresDiv.appendChild(containerDiv);
        });
      });
    </script>
  </body>
</html>
"""

# ---------------------------
# TEMPLATE PER LA REVISIONE DELLE CHECKLIST (AREA ADMIN)
# ---------------------------
review_template = """
<!doctype html>
<html lang="it">
  <head>
    <meta charset="utf-8">
    <title>Revisione Checklist</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
      .filter-container { margin-bottom: 20px; }
    </style>
  </head>
  <body>
    <div class="container my-5">
      <h1>Checklist da revisionare</h1>
      <div class="filter-container">
        <label for="topicFilter" class="form-label">Filtra per Argomento:</label>
        <select id="topicFilter" class="form-select" style="width: 300px;">
          <option value="all">Tutti</option>
        </select>
      </div>
      {% if checklists %}
      <table class="table table-bordered" id="checklistTable">
        <thead>
          <tr>
            <th>ID</th>
            <th>File</th>
            <th>Generata il</th>
            <th>Argomenti</th>
            <th>Status</th>
            <th>Azioni</th>
          </tr>
        </thead>
        <tbody>
          {% for checklist in checklists %}
          <tr data-topics="{{ checklist.topics | lower }}">
            <td>{{ checklist.id }}</td>
            <td>{{ checklist.file_name }}</td>
            <td>{{ checklist.generated_at }}</td>
            <td>{{ checklist.topics }}</td>
            <td>{{ checklist.status }}</td>
            <td>
              <form action="{{ url_for('approve_checklist', checklist_id=checklist.id) }}" method="post" style="display:inline;">
                <button class="btn btn-success btn-sm" type="submit">Approva</button>
              </form>
              <form action="{{ url_for('reject_checklist', checklist_id=checklist.id) }}" method="post" style="display:inline;">
                <button class="btn btn-danger btn-sm" type="submit">Rifiuta</button>
              </form>
            </td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
      {% else %}
      <p>Nessuna checklist da revisionare.</p>
      {% endif %}
      <a href="/" class="btn btn-secondary">Torna alla Home</a>
    </div>
    <script>
      const rows = document.querySelectorAll("#checklistTable tbody tr");
      const topicSet = new Set();
      rows.forEach(row => {
          const topics = row.getAttribute("data-topics").split(",").map(t => t.trim());
          topics.forEach(t => { if(t) topicSet.add(t); });
      });
      const topicFilter = document.getElementById("topicFilter");
      topicSet.forEach(topic => {
          const option = document.createElement("option");
          option.value = topic.toLowerCase();
          option.textContent = topic;
          topicFilter.appendChild(option);
      });

      topicFilter.addEventListener("change", function(){
          const selected = this.value;
          rows.forEach(row => {
              if(selected === "all"){
                  row.style.display = "";
              } else {
                  const topics = row.getAttribute("data-topics").toLowerCase();
                  row.style.display = topics.includes(selected) ? "" : "none";
              }
          });
      });
    </script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
  </body>
</html>
"""

# ---------------------------
# PAGINA ADMIN: Aggiunta del pulsante "Torna alla Home" nel template admin_index
# ---------------------------
admin_nav = """
<nav class="navbar navbar-expand-lg navbar-dark bg-dark mb-4">
  <div class="container-fluid">
    <a class="navbar-brand" href="{{ url_for('admin_index') }}">Admin Panel</a>
    <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
      <span class="navbar-toggler-icon"></span>
    </button>
    <div class="collapse navbar-collapse" id="navbarNav">
      <ul class="navbar-nav">
        <li class="nav-item"><a class="nav-link" href="{{ url_for('manage_users') }}">Utenti</a></li>
        <li class="nav-item"><a class="nav-link" href="{{ url_for('manage_zones') }}">Zone</a></li>
        <li class="nav-item"><a class="nav-link" href="{{ url_for('manage_devices') }}">Dispositivi</a></li>
        <li class="nav-item"><a class="nav-link" href="{{ url_for('manage_alarms') }}">Allarmi</a></li>
        <li class="nav-item"><a class="nav-link" href="{{ url_for('assignments') }}">Assegnazioni</a></li>
      </ul>
    </div>
  </div>
</nav>
"""

admin_index_template = """
<!doctype html>
<html lang="it">
  <head>
    <meta charset="utf-8">
    <title>Admin Panel</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  </head>
  <body>
    """ + admin_nav + """
    <div class="container">
      <h1>Benvenuto nell'Admin Panel</h1>
      <p>Qui puoi controllare il sistema idraulico e gestire la piattaforma.</p>
      <!-- Placeholder per il controllo del sistema idraulico -->
      <div>
        <h3>Sistema Idraulico</h3>
        <p>[Qui importa e integra il controllo del sistema idraulico da un altro file]</p>
      </div>
      <a href="/" class="btn btn-primary mt-3">Torna alla Home</a>
      <!-- <a href="/logout" class="btn btn-secondary">Logout</a> -->
    </div>
  </body>
</html>
"""

# (I template per gestione utenti, zone, dispositivi, allarmi, assignments sono mantenuti invariati)
# ---------------------------
# ROTTE PER ESTRARRE, ANALIZZARE E GENERARE CHECKLIST
# ---------------------------
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
                    img_obj.verify()  # Verifica integrità
                    img_obj = Image.open(io.BytesIO(image_bytes))
                    images.append(img_obj)
                except Exception as e:
                    logging.error(f"Errore elaborando immagine {file_path} pagina {page_num}, xref {xref}: {e}")
        logging.info(f"Immagini estratte da {file_path}")
        return images
    except Exception as e:
        logging.error(f"Errore nell'estrazione immagini da {file_path}: {e}")
        return []


def filter_useful_images(images):
    useful_images = []
    keywords = ["disegno", "planta", "schema", "componente", "elenco"]
    for img in images:
        try:
            ocr_text = pytesseract.image_to_string(img, lang="ita")
            if any(keyword in ocr_text.lower() for keyword in keywords):
                useful_images.append({"image": img, "ocr_text": ocr_text})
        except Exception as e:
            logging.error(f"Errore OCR: {e}")
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
        logging.info(f"Documento {file_path} caricato")
        return text
    except Exception as e:
        logging.error(f"Errore caricamento documento {file_path}: {e}")
        return None


def analyze_text_combined(text, chunk_size, overlap_size, temperature):
    llm.temperature = temperature
    prompt_template = (
        "Sei un esperto di manutenzione industriale e sicurezza operativa. "
        "Analizza il seguente testo e restituisci un JSON conforme allo schema:\n\n"
        "{{\n"
        "  \"problemi\": [\n"
        "    {{\n"
        "      \"titolo\": \"Titolo specifico del problema\",\n"
        "      \"descrizione\": \"Descrizione dettagliata del problema\",\n"
        "      \"cause\": [\"causa1\", \"causa2\"],\n"
        "      \"procedure\": [\"procedura1\", \"procedura2\"]\n"
        "    }}\n"
        "  ]\n"
        "}}\n\n"
        "Testo:\n{0}"
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap_size)
    split_docs = text_splitter.split_text(text)
    analyses = []
    for idx, doc in enumerate(split_docs):
        try:
            prompt = prompt_template.format(doc)
            logging.info(f"Invio chunk {idx + 1} al LLM...")
            analysis = llm.invoke(prompt)
            analyses.append(analysis)
        except Exception as e:
            logging.error(f"Errore LLM per chunk {idx + 1}: {e}")
    full_analysis = "\n".join(analyses)
    return full_analysis, analyses


def generate_checklist(analyses):
    problems = {}
    # Parsing dei risultati JSON
    for analysis in analyses:
        try:
            data = json.loads(analysis)
            for problem in data.get("problemi", []):
                title = problem.get("titolo", "").strip()
                if title:
                    # Qui viene creata la checklist; si usano join per avere una stringa
                    problems[title] = {
                        'description': problem.get("descrizione", "").strip(),
                        'causes': "\n".join(problem.get("cause", [])),
                        'procedures': "\n".join(problem.get("procedure", [])),
                        'tipo_intervento': "Intervento Generico",
                        'associated_images': []
                    }
        except Exception as e:
            logging.error(f"Errore nel parsing JSON: {e}")
    problems = apply_rules(problems)
    return problems


def apply_rules(checklist):
    # Definizione delle regole per dispositivo
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
        # Dividi le procedure originali in righe
        original_procedures = [line.strip() for line in value['procedures'].split("\n") if line.strip()]
        # Inizializza la suddivisione per dispositivo: il gruppo "Generale" con le procedure originali
        proc_by_device = {}
        proc_by_device["Generale"] = original_procedures
        # Applica le regole basate sulle keyword
        for rule_keyword, (device, rule_text) in rules.items():
            if rule_keyword in lower_title or rule_keyword in lower_description:
                tipo_intervento = device
                if device not in proc_by_device:
                    proc_by_device[device] = []
                if rule_text not in proc_by_device[device]:
                    proc_by_device[device].append(rule_text)
        value['tipo_intervento'] = tipo_intervento
        # Crea una stringa formattata che raggruppa le procedure per dispositivo
        formatted_procs = ""
        for device, procs in proc_by_device.items():
            if procs:
                formatted_procs += f"{device}:\n"
                for proc in procs:
                    formatted_procs += f"- {proc}\n"
                formatted_procs += "\n"
        value['procedures'] = formatted_procs.strip()
    return checklist


def associate_images_to_checklist(checklist, filtered_images):
    for item in filtered_images:
        ocr_text = item["ocr_text"].lower()
        keywords = ["disegno", "planta", "schema", "componente", "elenco"]
        for title, details in checklist.items():
            text_to_check = (title + " " + details["description"]).lower()
            if any(kw in ocr_text for kw in keywords) and any(kw in text_to_check for kw in keywords):
                details["associated_images"].append(item.get("image_filename", ""))
    return checklist


def save_checklist_to_db(file_key, analysis_text, checklist, image_analysis):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    generated_at = datetime.now().isoformat()
    checklist_json = json.dumps(checklist, ensure_ascii=False)
    image_analysis_json = json.dumps(image_analysis, ensure_ascii=False)
    c.execute(
        "INSERT INTO checklists (file_name, generated_at, analysis_text, checklist_json, image_analysis, status) VALUES (?, ?, ?, ?, ?, ?)",
        (file_key, generated_at, analysis_text, checklist_json, image_analysis_json, 'da_revisionare'))
    conn.commit()
    conn.close()


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
            causes = details.get("causes", "").split("\n") if isinstance(details.get("causes", ""), str) else details.get("causes", [])
            procedures = details.get("procedures", "").split("\n") if isinstance(details.get("procedures", ""), str) else details.get("procedures", [])
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
    logging.info(f"Esportazione JSON completata: {json_filename}")
    return send_from_directory(os.getcwd(), json_filename, as_attachment=True)

# ---------------------------
# ROTTE PRINCIPALI
# ---------------------------
@app.route('/')
def index():
    return render_template_string(index_template, default_pdf=DEFAULT_PDF, component_pdfs=COMPONENT_PDFS)


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

    filtered_images = filter_useful_images(combined_images)
    useful_classes = [2, 3, 4]
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
    checklist = associate_images_to_checklist(checklist, filtered_images)
    save_checklist_to_db(file_key, analysis_text, checklist, image_analysis)
    save_analysis_memory(file_key, analysis_text, checklist, image_analysis)

    return render_template_string(result_template, analysis_text=analysis_text, checklist=checklist,
                                  image_analysis=image_analysis, file_key=file_key)


@app.route('/checklists')
def view_checklists():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("SELECT id, file_name, generated_at, checklist_json, status FROM checklists ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    checklists = []
    for row in rows:
        checklists.append({
            'id': row[0],
            'file_name': row[1],
            'generated_at': row[2],
            'checklist': json.loads(row[3]),
            'status': row[4]
        })
    return render_template_string(checklists_template, checklists=checklists)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/export_json_structured')
def export_json_structured():
    return export_checklists_to_structured_json()


@app.route('/upload_ftp')
def upload_ftp():
    json_filename = "checklists_structured.json"
    ftp_host = "ftp.water4.altervista.org"
    ftp_port = 21
    ftp_username = "hespanel@water4"
    ftp_password = "saoihj$q1"
    remote_directory = "procedures"
    remote_filename = "checklists_structured.json"
    export_checklists_to_structured_json(json_filename)
    try:
        ftp = FTP(timeout=30)
        ftp.encoding = "latin-1"
        ftp.connect(ftp_host, ftp_port)
        ftp.login(ftp_username, ftp_password)
        try:
            ftp.cwd(remote_directory)
        except Exception as e:
            logging.info(f"Creazione directory {remote_directory}")
            ftp.mkd(remote_directory)
            ftp.cwd(remote_directory)
        with open(json_filename, "rb") as f:
            ftp.storbinary(f"STOR {remote_filename}", f)
        ftp.quit()
        return f"File {json_filename} caricato via FTP."
    except Exception as e:
        return f"Errore FTP: {e}"


@app.route('/review_checklists')
def review_checklists():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute(
        "SELECT id, file_name, generated_at, checklist_json, status FROM checklists WHERE status = 'da_revisionare'")
    rows = c.fetchall()
    conn.close()
    checklists = []
    for row in rows:
        checklist_data = json.loads(row[3])
        topics_set = set()
        for title, details in checklist_data.items():
            topics_set.add(details.get("tipo_intervento", ""))
        topics_str = ", ".join(topics_set)
        checklists.append({
            'id': row[0],
            'file_name': row[1],
            'generated_at': row[2],
            'checklist': checklist_data,
            'status': row[4],
            'topics': topics_str
        })
    return render_template_string(review_template, checklists=checklists)


@app.route('/approve_checklist/<int:checklist_id>', methods=['POST'])
def approve_checklist(checklist_id):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("UPDATE checklists SET status = 'approvata' WHERE id = ?", (checklist_id,))
    conn.commit()
    conn.close()
    flash("Checklist approvata!")
    return redirect(url_for('review_checklists'))


@app.route('/reject_checklist/<int:checklist_id>', methods=['POST'])
def reject_checklist(checklist_id):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("UPDATE checklists SET status = 'rifiutata' WHERE id = ?", (checklist_id,))
    conn.commit()
    conn.close()
    flash("Checklist rifiutata!")
    return redirect(url_for('review_checklists'))

# ---------------------------
# ROTTE PER LA GESTIONE AMMINISTRATIVA
# ---------------------------
@app.route('/admin')
def admin_index():
    return render_template_string(admin_index_template)

manage_users_template = """
<!doctype html>
<html lang="it">
  <head>
    <meta charset="utf-8">
    <title>Gestione Utenti</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  </head>
  <body>
    """ + admin_nav + """
    <div class="container">
      <h1>Gestione Utenti</h1>
      <form method="post" action="{{ url_for('manage_users') }}">
        <div class="mb-3">
          <label class="form-label">Nome</label>
          <input type="text" class="form-control" name="nome" required>
        </div>
        <div class="mb-3">
          <label class="form-label">Ruolo</label>
          <input type="text" class="form-control" name="ruolo">
        </div>
        <div class="mb-3">
          <label class="form-label">Email</label>
          <input type="email" class="form-control" name="email">
        </div>
        <button type="submit" class="btn btn-primary">Aggiungi Utente</button>
      </form>
      <hr>
      <h2>Lista Utenti</h2>
      <table class="table table-bordered">
        <thead>
          <tr><th>ID</th><th>Nome</th><th>Ruolo</th><th>Email</th></tr>
        </thead>
        <tbody>
          {% for user in users %}
          <tr>
            <td>{{ user.id }}</td>
            <td>{{ user.nome }}</td>
            <td>{{ user.ruolo }}</td>
            <td>{{ user.email }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
      <a href="{{ url_for('admin_index') }}" class="btn btn-secondary">Torna al Admin Panel</a>
    </div>
  </body>
</html>
"""

@app.route('/admin/users', methods=['GET', 'POST'])
def manage_users():
    if request.method == 'POST':
        nome = request.form.get('nome')
        ruolo = request.form.get('ruolo')
        email = request.form.get('email')
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute("INSERT INTO users (nome, ruolo, email) VALUES (?, ?, ?)", (nome, ruolo, email))
        conn.commit()
        conn.close()
        flash("Utente aggiunto!")
        return redirect(url_for('manage_users'))
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("SELECT id, nome, ruolo, email FROM users")
    rows = c.fetchall()
    conn.close()
    users = [{"id": r[0], "nome": r[1], "ruolo": r[2], "email": r[3]} for r in rows]
    return render_template_string(manage_users_template, users=users)

manage_zones_template = """
<!doctype html>
<html lang="it">
  <head>
    <meta charset="utf-8">
    <title>Gestione Zone</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  </head>
  <body>
    """ + admin_nav + """
    <div class="container">
      <h1>Gestione Zone</h1>
      <form method="post" action="{{ url_for('manage_zones') }}">
        <div class="mb-3">
          <label class="form-label">Nome Zona</label>
          <input type="text" class="form-control" name="nome" required>
        </div>
        <button type="submit" class="btn btn-primary">Aggiungi Zona</button>
      </form>
      <hr>
      <h2>Lista Zone</h2>
      <table class="table table-bordered">
        <thead>
          <tr><th>ID</th><th>Nome</th></tr>
        </thead>
        <tbody>
          {% for zone in zones %}
          <tr>
            <td>{{ zone.id }}</td>
            <td>{{ zone.nome }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
      <a href="{{ url_for('admin_index') }}" class="btn btn-secondary">Torna al Admin Panel</a>
    </div>
  </body>
</html>
"""

@app.route('/admin/zones', methods=['GET', 'POST'])
def manage_zones():
    if request.method == 'POST':
        nome = request.form.get('nome')
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute("INSERT INTO zones (nome) VALUES (?)", (nome,))
        conn.commit()
        conn.close()
        flash("Zona aggiunta!")
        return redirect(url_for('manage_zones'))
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("SELECT id, nome FROM zones")
    rows = c.fetchall()
    conn.close()
    zones = [{"id": r[0], "nome": r[1]} for r in rows]
    return render_template_string(manage_zones_template, zones=zones)

manage_devices_template = """
<!doctype html>
<html lang="it">
  <head>
    <meta charset="utf-8">
    <title>Gestione Dispositivi</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  </head>
  <body>
    """ + admin_nav + """
    <div class="container">
      <h1>Gestione Dispositivi</h1>
      <form method="post" action="{{ url_for('manage_devices') }}">
        <div class="mb-3">
          <label class="form-label">Nome Dispositivo</label>
          <input type="text" class="form-control" name="nome" required>
        </div>
        <div class="mb-3">
          <label class="form-label">Zona</label>
          <select name="zone_id" class="form-select" required>
            {% for zone in zones %}
            <option value="{{ zone.id }}">{{ zone.nome }}</option>
            {% endfor %}
          </select>
        </div>
        <button type="submit" class="btn btn-primary">Aggiungi Dispositivo</button>
      </form>
      <hr>
      <h2>Lista Dispositivi</h2>
      <table class="table table-bordered">
        <thead>
          <tr><th>ID</th><th>Nome</th><th>Zona</th></tr>
        </thead>
        <tbody>
          {% for device in devices %}
          <tr>
            <td>{{ device.id }}</td>
            <td>{{ device.nome }}</td>
            <td>{{ device.zone_nome }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
      <a href="{{ url_for('admin_index') }}" class="btn btn-secondary">Torna al Admin Panel</a>
    </div>
  </body>
</html>
"""

@app.route('/admin/devices', methods=['GET', 'POST'])
def manage_devices():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    if request.method == 'POST':
        nome = request.form.get('nome')
        zone_id = request.form.get('zone_id')
        c.execute("INSERT INTO devices (nome, zone_id) VALUES (?, ?)", (nome, zone_id))
        conn.commit()
        flash("Dispositivo aggiunto!")
        return redirect(url_for('manage_devices'))
    c.execute("SELECT id, nome FROM zones")
    zones_rows = c.fetchall()
    zones = [{"id": r[0], "nome": r[1]} for r in zones_rows]
    c.execute("SELECT devices.id, devices.nome, zones.nome FROM devices LEFT JOIN zones ON devices.zone_id = zones.id")
    rows = c.fetchall()
    conn.close()
    devices = [{"id": r[0], "nome": r[1], "zone_nome": r[2] if r[2] else "N/A"} for r in rows]
    return render_template_string(manage_devices_template, devices=devices, zones=zones)

manage_alarms_template = """
<!doctype html>
<html lang="it">
  <head>
    <meta charset="utf-8">
    <title>Gestione Allarmi</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  </head>
  <body>
    """ + admin_nav + """
    <div class="container">
      <h1>Gestione Allarmi</h1>
      <form method="post" action="{{ url_for('manage_alarms') }}">
        <div class="mb-3">
          <label class="form-label">Dispositivo</label>
          <select name="device_id" class="form-select" required>
            {% for device in devices %}
            <option value="{{ device.id }}">{{ device.nome }} (Zona: {{ device.zone_nome }})</option>
            {% endfor %}
          </select>
        </div>
        <div class="mb-3">
          <label class="form-label">Checklist ID (opzionale)</label>
          <input type="text" class="form-control" name="checklist_id">
        </div>
        <div class="mb-3">
          <label class="form-label">Descrizione Allarme</label>
          <textarea name="descrizione" class="form-control" required></textarea>
        </div>
        <button type="submit" class="btn btn-primary">Aggiungi Allarme</button>
      </form>
      <hr>
      <h2>Lista Allarmi</h2>
      <table class="table table-bordered">
        <thead>
          <tr><th>ID</th><th>Dispositivo</th><th>Checklist ID</th><th>Descrizione</th><th>Stato</th></tr>
        </thead>
        <tbody>
          {% for alarm in alarms %}
          <tr>
            <td>{{ alarm.id }}</td>
            <td>{{ alarm.device_nome }}</td>
            <td>{{ alarm.checklist_id if alarm.checklist_id else 'N/A' }}</td>
            <td>{{ alarm.descrizione }}</td>
            <td>{{ alarm.stato }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
      <a href="{{ url_for('admin_index') }}" class="btn btn-secondary">Torna al Admin Panel</a>
    </div>
  </body>
</html>
"""

@app.route('/admin/alarms', methods=['GET', 'POST'])
def manage_alarms():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    if request.method == 'POST':
        device_id = request.form.get('device_id')
        checklist_id = request.form.get('checklist_id')
        descrizione = request.form.get('descrizione')
        c.execute("INSERT INTO alarms (device_id, checklist_id, descrizione) VALUES (?, ?, ?)",
                  (device_id, checklist_id, descrizione))
        conn.commit()
        flash("Allarme aggiunto!")
        return redirect(url_for('manage_alarms'))
    c.execute("SELECT devices.id, devices.nome, zones.nome FROM devices LEFT JOIN zones ON devices.zone_id = zones.id")
    devices_rows = c.fetchall()
    devices = [{"id": r[0], "nome": r[1], "zone_nome": r[2] if r[2] else "N/A"} for r in devices_rows]
    c.execute(
        "SELECT alarms.id, devices.nome, alarms.checklist_id, alarms.descrizione, alarms.stato FROM alarms LEFT JOIN devices ON alarms.device_id = devices.id")
    rows = c.fetchall()
    conn.close()
    alarms = [{"id": r[0], "device_nome": r[1], "checklist_id": r[2], "descrizione": r[3], "stato": r[4]} for r in rows]
    return render_template_string(manage_alarms_template, alarms=alarms, devices=devices)

assignments_template = """
<!doctype html>
<html lang="it">
  <head>
    <meta charset="utf-8">
    <title>Assegnazione Utenti ad Allarmi</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  </head>
  <body>
    """ + admin_nav + """
    <div class="container">
      <h1>Assegnazione Utenti ad Allarmi</h1>
      <form method="post" action="{{ url_for('assign_user_to_alarm') }}">
        <div class="mb-3">
          <label class="form-label">Utente</label>
          <select name="user_id" class="form-select" required>
            {% for user in users %}
            <option value="{{ user.id }}">{{ user.nome }}</option>
            {% endfor %}
          </select>
        </div>
        <div class="mb-3">
          <label class="form-label">Allarme</label>
          <select name="alarm_id" class="form-select" required>
            {% for alarm in alarms %}
            <option value="{{ alarm.id }}">ID: {{ alarm.id }} - {{ alarm.device_nome }}</option>
            {% endfor %}
          </select>
        </div>
        <button type="submit" class="btn btn-primary">Assegna Utente</button>
      </form>
      <hr>
      <h2>Lista Assegnazioni</h2>
      <table class="table table-bordered">
        <thead>
          <tr><th>ID</th><th>Utente</th><th>Allarme</th></tr>
        </thead>
        <tbody>
          {% for assignment in assignments %}
          <tr>
            <td>{{ assignment.id }}</td>
            <td>{{ assignment.user_nome }}</td>
            <td>{{ assignment.alarm_id }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
      <a href="{{ url_for('admin_index') }}" class="btn btn-secondary">Torna al Admin Panel</a>
    </div>
  </body>
</html>
"""

@app.route('/admin/assignments', methods=['GET', 'POST'])
def assignments():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        alarm_id = request.form.get('alarm_id')
        c.execute("INSERT INTO user_alarms (user_id, alarm_id) VALUES (?, ?)", (user_id, alarm_id))
        conn.commit()
        flash("Assegnazione completata!")
        return redirect(url_for('assignments'))
    c.execute("SELECT id, nome FROM users")
    users_rows = c.fetchall()
    users = [{"id": r[0], "nome": r[1]} for r in users_rows]
    c.execute("SELECT ua.id, u.nome, ua.alarm_id FROM user_alarms ua JOIN users u ON ua.user_id = u.id")
    assignments_display = [{"id": r[0], "user_nome": r[1], "alarm_id": r[2]} for r in c.fetchall()]
    c.execute("SELECT alarms.id, devices.nome FROM alarms JOIN devices ON alarms.device_id = devices.id")
    alarms_rows = c.fetchall()
    alarms = [{"id": r[0], "device_nome": r[1]} for r in alarms_rows]
    conn.close()
    return render_template_string(assignments_template, users=users, alarms=alarms, assignments=assignments_display)

@app.route('/admin/assign_user', methods=['POST'])
def assign_user_to_alarm():
    return assignments()

# ---------------------------
# AREA UTENTE: Visualizzazione allarmi e checklist assegnate all'utente
# ---------------------------
@app.route('/user')
def user_panel():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("SELECT alarm_id FROM user_alarms")
    assignment_rows = c.fetchall()
    alarm_ids = [row[0] for row in assignment_rows]
    alarms = []
    for alarm_id in alarm_ids:
        c.execute(
            "SELECT alarms.id, devices.nome, alarms.descrizione, alarms.checklist_id FROM alarms LEFT JOIN devices ON alarms.device_id = devices.id WHERE alarms.id = ?",
            (alarm_id,))
        alarm_row = c.fetchone()
        if alarm_row:
            alarm = {
                'id': alarm_row[0],
                'device_nome': alarm_row[1],
                'descrizione': alarm_row[2],
                'checklist_id': alarm_row[3]
            }
            alarms.append(alarm)
    checklists = {}
    for alarm in alarms:
        if alarm['checklist_id']:
            c.execute("SELECT checklist_json FROM checklists WHERE id = ?", (alarm['checklist_id'],))
            cl_row = c.fetchone()
            if cl_row:
                checklists[alarm['id']] = json.loads(cl_row[0])
    conn.close()
    return render_template_string(user_panel_template, alarms=alarms, checklists=checklists)

# ---------------------------
# AVVIO DELL'APPLICAZIONE
# ---------------------------
if __name__ == '__main__':
    app.run(debug=True)
