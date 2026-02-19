# CPR-Analyser

Erste lauffähige Python-Implementierung für die Analyse von CPR-Übungen aus:

- **Live-Kamera** (Webcam via OpenCV)
- **lokalem Video** (Dateiupload in der UI)

Der Fokus liegt aktuell auf:

1. **Kompressionsfrequenz (CPM)** auf Basis der Handgelenk-Bewegung (MediaPipe Hands)
2. **Regelmäßigkeit** der Kompressionen
3. **Anzahl der Kompressionen**
4. **Beatmungs-Detektion (heuristisch)** und **30:2-Zyklus-Nähe**
5. **Körperhaltung (heuristisch über Pose)**

---

## Für Einsteiger: Schritt-für-Schritt mit Anaconda + VS Code (Windows)

> Ziel: Du kannst das Projekt lokal starten und bekommst in VS Code die korrekte Python-Umgebung.

## 1) Voraussetzungen installieren

1. **Anaconda** installieren
2. **Visual Studio Code** installieren
3. In VS Code die Erweiterung **Python (Microsoft)** installieren

Optional (empfohlen):
- VS Code Erweiterung **Pylance**

---

## 2) Projektordner in VS Code öffnen

- In VS Code: `Datei -> Ordner öffnen...`
- Den Projektordner `CPR-Analyser` auswählen

---

## 3) Conda Environment erstellen

Öffne ein **Anaconda Prompt** oder VS-Code-Terminal und führe aus:

```bash
conda env create -f environment.yml
```

Danach aktivieren:

```bash
conda activate cpr_vision310
```

Prüfen, ob die richtige Python-Version aktiv ist:

```bash
python --version
```

Sollte `Python 3.10.x` anzeigen.

---

## 4) VS Code auf die richtige Environment stellen

1. `Strg+Shift+P`
2. `Python: Select Interpreter`
3. Interpreter mit Namen **cpr_vision310** auswählen

Falls der Interpreter nicht erscheint:

- VS Code neu starten
- erneut `Python: Select Interpreter`
- alternativ im Terminal prüfen:

```bash
where python
```

und den Pfad aus der Conda-Env manuell auswählen.

---

## 5) Installation testen

Im aktivierten Environment:

```bash
python -c "import cv2, mediapipe, scipy, streamlit; print('OK')"
```

Wenn `OK` kommt, passt die Basis.

---

## 6) App starten

Im aktivierten Environment:

```bash
streamlit run app.py
```

Danach im Browser (normalerweise automatisch):

- `http://localhost:8501`

In der Sidebar kannst du wählen:
- **Lokales Video**
- **Live Kamera**

---

## 7) Typischer Workflow im Projekt

1. Environment aktivieren:
   ```bash
   conda activate cpr_vision310
   ```
2. App starten:
   ```bash
   streamlit run app.py
   ```
3. Optional Tests ausführen:
   ```bash
   pytest -q
   ```

---

## 8) Häufige Probleme + Lösungen

### Problem A: `ModuleNotFoundError` (z. B. `cv2` oder `mediapipe`)
Ursache: Falscher Interpreter in VS Code.

Lösung:
- Interpreter erneut auf **cpr_vision310** stellen
- Terminal schließen/neu öffnen
- erneut testen mit:
  ```bash
  python -c "import cv2, mediapipe"
  ```

### Problem B: Kamera wird nicht erkannt
- Prüfen, ob Kamera von anderer App blockiert ist (Teams/Zoom etc.)
- In Windows Kamera-Berechtigung aktivieren
- In der App auf `Live Kamera` wechseln und neu starten

### Problem C: Streamlit startet, aber langsam
- In der Sidebar:
  - `Auswertungs-FPS` reduzieren (z. B. 10-15)
  - `Jede N-te Frame auswerten` auf 2-3 setzen

### Problem D: `conda env create` schlägt fehl
- Anaconda aktualisieren:
  ```bash
  conda update -n base -c defaults conda
  ```
- Dann erneut:
  ```bash
  conda env create -f environment.yml
  ```

---

## 9) Projektstruktur

- `app.py` – Streamlit UI (Live/Video-Auswahl, Anzeige, Metriken)
- `cpr_analyser/analyzer.py` – Kernanalyse (Handgelenk-Signal, CPM, Pose/Haltung)
- `cpr_analyser/metrics.py` – Hilfsfunktionen für Metrikbewertung
- `tests/` – erste Unit-Tests
- `environment.yml` – Conda-Environment (Python 3.10)

---

## 10) Nächste sinnvolle Schritte für dein Projekt

1. Eigene Testvideos sammeln (verschiedene Winkel, Lichtbedingungen)
2. Schwellenwerte kalibrieren (`prominence`, Peak-Abstände)
3. Beatmungs-Detektion robuster machen (State Machine statt einfacher Heuristik)
4. 30:2-Zyklus pro Abschnitt explizit auswerten
5. Optional externes CPR-Projekt integrieren (nur nach Kompatibilitätsprüfung)

