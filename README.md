# CPR-Analyser

Erste lauffähige Python-Implementierung für die Analyse von CPR-Übungen aus:

- **Live-Kamera** (Webcam via OpenCV)
- **lokalem Video** (Dateiupload in der UI)

Der Fokus liegt auf:

1. **Kompressionsfrequenz (CPM)** auf Basis der Handgelenk-Bewegung (MediaPipe Hands)
2. **Regelmäßigkeit** der Kompressionen
3. **Anzahl der Kompressionen**
4. **Beatmungs-Detektion (heuristisch)** und **30:2-Zyklus-Nähe**
5. **Körperhaltung (heuristisch über Pose)**

## Setup (Conda, Python 3.10)

```bash
conda env create -f environment.yml
conda activate cpr_vision310
streamlit run app.py
```

## Hinweise zur Echtzeitfähigkeit

- Verarbeitung kann über `target_fps` begrenzt werden (z. B. 15 FPS)
- `model_complexity` und Bildgröße sind reduziert
- Nur jede `N`-te Frame-Auswertung möglich (Parameter `frame_skip`)

## Start

```bash
streamlit run app.py
```

Dann in der Sidebar Quelle auswählen:
- `Lokales Video`
- `Live Kamera`
