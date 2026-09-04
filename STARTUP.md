# Code_Down_ML — Startup & Launch

*Intelligent Transaction Classification Automation*

This is the quickest way to run the app locally. It is a Streamlit application;
the entry point is **`main.py`**.

---

## 1. Prerequisites

- **Python 3.11+**
- A terminal (PowerShell on Windows, Terminal on macOS/Linux)

---

## 2. Set up a virtual environment & install

### Windows (PowerShell)

```powershell
cd code_down_ML
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
cd code_down_ML
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> On Debian/Ubuntu, if `python3 -m venv` fails, run `sudo apt install python3-venv`
> once, then retry.

---

## 3. Launch

```bash
streamlit run main.py
```

Then open the URL it prints (usually <http://localhost:8501>).

To stop the server, press `Ctrl+C` in the terminal.

---

## 4. Optional: higher-accuracy AI engines

The app runs fully on the lightweight, built-in engines (TF-IDF + Logistic
Regression). To enable optional semantic matching + SetFit:

```bash
pip install -r requirements-semantic.txt
```

These are heavier downloads and are not required for the demo.

---

## 5. Useful options

```bash
# Run on a different port
streamlit run main.py --server.port 8600

# Use a throwaway database (keeps the shipped data/ clean)
FILLDOWN_DB_PATH=/tmp/demo.db streamlit run main.py

# Run in public-demo mode (demo banner + fresh-state reset)
FILLDOWN_DEMO=1 streamlit run main.py
```

---

## 6. Run with Docker (same image Hugging Face uses)

```bash
docker build -t code-down-ml .
docker run -p 8501:8501 code-down-ml
```

See `README.md` for the full feature tour and configuration reference.
