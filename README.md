# Trading Bot - Secure Package

This package contains a secure starter version of the **Trading Bot** (Streamlit UI + analysis core).
It reads sensitive credentials from a `.env` file locally or from Streamlit secrets when deployed.

## Files
- `app.py` - Streamlit app (UI + integration)
- `bot_core.py` - Core logic for signals, data collection, analysis
- `.env` - Template for your secrets (DO NOT commit this to GitHub)
- `requirements.txt` - Python dependencies
- `.gitignore` - ignores `.env` and venv
- `.streamlit/config.toml` - Streamlit UI config

## Setup (Windows)
1. Extract this package to your project folder.
2. Open PowerShell in the folder.
3. Create virtual env and install deps:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. Edit `.env` and paste your token & chat id.
5. Run locally:
   ```powershell
   streamlit run app.py
   ```
6. When deploying to Streamlit Cloud, do NOT upload `.env`. Instead, add the same keys under **Manage app -> Secrets** in Streamlit Cloud:
   - TELEGRAM_TOKEN
   - TELEGRAM_CHAT_ID
   - LOGIN_PASSWORD

## Notes
- This starter uses a mix of mock data and ccxt (if available). For live trading, configure real exchange API keys securely.
- Always test with paper trading before using any real funds.
