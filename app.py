import subprocess
import sys
import os

# Auto-install required packages
REQUIRED_PACKAGES = ['pandas_ta', 'ccxt', 'streamlit', 'scikit-learn', 'plotly', 'kaleido']

for package in REQUIRED_PACKAGES:
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Now continue with your normal imports
import time
import streamlit as st
import pandas as pd
from datetime import datetime
from bot_core import (
    SUPPORTED_EXCHANGES, ExchangeManager, generate_signal, suggest_pairs,
    format_signal_for_display, bitnodes_snapshot, send_telegram, fetch_htf_alignment,
    run_backtest
)
