# -*- coding: utf-8 -*-
import sys
import subprocess
import os

# ===== AUTO-INSTALLER =====
def ensure_import(package, min_version=None):
    try:
        mod = __import__(package)
        if min_version:
            from packaging import version
            if version.parse(mod.__version__) < version.parse(min_version):
                raise ImportError(f"Need {package}>={min_version}")
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}>={min_version}" if min_version else package])

# Critical dependencies with minimum versions
ensure_import("pandas_ta", "0.3.14")
ensure_import("ccxt", "3.0.0")
ensure_import("scikit-learn", "1.2.0")
ensure_import("kaleido", "0.2.1")

# ===== YOUR ORIGINAL IMPORTS =====
# [KEEP ALL YOUR EXISTING IMPORTS BELOW THIS LINE]
import time
import streamlit as st
import pandas as pd
from datetime import datetime
from bot_core import (
    SUPPORTED_EXCHANGES, ExchangeManager, generate_signal, suggest_pairs,
    format_signal_for_display, bitnodes_snapshot, send_telegram, fetch_htf_alignment,
    run_backtest
)
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
