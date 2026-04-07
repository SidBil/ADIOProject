"""Vercel serverless entry point — re-exports the FastAPI app."""

import sys
import os
from pathlib import Path

web_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(web_dir))

os.environ["WEB_DIR"] = str(web_dir)

from app import app
