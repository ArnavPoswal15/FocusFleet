"""Entrypoint: initializes the DB and launches the FocusFleet Gradio app."""

import _path_setup  # noqa: F401

from focusfleet.database import init_db
from focusfleet.ui.app import build_app

if __name__ == "__main__":
    init_db()
    demo = build_app()
    demo.launch()
