"""API module for Media Recommender."""

try:
    from .main import app
    __all__ = ["app"]
except ImportError as e:
    import logging
    logging.warning(f"Could not import FastAPI app: {e}")
    app = None
    __all__ = []
