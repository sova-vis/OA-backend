"""FastAPI app entrypoint for O/A Levels pipeline (use: oa_levels_pipeline.api:app)."""

from dotenv import load_dotenv

load_dotenv()

from oa_main_pipeline.api import create_app

app = create_app()
