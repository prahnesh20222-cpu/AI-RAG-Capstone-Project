import sys
for path in sys.path:
    print(path)

from src.rag.md_loader import load_documents
from config.logging_config import setup_logger


logger = setup_logger(__name__)
logger.info("Starting tests for module.")
print("Entry point is running")
print("Entry point name:", __name__)