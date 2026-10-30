"""Common interface for asynchronous file loaders."""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseDocumentLoader(ABC):
    """Base class for loaders that return extracted document text."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    @abstractmethod
    async def load(self) -> str:
        """Read or extract text from the source file."""
        raise NotImplementedError
