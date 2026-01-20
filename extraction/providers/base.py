"""Abstract base class for LLM providers for medical exam extraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MedicalExamResult:
    """Result from LLM medical exam extraction."""

    data: dict
    raw_response: str | None = None
    metadata: dict | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LLMProvider(ABC):
    """Abstract base class for LLM providers that extract medical exam data from PDFs."""

    model_suffix: str = ""

    @abstractmethod
    def extract_from_pdf(
        self,
        pdf_path: str | Path,
        **kwargs
    ) -> MedicalExamResult:
        """
        Extract medical exam data from a PDF file.

        Args:
            pdf_path: Path to the PDF file
            **kwargs: Provider-specific parameters

        Returns:
            MedicalExamResult containing extracted data

        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If extraction fails
        """
        pass
