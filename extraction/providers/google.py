"""Google Gemini provider for medical exam extraction."""

import os
from pathlib import Path

try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError(
        "google-genai package is required for GoogleProvider. "
        "Install with: pip install google-genai"
    )

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from .base import LLMProvider, MedicalExamResult
from .schema import EXTRACTION_SYSTEM_PROMPT, SAVE_MEDICAL_EXAM_FUNCTION_GOOGLE, get_extraction_prompt


class GoogleProvider(LLMProvider):
    """Google Gemini provider for extracting medical exam data from PDFs."""

    model_suffix = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.0,
        dotenv_path: str | None = None,
    ):
        # Load .env file if dotenv is available
        if load_dotenv is not None:
            if dotenv_path:
                load_dotenv(dotenv_path)
            else:
                current_dir = Path(__file__).resolve().parent
                for parent in [current_dir] + list(current_dir.parents):
                    env_file = parent / ".env"
                    if env_file.exists():
                        load_dotenv(env_file)
                        break

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set GOOGLE_API_KEY or GEMINI_API_KEY env var or pass api_key parameter."
            )

        self.api_key = self.api_key.strip('"\'')
        self.model = model
        self.temperature = temperature

        # Google GenAI client
        self.client = genai.Client(api_key=self.api_key)

    def extract_from_pdf(
        self,
        pdf_path: str | Path,
        **kwargs
    ) -> MedicalExamResult:
        """
        Extract medical exam data from a PDF file using Gemini.

        Uses a two-step approach:
        1. Extract raw text/content from PDF using Gemini vision
        2. Parse into structured data using function calling
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Step 1: Extract raw content from PDF
        raw_text, extract_in, extract_out = self._extract_content_from_pdf(pdf_path)

        # Step 2: Parse into structured data using function calling
        data, parse_in, parse_out = self._parse_medical_data(raw_text, pdf_path.name)

        # Add source file info
        if "source_files" not in data:
            data["source_files"] = []
        data["source_files"].append({
            "filename": pdf_path.name,
            "record_number": data.get("exam_info", {}).get("record_number", ""),
            "exam_date": data.get("exam_info", {}).get("exam_date", "")
        })

        return MedicalExamResult(
            data=data,
            raw_response=raw_text,
            metadata={
                "model": self.model,
                "method": "two_step_pdf",
                "input_tokens": extract_in + parse_in,
                "output_tokens": extract_out + parse_out,
                "extract_tokens": {"input": extract_in, "output": extract_out},
                "parse_tokens": {"input": parse_in, "output": parse_out},
            }
        )

    def _extract_content_from_pdf(self, pdf_path: Path) -> tuple[str, int, int]:
        """
        Step 1: Extract raw content from PDF using Gemini vision.
        Returns: (text, input_tokens, output_tokens)
        """
        pdf_data = pdf_path.read_bytes()

        extraction_prompt = """Extract ALL text and data from this medical exam document.
Include:
- Patient information (name, date of birth, etc.)
- Physician and facility information
- All test results with exact values, units, and reference ranges
- Any conclusions, findings, or observations
- Dates and timestamps

Preserve the exact numeric values as shown. Format as structured text."""

        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=pdf_data, mime_type="application/pdf"),
                extraction_prompt,
            ],
        )

        input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
        output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0

        return response.text, input_tokens, output_tokens

    def _parse_medical_data(self, raw_text: str, filename: str) -> tuple[dict, int, int]:
        """
        Step 2: Parse extracted text into structured medical data using function calling.
        Returns: (data, input_tokens, output_tokens)
        """
        tools = types.Tool(function_declarations=[SAVE_MEDICAL_EXAM_FUNCTION_GOOGLE])
        config = types.GenerateContentConfig(
            tools=[tools],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=["save_medical_exam_data"],
                )
            ),
        )

        prompt = f"""{EXTRACTION_SYSTEM_PROMPT}

{get_extraction_prompt(f"Source file: {filename}")}

Raw extracted content:
{raw_text}

Extract and structure all medical exam data according to the schema. Call the save_medical_exam_data function with the extracted data."""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )

        input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
        output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0

        # Extract data from function call
        data = {}
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    if part.function_call.name == "save_medical_exam_data":
                        args = dict(part.function_call.args) if part.function_call.args else {}
                        data = args
                        break

        return data, input_tokens, output_tokens
