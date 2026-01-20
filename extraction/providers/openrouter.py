"""OpenRouter provider for medical exam extraction."""

import base64
import os
import json
from pathlib import Path

import httpx

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from .base import LLMProvider, MedicalExamResult
from .schema import EXTRACTION_SYSTEM_PROMPT, get_extraction_prompt


class OpenRouterProvider(LLMProvider):
    """OpenRouter provider for extracting medical exam data from PDFs."""

    model_suffix = "openrouter"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "google/gemini-2.5-flash",
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

        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set OPENROUTER_API_KEY env var or pass api_key parameter."
            )

        self.api_key = self.api_key.strip('"\'')
        self.model = model
        self.temperature = temperature
        self.base_url = "https://openrouter.ai/api/v1"

    def extract_from_pdf(
        self,
        pdf_path: str | Path,
        **kwargs
    ) -> MedicalExamResult:
        """
        Extract medical exam data from a PDF file using OpenRouter.

        Uses a two-step approach:
        1. Extract raw text/content from PDF
        2. Parse into structured data
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Step 1: Extract raw content from PDF
        raw_text, extract_tokens = self._extract_content_from_pdf(pdf_path)

        # Step 2: Parse into structured data
        data, parse_tokens = self._parse_medical_data(raw_text, pdf_path.name)

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
                "input_tokens": extract_tokens.get("input", 0) + parse_tokens.get("input", 0),
                "output_tokens": extract_tokens.get("output", 0) + parse_tokens.get("output", 0),
                "extract_tokens": extract_tokens,
                "parse_tokens": parse_tokens,
            }
        )

    def _extract_content_from_pdf(self, pdf_path: Path) -> tuple[str, dict]:
        """
        Step 1: Extract raw content from PDF.
        Returns: (text, token_counts)
        """
        # Read and encode PDF as base64
        pdf_data = pdf_path.read_bytes()
        pdf_base64 = base64.standard_b64encode(pdf_data).decode("utf-8")

        extraction_prompt = """Extract ALL text and data from this medical exam document.
Include:
- Patient information (name, date of birth, etc.)
- Physician and facility information
- All test results with exact values, units, and reference ranges
- Any conclusions, findings, or observations
- Dates and timestamps

Preserve the exact numeric values as shown. Format as structured text."""

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "filename": pdf_path.name,
                            "file_data": f"data:application/pdf;base64,{pdf_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": extraction_prompt
                    }
                ]
            }
        ]

        response = self._call_api(messages)
        text = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = response.get("usage", {})

        return text, {
            "input": usage.get("prompt_tokens", 0),
            "output": usage.get("completion_tokens", 0)
        }

    def _parse_medical_data(self, raw_text: str, filename: str) -> tuple[dict, dict]:
        """
        Step 2: Parse extracted text into structured medical data.
        Returns: (data, token_counts)
        """
        prompt = f"""{EXTRACTION_SYSTEM_PROMPT}

{get_extraction_prompt(f"Source file: {filename}")}

Raw extracted content:
{raw_text}

Extract and structure all medical exam data according to the schema. Return ONLY valid JSON with the extracted data, no markdown formatting or explanations."""

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        response = self._call_api(messages, json_mode=True)
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = response.get("usage", {})

        # Parse JSON from response
        data = {}
        try:
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            data = json.loads(content.strip())
        except (json.JSONDecodeError, IndexError) as e:
            print(f"Warning: Failed to parse JSON response: {e}")
            data = {"raw_content": content}

        return data, {
            "input": usage.get("prompt_tokens", 0),
            "output": usage.get("completion_tokens", 0)
        }

    def _call_api(self, messages: list, json_mode: bool = False) -> dict:
        """Make API call to OpenRouter."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/health-pilot",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        with httpx.Client(timeout=300.0) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
