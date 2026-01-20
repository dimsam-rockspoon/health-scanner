# Health Scanner

A CLI tool for extracting structured medical exam data from PDF files using vision-capable LLMs.

## Features

- Extract medical exam data from PDF documents using AI vision models
- Support for multiple LLM providers (Google Gemini, OpenRouter)
- Handles various exam types:
  - Blood tests (CBC, biochemistry, coagulation)
  - Imaging studies (MRI, CT, X-ray, Ultrasound)
  - Electrocardiograms (EKG/ECG)
  - Polysomnography (sleep studies)
- Batch processing with automatic result merging
- Proto-compatible JSON output format
- Optional Azure Blob Storage upload

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Add your API keys to `.env`

## Usage

### Single PDF extraction

```bash
python extract_exam_info.py path/to/exam.pdf
```

### Batch processing (directory of PDFs)

```bash
python extract_exam_info.py path/to/pdf/directory/
```

### Options

```
positional arguments:
  pdf_path              Path to PDF file or directory containing PDFs

optional arguments:
  -o, --output          Output JSON path (single PDF) or directory (batch)
  --provider            LLM provider: google, openrouter (default: openrouter)
  --model               Model to use (default: provider-specific)
  --api-key             API key for the provider (or set via env var)
  --save-text           Save extracted raw text to .txt files
  --merge               Merge all PDFs into a single patient record (default: true)
  --upload-azure        Upload results to Azure Blob Storage
  --no-upload-azure     Disable Azure upload
  --azure-container     Azure blob container name (default: health-pilot-data)
  --azure-folder        Folder path within the Azure container
  -v, --verbose         Verbose output
```

### Examples

Extract with Google Gemini:
```bash
python extract_exam_info.py exam.pdf --provider google
```

Batch process with custom output directory:
```bash
python extract_exam_info.py ./exams/ -o ./results/
```

Disable Azure upload:
```bash
python extract_exam_info.py ./exams/ --no-upload-azure
```

## Output Format

Extracted data follows a proto-compatible JSON schema. Individual exam files are saved in the `exams/` subdirectory with the format:

```
{exam_type}_{exam_code}_{id}.json
```

Example output structure:
```json
{
  "id": "uuid",
  "patient_id": "PAT-XXXXXXXXXXXX",
  "exam_type": "blood",
  "exam_code": "HEMOGRAMA",
  "exam_name": "Complete Blood Count",
  "collected_at": "2024-01-15T00:00:00Z",
  "status": "FINAL",
  "priority": "ROUTINE",
  "json": {
    "analytes": [...]
  }
}
```

## Project Structure

```
health-scanner/
├── extract_exam_info.py      # CLI entry point
├── extraction/
│   ├── __init__.py
│   ├── medical_exam_extractor.py  # Main extractor class
│   ├── proto_models.py       # Pydantic models (proto-compatible)
│   ├── proto_transformer.py  # Data transformation logic
│   └── providers/
│       ├── __init__.py
│       ├── base.py           # Provider interface
│       ├── schema.py         # Extraction schema
│       ├── google.py         # Google Gemini provider
│       └── openrouter.py     # OpenRouter provider
└── data/
    └── extracted/            # Output directory
        └── exams/            # Individual exam files
```

## License

Proprietary - Rockspoon Inc.
