#!/usr/bin/env python3
"""CLI tool for extracting medical exam data from PDFs using LLM providers."""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

try:
    from azure.storage.blob import BlobServiceClient
except ImportError:
    print("Warning: azure-storage-blob not installed. Run: pip install azure-storage-blob")
    exit(1)

from extraction import MedicalExamExtractor
from extraction.providers import GoogleProvider, OpenRouterProvider


def upload_to_azure_blob(
    local_path: Path,
    container_name: str,
    blob_folder: str | None = None,
    connection_string: str | None = None,
) -> str | None:
    """
    Upload a file to Azure Blob Storage.

    Args:
        local_path: Path to the local file to upload
        container_name: Name of the Azure blob container
        blob_folder: Optional folder path within the container
        connection_string: Azure storage connection string (or uses env var)

    Returns:
        The blob URL if successful, None if failed
    """
    conn_string = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_string:
        print("Warning: AZURE_STORAGE_CONNECTION_STRING not set, skipping upload")
        return None

    try:
        blob_service_client = BlobServiceClient.from_connection_string(conn_string)
        container_client = blob_service_client.get_container_client(container_name)

        # Create container if it doesn't exist
        try:
            container_client.create_container()
        except Exception:
            pass  # Container already exists

        # Build blob name with optional folder
        blob_name = local_path.name
        if blob_folder:
            blob_folder = blob_folder.strip("/")
            blob_name = f"{blob_folder}/{blob_name}"

        # Upload the file
        blob_client = container_client.get_blob_client(blob_name)
        with open(local_path, "rb") as f:
            blob_client.upload_blob(f, overwrite=True)

        return blob_client.url
    except Exception as e:
        print(f"Warning: Failed to upload {local_path.name} to Azure: {e}")
        return None


def upload_directory_to_azure(
    local_dir: Path,
    container_name: str,
    blob_folder: str | None = None,
    pattern: str = "*.json",
) -> list[str]:
    """
    Upload all files matching pattern from a directory to Azure Blob Storage.

    Args:
        local_dir: Local directory containing files to upload
        container_name: Name of the Azure blob container
        blob_folder: Optional folder path within the container
        pattern: Glob pattern for files to upload

    Returns:
        List of successfully uploaded blob URLs
    """
    uploaded = []
    for file_path in local_dir.glob(pattern):
        if file_path.is_file():
            url = upload_to_azure_blob(file_path, container_name, blob_folder)
            if url:
                uploaded.append(url)
                print(f"  Uploaded: {file_path.name}")
    return uploaded


def main():
    parser = argparse.ArgumentParser(
        description="Extract medical exam data from PDF files using vision-capable LLMs"
    )
    parser.add_argument(
        "--pdf_path",
        default="./data",
        type=str,
        help="Path to PDF file or directory containing PDFs"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output JSON path (for single PDF) or directory (for batch)"
    )
    parser.add_argument(
        "--provider",
        choices=["google", "openrouter"],
        default="openrouter",
        help="LLM provider to use (default: openrouter)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (default: provider-specific)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the provider (or set via env var)"
    )
    parser.add_argument(
        "--save-text",
        action="store_true",
        default=True,
        help="Save extracted raw text to .txt files alongside output"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        default=True,
        help="Merge all PDFs into a single patient record (for batch processing)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--upload-azure",
        action="store_true",
        default=True,
        help="Upload results to Azure Blob Storage (default: True for merge mode)"
    )
    parser.add_argument(
        "--no-upload-azure",
        action="store_true",
        help="Disable Azure upload"
    )
    parser.add_argument(
        "--azure-container",
        type=str,
        default="health-pilot-data",
        help="Azure blob container name (default: health-pilot-data)"
    )
    parser.add_argument(
        "--azure-folder",
        type=str,
        default=None,
        help="Folder path within the Azure container (optional)"
    )

    args = parser.parse_args()

    # Handle --no-upload-azure flag
    if args.no_upload_azure:
        args.upload_azure = False

    # Initialize provider
    if args.provider == "google":
        api_key = args.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("Error: API key required. Set GOOGLE_API_KEY or GEMINI_API_KEY env var or use --api-key")
            sys.exit(1)

        model = args.model or "gemini-2.5-flash"
        provider = GoogleProvider(
            api_key=api_key,
            model=model,
            temperature=0.0
        )
    elif args.provider == "openrouter":
        api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("Error: API key required. Set OPENROUTER_API_KEY env var or use --api-key")
            sys.exit(1)

        model = args.model or "google/gemini-2.5-flash"
        provider = OpenRouterProvider(
            api_key=api_key,
            model=model,
            temperature=0.0
        )
    else:
        print(f"Error: Provider '{args.provider}' not implemented")
        sys.exit(1)

    # Create extractor
    extractor = MedicalExamExtractor(provider=provider)

    # Process PDF(s)
    pdf_path = Path(args.pdf_path)

    if pdf_path.is_dir():
        # Batch processing
        pdf_paths = list(pdf_path.glob("*.pdf"))
        if not pdf_paths:
            print(f"Error: No PDFs found in {pdf_path}")
            sys.exit(1)

        print(f"Processing {len(pdf_paths)} PDFs...")

        # Determine output directory
        output_dir = Path(args.output) if args.output else pdf_path / "extracted"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        total_input_tokens = 0
        total_output_tokens = 0

        for i, p in enumerate(pdf_paths, 1):
            print(f"\n[{i}/{len(pdf_paths)}] Processing: {p.name}")
            try:
                result = extractor.extract(
                    p,
                    save_json=not args.merge,  # Don't save individual JSONs if merging
                    save_text=args.save_text,
                    output_dir=output_dir,
                )
                results.append(result)

                # Track tokens
                metadata = result.get("metadata", {})
                in_tokens = metadata.get("input_tokens", 0)
                out_tokens = metadata.get("output_tokens", 0)
                total_input_tokens += in_tokens
                total_output_tokens += out_tokens

                # Show summary
                data = result.get("data", {})
                patient = data.get("patient_info", {}).get("name", "Unknown")
                exam_date = data.get("exam_info", {}).get("exam_date", "N/A")
                print(f"  Patient: {patient}")
                print(f"  Exam date: {exam_date}")
                print(f"  Tokens: {in_tokens} in, {out_tokens} out")

                if args.verbose:
                    # Show what was extracted
                    sections = []
                    if "biochemistry" in data:
                        sections.append("biochemistry")
                    if "coagulation" in data:
                        sections.append("coagulation")
                    if "complete_blood_count" in data:
                        sections.append("CBC")
                    if "electrocardiogram" in data:
                        sections.append("ECG")
                    if "imaging_studies" in data:
                        sections.append(f"imaging({len(data['imaging_studies'])})")
                    if "polysomnography" in data:
                        sections.append("sleep study")
                    print(f"  Sections: {', '.join(sections) if sections else 'none'}")

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    "source_pdf": str(p),
                    "error": str(e),
                    "data": {},
                })

        # Merge results if requested
        if args.merge and results:
            print("\nMerging results...")
            merged = extractor.merge_results(results)
            merged_path = output_dir / "merged_patient_data.json"
            extractor.save_to_json(merged, merged_path)
            print(f"Merged data saved to: {merged_path}")

            # Format individual exams from merged data (using proto-compatible format)
            print("\nFormatting individual exams (proto format)...")
            exams_dir = output_dir / "exams"
            individual_exams = extractor.format_individual_exam_proto(merged, exams_dir)
            print(f"Created {len(individual_exams)} individual exam files in: {exams_dir}")

            # Upload to Azure Blob Storage
            if args.upload_azure:
                print("\nUploading to Azure Blob Storage...")
                uploaded_urls = []

                # Upload merged file
                url = upload_to_azure_blob(
                    merged_path,
                    args.azure_container,
                    args.azure_folder,
                )
                if url:
                    uploaded_urls.append(url)
                    print(f"  Uploaded: {merged_path.name}")

                # Upload individual exam files
                exams_folder = f"{args.azure_folder}/exams" if args.azure_folder else "exams"
                exam_urls = upload_directory_to_azure(
                    exams_dir,
                    args.azure_container,
                    exams_folder,
                )
                uploaded_urls.extend(exam_urls)

                print(f"Uploaded {len(uploaded_urls)} files to Azure container: {args.azure_container}")

        print(f"\n{'='*50}")
        print(f"Total PDFs processed: {len(results)}")
        print(f"Total tokens: {total_input_tokens:,} input, {total_output_tokens:,} output")

    else:
        # Single PDF processing
        if not pdf_path.exists():
            print(f"Error: PDF not found: {pdf_path}")
            sys.exit(1)

        print(f"Extracting from: {pdf_path}")

        result = extractor.extract(
            pdf_path,
            save_json=True,
            save_text=args.save_text,
            output_dir=Path(args.output).parent if args.output else None,
        )

        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = pdf_path.with_suffix(f".{provider.model_suffix}.json")

        extractor.save_to_json(result["data"], output_path)

        # Show results
        data = result.get("data", {})
        metadata = result.get("metadata", {})

        print(f"\nExtraction complete!")
        print(f"Output: {output_path}")

        patient = data.get("patient_info", {})
        print(f"\nPatient: {patient.get('name', 'Unknown')}")
        print(f"DOB: {patient.get('date_of_birth', 'N/A')}")

        exam = data.get("exam_info", {})
        print(f"Exam date: {exam.get('exam_date', 'N/A')}")
        print(f"Record #: {exam.get('record_number', 'N/A')}")

        print(f"\nTokens: {metadata.get('input_tokens', 0)} input, {metadata.get('output_tokens', 0)} output")

        if args.verbose:
            print("\nExtracted data preview:")
            print(json.dumps(data, indent=2, ensure_ascii=False)[:2000])
            if len(json.dumps(data)) > 2000:
                print("... (truncated)")


if __name__ == "__main__":
    main()
