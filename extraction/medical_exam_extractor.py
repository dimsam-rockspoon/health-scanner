"""Provider-agnostic medical exam extractor."""

import json
import uuid
from pathlib import Path

from .providers.base import LLMProvider, MedicalExamResult
from .proto_models import LabTestResult
from .proto_transformer import ProtoTransformer


class MedicalExamExtractor:
    """
    Provider-agnostic extractor for medical exam data from PDFs.

    This class orchestrates the extraction process using a pluggable LLM provider.
    """

    def __init__(self, provider: LLMProvider):
        """
        Initialize the extractor with a provider.

        Args:
            provider: An LLM provider implementing the LLMProvider interface
        """
        self.provider = provider

    def extract(
        self,
        pdf_path: str | Path,
        save_json: bool = False,
        save_text: bool = False,
        output_dir: str | Path | None = None,
        **kwargs
    ) -> dict:
        """
        Extract medical exam data from a PDF file.

        Args:
            pdf_path: Path to the PDF file
            save_json: If True, save extracted data to a JSON file
            save_text: If True, save raw extracted text to a .txt file
            output_dir: Directory for output files. If None, uses PDF's directory
            **kwargs: Additional parameters passed to the provider

        Returns:
            Dict containing extracted medical exam data and metadata
        """
        pdf_path = Path(pdf_path)

        # Extract using provider
        result = self.provider.extract_from_pdf(pdf_path, **kwargs)

        # Determine output directory
        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = pdf_path.parent

        # Save JSON if requested
        if save_json:
            json_path = out_dir / f"{pdf_path.stem}.{self.provider.model_suffix}.json"
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(result.data, f, ensure_ascii=False, indent=2)

        # Save raw text if requested
        if save_text and result.raw_response:
            txt_path = out_dir / f"{pdf_path.stem}.{self.provider.model_suffix}.txt"
            txt_path.write_text(result.raw_response, encoding="utf-8")

        # Build output
        return {
            "source_pdf": str(pdf_path),
            "data": result.data,
            "metadata": result.metadata,
            "raw_response": result.raw_response,
        }

    def extract_batch(
        self,
        pdf_paths: list[str | Path],
        save_json: bool = False,
        save_text: bool = False,
        output_dir: str | Path | None = None,
        **kwargs
    ) -> list[dict]:
        """
        Extract medical exam data from multiple PDF files.

        Args:
            pdf_paths: List of paths to PDF files
            save_json: If True, save extracted data to JSON files
            save_text: If True, save raw extracted text to .txt files
            output_dir: Directory for output files
            **kwargs: Additional parameters passed to the provider

        Returns:
            List of extraction results (one per PDF)
        """
        results = []
        for pdf_path in pdf_paths:
            try:
                result = self.extract(
                    pdf_path,
                    save_json=save_json,
                    save_text=save_text,
                    output_dir=output_dir,
                    **kwargs
                )
                results.append(result)
            except Exception as e:
                print(f"Error extracting {pdf_path}: {e}")
                results.append({
                    "source_pdf": str(pdf_path),
                    "error": str(e),
                    "data": {},
                })
        return results

    def merge_results(self, results: list[dict]) -> dict:
        """
        Merge multiple extraction results into a single patient record.

        Useful when a patient has multiple exam PDFs that should be combined.

        Args:
            results: List of extraction results from extract() or extract_batch()

        Returns:
            Merged data dict with combined source_files and all exam data
        """
        if not results:
            return {}

        # Start with the first result's data
        merged = results[0].get("data", {}).copy()

        # Ensure source_files is a list
        if "source_files" not in merged:
            merged["source_files"] = []

        # Keys that should be merged as arrays (append items)
        array_merge_keys = {"source_files", "imaging_studies", "exams"}

        # Keys that are patient/facility info - prefer first non-empty value
        info_keys = {"patient_info", "physician_info", "facility_info", "general_note"}

        # Merge data from additional results
        for result in results[1:]:
            data = result.get("data", {})

            for key, value in data.items():
                if not value:
                    continue

                # Array merge: extend existing arrays
                if key in array_merge_keys:
                    if key not in merged:
                        merged[key] = []
                    if isinstance(value, list):
                        merged[key].extend(value)
                    else:
                        merged[key].append(value)

                # Info keys: prefer first non-empty value
                elif key in info_keys:
                    if key not in merged or not merged[key]:
                        merged[key] = value

                # All other keys: prefer first non-empty, or merge dicts
                elif key not in merged:
                    merged[key] = value
                elif isinstance(merged[key], dict) and isinstance(value, dict):
                    # Shallow merge dicts, preferring existing values
                    for k, v in value.items():
                        if k not in merged[key]:
                            merged[key][k] = v

        return merged

    def save_to_json(self, data: dict, output_path: str | Path) -> None:
        """
        Save extraction data to a JSON file.

        Args:
            data: Extraction data dict
            output_path: Path where JSON should be saved
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def format_individual_exams(
        self,
        merged_data: dict,
        output_dir: str | Path,
    ) -> list[dict]:
        """
        Convert merged patient data into individual exam files.

        Decomposes the merged dict into separate exam files following a standardized schema.
        Each exam is saved as a separate JSON in the output directory.

        Sources of exams in merged data:
        - exam_results[]: individual lab/diagnostic tests (blood, chemistry, etc.)
        - polysomnography_results: sleep study exam
        - exam_metadata + findings/conclusion: imaging exams (MRI, CT, etc.)
        - electrocardiogram in exam_results: EKG exam

        Args:
            merged_data: Merged patient data dict from merge_results()
            output_dir: Directory where individual exam JSONs will be saved

        Returns:
            List of formatted exam dicts that were saved
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exams = []

        # Extract patient info for all exams
        patient_info = merged_data.get("patient_info") or merged_data.get("patient_information", {})
        patient_id = self._generate_patient_id(patient_info)
        facility_info = merged_data.get("facility_info") or merged_data.get("facility_information", {})

        # 1. Process exam_results[] - individual lab tests
        exam_results = merged_data.get("exam_results", [])
        for exam_result in exam_results:
            exam = self._format_lab_exam(exam_result, patient_id, facility_info)
            if exam:
                exams.append(exam)

        # 2. Process polysomnography_results - sleep study
        polysom = merged_data.get("polysomnography_results")
        if polysom:
            exam = self._format_polysomnography_exam(
                polysom,
                merged_data.get("conclusions", []),
                patient_id,
                facility_info,
                merged_data.get("exam_metadata", {}),
            )
            if exam:
                exams.append(exam)

        # 3. Process imaging exams (MRI, CT, etc.) from exam_metadata + findings
        exam_meta = merged_data.get("exam_metadata", {})
        findings = merged_data.get("findings", [])
        conclusion = merged_data.get("conclusion", [])
        if exam_meta.get("exam_type") in ("MRI", "CT", "X-ray", "Ultrasound"):
            exam = self._format_imaging_exam(
                exam_meta, findings, conclusion, patient_id, facility_info
            )
            if exam:
                exams.append(exam)

        # Save each exam to a separate file
        for exam in exams:
            exam_id = exam["_id"]
            exam_type = exam["exam_type"]
            exam_code = exam.get("exam_code", "unknown").lower().replace(" ", "_")
            filename = f"{exam_type}_{exam_code}_{exam_id[:8]}.json"
            output_path = output_dir / filename
            self.save_to_json(exam, output_path)

        return exams

    def _generate_patient_id(self, patient_info: dict) -> str:
        """Generate a patient ID from patient info or create a new UUID."""
        if not patient_info:
            return f"PAT-{uuid.uuid4().hex[:12].upper()}"

        # Use name hash if available for consistency
        name = patient_info.get("name", "")
        dob = patient_info.get("date_of_birth", "")
        if name and dob:
            import hashlib
            hash_input = f"{name}_{dob}".encode()
            return f"PAT-{hashlib.sha256(hash_input).hexdigest()[:12].upper()}"

        return f"PAT-{uuid.uuid4().hex[:12].upper()}"

    def _normalize_exam_type(self, exam_type_raw: str) -> str:
        """Normalize exam type to standard modality categories."""
        exam_type_upper = exam_type_raw.upper()

        # Blood/chemistry tests
        blood_tests = {
            "CREATININA", "POTASSIO", "SODIO", "UREIA", "GLICOSE",
            "HEMOGRAMA", "TEMPO DE PROTROMBINA", "TEMPO DE TROMBOPLASTINA",
        }
        if any(bt in exam_type_upper for bt in blood_tests):
            return "blood"

        # Imaging
        if "MRI" in exam_type_upper or "RESSONANCIA" in exam_type_upper:
            return "mri"
        if "CT" in exam_type_upper or "TOMOGRAFIA" in exam_type_upper:
            return "ct"
        if "RAIO" in exam_type_upper or "X-RAY" in exam_type_upper:
            return "xray"
        if "ULTRASSOM" in exam_type_upper or "ULTRASOUND" in exam_type_upper:
            return "ultrasound"

        # EKG
        if "ELETROCARDIOGRAMA" in exam_type_upper or "ECG" in exam_type_upper or "EKG" in exam_type_upper:
            return "ekg"

        # Sleep study
        if "POLISSONOGRAFIA" in exam_type_upper or "POLYSOMNOGRAPHY" in exam_type_upper:
            return "polysomnography"

        return "other"

    def _format_lab_exam(self, exam_result: dict, patient_id: str, facility_info: dict) -> dict | None:
        """Format a lab/blood test exam from exam_results."""
        if not exam_result:
            return None

        exam_type_raw = exam_result.get("exam_type", "")
        normalized_type = self._normalize_exam_type(exam_type_raw)

        # Handle EKG separately
        if normalized_type == "ekg":
            return self._format_ekg_exam(exam_result, patient_id, facility_info)

        exam = {
            "_id": str(uuid.uuid4()),
            "patient_id": patient_id,
            "exam_type": normalized_type,
            "exam_code": exam_type_raw,
            "exam_name": exam_result.get("test_name", exam_type_raw),
            "collected_at": self._format_date(exam_result.get("collected_date")),
            "reported_at": exam_result.get("released_date"),
            "status": "final",
            "priority": "routine",
            "location": facility_info.get("name", ""),
            "details": {},
        }

        # Build details based on exam structure
        details = {
            "method": exam_result.get("method"),
        }

        # Handle single result vs multiple results
        if "result" in exam_result:
            details["measurements"] = [{
                "name": exam_result.get("test_name", exam_type_raw),
                "value": exam_result.get("result"),
                "unit": exam_result.get("unit"),
                "reference_range": exam_result.get("reference_range"),
            }]
        elif "results" in exam_result:
            results = exam_result["results"]
            if isinstance(results, list):
                details["measurements"] = [
                    {
                        "name": r.get("name"),
                        "value": r.get("value"),
                        "unit": r.get("unit"),
                        "reference_range": r.get("reference_range"),
                    }
                    for r in results
                ]
            elif isinstance(results, dict):
                # Handle hemogram-style nested structure
                details["measurements"] = []
                for group_name, group_items in results.items():
                    if isinstance(group_items, list) and group_items:
                        # Check if it's a list of dicts (measurements) or list of strings (notes)
                        if isinstance(group_items[0], dict):
                            for item in group_items:
                                measurement = {
                                    "name": item.get("name"),
                                    "value": item.get("value"),
                                    "unit": item.get("unit"),
                                    "reference_range": item.get("reference_range"),
                                    "group": group_name,
                                }
                                if "absolute_value" in item:
                                    measurement["absolute_value"] = item["absolute_value"]
                                    measurement["absolute_unit"] = item.get("absolute_unit")
                                details["measurements"].append(measurement)
                        else:
                            # List of strings (notes)
                            details[group_name] = group_items
                    elif isinstance(group_items, str):
                        details[group_name] = group_items

        # Add notes/observations
        if exam_result.get("notes"):
            details["notes"] = exam_result["notes"]
        if exam_result.get("observations"):
            details["observations"] = exam_result["observations"]

        # Responsible professional
        if exam_result.get("responsible_professional"):
            exam["performing_physician"] = {
                "name": exam_result["responsible_professional"],
                "crm": exam_result.get("responsible_professional_crm"),
            }

        # Clean None values from details
        exam["details"] = {k: v for k, v in details.items() if v is not None}

        return exam

    def _format_ekg_exam(self, exam_result: dict, patient_id: str, facility_info: dict) -> dict:
        """Format an EKG exam."""
        exam = {
            "_id": str(uuid.uuid4()),
            "patient_id": patient_id,
            "exam_type": "ekg",
            "exam_code": "ECG",
            "exam_name": exam_result.get("test_name", "Electrocardiogram"),
            "collected_at": self._format_date(exam_result.get("collected_date")),
            "reported_at": exam_result.get("released_date"),
            "status": "final",
            "priority": "routine",
            "location": facility_info.get("name", ""),
            "indication": exam_result.get("indication"),
            "clinical_notes": exam_result.get("medication"),
            "result_summary": exam_result.get("conclusion"),
            "details": {
                "parameters": exam_result.get("parameters", []),
                "morphological_description": exam_result.get("morphological_description_and_comment"),
                "conclusion": exam_result.get("conclusion"),
                "observations": exam_result.get("observations"),
            },
        }

        if exam_result.get("laudado_by"):
            exam["performing_physician"] = {
                "name": exam_result["laudado_by"],
                "crm": exam_result.get("laudado_by_crm"),
            }

        return exam

    def _format_polysomnography_exam(
        self,
        polysom: dict,
        conclusions: list,
        patient_id: str,
        facility_info: dict,
        exam_metadata: dict,
    ) -> dict:
        """Format a polysomnography (sleep study) exam."""
        # Try to determine date from monitoring timestamps or exam metadata
        collected_at = None
        if polysom.get("monitoring_start_time"):
            # Infer from exam_metadata if available
            collected_at = exam_metadata.get("exam_date")

        exam = {
            "_id": str(uuid.uuid4()),
            "patient_id": patient_id,
            "exam_type": "polysomnography",
            "exam_code": "PSG",
            "exam_name": "Polysomnography (Sleep Study)",
            "collected_at": collected_at,
            "status": "final",
            "priority": "routine",
            "location": polysom.get("monitoring_location", facility_info.get("name", "")),
            "result_summary": "; ".join(conclusions) if conclusions else None,
            "details": {
                "monitoring_equipment": polysom.get("monitoring_equipment"),
                "monitoring_duration_minutes": polysom.get("monitoring_duration_minutes"),
                "monitoring_start_time": polysom.get("monitoring_start_time"),
                "monitoring_end_time": polysom.get("monitoring_end_time"),
                "electrophysiological_parameters": polysom.get("electrophysiological_parameters"),
                "sleep_metrics": {
                    "total_recording_time_minutes": polysom.get("total_recording_time_minutes"),
                    "total_sleep_time_minutes": polysom.get("total_sleep_time_minutes"),
                    "sleep_efficiency_percent": polysom.get("sleep_efficiency_percent"),
                    "sleep_distribution": polysom.get("sleep_distribution"),
                    "sleep_latency_minutes": polysom.get("sleep_latency_minutes"),
                    "rem_sleep_latency_minutes": polysom.get("rem_sleep_latency_minutes"),
                    "wake_time_during_sleep_minutes": polysom.get("wake_time_during_sleep_minutes"),
                },
                "sleep_fragmentation": polysom.get("sleep_fragmentation"),
                "respiratory_events": polysom.get("respiratory_events_during_sleep"),
                "oxygen_saturation": polysom.get("oxygen_saturation_SaO2"),
                "cardiac": {
                    "average_heart_rate_bpm": polysom.get("average_heart_rate_bpm"),
                    "arrhythmias": polysom.get("cardiac_arrhythmias"),
                },
                "other_findings": {
                    "periodic_limb_movements": polysom.get("periodic_limb_movements"),
                    "snoring": polysom.get("snoring"),
                    "event_position": polysom.get("event_position"),
                },
                "epworth_sleepiness_scale": polysom.get("epworth_sleepiness_scale"),
            },
        }

        return exam

    def _format_imaging_exam(
        self,
        exam_metadata: dict,
        findings: list,
        conclusion: list,
        patient_id: str,
        facility_info: dict,
    ) -> dict:
        """Format an imaging exam (MRI, CT, X-ray, etc.)."""
        exam_type_raw = exam_metadata.get("exam_type", "imaging")
        normalized_type = self._normalize_exam_type(exam_type_raw)

        exam = {
            "_id": str(uuid.uuid4()),
            "patient_id": patient_id,
            "exam_type": normalized_type,
            "exam_code": exam_metadata.get("record_number"),
            "exam_name": exam_metadata.get("exam_title") or exam_metadata.get("specific_exam"),
            "collected_at": self._format_date(exam_metadata.get("exam_date")),
            "reported_at": f"{exam_metadata.get('release_date')} {exam_metadata.get('release_time', '')}".strip(),
            "status": "final",
            "priority": "routine",
            "location": facility_info.get("name", ""),
            "result_summary": "; ".join(conclusion) if conclusion else None,
            "details": {
                "specific_exam": exam_metadata.get("specific_exam"),
                "findings": findings,
                "conclusion": conclusion,
                "digital_signature": exam_metadata.get("digital_signature"),
            },
        }

        if exam_metadata.get("performed_by"):
            exam["performing_physician"] = exam_metadata["performed_by"]
        if exam_metadata.get("signed_by"):
            exam["signing_physician"] = exam_metadata["signed_by"]

        return exam

    def _format_date(self, date_str: str | None) -> str | None:
        """Format date string to ISO format if needed."""
        if not date_str:
            return None
        # Already in ISO format (YYYY-MM-DD)
        if len(date_str) == 10 and date_str[4] == "-":
            return f"{date_str}T00:00:00Z"
        return date_str

    def _normalize_test_format(self, test: dict) -> dict:
        """
        Normalize test data from 'tests' format to 'exam_results' format.

        The 'tests' format (from some extractions) uses:
        - name: test name
        - result: {value, unit} or results: [...]
        - reference_ranges: {min, max, unit}
        - timestamps: {received_collected, released}

        The 'exam_results' format expects:
        - exam_type: test type
        - test_name: test name
        - result: value (string)
        - unit: unit
        - reference_range: "min - max"
        - collected_date: date
        - released_date: date
        """
        # If already in exam_results format, return as-is
        if "exam_type" in test:
            return test

        normalized = {
            "exam_type": test.get("name", ""),
            "test_name": test.get("name", ""),
            "method": test.get("method"),
        }

        # Handle result - can be dict {value, unit} or direct value
        result = test.get("result")
        if isinstance(result, dict):
            normalized["result"] = result.get("value")
            normalized["unit"] = result.get("unit")
        elif result is not None:
            normalized["result"] = result

        # Handle multiple results (e.g., hemogram series, coagulation tests)
        if "results" in test:
            results_list = test["results"]
            if isinstance(results_list, list):
                normalized["results"] = [
                    {
                        "name": r.get("parameter", r.get("name")),
                        "value": r.get("value"),
                        "unit": r.get("unit"),
                        "reference_range": self._format_ref_range(r.get("reference_range")),
                    }
                    for r in results_list
                ]

        # Handle hemogram series format
        if "series" in test:
            all_results = []
            for series in test["series"]:
                series_name = series.get("name", "")
                for r in series.get("results", []):
                    all_results.append({
                        "name": r.get("parameter"),
                        "value": r.get("value"),
                        "unit": r.get("unit"),
                        "reference_range": self._format_ref_range(r.get("reference_range")),
                        "group": series_name,
                    })
            if all_results:
                normalized["results"] = all_results

        # Handle reference ranges
        ref_ranges = test.get("reference_ranges")
        if ref_ranges and isinstance(ref_ranges, dict):
            normalized["reference_range"] = self._format_ref_range(ref_ranges)

        # Handle timestamps
        timestamps = test.get("timestamps", {})
        if timestamps:
            normalized["collected_date"] = timestamps.get("received_collected")
            released = timestamps.get("released", "")
            # Extract just the date part if datetime
            if released and " " in released:
                normalized["released_date"] = released.split(" ")[0]
            else:
                normalized["released_date"] = released

        # Handle EKG-specific fields
        if "parameters" in test and isinstance(test["parameters"], dict):
            # This is an EKG exam
            normalized["parameters"] = [
                {"name": k, "value": v}
                for k, v in test["parameters"].items()
            ]
            normalized["conclusion"] = test.get("conclusion")
            normalized["morphological_description_and_comment"] = test.get("morphological_description_comment")
            normalized["indication"] = test.get("indication")
            normalized["medication"] = test.get("medication")
            if test.get("lauded_by"):
                normalized["laudado_by"] = test["lauded_by"]

        return normalized

    def _format_ref_range(self, ref_range: dict | None) -> str | None:
        """Format reference range dict to string."""
        if not ref_range or not isinstance(ref_range, dict):
            return None
        min_val = ref_range.get("min")
        max_val = ref_range.get("max")
        if min_val is not None and max_val is not None:
            return f"{min_val} - {max_val}"
        elif min_val is not None:
            return f"> {min_val}"
        elif max_val is not None:
            return f"< {max_val}"
        return None

    def format_individual_exam_proto(
        self,
        merged_data: dict,
        output_dir: str | Path,
        use_ai_classification: bool = False,
    ) -> list[LabTestResult]:
        """
        Convert merged patient data into individual exam files in proto-compatible format.

        Decomposes the merged dict into separate exam files following the LabTestResult
        proto message schema. Each exam is saved as a separate JSON in the output directory.

        Sources of exams in merged data:
        - exam_results[]: individual lab/diagnostic tests (blood, chemistry, etc.)
        - polysomnography_results: sleep study exam
        - exam_metadata + findings/conclusion: imaging exams (MRI, CT, X-ray, etc.)
        - electrocardiogram in exam_results: EKG exam

        Args:
            merged_data: Merged patient data dict from merge_results()
            output_dir: Directory where individual exam JSONs will be saved
            use_ai_classification: Whether to use pydantic-ai for semantic classification

        Returns:
            List of LabTestResult models that were saved
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exams: list[LabTestResult] = []
        transformer = ProtoTransformer(use_ai_classification=use_ai_classification)

        # Extract patient info for all exams
        patient_info = merged_data.get("patient_info") or merged_data.get("patient_information", {})
        patient_id = transformer.generate_patient_id(patient_info)
        facility_info = merged_data.get("facility_info") or merged_data.get("facility_information", {})

        # 1. Process exam_results[] and tests[] - individual lab tests
        # Separate blood exams from other exams for grouping
        # Check both "exam_results" and "tests" keys (different extraction formats)
        exam_results = merged_data.get("exam_results", []) + merged_data.get("tests", [])
        blood_exams_by_date: dict[str, list[dict]] = {}
        other_exams: list[dict] = []

        for exam_result in exam_results:
            if not exam_result:
                continue

            # Normalize different data formats
            # "tests" format uses "name" instead of "exam_type"
            exam_type_raw = exam_result.get("exam_type") or exam_result.get("name", "")
            normalized_type = transformer.normalize_exam_type(exam_type_raw)

            # Normalize collection date (different formats)
            collected_date = (
                exam_result.get("collected_date")
                or (exam_result.get("timestamps", {}).get("received_collected"))
                or "unknown"
            )

            # Normalize the exam_result structure for transformer compatibility
            normalized_result = self._normalize_test_format(exam_result)

            if normalized_type == "blood":
                # Group blood exams by collection date
                if collected_date not in blood_exams_by_date:
                    blood_exams_by_date[collected_date] = []
                blood_exams_by_date[collected_date].append(normalized_result)
            else:
                other_exams.append((normalized_result, normalized_type))

        # Process grouped blood exams - one LabTestResult per collection date
        for collected_date, blood_exam_group in blood_exams_by_date.items():
            exam = transformer.transform_blood_exams_grouped(
                blood_exam_group, patient_id, facility_info
            )
            exams.append(exam)

        # Process other (non-blood) exams individually
        for exam_result, normalized_type in other_exams:
            if normalized_type == "ekg":
                exam = transformer.transform_ekg_exam(exam_result, patient_id, facility_info)
            else:
                exam = transformer.transform_generic_exam(exam_result, patient_id, facility_info)

            exams.append(exam)

        # 2. Process polysomnography_results - sleep study
        polysom = merged_data.get("polysomnography_results")
        if polysom:
            exam = transformer.transform_polysomnography_exam(
                polysom,
                merged_data.get("conclusions", []),
                patient_id,
                facility_info,
                merged_data.get("exam_metadata", {}),
            )
            exams.append(exam)

        # 3. Process imaging exams (MRI, CT, X-ray, etc.) from exam_metadata/exam_details
        # Check BOTH keys since they may contain different exams
        imaging_sources = [
            (merged_data.get("exam_metadata", {}), merged_data.get("findings", []), merged_data.get("conclusion", [])),
            (merged_data.get("exam_details", {}), merged_data.get("analysis", []), merged_data.get("opinion", [])),
        ]

        for exam_meta, findings, conclusion in imaging_sources:
            if not exam_meta:
                continue

            exam_type_raw = exam_meta.get("exam_type", "")
            if not exam_type_raw:
                continue

            normalized_type = transformer.normalize_exam_type(exam_type_raw)

            if normalized_type == "mri":
                exam = transformer.transform_mri_exam(
                    exam_meta, findings, conclusion, patient_id, facility_info
                )
                exams.append(exam)
            elif normalized_type == "xray":
                exam = transformer.transform_xray_exam(
                    exam_meta, findings, conclusion, patient_id, facility_info
                )
                exams.append(exam)
            elif normalized_type == "ct_scan":
                exam = transformer.transform_ct_exam(
                    exam_meta, findings, conclusion, patient_id, facility_info
                )
                exams.append(exam)

        # 4. Process imaging_studies array if present
        imaging_studies = merged_data.get("imaging_studies", [])
        for study in imaging_studies:
            if not study:
                continue

            study_type = study.get("exam_type", "")
            normalized_type = transformer.normalize_exam_type(study_type)

            study_meta = {
                "exam_type": study_type,
                "specific_exam": study.get("exam_name") or study.get("body_part"),
                "exam_title": study.get("exam_name"),
                "record_number": study.get("record_number"),
                "exam_date": study.get("exam_date"),
                "release_date": study.get("reported_date"),
                "signed_by": study.get("reported_by"),
            }
            study_findings = study.get("findings", [])
            study_conclusion = study.get("conclusion", [])

            if normalized_type == "mri":
                exam = transformer.transform_mri_exam(
                    study_meta, study_findings, study_conclusion, patient_id, facility_info
                )
                exams.append(exam)
            elif normalized_type == "xray":
                exam = transformer.transform_xray_exam(
                    study_meta, study_findings, study_conclusion, patient_id, facility_info
                )
                exams.append(exam)
            elif normalized_type == "ct_scan":
                exam = transformer.transform_ct_exam(
                    study_meta, study_findings, study_conclusion, patient_id, facility_info
                )
                exams.append(exam)

        # Save each exam to a separate file in proto-compatible JSON format
        for exam in exams:
            exam_id = exam.id
            exam_type = exam.exam_type
            exam_code = (exam.exam_code or "unknown").lower().replace(" ", "_")[:20]
            filename = f"{exam_type}_{exam_code}_{exam_id[:8]}.json"
            output_path = output_dir / filename

            # Save as proto-compatible JSON
            proto_dict = exam.model_dump_proto()
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(proto_dict, f, ensure_ascii=False, indent=2)

        return exams
