"""Transformer module for converting extracted medical data to proto format.

Uses pydantic-ai for semantic transformations when needed.
"""

import os
import uuid
import hashlib
from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import Agent

from .proto_models import (
    LabTestResult,
    LabTestStatus,
    LabPriority,
    Address,
    BloodExam,
    Analyte,
    ReferenceRange,
    MRIExam,
    MRIFinding,
    XRayExam,
    XRayFinding,
    CTScanExam,
    CTFinding,
    PolysomnographyExam,
    Duration,
    SleepEfficiency,
    SleepDistribution,
    SleepStage,
    SleepFragmentation,
    Microarousals,
    CompleteArousals,
    RespiratoryEvents,
    ObstructiveApneas,
    Hypopneas,
    ApneaHypopneaIndex,
    OxygenSaturation,
    OxygenSaturationValue,
    HeartRate,
    HeartRateValue,
    EpworthSleepinessScale,
    GenericExamDetails,
)


class ExamTypeClassification(BaseModel):
    """Classification result for exam type."""
    exam_type: str  # blood, mri, xray, ct_scan, polysomnography, ekg, generic
    exam_code: str | None = None  # LOINC or other standard code
    exam_name: str
    confidence: float


class SeverityAssessment(BaseModel):
    """Severity assessment for exam results."""
    severity_flag: str  # Normal, Abnormal, Critical
    interpretation: str | None = None
    confidence: float


# Create pydantic-ai agent for semantic transformations
_exam_classifier_agent: Agent | None = None
_severity_agent: Agent | None = None


def _get_exam_classifier_agent() -> Agent:
    """Get or create the exam classification agent."""
    global _exam_classifier_agent
    if _exam_classifier_agent is None:
        _exam_classifier_agent = Agent(
            model="openai:gpt-4o-mini",
            result_type=ExamTypeClassification,
            system_prompt="""You are a medical exam classification expert.
Given exam information, classify it into one of these types:
- blood: Blood tests, chemistry panels, CBC, coagulation tests
- mri: MRI scans, magnetic resonance imaging
- xray: X-ray imaging
- ct_scan: CT scans, computed tomography
- polysomnography: Sleep studies
- ekg: Electrocardiograms, ECG
- generic: Other exam types

Provide a LOINC code if applicable, and a standardized exam name.
Set confidence between 0 and 1 based on how certain you are.""",
        )
    return _exam_classifier_agent


def _get_severity_agent() -> Agent:
    """Get or create the severity assessment agent."""
    global _severity_agent
    if _severity_agent is None:
        _severity_agent = Agent(
            model="openai:gpt-4o-mini",
            result_type=SeverityAssessment,
            system_prompt="""You are a medical severity assessment expert.
Given exam results and reference ranges, assess the overall severity:
- Normal: All values within reference ranges, no concerning findings
- Abnormal: Some values outside reference ranges or minor findings
- Critical: Values significantly outside ranges or concerning findings

Provide a brief interpretation and confidence level (0-1).""",
        )
    return _severity_agent


class ProtoTransformer:
    """Transforms extracted medical exam data to LabTestResult proto format."""

    def __init__(self, use_ai_classification: bool = True):
        """
        Initialize the transformer.

        Args:
            use_ai_classification: Whether to use pydantic-ai for semantic transformations.
                                   If False, uses rule-based classification only.
        """
        self.use_ai_classification = use_ai_classification

    def generate_patient_id(self, patient_info: dict) -> str:
        """Generate a patient ID from patient info or create a new UUID."""
        if not patient_info:
            return f"PAT-{uuid.uuid4().hex[:12].upper()}"

        name = patient_info.get("name", "")
        dob = patient_info.get("date_of_birth", "")
        if name and dob:
            hash_input = f"{name}_{dob}".encode()
            return f"PAT-{hashlib.sha256(hash_input).hexdigest()[:12].upper()}"

        return f"PAT-{uuid.uuid4().hex[:12].upper()}"

    def normalize_exam_type(self, exam_type_raw: str) -> str:
        """Normalize exam type to standard categories."""
        exam_type_upper = exam_type_raw.upper()

        # Blood/chemistry tests
        blood_tests = {
            "CREATININA", "POTASSIO", "SODIO", "UREIA", "GLICOSE",
            "HEMOGRAMA", "TEMPO DE PROTROMBINA", "TEMPO DE TROMBOPLASTINA",
            "CBC", "BLOOD", "CHEMISTRY", "BIOCHEMISTRY", "COAGULATION",
            # Hormones and metabolic tests
            "INSULINA", "HEMOGLOBINA GLICADA", "A1C",
            "TSH", "TIROESTIMULANTE", "T3", "T4", "TIROXINA", "TRIIODOTIRONINA",
            "TESTOSTERONA", "CORTISOL", "SHBG", "GLOBULINA LIGADORA",
            # Iron studies
            "FERRO", "FERRITINA",
            # Lipid panel
            "COLESTEROL", "TRIGLICERID", "HDL", "LDL", "VLDL",
            # Vitamins and minerals
            "VITAMINA", "ACIDO URICO", "ACIDO FOLICO", "ACIDO ASCORBICO",
            "HOMOCISTE", "CALCIO", "ZINCO", "MAGNESIO",
            # Liver and pancreatic enzymes
            "BILIRRUBINA", "TGO", "TGP", "FOSFATASE", "AMILASE", "LIPASE",
            # Proteins
            "PROTEINA", "ALBUMINA", "GLOBULINA",
            # Other
            "HIDROXI", "B-12", "B12",
        }
        if any(bt in exam_type_upper for bt in blood_tests):
            return "blood"

        # Imaging
        if "MRI" in exam_type_upper or "RESSONANCIA" in exam_type_upper or "MAGNETIC" in exam_type_upper:
            return "mri"
        if "CT" in exam_type_upper or "TOMOGRAFIA" in exam_type_upper or "COMPUTED" in exam_type_upper:
            return "ct_scan"
        if "RAIO" in exam_type_upper or "X-RAY" in exam_type_upper or "XRAY" in exam_type_upper:
            return "xray"
        if "ULTRASSOM" in exam_type_upper or "ULTRASOUND" in exam_type_upper:
            return "ultrasound"

        # EKG
        if "ELETROCARDIOGRAMA" in exam_type_upper or "ECG" in exam_type_upper or "EKG" in exam_type_upper:
            return "ekg"

        # Sleep study
        if "POLISSONOGRAFIA" in exam_type_upper or "POLYSOMNOGRAPHY" in exam_type_upper or "SLEEP" in exam_type_upper:
            return "polysomnography"

        return "generic"

    async def classify_exam_ai(self, exam_data: dict) -> ExamTypeClassification:
        """Use AI to classify exam type when rule-based classification is uncertain."""
        agent = _get_exam_classifier_agent()
        exam_desc = f"""
Exam type: {exam_data.get('exam_type', 'unknown')}
Test name: {exam_data.get('test_name', exam_data.get('exam_name', 'unknown'))}
Method: {exam_data.get('method', 'unknown')}
Results available: {bool(exam_data.get('result') or exam_data.get('results'))}
"""
        result = await agent.run(exam_desc)
        return result.data

    async def assess_severity_ai(self, exam_data: dict, measurements: list[dict]) -> SeverityAssessment:
        """Use AI to assess the severity of exam results."""
        agent = _get_severity_agent()
        measurements_desc = "\n".join([
            f"- {m.get('name')}: {m.get('value')} {m.get('unit', '')} (ref: {m.get('reference_range', 'N/A')})"
            for m in measurements[:20]  # Limit to first 20 measurements
        ])
        result = await agent.run(f"Exam results:\n{measurements_desc}")
        return result.data

    def format_date(self, date_str: str | None) -> str | None:
        """Format date string to ISO format."""
        if not date_str:
            return None
        # Already in ISO format (YYYY-MM-DD)
        if len(date_str) == 10 and date_str[4] == "-":
            return f"{date_str}T00:00:00Z"
        return date_str

    def _create_address(self, facility_info: dict) -> Address | None:
        """Create an Address from facility info."""
        if not facility_info:
            return None
        return Address(
            name=facility_info.get("name"),
            street=facility_info.get("address"),
            city=facility_info.get("city"),
            state=facility_info.get("state"),
            postal_code=facility_info.get("postal_code"),
            country=facility_info.get("country", "Brazil"),
        )

    def _parse_reference_range(self, ref_range_str: str | None) -> ReferenceRange | None:
        """Parse a reference range string into ReferenceRange model."""
        if not ref_range_str:
            return None

        try:
            # Handle formats like "3.5 - 5.5", "3.5-5.5", "> 60", "< 100"
            ref_str = str(ref_range_str).strip()

            if " - " in ref_str:
                parts = ref_str.split(" - ")
                low = float(parts[0].strip().replace(",", "."))
                high = float(parts[1].strip().replace(",", "."))
                return ReferenceRange(low=low, high=high)
            elif "-" in ref_str and not ref_str.startswith("-"):
                parts = ref_str.split("-")
                low = float(parts[0].strip().replace(",", "."))
                high = float(parts[1].strip().replace(",", "."))
                return ReferenceRange(low=low, high=high)
            elif ref_str.startswith(">"):
                val = float(ref_str[1:].strip().replace(",", "."))
                return ReferenceRange(low=val)
            elif ref_str.startswith("<"):
                val = float(ref_str[1:].strip().replace(",", "."))
                return ReferenceRange(high=val)
        except (ValueError, IndexError):
            pass

        return None

    def _determine_flag(self, value: float | None, ref_range: ReferenceRange | None) -> str | None:
        """Determine the result flag based on value and reference range."""
        if value is None or ref_range is None:
            return None

        if ref_range.low is not None and ref_range.high is not None:
            if value < ref_range.low:
                return "low" if value >= ref_range.low * 0.7 else "critical_low"
            elif value > ref_range.high:
                return "high" if value <= ref_range.high * 1.3 else "critical_high"
            else:
                return "normal"
        elif ref_range.low is not None:
            return "low" if value < ref_range.low else "normal"
        elif ref_range.high is not None:
            return "high" if value > ref_range.high else "normal"

        return None

    def _extract_analytes_from_exam(self, exam_result: dict) -> list[Analyte]:
        """Extract analytes from a single exam result dict."""
        analytes = []
        exam_type_raw = exam_result.get("exam_type", "")

        # Handle single result
        if "result" in exam_result:
            try:
                value = float(str(exam_result["result"]).replace(",", "."))
            except (ValueError, TypeError):
                value = None

            ref_range = self._parse_reference_range(exam_result.get("reference_range"))
            flag = self._determine_flag(value, ref_range)

            analytes.append(Analyte(
                name=exam_result.get("test_name", exam_type_raw),
                value=value,
                unit=exam_result.get("unit"),
                reference_range=ref_range,
                flag=flag,
            ))

        # Handle multiple results
        elif "results" in exam_result:
            results = exam_result["results"]
            if isinstance(results, list):
                for r in results:
                    try:
                        value = float(str(r.get("value", "")).replace(",", "."))
                    except (ValueError, TypeError):
                        value = None

                    ref_range = self._parse_reference_range(r.get("reference_range"))
                    flag = self._determine_flag(value, ref_range)

                    analytes.append(Analyte(
                        name=r.get("name"),
                        value=value,
                        unit=r.get("unit"),
                        reference_range=ref_range,
                        flag=flag,
                    ))
            elif isinstance(results, dict):
                # Handle hemogram-style nested structure
                for group_name, group_items in results.items():
                    if isinstance(group_items, list) and group_items:
                        if isinstance(group_items[0], dict):
                            for item in group_items:
                                try:
                                    value = float(str(item.get("value", "")).replace(",", "."))
                                except (ValueError, TypeError):
                                    value = None

                                ref_range = self._parse_reference_range(item.get("reference_range"))
                                flag = self._determine_flag(value, ref_range)

                                analytes.append(Analyte(
                                    code=f"{group_name}:{item.get('name', '')}",
                                    name=item.get("name"),
                                    value=value,
                                    unit=item.get("unit"),
                                    reference_range=ref_range,
                                    flag=flag,
                                ))

        return analytes

    def transform_blood_exam(
        self,
        exam_result: dict,
        patient_id: str,
        facility_info: dict,
    ) -> LabTestResult:
        """Transform a blood/lab exam to LabTestResult proto format."""
        exam_type_raw = exam_result.get("exam_type", "")
        exam_id = str(uuid.uuid4())

        # Build analytes list
        analytes = self._extract_analytes_from_exam(exam_result)

        # Determine overall severity
        severity_flag = "Normal"
        for a in analytes:
            if a.flag and "critical" in a.flag:
                severity_flag = "Critical"
                break
            elif a.flag in ("low", "high"):
                severity_flag = "Abnormal"

        blood_exam = BloodExam(
            sample_type=exam_result.get("sample_type", "venous_blood"),
            lab_panel_code=exam_type_raw,
            analytes=analytes,
        )

        return LabTestResult(
            id=exam_id,
            patient_id=patient_id,
            exam_type="blood",
            exam_code=exam_type_raw,
            exam_name=exam_result.get("test_name", exam_type_raw),
            collected_at=self.format_date(exam_result.get("collected_date")),
            reported_at=exam_result.get("released_date"),
            status=LabTestStatus.FINAL,
            priority=LabPriority.ROUTINE,
            location=self._create_address(facility_info),
            severity_flag=severity_flag,
            json=blood_exam,
        )

    def transform_blood_exams_grouped(
        self,
        exam_results: list[dict],
        patient_id: str,
        facility_info: dict,
    ) -> LabTestResult:
        """
        Transform multiple blood exam results into a single LabTestResult.

        Groups all analytes from multiple blood tests (e.g., Creatinine, Potassium,
        Sodium, CBC) collected on the same date into one consolidated blood exam.

        Args:
            exam_results: List of blood exam result dicts to combine
            patient_id: Patient identifier
            facility_info: Facility information dict

        Returns:
            Single LabTestResult containing all analytes from the grouped exams
        """
        if not exam_results:
            raise ValueError("exam_results cannot be empty")

        exam_id = str(uuid.uuid4())

        # Collect all analytes from all exam results
        all_analytes = []
        panel_codes = []
        test_names = []

        for exam_result in exam_results:
            analytes = self._extract_analytes_from_exam(exam_result)
            all_analytes.extend(analytes)

            exam_type_raw = exam_result.get("exam_type", "")
            if exam_type_raw:
                panel_codes.append(exam_type_raw)

            test_name = exam_result.get("test_name", "")
            if test_name:
                test_names.append(test_name)

        # Use the first exam's dates (they should all be the same for grouped exams)
        first_exam = exam_results[0]
        collected_date = first_exam.get("collected_date")
        released_date = first_exam.get("released_date")

        # Determine overall severity from all analytes
        severity_flag = "Normal"
        for a in all_analytes:
            if a.flag and "critical" in a.flag:
                severity_flag = "Critical"
                break
            elif a.flag in ("low", "high"):
                severity_flag = "Abnormal"

        # Create combined panel code and name
        combined_panel_code = "blood_panel" if len(panel_codes) > 1 else (panel_codes[0] if panel_codes else "blood")
        combined_exam_name = "Blood Panel" if len(test_names) > 1 else (test_names[0] if test_names else "Blood Exam")

        blood_exam = BloodExam(
            sample_type=first_exam.get("sample_type", "venous_blood"),
            lab_panel_code=combined_panel_code,
            analytes=all_analytes,
        )

        return LabTestResult(
            id=exam_id,
            patient_id=patient_id,
            exam_type="blood",
            exam_code=combined_panel_code,
            exam_name=combined_exam_name,
            collected_at=self.format_date(collected_date),
            reported_at=released_date,
            status=LabTestStatus.FINAL,
            priority=LabPriority.ROUTINE,
            location=self._create_address(facility_info),
            severity_flag=severity_flag,
            json=blood_exam,
        )

    def transform_mri_exam(
        self,
        exam_metadata: dict,
        findings: list,
        conclusion: list,
        patient_id: str,
        facility_info: dict,
    ) -> LabTestResult:
        """Transform an MRI exam to LabTestResult proto format."""
        exam_id = str(uuid.uuid4())

        mri_findings = []
        for f in findings:
            if isinstance(f, dict):
                mri_findings.append(MRIFinding(
                    title=f.get("structure", f.get("title")),
                    description=f.get("description", f.get("status")),
                    severity=f.get("severity"),
                ))
            elif isinstance(f, str):
                mri_findings.append(MRIFinding(description=f))

        # Handle alternative field names from different extraction formats
        record_number = exam_metadata.get("record_number") or exam_metadata.get("ficha_number")
        release_date = exam_metadata.get("release_date") or exam_metadata.get("liberated_on", "").split(" ")[0] if exam_metadata.get("liberated_on") else ""
        release_time = exam_metadata.get("release_time") or (exam_metadata.get("liberated_on", "").split(" ")[1] if " " in exam_metadata.get("liberated_on", "") else "")
        reported_at = f"{release_date} {release_time}".strip() or None

        mri_exam = MRIExam(
            body_region=exam_metadata.get("specific_exam") or exam_metadata.get("body_part") or exam_metadata.get("exam_title"),
            findings=mri_findings,
            impression=conclusion if isinstance(conclusion, list) else [conclusion] if conclusion else [],
            report_signed_by=exam_metadata.get("signed_by", {}).get("name") if isinstance(exam_metadata.get("signed_by"), dict) else exam_metadata.get("signed_by"),
            report_signed_at=reported_at,
        )

        return LabTestResult(
            id=exam_id,
            patient_id=patient_id,
            exam_type="mri",
            exam_code=record_number,
            exam_name=exam_metadata.get("exam_title") or exam_metadata.get("specific_exam"),
            collected_at=self.format_date(exam_metadata.get("exam_date")),
            reported_at=reported_at,
            status=LabTestStatus.FINAL,
            priority=LabPriority.ROUTINE,
            location=self._create_address(facility_info),
            result_summary="; ".join(conclusion) if conclusion else None,
            mri_json=mri_exam,
        )

    def transform_xray_exam(
        self,
        exam_metadata: dict,
        findings: list,
        conclusion: list,
        patient_id: str,
        facility_info: dict,
    ) -> LabTestResult:
        """Transform an X-ray exam to LabTestResult proto format."""
        exam_id = str(uuid.uuid4())

        xray_findings = []
        for f in findings:
            if isinstance(f, dict):
                xray_findings.append(XRayFinding(
                    title=f.get("structure", f.get("title")),
                    description=f.get("description", f.get("status")),
                    severity=f.get("severity"),
                ))
            elif isinstance(f, str):
                xray_findings.append(XRayFinding(description=f))

        xray_exam = XRayExam(
            findings=xray_findings,
            impression=conclusion if isinstance(conclusion, list) else [conclusion] if conclusion else [],
            report_signed_by=exam_metadata.get("signed_by", {}).get("name") if isinstance(exam_metadata.get("signed_by"), dict) else exam_metadata.get("signed_by"),
            report_signed_at=f"{exam_metadata.get('release_date', '')} {exam_metadata.get('release_time', '')}".strip() or None,
        )

        return LabTestResult(
            id=exam_id,
            patient_id=patient_id,
            exam_type="xray",
            exam_code=exam_metadata.get("record_number"),
            exam_name=exam_metadata.get("exam_title") or exam_metadata.get("specific_exam"),
            collected_at=self.format_date(exam_metadata.get("exam_date")),
            reported_at=f"{exam_metadata.get('release_date', '')} {exam_metadata.get('release_time', '')}".strip() or None,
            status=LabTestStatus.FINAL,
            priority=LabPriority.ROUTINE,
            location=self._create_address(facility_info),
            result_summary="; ".join(conclusion) if conclusion else None,
            xray_json=xray_exam,
        )

    def transform_ct_exam(
        self,
        exam_metadata: dict,
        findings: list,
        conclusion: list,
        patient_id: str,
        facility_info: dict,
    ) -> LabTestResult:
        """Transform a CT scan exam to LabTestResult proto format."""
        exam_id = str(uuid.uuid4())

        ct_findings = []
        for f in findings:
            if isinstance(f, dict):
                ct_findings.append(CTFinding(
                    title=f.get("structure", f.get("title")),
                    description=f.get("description", f.get("status")),
                    severity=f.get("severity"),
                ))
            elif isinstance(f, str):
                ct_findings.append(CTFinding(description=f))

        ct_exam = CTScanExam(
            modality="CT",
            findings=ct_findings,
            impression=conclusion if isinstance(conclusion, list) else [conclusion] if conclusion else [],
        )

        return LabTestResult(
            id=exam_id,
            patient_id=patient_id,
            exam_type="ct_scan",
            exam_code=exam_metadata.get("record_number"),
            exam_name=exam_metadata.get("exam_title") or exam_metadata.get("specific_exam"),
            collected_at=self.format_date(exam_metadata.get("exam_date")),
            reported_at=f"{exam_metadata.get('release_date', '')} {exam_metadata.get('release_time', '')}".strip() or None,
            status=LabTestStatus.FINAL,
            priority=LabPriority.ROUTINE,
            location=self._create_address(facility_info),
            result_summary="; ".join(conclusion) if conclusion else None,
            ct_scan_json=ct_exam,
        )

    def transform_polysomnography_exam(
        self,
        polysom: dict,
        conclusions: list,
        patient_id: str,
        facility_info: dict,
        exam_metadata: dict,
    ) -> LabTestResult:
        """Transform a polysomnography exam to LabTestResult proto format."""
        exam_id = str(uuid.uuid4())

        # Build sleep distribution
        sleep_dist_data = polysom.get("sleep_distribution", {})
        sleep_distribution = None
        if sleep_dist_data and isinstance(sleep_dist_data, dict):
            n1 = sleep_dist_data.get("n1")
            n2 = sleep_dist_data.get("n2")
            n3 = sleep_dist_data.get("n3")
            rem = sleep_dist_data.get("rem")

            sleep_distribution = SleepDistribution(
                stage_1_n1=SleepStage(
                    value=n1.get("value") if isinstance(n1, dict) else n1,
                    unit=n1.get("unit", "%") if isinstance(n1, dict) else "%",
                ) if n1 else None,
                stage_2_n2=SleepStage(
                    value=n2.get("value") if isinstance(n2, dict) else n2,
                    unit=n2.get("unit", "%") if isinstance(n2, dict) else "%",
                ) if n2 else None,
                stage_3_n3=SleepStage(
                    value=n3.get("value") if isinstance(n3, dict) else n3,
                    unit=n3.get("unit", "%") if isinstance(n3, dict) else "%",
                ) if n3 else None,
                rem_sleep=SleepStage(
                    value=rem.get("value") if isinstance(rem, dict) else rem,
                    unit=rem.get("unit", "%") if isinstance(rem, dict) else "%",
                ) if rem else None,
            )

        # Build sleep fragmentation
        frag_data = polysom.get("sleep_fragmentation", {})
        sleep_fragmentation = None
        if frag_data and isinstance(frag_data, dict):
            micro_data = frag_data.get("microarousals")
            complete_data = frag_data.get("complete_arousals")

            sleep_fragmentation = SleepFragmentation(
                microarousals=Microarousals(
                    count=micro_data.get("count") if isinstance(micro_data, dict) else micro_data,
                    rate=micro_data.get("rate") if isinstance(micro_data, dict) else None,
                ) if micro_data else None,
                complete_arousals=CompleteArousals(
                    count=complete_data.get("count") if isinstance(complete_data, dict) else complete_data,
                ) if complete_data else None,
            )

        # Build respiratory events
        resp_data = polysom.get("respiratory_events_during_sleep", {})
        respiratory_events = None
        if resp_data and isinstance(resp_data, dict):
            obs_apneas = resp_data.get("obstructive_apneas")
            hypop = resp_data.get("hypopneas")

            respiratory_events = RespiratoryEvents(
                obstructive_apneas=ObstructiveApneas(
                    count=obs_apneas.get("count") if isinstance(obs_apneas, dict) else obs_apneas,
                ) if obs_apneas else None,
                hypopneas=Hypopneas(
                    count=hypop.get("count") if isinstance(hypop, dict) else hypop,
                ) if hypop else None,
            )

        # Build AHI
        ahi_data = polysom.get("apnea_hypopnea_index", {})
        ahi = None
        if ahi_data:
            if isinstance(ahi_data, dict):
                ahi = ApneaHypopneaIndex(
                    value=ahi_data.get("value"),
                    unit=ahi_data.get("unit", "events/hour"),
                    apnea_per_hour=ahi_data.get("apnea_per_hour"),
                    hypopnea_per_hour=ahi_data.get("hypopnea_per_hour"),
                )
            else:
                ahi = ApneaHypopneaIndex(
                    value=float(ahi_data),
                    unit="events/hour",
                )

        # Build oxygen saturation
        sao2_data = polysom.get("oxygen_saturation_SaO2", {})
        oxygen_saturation = None
        if sao2_data and isinstance(sao2_data, dict):
            wake_val = sao2_data.get("wakefulness")
            avg_val = sao2_data.get("average")
            min_val = sao2_data.get("minimum")

            oxygen_saturation = OxygenSaturation(
                wakefulness=OxygenSaturationValue(
                    value=wake_val.get("value") if isinstance(wake_val, dict) else wake_val,
                    unit="%",
                ) if wake_val else None,
                average=OxygenSaturationValue(
                    value=avg_val.get("value") if isinstance(avg_val, dict) else avg_val,
                    unit="%",
                ) if avg_val else None,
                minimum=OxygenSaturationValue(
                    value=min_val.get("value") if isinstance(min_val, dict) else min_val,
                    unit="%",
                ) if min_val else None,
            )

        # Build heart rate
        hr_data = polysom.get("heart_rate", polysom.get("average_heart_rate_bpm"))
        heart_rate = None
        if hr_data:
            if isinstance(hr_data, dict):
                heart_rate = HeartRate(
                    average=HeartRateValue(
                        value=hr_data.get("value"),
                        unit=hr_data.get("unit", "bpm"),
                    ),
                    arrhythmias=polysom.get("cardiac_arrhythmias"),
                )
            else:
                heart_rate = HeartRate(
                    average=HeartRateValue(value=float(hr_data), unit="bpm"),
                    arrhythmias=polysom.get("cardiac_arrhythmias"),
                )

        # Build epworth scale
        epworth = polysom.get("epworth_sleepiness_scale")
        epworth_scale = None
        if epworth:
            if isinstance(epworth, dict):
                epworth_scale = EpworthSleepinessScale(value=epworth.get("value"))
            else:
                epworth_scale = EpworthSleepinessScale(value=int(epworth))

        # Handle electrophysiological_parameters which can be a list or string
        electro_params = polysom.get("electrophysiological_parameters")
        if isinstance(electro_params, list):
            electro_params = "; ".join(str(p) for p in electro_params)

        polysom_exam = PolysomnographyExam(
            equipment=polysom.get("monitoring_equipment"),
            monitoring_duration=Duration(
                value=polysom.get("monitoring_duration_minutes"),
                unit="minutes",
            ) if polysom.get("monitoring_duration_minutes") else None,
            start_time=polysom.get("monitoring_start_time"),
            end_time=polysom.get("monitoring_end_time"),
            electrophysiological_parameters=electro_params,
            total_registration_time=Duration(
                value=polysom.get("total_recording_time_minutes"),
                unit="minutes",
            ) if polysom.get("total_recording_time_minutes") else None,
            total_sleep_time=Duration(
                value=polysom.get("total_sleep_time_minutes"),
                unit="minutes",
            ) if polysom.get("total_sleep_time_minutes") else None,
            sleep_efficiency=SleepEfficiency(
                value=polysom.get("sleep_efficiency_percent"),
                unit="%",
            ) if polysom.get("sleep_efficiency_percent") else None,
            sleep_distribution=sleep_distribution,
            sleep_latency=Duration(
                value=polysom.get("sleep_latency_minutes"),
                unit="minutes",
            ) if polysom.get("sleep_latency_minutes") else None,
            rem_sleep_latency=Duration(
                value=polysom.get("rem_sleep_latency_minutes"),
                unit="minutes",
            ) if polysom.get("rem_sleep_latency_minutes") else None,
            wakefulness_during_total_sleep_time=Duration(
                value=polysom.get("wake_time_during_sleep_minutes"),
                unit="minutes",
            ) if polysom.get("wake_time_during_sleep_minutes") else None,
            sleep_fragmentation=sleep_fragmentation,
            periodic_limb_movements=polysom.get("periodic_limb_movements"),
            respiratory_events_during_sleep=respiratory_events,
            apnea_hypopnea_index_ahi=ahi,
            oxygen_saturation_sao2=oxygen_saturation,
            snoring=polysom.get("snoring"),
            body_position=polysom.get("event_position"),
            heart_rate=heart_rate,
            epworth_sleepiness_scale=epworth_scale,
            conclusions=conclusions if isinstance(conclusions, list) else [conclusions] if conclusions else [],
        )

        return LabTestResult(
            id=exam_id,
            patient_id=patient_id,
            exam_type="polysomnography",
            exam_code="PSG",
            exam_name="Polysomnography (Sleep Study)",
            collected_at=exam_metadata.get("exam_date"),
            status=LabTestStatus.FINAL,
            priority=LabPriority.ROUTINE,
            location=self._create_address(facility_info) or Address(name=polysom.get("monitoring_location")),
            result_summary="; ".join(conclusions) if conclusions else None,
            polysomnography_json=polysom_exam,
        )

    def transform_ekg_exam(
        self,
        exam_result: dict,
        patient_id: str,
        facility_info: dict,
    ) -> LabTestResult:
        """Transform an EKG exam to LabTestResult proto format."""
        exam_id = str(uuid.uuid4())

        # Build generic details for EKG
        fields = {}
        notes = []

        parameters = exam_result.get("parameters", [])
        if parameters:
            for param in parameters:
                if isinstance(param, dict):
                    name = param.get("name", "")
                    value = param.get("value", "")
                    unit = param.get("unit", "")
                    fields[name] = f"{value} {unit}".strip()
                elif isinstance(param, str):
                    notes.append(param)

        if exam_result.get("morphological_description_and_comment"):
            notes.append(f"Morphological description: {exam_result['morphological_description_and_comment']}")
        if exam_result.get("conclusion"):
            notes.append(f"Conclusion: {exam_result['conclusion']}")
        if exam_result.get("observations"):
            notes.append(f"Observations: {exam_result['observations']}")

        generic_details = GenericExamDetails(
            fields=fields,
            notes=notes,
        )

        return LabTestResult(
            id=exam_id,
            patient_id=patient_id,
            exam_type="ekg",
            exam_code="ECG",
            exam_name=exam_result.get("test_name", "Electrocardiogram"),
            indication=exam_result.get("indication"),
            clinical_notes=exam_result.get("medication"),
            collected_at=self.format_date(exam_result.get("collected_date")),
            reported_at=exam_result.get("released_date"),
            status=LabTestStatus.FINAL,
            priority=LabPriority.ROUTINE,
            location=self._create_address(facility_info),
            result_summary=exam_result.get("conclusion"),
            generic_json=generic_details,
        )

    def transform_generic_exam(
        self,
        exam_result: dict,
        patient_id: str,
        facility_info: dict,
    ) -> LabTestResult:
        """Transform a generic/unknown exam to LabTestResult proto format."""
        exam_id = str(uuid.uuid4())
        exam_type_raw = exam_result.get("exam_type", "unknown")

        fields = {}
        notes = []

        # Extract any structured data as fields
        for key, value in exam_result.items():
            if key in ("exam_type", "test_name", "collected_date", "released_date"):
                continue
            if isinstance(value, (str, int, float)):
                fields[key] = str(value)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, str):
                        notes.append(item)
                    elif isinstance(item, dict):
                        for k, v in item.items():
                            fields[f"{key}_{i}_{k}"] = str(v)

        generic_details = GenericExamDetails(
            fields=fields,
            notes=notes,
        )

        return LabTestResult(
            id=exam_id,
            patient_id=patient_id,
            exam_type="generic",
            exam_code=exam_type_raw,
            exam_name=exam_result.get("test_name", exam_type_raw),
            collected_at=self.format_date(exam_result.get("collected_date")),
            reported_at=exam_result.get("released_date"),
            status=LabTestStatus.FINAL,
            priority=LabPriority.ROUTINE,
            location=self._create_address(facility_info),
            generic_json=generic_details,
        )
