"""Pydantic models matching LabTestResult proto message structure."""

from enum import IntEnum
from pydantic import BaseModel, Field


class LabTestStatus(IntEnum):
    """Status of a lab test."""
    LAB_TEST_STATUS_UNSPECIFIED = 0
    PENDING = 1
    IN_PROGRESS = 2
    FINAL = 3
    CORRECTED = 4
    CANCELED = 5


class LabPriority(IntEnum):
    """Priority of a lab test."""
    LAB_PRIORITY_UNSPECIFIED = 0
    URGENT = 1
    ROUTINE = 2


class Address(BaseModel):
    """Address model for location fields."""
    street: str | None = None
    city: str | None = None
    state: str | None = None
    postal_code: str | None = None
    country: str | None = None
    name: str | None = None  # Facility name


class ComparisonToPrevious(BaseModel):
    """Comparison to previous exam results."""
    previous_exam_id: str | None = None
    summary: str | None = None


# Blood Exam nested models
class ReferenceRange(BaseModel):
    """Reference range for an analyte."""
    low: float | None = None
    high: float | None = None
    age_specific: bool = False
    sex_specific: bool = False


class Analyte(BaseModel):
    """A single analyte measurement in a blood exam."""
    code: str | None = None
    name: str | None = None
    value: float | None = None
    unit: str | None = None
    reference_range: ReferenceRange | None = None
    flag: str | None = None  # normal | low | high | critical_low | critical_high
    comments: str | None = None


class QualityControl(BaseModel):
    """Quality control metrics for blood sample."""
    hemolysis_index: float | None = None
    lipemia_index: float | None = None
    icterus_index: float | None = None
    flags: list[str] = Field(default_factory=list)


class BloodExam(BaseModel):
    """Blood exam details."""
    sample_type: str | None = None
    fasting_status: str | None = None  # fasting | non_fasting | unknown
    collection_site: Address | None = None
    container_type: str | None = None
    lab_panel_code: str | None = None
    analytes: list[Analyte] = Field(default_factory=list)
    quality_control: QualityControl | None = None


# MRI Exam nested models
class ContrastDose(BaseModel):
    """Contrast dose information."""
    value: float | None = None
    unit: str | None = None


class ImageSeries(BaseModel):
    """Image series information."""
    series_uid: str | None = None
    description: str | None = None
    modality: str | None = None
    num_images: int | None = None
    storage_url: str | None = None
    thumbnail_url: str | None = None


class FindingCode(BaseModel):
    """Standardized code for a finding."""
    system: str | None = None
    code: str | None = None


class MRIFinding(BaseModel):
    """A finding from an MRI exam."""
    title: str | None = None
    description: str | None = None
    location: Address | None = None
    severity: str | None = None
    codes: list[FindingCode] = Field(default_factory=list)


class MeasurementData(BaseModel):
    """Quantitative measurement data."""
    name: str | None = None
    value: float | None = None
    unit: str | None = None


class MRIExam(BaseModel):
    """MRI exam details."""
    body_region: str | None = None
    contrast_used: bool = False
    contrast_agent: str | None = None
    contrast_dose: ContrastDose | None = None
    sequences: list[str] = Field(default_factory=list)
    positioning: str | None = None
    image_series: list[ImageSeries] = Field(default_factory=list)
    findings: list[MRIFinding] = Field(default_factory=list)
    impression: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    measurement_data: list[MeasurementData] = Field(default_factory=list)
    radiation_dose: float | None = None
    report_signed_by: str | None = None
    report_signed_at: str | None = None


# X-Ray Exam nested models
class RadiationDose(BaseModel):
    """Radiation dose information for X-ray."""
    d_ap: float | None = None
    unit: str | None = None
    exposure_time_ms: int | None = None
    kvp: int | None = None
    mAs: int | None = None


class XRayFinding(BaseModel):
    """A finding from an X-ray exam."""
    title: str | None = None
    description: str | None = None
    location: str | None = None
    severity: str | None = None


class XRayExam(BaseModel):
    """X-Ray exam details."""
    body_region: Address | None = None
    projection: list[str] = Field(default_factory=list)
    portable: bool = False
    radiation_dose: RadiationDose | None = None
    image_series: list[ImageSeries] = Field(default_factory=list)
    findings: list[XRayFinding] = Field(default_factory=list)
    impression: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    report_signed_by: str | None = None
    report_signed_at: str | None = None


# CT Scan Exam nested models
class CTMeasurement(BaseModel):
    """Measurement from CT scan."""
    name: str | None = None
    value: float | None = None
    unit: str | None = None


class CTFinding(BaseModel):
    """A finding from a CT scan."""
    title: str | None = None
    description: str | None = None
    location: Address | None = None
    severity: str | None = None


class CTRadiationDose(BaseModel):
    """Radiation dose information for CT scan."""
    value: float | None = None
    unit: str | None = None
    exposure_time_ms: int | None = None
    kvp: int | None = None
    mAs: int | None = None


class CTScanExam(BaseModel):
    """CT Scan exam details."""
    modality: str | None = None
    body_region: Address | None = None
    contrast_used: bool = False
    contrast_route: str | None = None
    protocol: str | None = None
    image_series: list[ImageSeries] = Field(default_factory=list)
    measurements: list[CTMeasurement] = Field(default_factory=list)
    findings: list[CTFinding] = Field(default_factory=list)
    impression: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    radiation_dose: CTRadiationDose | None = None


# Polysomnography Exam nested models
class Duration(BaseModel):
    """Duration value with unit."""
    value: float | None = None
    unit: str | None = None


class SleepEfficiency(BaseModel):
    """Sleep efficiency measurement."""
    value: float | None = None
    unit: str | None = None
    interpretation: str | None = None


class SleepStage(BaseModel):
    """Individual sleep stage."""
    value: float | None = None
    unit: str | None = None


class SleepDistribution(BaseModel):
    """Sleep stage distribution."""
    stage_1_n1: SleepStage | None = None
    stage_2_n2: SleepStage | None = None
    stage_3_n3: SleepStage | None = None
    rem_sleep: SleepStage | None = None


class Microarousals(BaseModel):
    """Microarousal data."""
    count: int | None = None
    rate: str | None = None


class CompleteArousals(BaseModel):
    """Complete arousal data."""
    count: int | None = None


class SleepFragmentation(BaseModel):
    """Sleep fragmentation data."""
    microarousals: Microarousals | None = None
    complete_arousals: CompleteArousals | None = None


class ObstructiveApneas(BaseModel):
    """Obstructive apnea data."""
    count: int | None = None


class Hypopneas(BaseModel):
    """Hypopnea data."""
    count: int | None = None


class RespiratoryEvents(BaseModel):
    """Respiratory events during sleep."""
    obstructive_apneas: ObstructiveApneas | None = None
    hypopneas: Hypopneas | None = None


class ApneaHypopneaIndex(BaseModel):
    """Apnea-hypopnea index data."""
    value: float | None = None
    unit: str | None = None
    apnea_per_hour: float | None = None
    hypopnea_per_hour: float | None = None


class OxygenSaturationValue(BaseModel):
    """Oxygen saturation measurement."""
    value: float | None = None
    unit: str | None = None


class OxygenSaturation(BaseModel):
    """Oxygen saturation data."""
    wakefulness: OxygenSaturationValue | None = None
    average: OxygenSaturationValue | None = None
    minimum: OxygenSaturationValue | None = None


class HeartRateValue(BaseModel):
    """Heart rate measurement."""
    value: float | None = None
    unit: str | None = None


class HeartRate(BaseModel):
    """Heart rate data."""
    average: HeartRateValue | None = None
    arrhythmias: str | None = None


class EpworthSleepinessScale(BaseModel):
    """Epworth Sleepiness Scale."""
    value: int | None = None


class PolysomnographyExam(BaseModel):
    """Polysomnography (sleep study) exam details."""
    equipment: str | None = None
    monitoring_duration: Duration | None = None
    start_time: str | None = None
    end_time: str | None = None
    electrophysiological_parameters: str | None = None
    total_registration_time: Duration | None = None
    total_sleep_time: Duration | None = None
    sleep_efficiency: SleepEfficiency | None = None
    sleep_distribution: SleepDistribution | None = None
    sleep_latency: Duration | None = None
    rem_sleep_latency: Duration | None = None
    wakefulness_during_total_sleep_time: Duration | None = None
    sleep_fragmentation: SleepFragmentation | None = None
    periodic_limb_movements: str | None = None
    respiratory_events_during_sleep: RespiratoryEvents | None = None
    oxygen_desaturations: str | None = None
    apnea_hypopnea_index_ahi: ApneaHypopneaIndex | None = None
    oxygen_saturation_sao2: OxygenSaturation | None = None
    snoring: str | None = None
    body_position: str | None = None
    heart_rate: HeartRate | None = None
    epworth_sleepiness_scale: EpworthSleepinessScale | None = None
    conclusions: list[str] = Field(default_factory=list)


class GenericExamDetails(BaseModel):
    """Generic exam details for exams that don't fit other categories."""
    fields: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class LabTestResult(BaseModel):
    """
    Main LabTestResult model matching the proto message.

    This is the primary output format for medical exam data.
    """
    id: str
    patient_id: str
    exam_type: str  # e.g., "blood_test", "mri", "xray", "ct_scan", "polysomnography"
    exam_code: str | None = None  # Code according to standard coding system (e.g., LOINC)
    exam_name: str | None = None
    ordering_physician_id: str | None = None
    ordering_department: str | None = None
    indication: str | None = None
    clinical_notes: str | None = None
    ordered_at: str | None = None
    collected_at: str | None = None
    received_at: str | None = None
    reported_at: str | None = None
    status: LabTestStatus = LabTestStatus.FINAL
    priority: LabPriority = LabPriority.ROUTINE
    location: Address | None = None
    result_summary: str | None = None
    interpretation: str | None = None
    severity_flag: str | None = None  # Normal, Abnormal, Critical
    comparison_to_previous: ComparisonToPrevious | None = None

    # OneOf details - only one should be set
    json: BloodExam | None = Field(default=None, description="Blood exam details")
    mri_json: MRIExam | None = Field(default=None, description="MRI exam details")
    xray_json: XRayExam | None = Field(default=None, description="X-Ray exam details")
    ct_scan_json: CTScanExam | None = Field(default=None, description="CT Scan exam details")
    polysomnography_json: PolysomnographyExam | None = Field(default=None, description="Polysomnography exam details")
    generic_json: GenericExamDetails | None = Field(default=None, description="Generic exam details")

    tags: list[str] = Field(default_factory=list)
    sensitive: bool = False

    def model_dump_proto(self) -> dict:
        """
        Dump the model to a dict format compatible with proto JSON serialization.

        Converts enum values to their proto representation and removes None fields.
        """
        data = self.model_dump(exclude_none=True, by_alias=True)

        # Convert enums to proto format
        if "status" in data:
            data["status"] = LabTestStatus(data["status"]).name
        if "priority" in data:
            data["priority"] = LabPriority(data["priority"]).name

        return data
