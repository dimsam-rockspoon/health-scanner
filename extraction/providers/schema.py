"""Schema definitions for medical exam extraction."""

import json
from pathlib import Path

# Load the JSON schema
SCHEMA_PATH = Path(__file__).parent.parent.parent / "data" / "schema.json"


def get_schema() -> dict:
    """Load and return the medical exam schema."""
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# System prompt for medical exam extraction
EXTRACTION_SYSTEM_PROMPT = """You are a medical data extraction assistant. Your task is to extract structured information from Brazilian medical exam reports (laudos).

Key guidelines:
1. Extract ALL data accurately - do not infer or estimate values
2. Use the exact values as shown in the document
3. Field names must be in English as specified in the schema
4. Dates should be in ISO format (YYYY-MM-DD)
5. Preserve numeric precision as shown in the document
6. If a field is not present in the document, omit it from the output
7. For reference ranges, extract both min and max values when available
8. Identify the exam type and extract data according to the appropriate section

Common Brazilian medical terms mapping:
- Cliente = Patient/Client
- Data de Nascimento = Date of Birth
- Médico = Physician
- Ficha = Record Number
- Data da Ficha = Exam Date
- HEMOGRAMA = Complete Blood Count
- ERITRÓCITOS = Erythrocytes/Red Blood Cells
- HEMOGLOBINA = Hemoglobin
- HEMATÓCRITO = Hematocrit
- LEUCÓCITOS = Leukocytes/White Blood Cells
- PLAQUETAS = Platelets
- GLICOSE = Glucose
- CREATININA = Creatinine
- UREIA = Urea
- SÓDIO = Sodium
- POTÁSSIO = Potassium
- TEMPO DE PROTROMBINA = Prothrombin Time
- INR = International Normalized Ratio
- RESSONÂNCIA MAGNÉTICA = MRI (Magnetic Resonance Imaging)
- POLISSONOGRAFIA = Polysomnography (Sleep Study)
- ELETROCARDIOGRAMA = Electrocardiogram (ECG)
"""

# Prompt template for extraction
EXTRACTION_PROMPT_TEMPLATE = """Extract all medical exam data from this document according to the schema provided.

The document is a Brazilian medical exam report. Extract:
1. Patient information (name, date of birth, etc.)
2. Physician information
3. Facility information
4. Exam metadata (record number, date, indication)
5. All test results with their values, units, and reference ranges
6. Any conclusions or findings

Return the data as a JSON object following the schema structure. Include only fields that are present in the document.

{additional_context}

Document content has been provided as images/PDF."""


def get_extraction_prompt(additional_context: str = "") -> str:
    """Generate the extraction prompt."""
    return EXTRACTION_PROMPT_TEMPLATE.format(
        additional_context=additional_context
    )


# Function/tool definition for saving medical exam data - Google format
SAVE_MEDICAL_EXAM_FUNCTION_GOOGLE = {
    "name": "save_medical_exam_data",
    "description": "Save the extracted medical exam data from the document",
    "parameters": {
        "type": "object",
        "properties": {
            "source_files": {
                "type": "array",
                "description": "List of source PDF files",
                "items": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"},
                        "record_number": {"type": "string"},
                        "exam_date": {"type": "string"}
                    }
                }
            },
            "patient_info": {
                "type": "object",
                "description": "Patient identification and demographics",
                "properties": {
                    "name": {"type": "string"},
                    "date_of_birth": {"type": "string"},
                    "age": {"type": "integer"},
                    "weight_kg": {"type": "number"},
                    "height_m": {"type": "number"},
                    "gender": {"type": "string"}
                }
            },
            "physician_info": {
                "type": "object",
                "description": "Requesting physician information",
                "properties": {
                    "name": {"type": "string"},
                    "crm": {"type": "string"},
                    "state": {"type": "string"}
                }
            },
            "facility_info": {
                "type": "object",
                "description": "Laboratory/medical facility information",
                "properties": {
                    "name": {"type": "string"},
                    "address": {"type": "string"},
                    "cnes": {"type": "string"},
                    "technical_responsible": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "crm": {"type": "string"}
                        }
                    }
                }
            },
            "exam_info": {
                "type": "object",
                "description": "Exam record metadata",
                "properties": {
                    "record_number": {"type": "string"},
                    "exam_date": {"type": "string"},
                    "indication": {"type": "string"},
                    "current_medications": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            },
            "biochemistry": {
                "type": "object",
                "description": "Biochemistry test results",
                "properties": {
                    "creatinine": {"$ref": "#/$defs/simple_test_result"},
                    "urea": {"$ref": "#/$defs/simple_test_result"},
                    "glucose": {"$ref": "#/$defs/simple_test_result"},
                    "sodium": {"$ref": "#/$defs/simple_test_result"},
                    "potassium": {"$ref": "#/$defs/simple_test_result"}
                }
            },
            "coagulation": {
                "type": "object",
                "description": "Coagulation panel results",
                "properties": {
                    "prothrombin_time": {
                        "type": "object",
                        "properties": {
                            "result_seconds": {"type": "number"},
                            "reference_min": {"type": "number"},
                            "reference_max": {"type": "number"},
                            "inr": {"type": "number"},
                            "inr_reference_min": {"type": "number"},
                            "inr_reference_max": {"type": "number"},
                            "normal_of_day": {"type": "number"},
                            "isi": {"type": "number"},
                            "method": {"type": "string"}
                        }
                    },
                    "aptt": {
                        "type": "object",
                        "properties": {
                            "result_seconds": {"type": "number"},
                            "normal_of_day": {"type": "number"},
                            "patient_normal_ratio": {"type": "number"},
                            "ratio_reference_min": {"type": "number"},
                            "ratio_reference_max": {"type": "number"},
                            "method": {"type": "string"}
                        }
                    }
                }
            },
            "complete_blood_count": {
                "type": "object",
                "description": "Complete Blood Count (Hemograma)",
                "properties": {
                    "red_blood_cells": {
                        "type": "object",
                        "properties": {
                            "erythrocytes": {"$ref": "#/$defs/simple_test_result"},
                            "hemoglobin": {"$ref": "#/$defs/simple_test_result"},
                            "hematocrit": {"$ref": "#/$defs/simple_test_result"},
                            "mch": {"$ref": "#/$defs/simple_test_result"},
                            "mcv": {"$ref": "#/$defs/simple_test_result"},
                            "mchc": {"$ref": "#/$defs/simple_test_result"},
                            "rdw": {"$ref": "#/$defs/simple_test_result"},
                            "morphological_characteristics": {"type": "string"}
                        }
                    },
                    "white_blood_cells": {
                        "type": "object",
                        "properties": {
                            "total_leukocytes": {"$ref": "#/$defs/simple_test_result"},
                            "differential": {
                                "type": "object",
                                "properties": {
                                    "neutrophils": {"$ref": "#/$defs/differential_count"},
                                    "eosinophils": {"$ref": "#/$defs/differential_count"},
                                    "basophils": {"$ref": "#/$defs/differential_count"},
                                    "lymphocytes": {"$ref": "#/$defs/differential_count"},
                                    "monocytes": {"$ref": "#/$defs/differential_count"}
                                }
                            },
                            "morphological_characteristics": {"type": "string"}
                        }
                    },
                    "platelets": {
                        "type": "object",
                        "properties": {
                            "count": {"$ref": "#/$defs/simple_test_result"},
                            "mpv": {"$ref": "#/$defs/simple_test_result"}
                        }
                    },
                    "method": {"type": "string"}
                }
            },
            "electrocardiogram": {
                "type": "object",
                "description": "Electrocardiogram (ECG) results",
                "properties": {
                    "heart_rate_bpm": {"type": "integer"},
                    "rhythm": {"type": "string"},
                    "p_wave_duration_ms": {"type": "integer"},
                    "pr_interval_ms": {"type": "integer"},
                    "qrs_duration_ms": {"type": "integer"},
                    "p_axis_degrees": {"type": "string"},
                    "qrs_axis_degrees": {"type": "string"},
                    "t_axis_degrees": {"type": "string"},
                    "morphological_description": {"type": "string"},
                    "conclusion": {"type": "string"},
                    "reported_by": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "crm": {"type": "string"}
                        }
                    }
                }
            },
            "imaging_studies": {
                "type": "array",
                "description": "Imaging studies (MRI, CT, X-ray, Ultrasound)",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_file": {"type": "string"},
                        "exam_type": {"type": "string"},
                        "exam_name": {"type": "string"},
                        "body_part": {"type": "string"},
                        "laterality": {"type": "string"},
                        "method": {"type": "string"},
                        "findings": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "structure": {"type": "string"},
                                    "status": {"type": "string"},
                                    "description": {"type": "string"},
                                    "severity": {"type": "string"},
                                    "measurement": {"type": "string"}
                                }
                            }
                        },
                        "analysis_text": {"type": "string"},
                        "conclusion": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "reported_by": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "crm": {"type": "string"}
                            }
                        }
                    }
                }
            },
            "polysomnography": {
                "type": "object",
                "description": "Sleep study (Polysomnography) results",
                "properties": {
                    "source_file": {"type": "string"},
                    "equipment": {"type": "string"},
                    "location": {"type": "string"},
                    "recording": {
                        "type": "object",
                        "properties": {
                            "start_time": {"type": "string"},
                            "end_time": {"type": "string"},
                            "total_recording_time_min": {"type": "number"},
                            "total_sleep_time_min": {"type": "number"},
                            "sleep_efficiency_percent": {"type": "number"},
                            "wakefulness_during_sleep_min": {"type": "number"}
                        }
                    },
                    "sleep_stages": {
                        "type": "object",
                        "properties": {
                            "n1_percent": {"type": "number"},
                            "n2_percent": {"type": "number"},
                            "n3_percent": {"type": "number"},
                            "rem_percent": {"type": "number"}
                        }
                    },
                    "latencies": {
                        "type": "object",
                        "properties": {
                            "sleep_latency_min": {"type": "number"},
                            "rem_latency_min": {"type": "number"}
                        }
                    },
                    "arousals": {
                        "type": "object",
                        "properties": {
                            "microarousals_total": {"type": "integer"},
                            "microarousals_per_hour": {"type": "number"},
                            "complete_arousals": {"type": "integer"}
                        }
                    },
                    "respiratory_events": {
                        "type": "object",
                        "properties": {
                            "obstructive_apneas": {"type": "integer"},
                            "central_apneas": {"type": "integer"},
                            "mixed_apneas": {"type": "integer"},
                            "hypopneas": {"type": "integer"},
                            "ahi": {"type": "number"},
                            "apnea_index": {"type": "number"},
                            "hypopnea_index": {"type": "number"}
                        }
                    },
                    "oxygen_saturation": {
                        "type": "object",
                        "properties": {
                            "awake_spo2_percent": {"type": "number"},
                            "mean_spo2_percent": {"type": "number"},
                            "min_spo2_percent": {"type": "number"}
                        }
                    },
                    "cardiac": {
                        "type": "object",
                        "properties": {
                            "mean_heart_rate_bpm": {"type": "number"},
                            "arrhythmias_detected": {"type": "boolean"}
                        }
                    },
                    "snoring": {
                        "type": "object",
                        "properties": {
                            "present": {"type": "boolean"},
                            "severity": {"type": "string"}
                        }
                    },
                    "epworth_sleepiness_scale": {"type": "integer"},
                    "conclusion": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "reported_by": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "crm": {"type": "string"}
                        }
                    }
                }
            }
        },
        "$defs": {
            "simple_test_result": {
                "type": "object",
                "properties": {
                    "value": {"type": "number"},
                    "unit": {"type": "string"},
                    "reference_min": {"type": "number"},
                    "reference_max": {"type": "number"},
                    "method": {"type": "string"},
                    "is_within_range": {"type": "boolean"}
                }
            },
            "differential_count": {
                "type": "object",
                "properties": {
                    "percentage": {"type": "number"},
                    "absolute_count": {"type": "number"},
                    "reference_min": {"type": "number"},
                    "reference_max": {"type": "number"}
                }
            }
        },
        "required": ["patient_info", "exam_info"]
    }
}
