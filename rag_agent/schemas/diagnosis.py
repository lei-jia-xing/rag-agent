from typing import Literal

from pydantic import BaseModel, Field


class DiagnosisFields(BaseModel):
    title: str = Field(default="设备健康诊断报告")
    report_id: str = Field(default="DX-00000000-001", description="报告编号 DX-YYYYMMDD-NNN")
    device_name: str
    device_model: str = Field(default="未知型号")
    location: str = Field(default="未知位置")
    diagnosis_date: str = Field(default="")
    data_range: str = Field(default="")

    health_score: int = Field(ge=0, le=100, default=0)
    health_status: Literal["正常", "警告", "异常", "严重"] = Field(default="正常")
    risk_level: Literal["低", "中", "高"] = Field(default="低")
    issue_count: int = Field(ge=0, default=0)

    abstract: str = Field(default="")
    device_basic_info: str = Field(default="")
    operating_environment: str = Field(default="")
    maintenance_history: str = Field(default="")

    monitoring_data_summary: str = Field(default="")
    key_metrics_analysis: str = Field(default="")
    trend_analysis: str = Field(default="")
    anomaly_detection: str = Field(default="")

    fault_description: str = Field(default="")
    fault_cause_analysis: str = Field(default="")
    fault_location: str = Field(default="")

    urgent_measures: str = Field(default="")
    maintenance_plan: str = Field(default="")
    spare_parts_suggestion: str = Field(default="")

    current_risks: str = Field(default="")
    potential_risks: str = Field(default="")
    risk_control: str = Field(default="")

    conclusion_and_recommendations: str = Field(default="")
    technical_parameters: str = Field(default="")
    related_standards: str = Field(default="")
    diagnosis_method: str = Field(default="")

    @classmethod
    def from_llm_response(cls, data: dict) -> "DiagnosisFields":
        """Parse LLM response with fallback defaults for missing/invalid fields"""
        if isinstance(data.get("health_score"), str):
            try:
                data["health_score"] = int(data["health_score"])
            except ValueError:
                data["health_score"] = 0

        if isinstance(data.get("issue_count"), str):
            try:
                data["issue_count"] = int(data["issue_count"])
            except ValueError:
                data["issue_count"] = 0

        return cls.model_validate(data)
