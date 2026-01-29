"""模板管理器

管理 LaTeX 模板的加载、验证和渲染。
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)


LATEX_SPECIAL_CHARS = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _escape_latex(text: str) -> str:
    """Escape LaTeX special characters."""
    for char, escaped in LATEX_SPECIAL_CHARS.items():
        text = text.replace(char, escaped)
    text = text.replace("\\", r"\textbackslash{}")
    return text


def _convert_numbered_list(text: str) -> str:
    """Convert numbered lists (1. 2. 3.) to LaTeX enumerate."""
    lines = text.split("\n")
    result = []
    in_list = False
    list_buffer = []

    for line in lines:
        stripped = line.strip()
        match = re.match(r"^(\d+)[.、]\s*(.+)$", stripped)

        if match:
            if not in_list:
                in_list = True
                list_buffer = []
            list_buffer.append(match.group(2))
        else:
            if in_list:
                result.append(r"\begin{enumerate}")
                for item in list_buffer:
                    result.append(rf"  \item {item}")
                result.append(r"\end{enumerate}")
                in_list = False
                list_buffer = []
            result.append(line)

    if in_list:
        result.append(r"\begin{enumerate}")
        for item in list_buffer:
            result.append(rf"  \item {item}")
        result.append(r"\end{enumerate}")

    return "\n".join(result)


def _convert_paragraphs(text: str) -> str:
    """Convert double newlines to LaTeX paragraphs."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\n\n+", "\n\n", text)
    return text


def text_to_latex(text: str) -> str:
    """Convert plain text to LaTeX-formatted content."""
    if not isinstance(text, str) or not text.strip():
        return text

    text = _escape_latex(text)
    text = _convert_numbered_list(text)
    text = _convert_paragraphs(text)

    return text


def convert_data_to_latex(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively convert all string values in dict to LaTeX format."""
    skip_fields = {
        "title",
        "report_id",
        "device_name",
        "device_model",
        "location",
        "diagnosis_date",
        "data_range",
        "health_status",
        "risk_level",
    }

    result = {}
    for key, value in data.items():
        if key in skip_fields:
            if isinstance(value, str):
                result[key] = _escape_latex(value)
            else:
                result[key] = value
        elif isinstance(value, str):
            result[key] = text_to_latex(value)
        elif isinstance(value, dict):
            result[key] = convert_data_to_latex(value)
        elif isinstance(value, list):
            result[key] = [text_to_latex(item) if isinstance(item, str) else item for item in value]
        else:
            result[key] = value
    return result


class TemplateManager:
    """LaTeX 模板管理器"""

    def __init__(self, templates_dir: str | Path | None = None):
        """初始化模板管理器

        Args:
            templates_dir: 模板目录路径，默认为 ./templates
        """
        if templates_dir is None:
            templates_dir = Path(__file__).parent / "templates"
        else:
            templates_dir = Path(templates_dir)

        self.templates_dir = Path(templates_dir)
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            variable_start_string="[[",  # 使用 [[ 作为变量开始
            variable_end_string="]]",  # 使用 ]] 作为变量结束
        )

        logger.info(f"模板管理器初始化: {self.templates_dir}")

    def list_templates(self) -> list[str]:
        """列出所有可用模板"""
        if not self.templates_dir.exists():
            return []

        templates = []
        for item in self.templates_dir.iterdir():
            if item.is_dir() and (item / "template.tex").exists():
                templates.append(item.name)

        return sorted(templates)

    def load_schema(self, template_id: str) -> dict[str, Any] | None:
        """加载模板的 JSON Schema

        Args:
            template_id: 模板 ID

        Returns:
            Schema 字典，如果不存在则返回 None
        """
        schema_path = self.templates_dir / template_id / "schema.json"
        if not schema_path.exists():
            return None

        with open(schema_path, encoding="utf-8") as f:
            return json.load(f)

    def load_metadata(self, template_id: str) -> dict[str, Any] | None:
        """加载模板的元数据

        Args:
            template_id: 模板 ID

        Returns:
            元数据字典，如果不存在则返回 None
        """
        metadata_path = self.templates_dir / template_id / "metadata.json"
        if not metadata_path.exists():
            return None

        with open(metadata_path, encoding="utf-8") as f:
            return json.load(f)

    def validate_data(self, template_id: str, data: dict[str, Any]) -> tuple[bool, list[str]]:
        """验证数据是否符合模板的 Schema

        Args:
            template_id: 模板 ID
            data: 要验证的数据

        Returns:
            (是否有效, 错误列表)
        """
        schema = self.load_schema(template_id)
        if not schema:
            # 没有 schema 则跳过验证
            return True, []

        errors = []

        # 检查必填字段
        required = schema.get("required", [])
        for field in required:
            if field not in data:
                errors.append(f"缺少必填字段: {field}")

        # 检查字段类型和枚举值
        properties = schema.get("properties", {})
        for field, value in data.items():
            if field not in properties:
                continue

            prop_def = properties[field]
            expected_type = prop_def.get("type")

            # 类型检查
            if expected_type == "integer":
                if not isinstance(value, int):
                    errors.append(f"字段 {field} 应为整数，实际为 {type(value).__name__}")
            elif expected_type == "string":
                if not isinstance(value, str):
                    errors.append(f"字段 {field} 应为字符串，实际为 {type(value).__name__}")

            # 枚举值检查
            enum_values = prop_def.get("enum")
            if enum_values and value not in enum_values:
                errors.append(f"字段 {field} 值 '{value}' 不在允许的值中: {enum_values}")

        return len(errors) == 0, errors

    def render_template(self, template_id: str, data: dict[str, Any]) -> str:
        """渲染模板

        Args:
            template_id: 模板 ID
            data: 模板数据

        Returns:
            渲染后的 LaTeX 内容

        Raises:
            FileNotFoundError: 模板不存在
            ValueError: 数据验证失败
        """
        logger.debug(f"开始渲染模板: {template_id}")
        logger.debug(f"提供的数据字段: {list(data.keys())}")

        template_path = self.templates_dir / template_id / "template.tex"
        if not template_path.exists():
            logger.error(f"模板文件不存在: {template_path}")
            raise FileNotFoundError(f"模板不存在: {template_id}")

        is_valid, errors = self.validate_data(template_id, data)
        if not is_valid:
            logger.error(f"数据验证失败: {errors}")
            raise ValueError(f"数据验证失败: {'; '.join(errors)}")

        logger.debug("数据验证通过")

        latex_data = convert_data_to_latex(data)

        template = self.jinja_env.get_template(f"{template_id}/template.tex")
        latex_content = template.render(**latex_data)

        logger.info(f"模板渲染成功: {template_id}, 填充 {len(data)} 个字段, 输出 {len(latex_content)} 字符")

        return latex_content

    def get_template_info(self, template_id: str) -> dict[str, Any]:
        """获取模板信息

        Args:
            template_id: 模板 ID

        Returns:
            模板信息字典
        """
        metadata = self.load_metadata(template_id) or {}
        schema = self.load_schema(template_id) or {}

        return {
            "id": template_id,
            "name": metadata.get("name", template_id),
            "description": metadata.get("description", ""),
            "version": metadata.get("version", "1.0.0"),
            "required_fields": schema.get("required", []),
            "total_fields": len(schema.get("properties", {})),
        }
