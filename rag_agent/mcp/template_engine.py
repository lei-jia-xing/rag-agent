import re
from pathlib import Path
from typing import Any


class TemplateEngine:
    """LaTeX模板引擎，支持字段填充"""

    def __init__(self, template_path: Path):
        """初始化模板引擎

        Args:
            template_path: LaTeX模板文件路径
        """
        self.template_path = template_path
        self.template: str = ""
        self._load_template()

    def _load_template(self) -> None:
        """加载模板文件"""
        if not self.template_path.exists():
            raise FileNotFoundError(f"模板文件不存在: {self.template_path}")

        with open(self.template_path, encoding="utf-8") as f:
            self.template = f.read()

    def render(self, data: dict[str, Any]) -> str:
        """渲染模板，替换占位符

        Args:
            data: 包含字段值的字典

        Returns:
            渲染后的LaTeX内容
        """
        rendered = self.template

        for key, value in data.items():
            placeholder = f"{{{key}}}"
            # 转义LaTeX特殊字符
            if isinstance(value, str):
                value = self._escape_latex(value)
            rendered = rendered.replace(placeholder, str(value))

        return rendered

    def _escape_latex(self, text: str) -> str:
        """转义LaTeX特殊字符

        Args:
            text: 原始文本

        Returns:
            转义后的文本
        """
        # 常见LaTeX特殊字符的转义
        escapes = {
            "&": r"\\&",
            "%": r"\\%",
            "$": r"\\$",
            "#": r"\\#",
            "_": r"\\_",
            "{": r"\\{",
            "}": r"\\}",
            "~": "\textasciitilde{}",
            "^": "\textasciicircum{}",
        }

        for char, escaped in escapes.items():
            text = text.replace(char, escaped)

        return text

    def validate_template(self) -> dict[str, Any]:
        """验证模板，找出所有占位符

        Returns:
            包含占位符列表的字典
        """
        # 使用正则表达式查找所有 {{placeholder}}
        pattern = r"{{([^}]+)}}"
        matches = re.findall(pattern, self.template)

        return {
            "placeholders": sorted(set(matches)),
            "count": len(set(matches))
        }
