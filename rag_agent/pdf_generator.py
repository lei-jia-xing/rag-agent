"""PDF 生成器

使用 WeasyPrint 将 Markdown 转换为支持中文的 PDF。
"""

from pathlib import Path
from typing import Any

import markdown
from weasyprint import CSS, HTML


class PDFGenerator:
    """PDF 生成器类"""

    def __init__(self) -> None:
        """初始化 PDF 生成器"""
        self._setup_markdown()

    def _setup_markdown(self) -> None:
        """设置 Markdown 处理器"""
        # 配置 Markdown 扩展
        self.md = markdown.Markdown(
            extensions=[
                "tables",  # 表格支持
                "toc",  # 目录支持
                "sane_lists",  # 列表支持
                "codehilite",  # 代码高亮
                "fenced_code",  # 围栏代码块
            ]
        )

    def _get_css(self) -> str:
        """获取 CSS 样式"""
        return """
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap');

        @page {
            size: A4;
            margin: 2cm;
            @bottom-center {
                content: counter(page);
                font-size: 10pt;
                color: #666;
            }
        }

        body {
            font-family: "Noto Sans SC", "Microsoft YaHei", "SimHei", "文泉驿正黑", sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 0;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: "Noto Sans SC", "Microsoft YaHei", "SimHei", "文泉驿正黑", sans-serif;
            font-weight: 700;
            color: #2c3e50;
            margin-top: 1.5em;
            margin-bottom: 0.8em;
        }

        h1 {
            font-size: 20pt;
            text-align: center;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }

        h2 {
            font-size: 16pt;
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 5px;
            page-break-before: auto;
        }

        h3 {
            font-size: 14pt;
        }

        h4 {
            font-size: 12pt;
        }

        p {
            margin-bottom: 0.8em;
            text-align: justify;
        }

        strong {
            font-weight: 700;
            color: #2c3e50;
        }

        a {
            color: #3498db;
            text-decoration: none;
        }

        a:hover {
            text-decoration: underline;
        }

        ul, ol {
            padding-left: 2em;
            margin-bottom: 0.8em;
        }

        li {
            margin-bottom: 0.3em;
        }

        blockquote {
            border-left: 4px solid #3498db;
            margin: 1em 0;
            padding: 0.5em 1em;
            background-color: #f8f9fa;
            color: #555;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1em 0;
            font-size: 10pt;
        }

        th, td {
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: left;
        }

        th {
            background-color: #3498db;
            color: white;
            font-weight: 700;
        }

        tr:nth-child(even) {
            background-color: #f8f9fa;
        }

        code {
            background-color: #f1f2f6;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: "Courier New", monospace;
            font-size: 0.9em;
        }

        pre {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 1em;
            overflow-x: auto;
            font-size: 0.9em;
        }

        pre code {
            background-color: transparent;
            padding: 0;
            border-radius: 0;
        }

        .metadata {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 1em;
            margin-bottom: 2em;
        }

        .metadata h3 {
            margin-top: 0;
            color: #495057;
            font-size: 14pt;
        }

        .metadata table {
            margin: 0;
            font-size: 10pt;
        }

        .metadata th {
            background-color: #6c757d;
            width: 30%;
        }

        /* 避免分页时的断行问题 */
        h1, h2, h3, h4, h5, h6 {
            page-break-after: avoid;
        }

        table, blockquote, pre {
            page-break-inside: avoid;
        }

        ul, ol {
            page-break-inside: avoid;
        }

        /* 打印样式 */
        @media print {
            body {
                font-size: 10pt;
            }

            h1 { font-size: 18pt; }
            h2 { font-size: 14pt; }
            h3 { font-size: 12pt; }
            h4 { font-size: 11pt; }
        }
        """

    def _generate_html(self, content: str, title: str, metadata: dict[str, Any] | None = None) -> str:
        """生成 HTML 内容

        Args:
            content: Markdown 内容
            title: 报告标题
            metadata: 元数据

        Returns:
            HTML 字符串
        """
        # 转换 Markdown 为 HTML
        html_content = self.md.convert(content)

        # 生成元数据 HTML
        metadata_html = ""
        if metadata:
            metadata_html = """
            <div class="metadata">
                <h3>报告信息</h3>
                <table>
            """
            for key, value in metadata.items():
                if isinstance(value, list):
                    value = ", ".join(str(v) for v in value)
                metadata_html += f"<tr><th>{key}</th><td>{value}</td></tr>"
            metadata_html += """
                </table>
            </div>
            """

        # 生成完整的 HTML
        html_template = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
        </head>
        <body>
            <h1>{title}</h1>
            {metadata_html}
            <div class="content">
                {html_content}
            </div>
        </body>
        </html>
        """

        return html_template

    def generate_pdf(
        self,
        content: str,
        output_path: str | Path,
        title: str = "技术报告",
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """生成 PDF 报告

        Args:
            content: 报告内容（Markdown 格式）
            output_path: 输出文件路径
            title: 报告标题
            metadata: 额外的元数据信息

        Returns:
            生成的 PDF 文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # 生成 HTML
            html_content = self._generate_html(content, title, metadata)

            # 创建 HTML 对象
            html = HTML(string=html_content)

            # 创建 CSS
            css = CSS(string=self._get_css())

            # 生成 PDF
            html.write_pdf(str(output_path), stylesheets=[css])

            return output_path

        except Exception as e:
            raise RuntimeError(f"PDF 生成失败: {e}") from e

    def generate_pdf_bytes(
        self,
        content: str,
        title: str = "技术报告",
        metadata: dict[str, Any] | None = None,
    ) -> bytes:
        """生成 PDF 字节数据

        Args:
            content: 报告内容（Markdown 格式）
            title: 报告标题
            metadata: 额外的元数据信息

        Returns:
            PDF 字节数据
        """
        try:
            # 生成 HTML
            html_content = self._generate_html(content, title, metadata)

            # 创建 HTML 对象
            html = HTML(string=html_content)

            # 创建 CSS
            css = CSS(string=self._get_css())

            # 生成 PDF 字节
            pdf_bytes = html.write_pdf(stylesheets=[css])

            if pdf_bytes is None:
                raise RuntimeError("PDF 生成失败: write_pdf 返回 None")

            return pdf_bytes

        except Exception as e:
            raise RuntimeError(f"PDF 生成失败: {e}") from e


# 创建全局实例
pdf_generator = PDFGenerator()


def generate_report_pdf(
    content: str,
    output_path: str | Path,
    title: str = "技术报告",
    metadata: dict[str, Any] | None = None,
) -> Path:
    """便捷函数：生成报告 PDF

    Args:
        content: 报告内容（Markdown 格式）
        output_path: 输出文件路径
        title: 报告标题
        metadata: 额外的元数据信息

    Returns:
        生成的 PDF 文件路径
    """
    return pdf_generator.generate_pdf(content, output_path, title, metadata)


def generate_report_pdf_bytes(
    content: str,
    title: str = "技术报告",
    metadata: dict[str, Any] | None = None,
) -> bytes:
    """便捷函数：生成报告 PDF 字节数据

    Args:
        content: 报告内容（Markdown 格式）
        title: 报告标题
        metadata: 额外的元数据信息

    Returns:
        PDF 字节数据
    """
    return pdf_generator.generate_pdf_bytes(content, title, metadata)
