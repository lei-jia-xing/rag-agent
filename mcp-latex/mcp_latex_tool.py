#!/usr/bin/env python3
"""
MCP Server with LaTeX and TikZ Integration
Provides document compilation and TikZ diagram rendering through the Model Context Protocol.
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import mcp.server.stdio
import mcp.types as types
from mcp.server import InitializationOptions, NotificationOptions, Server
from template_manager import TemplateManager

# Configure logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
date_format = '%Y-%m-%d %H:%M:%S'

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

# Get root logger
root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
root_logger.addHandler(console_handler)

# Create module logger
logger = logging.getLogger(__name__)

# Add file handler if logs directory exists and is writable
logs_dir = Path('/workspace/logs')
if logs_dir.exists():
    try:
        log_file = logs_dir / f"mcp_latex_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        root_logger.addHandler(file_handler)
        logger.info(f"日志文件: {log_file}")
    except (PermissionError, OSError) as e:
        # 如果无法写入日志文件，只使用控制台输出
        logger.warning(f"无法创建日志文件处理器，仅使用控制台输出: {e}")
else:
    logger.warning("logs 目录不存在，仅使用控制台输出")

logger.info("=" * 60)
logger.info("LaTeX MCP Server 启动")
logger.info(f"日志级别: {LOG_LEVEL}")
logger.info("=" * 60)


class LaTeXTool:
    """LaTeX document compilation and TikZ rendering tool for MCP."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.latex_output_dir = output_dir / "latex"
        self.latex_output_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"LaTeXTool 初始化完成，输出目录: {self.latex_output_dir}")

    async def compile_latex(
        self,
        content: str,
        format: str = "pdf",
        template: str = "article"
    ) -> dict[str, Any]:
        """
        Compile LaTeX document to various formats.

        Args:
            content: LaTeX document content
            format: Output format (pdf, dvi, ps)
            template: Document template to use

        Returns:
            Dictionary with compiled document path and metadata
        """
        logger.info(f"开始编译 LaTeX 文档 (格式={format}, 模板={template})")
        logger.debug(f"LaTeX 内容长度: {len(content)} 字符")

        try:
            # Add template wrapper if not custom
            if template != "custom" and not content.startswith("\\documentclass"):
                templates = {
                    "article": "\\documentclass{article}\n\\begin{document}\n%CONTENT%\n\\end{document}",
                    "report": "\\documentclass{report}\n\\begin{document}\n%CONTENT%\n\\end{document}",
                    "book": "\\documentclass{book}\n\\begin{document}\n%CONTENT%\n\\end{document}",
                    "beamer": "\\documentclass{beamer}\n\\begin{document}\n%CONTENT%\n\\end{document}",
                }
                if template in templates:
                    content = templates[template].replace("%CONTENT%", content)
                    logger.debug(f"应用模板包装: {template}")

            # Create temporary directory
            with tempfile.TemporaryDirectory() as tmpdir:
                # Write LaTeX file
                tex_file = os.path.join(tmpdir, "document.tex")
                with open(tex_file, "w") as f:
                    f.write(content)
                logger.debug(f"临时文件: {tex_file}")

                # Choose compiler based on format and content
                if format == "pdf":
                    # Use xelatex for documents with ctex (Chinese support)
                    if "\\usepackage{ctex}" in content or "\\usepackage[" in content and "ctex" in content:
                        compiler = "xelatex"
                    else:
                        compiler = "pdflatex"
                else:
                    compiler = "latex"

                cmd = [compiler, "-interaction=nonstopmode", tex_file]

                logger.info(f"Compiling LaTeX with: {' '.join(cmd)}")

                # Run compilation (twice for references)
                for i in range(2):
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        cwd=tmpdir,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    _stdout, _stderr = await process.communicate()
                    if process.returncode != 0 and i == 0:
                        # First compilation might fail due to references
                        logger.warning("First compilation pass had warnings")

                # Convert DVI to PS if needed
                if format == "ps" and process.returncode == 0:
                    dvi_file = os.path.join(tmpdir, "document.dvi")
                    ps_file = os.path.join(tmpdir, "document.ps")
                    await asyncio.create_subprocess_exec(
                        "dvips", dvi_file, "-o", ps_file,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )

                # Check for output
                output_file = os.path.join(tmpdir, f"document.{format}")
                if os.path.exists(output_file):
                    # Copy to output directory
                    output_path = os.path.join(
                        str(self.latex_output_dir),
                        f"document_{os.getpid()}.{format}"
                    )
                    shutil.copy(output_file, output_path)

                    # Also copy log file for debugging
                    log_file = os.path.join(tmpdir, "document.log")
                    log_path = None
                    if os.path.exists(log_file):
                        log_path = output_path.replace(f".{format}", ".log")
                        shutil.copy(log_file, log_path)

                    return {
                        "success": True,
                        "output_path": output_path,
                        "format": format,
                        "template": template,
                        "log_path": log_path,
                    }

                # Extract error from log file and save failed files for debugging
                log_file = os.path.join(tmpdir, "document.log")
                error_msg = "Compilation failed"
                failed_tex_path = None
                failed_log_path = None

                # Save failed .tex file for debugging
                if os.path.exists(tex_file):
                    failed_tex_path = os.path.join(
                        str(self.latex_output_dir),
                        f"failed_{os.getpid()}.tex"
                    )
                    shutil.copy(tex_file, failed_tex_path)

                if os.path.exists(log_file):
                    # Save log file for debugging
                    failed_log_path = os.path.join(
                        str(self.latex_output_dir),
                        f"failed_{os.getpid()}.log"
                    )
                    shutil.copy(log_file, failed_log_path)

                    # Extract error messages
                    with open(log_file) as f:
                        log_content = f.read()
                        # Look for error messages
                        if "! " in log_content:
                            error_lines = [
                                line for line in log_content.split("\n")
                                if line.startswith("!")
                            ]
                            if error_lines:
                                error_msg = "\n".join(error_lines[:5])

                return {
                    "success": False,
                    "error": error_msg,
                    "failed_tex_path": failed_tex_path,
                    "failed_log_path": failed_log_path
                }

        except FileNotFoundError:
            return {
                "success": False,
                "error": f"{compiler} not found. Please install LaTeX.",
            }
        except Exception as e:
            logger.error(f"LaTeX compilation error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def render_tikz(
        self,
        tikz_code: str,
        output_format: str = "pdf"
    ) -> dict[str, Any]:
        """
        Render TikZ diagram as standalone image.

        Args:
            tikz_code: TikZ code for the diagram
            output_format: Output format (pdf, png, svg)

        Returns:
            Dictionary with rendered diagram path
        """
        # Wrap TikZ code in standalone document
        latex_content = f"""
\\documentclass[tikz,border=10pt]{{standalone}}
\\usepackage{{tikz}}
\\usetikzlibrary{{arrows.meta,positioning,shapes,calc}}
\\begin{{document}}
{tikz_code}
\\end{{document}}
        """

        # First compile to PDF
        result = await self.compile_latex(
            latex_content,
            format="pdf",
            template="custom"
        )

        if not result["success"]:
            return result

        pdf_path = result["output_path"]

        # Convert to requested format if needed
        if output_format != "pdf":
            try:
                base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                output_path = os.path.join(
                    str(self.latex_output_dir),
                    f"{base_name}.{output_format}"
                )

                if output_format == "png":
                    # Use pdftoppm for PNG conversion
                    process = await asyncio.create_subprocess_exec(
                        "pdftoppm", "-png", "-singlefile",
                        pdf_path, output_path[:-4],
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    await process.communicate()
                elif output_format == "svg":
                    # Use pdf2svg for SVG conversion
                    process = await asyncio.create_subprocess_exec(
                        "pdf2svg", pdf_path, output_path,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    await process.communicate()

                if os.path.exists(output_path):
                    return {
                        "success": True,
                        "output_path": output_path,
                        "format": output_format,
                        "source_pdf": pdf_path,
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Conversion to {output_format} failed",
                    }

            except Exception as e:
                return {
                    "success": False,
                    "error": f"Format conversion error: {str(e)}"
                }

        return result


class MCPLaTeXServer:
    """MCP Server with LaTeX and TikZ integration."""

    def __init__(self, port: int = 8000):
        self.server = Server("mcp-latex-server")
        self.port = port
        self.project_root = Path(os.getenv('MCP_PROJECT_ROOT', '/workspace'))
        self.latex_tool = LaTeXTool(
            self.project_root / os.getenv('DOCUMENT_OUTPUT_DIR', 'documents')
        )
        # 初始化模板管理器
        templates_dir = Path(__file__).parent / "templates"
        self.template_manager = TemplateManager(templates_dir)

        logger.info(f"可用模板数量: {len(self.template_manager.list_templates())}")
        logger.info(f"模板列表: {self.template_manager.list_templates()}")

        self._setup_tools()
        logger.info("MCP 工具注册完成")

    def _setup_tools(self):
        """Register MCP tools."""
        logger.info("注册 MCP 工具...")

        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            """List available tools."""
            return [
                types.Tool(
                    name="compile_latex",
                    description="Compile LaTeX documents to various formats (PDF, DVI, PS)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "LaTeX document content"
                            },
                            "format": {
                                "type": "string",
                                "description": "Output format",
                                "enum": ["pdf", "dvi", "ps"],
                                "default": "pdf"
                            },
                            "template": {
                                "type": "string",
                                "description": "Document template",
                                "enum": ["article", "report", "book", "beamer", "custom"],
                                "default": "article"
                            }
                        },
                        "required": ["content"]
                    }
                ),
                types.Tool(
                    name="render_tikz",
                    description="Render TikZ diagrams as standalone images",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "tikz_code": {
                                "type": "string",
                                "description": "TikZ code for the diagram"
                            },
                            "output_format": {
                                "type": "string",
                                "description": "Output format",
                                "enum": ["pdf", "png", "svg"],
                                "default": "pdf"
                            }
                        },
                        "required": ["tikz_code"]
                    }
                ),
                types.Tool(
                    name="list_templates",
                    description="List all available report templates",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                    }
                ),
                types.Tool(
                    name="generate_diagnosis_report",
                    description="Generate device diagnosis report PDF from structured data",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "object",
                                "description": "Report data fields (32 fields including device_name, health_score, etc.)"
                            },
                            "template_id": {
                                "type": "string",
                                "description": "Template ID",
                                "default": "device_diagnosis"
                            }
                        },
                        "required": ["data"]
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_tool_call(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
            """
            Handle tool calls by routing to the appropriate tool implementation.
            """
            logger.info(f"接收到工具调用: {name}")
            logger.debug(f"参数: {json.dumps(arguments, ensure_ascii=False)[:200]}")

            if name == "compile_latex":
                # Compile LaTeX documents
                result = await self.latex_tool.compile_latex(
                    content=arguments.get('content', ''),
                    format=arguments.get('format', 'pdf'),
                    template=arguments.get('template', 'article')
                )

                # Format response
                if result['success']:
                    log_text = ""
                    if result.get('log_path'):
                        log_text = f"\n📋 Log: {result['log_path']}"

                    response = f"""📄 Document Compiled Successfully!

📄 File: {os.path.basename(result['output_path'])}
📁 Location: {result['output_path']}
📄 Format: {result['format']}
📋 Template: {result['template']}{log_text}"""
                else:
                    response = f"""❌ Document Compilation Failed

Error: {result['error']}"""

                return [types.TextContent(type="text", text=response)]

            elif name == "render_tikz":
                # Render TikZ diagrams
                result = await self.latex_tool.render_tikz(
                    tikz_code=arguments.get('tikz_code', ''),
                    output_format=arguments.get('output_format', 'pdf')
                )

                # Format response
                if result['success']:
                    source_text = ""
                    if result.get('source_pdf'):
                        source_text = f"\n📄 Source PDF: {result['source_pdf']}"

                    response = f"""🎨 TikZ Diagram Rendered Successfully!

🎨 File: {os.path.basename(result['output_path'])}
📁 Location: {result['output_path']}
📄 Format: {result['format']}{source_text}"""
                else:
                    response = f"""❌ TikZ Rendering Failed

Error: {result['error']}"""

                return [types.TextContent(type="text", text=response)]

            elif name == "list_templates":
                # List available templates
                templates = self.template_manager.list_templates()
                template_info_list = []
                for tid in templates:
                    info = self.template_manager.get_template_info(tid)
                    template_info_list.append(f"""
- **{tid}**
  - 名称: {info['name']}
  - 描述: {info['description']}
  - 版本: {info['version']}
  - 必填字段: {len(info['required_fields'])} 个
  - 总字段数: {info['total_fields']}
                """.strip())

                response = f"""📋 可用模板列表 ({len(templates)} 个模板):
{''.join(template_info_list)}"""
                return [types.TextContent(type="text", text=response)]

            elif name == "generate_diagnosis_report":
                # Generate diagnosis report
                data = arguments.get('data', {})
                template_id = arguments.get('template_id', 'device_diagnosis')

                logger.info(f"生成诊断报告: template_id={template_id}, 字段数={len(data)}")
                logger.debug(f"数据字段: {list(data.keys())}")

                try:
                    # 渲染模板
                    logger.info("步骤 1/2: 渲染模板")
                    latex_content = self.template_manager.render_template(template_id, data)
                    logger.info(f"模板渲染成功，LaTeX 内容长度: {len(latex_content)} 字符")

                    # 编译 LaTeX
                    logger.info("步骤 2/2: 编译 PDF")
                    result = await self.latex_tool.compile_latex(
                        content=latex_content,
                        format="pdf",
                        template="custom"
                    )

                    if result['success']:
                        logger.info(f"✓ 诊断报告生成成功: {result['output_path']}")
                        response = f"""✅ 诊断报告生成成功!

📄 模板: {template_id}
📁 位置: {result['output_path']}
📊 填充字段: {len(data)} 个"""
                    else:
                        logger.error(f"✗ 报告生成失败: {result.get('error', '未知错误')}")
                        response = f"""❌ 报告生成失败

错误: {result.get('error', '未知错误')}"""

                except Exception as e:
                    logger.exception(f"报告生成异常: {e}")
                    response = f"""❌ 报告生成失败

错误: {str(e)}"""

                return [types.TextContent(type="text", text=response)]

            else:
                # Unknown tool
                logger.warning(f"未知的工具调用: {name}")
                return [types.TextContent(
                    type="text",
                    text=f"❌ Unknown tool: {name}"
                )]

    def run(self):
        """Run the MCP server."""
        logger.info("=" * 60)
        logger.info("启动 MCP LaTeX 服务器")
        logger.info(f"端口: {self.port}")
        logger.info(f"项目根目录: {self.project_root}")
        logger.info("=" * 60)

        async def main():
            logger.info("等待客户端连接...")
            try:
                async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                    logger.info("✓ 客户端已连接")
                    await self.server.run(
                        read_stream,
                        write_stream,
                        InitializationOptions(
                            server_name="mcp-latex-server",
                            server_version="2.0.0",
                            capabilities=self.server.get_capabilities(
                                notification_options=NotificationOptions(),
                                experimental_capabilities={},
                            ),
                        ),
                    )
            except Exception as e:
                logger.exception(f"服务器运行异常: {e}")
                raise
            finally:
                logger.info("客户端断开连接")

        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在关闭服务器...")
        finally:
            logger.info("=" * 60)
            logger.info("MCP LaTeX 服务器已停止")
            logger.info("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MCP Server with LaTeX and TikZ Integration")
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on"
    )

    args = parser.parse_args()

    server = MCPLaTeXServer(port=args.port)
    server.run()
