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
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server, InitializationOptions
from pydantic import AnyUrl

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LaTeXTool:
    """LaTeX document compilation and TikZ rendering tool for MCP."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.latex_output_dir = output_dir / "latex"
        self.latex_output_dir.mkdir(exist_ok=True, parents=True)
        
    async def compile_latex(
        self,
        content: str,
        format: str = "pdf",
        template: str = "article"
    ) -> Dict[str, Any]:
        """
        Compile LaTeX document to various formats.
        
        Args:
            content: LaTeX document content
            format: Output format (pdf, dvi, ps)
            template: Document template to use
            
        Returns:
            Dictionary with compiled document path and metadata
        """
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

            # Create temporary directory
            with tempfile.TemporaryDirectory() as tmpdir:
                # Write LaTeX file
                tex_file = os.path.join(tmpdir, "document.tex")
                with open(tex_file, "w") as f:
                    f.write(content)

                # Choose compiler based on format
                if format == "pdf":
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
                    stdout, stderr = await process.communicate()
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

                # Extract error from log file
                log_file = os.path.join(tmpdir, "document.log")
                error_msg = "Compilation failed"
                if os.path.exists(log_file):
                    with open(log_file, "r") as f:
                        log_content = f.read()
                        # Look for error messages
                        if "! " in log_content:
                            error_lines = [
                                line for line in log_content.split("\n")
                                if line.startswith("!")
                            ]
                            if error_lines:
                                error_msg = "\n".join(error_lines[:5])

                return {"success": False, "error": error_msg}

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
    ) -> Dict[str, Any]:
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
        self._setup_tools()
        
    def _setup_tools(self):
        """Register MCP tools."""
        
        @self.server.call_tool()
        async def compile_latex(arguments: Dict[str, Any]) -> List[types.TextContent]:
            """
            Compile LaTeX documents to various formats.
            
            Parameters:
            - content: LaTeX document content
            - format: Output format ("pdf", "dvi", "ps", default: "pdf")
            - template: Document template ("article", "report", "book", "beamer", "custom", default: "article")
            """
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
            
        @self.server.call_tool()
        async def render_tikz(arguments: Dict[str, Any]) -> List[types.TextContent]:
            """
            Render TikZ diagrams as standalone images.
            
            Parameters:
            - tikz_code: TikZ code for the diagram
            - output_format: Output format ("pdf", "png", "svg", default: "pdf")
            """
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
    
    def run(self):
        """Run the MCP server."""
        logger.info(f"Starting MCP LaTeX server on port {self.port}")
        
        async def main():
            async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="mcp-latex-server",
                        server_version="1.0.0",
                        capabilities=self.server.get_capabilities(
                            notification_options=NotificationOptions(),
                            experimental_capabilities={},
                        ),
                    ),
                )
        
        asyncio.run(main())


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