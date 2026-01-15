"""LaTeX MCP Client - Async-first design for LangGraph integration."""

import asyncio
import logging
from pathlib import Path
from typing import Any

import mcp.types
from mcp import StdioServerParameters
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

from rag_agent.config import config

logger = logging.getLogger(__name__)


def convert_container_path_to_host(container_path: str) -> str:
    if not container_path.startswith("/workspace/documents/"):
        return container_path

    relative_path = container_path.replace("/workspace/documents/", "")
    project_root = Path(__file__).parent.parent.parent
    host_path = project_root / "mcp-latex" / "documents" / relative_path
    return str(host_path)


def create_server_params() -> StdioServerParameters:
    return StdioServerParameters(
        command="podman",
        args=["exec", "-i", "mcp-latex-server", "python", "-m", "mcp_latex_tool"],
        env=None,
    )


def _parse_mcp_response(result: Any) -> tuple[bool, str]:
    if not result.content:
        return False, "Empty response from server"

    content_block = result.content[0]
    if isinstance(content_block, mcp.types.TextContent):
        response_text = content_block.text
    else:
        response_text = str(content_block)

    is_error = "❌" in response_text or "Failed" in response_text or "失败" in response_text
    return not is_error, response_text


def _extract_path_from_response(response_text: str, marker: str) -> str | None:
    for line in response_text.split("\n"):
        line = line.strip()
        if marker in line:
            container_path = line.split(marker, 1)[1].strip()
            return convert_container_path_to_host(container_path)
    return None


async def compile_latex_async(
    content: str,
    output_format: str = "pdf",
    template: str = "article",
    server_params: StdioServerParameters | None = None,
) -> dict[str, Any]:
    if server_params is None:
        server_params = create_server_params()

    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                result = await session.call_tool(
                    "compile_latex",
                    arguments={"content": content, "format": output_format, "template": template},
                )

                success, response_text = _parse_mcp_response(result)
                if not success:
                    return {"success": False, "error": response_text}

                return {
                    "success": True,
                    "format": output_format,
                    "template": template,
                    "output_path": _extract_path_from_response(response_text, "📁 Location:"),
                    "log_path": _extract_path_from_response(response_text, "📋 Log:"),
                }

    except Exception as e:
        logger.error(f"LaTeX compilation error: {e}")
        return {"success": False, "error": str(e)}


async def render_tikz_async(
    tikz_code: str,
    output_format: str = "pdf",
    server_params: StdioServerParameters | None = None,
) -> dict[str, Any]:
    if server_params is None:
        server_params = create_server_params()

    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                result = await session.call_tool(
                    "render_tikz",
                    arguments={"tikz_code": tikz_code, "output_format": output_format},
                )

                success, response_text = _parse_mcp_response(result)
                if not success:
                    return {"success": False, "error": response_text}

                return {
                    "success": True,
                    "format": output_format,
                    "output_path": _extract_path_from_response(response_text, "📁 Location:"),
                }

    except Exception as e:
        logger.error(f"TikZ rendering error: {e}")
        return {"success": False, "error": str(e)}


async def generate_diagnosis_report_async(
    data: dict[str, Any],
    template_id: str | None = None,
    server_params: StdioServerParameters | None = None,
) -> dict[str, Any]:
    if template_id is None:
        template_id = config.DIAGNOSIS_TEMPLATE_ID
    if server_params is None:
        server_params = create_server_params()

    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                result = await session.call_tool(
                    "generate_diagnosis_report",
                    arguments={"data": data, "template_id": template_id},
                )

                success, response_text = _parse_mcp_response(result)
                if not success:
                    return {"success": False, "error": response_text}

                return {
                    "success": True,
                    "template_id": template_id,
                    "output_path": _extract_path_from_response(response_text, "📁 位置:"),
                }

    except Exception as e:
        logger.error(f"Diagnosis report generation error: {e}")
        return {"success": False, "error": str(e)}


def compile_latex(
    content: str,
    format: str = "pdf",
    template: str = "article",
) -> dict[str, Any]:
    return asyncio.run(compile_latex_async(content, format, template))
