"""LaTeX MCP Client

Simple client for interacting with the LaTeX MCP server for document compilation
and TikZ diagram rendering using MCP Python SDK.
Following the official MCP SDK example pattern.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

import mcp.types
from mcp import StdioServerParameters
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


def convert_container_path_to_host(container_path: str) -> str:
    """Convert container path to host path.

    Container: /workspace/documents/latex/document_49.pdf
    Host: /home/hunter/Workspace/rag-agent/mcp-latex/documents/latex/document_49.pdf
    """
    if container_path.startswith("/workspace/documents/"):
        # Get the relative path after /workspace/documents/
        relative_path = container_path.replace("/workspace/documents/", "")
        # Get the project root (parent of rag_agent package)
        current_file = Path(__file__)  # rag_agent/mcp/latex_client.py
        project_root = current_file.parent.parent.parent  # Go up 3 levels
        # Build host path
        host_path = project_root / "mcp-latex" / "documents" / relative_path
        return str(host_path)
    return container_path


def create_server_params() -> StdioServerParameters:
    """Create StdioServerParameters for connecting to LaTeX MCP server.

    Assumes the mcp-latex-server container is running via podman compose.
    """
    return StdioServerParameters(
        command="podman",
        args=["exec", "-i", "mcp-latex-server", "python", "-m", "mcp_latex_tool"],
        env=None,
    )


async def compile_latex_async(
    content: str,
    format: str = "pdf",
    template: str = "article",
    server_params: StdioServerParameters | None = None,
) -> dict[str, Any]:
    """Compile LaTeX document asynchronously.

    Args:
        content: LaTeX document content.
        format: Output format: "pdf", "dvi", or "ps".
        template: Document template: "article", "report", "book", "beamer", or "custom".
        server_params: Optional StdioServerParameters. If None, uses default.

    Returns:
        Dictionary with compilation results.
    """
    if server_params is None:
        server_params = create_server_params()

    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                # Initialize the connection
                await session.initialize()

                # Call the compile_latex tool
                result = await session.call_tool(
                    "compile_latex",
                    arguments={
                        "content": content,
                        "format": format,
                        "template": template,
                    },
                )

                # Parse the result
                if not result.content:
                    return {"success": False, "error": "Empty response from server"}

                # The server returns TextContent with formatted message
                # Extract the text from the first content block
                content_block = result.content[0]
                if isinstance(content_block, mcp.types.TextContent):
                    response_text = content_block.text
                else:
                    # Try to get text from any available attribute
                    response_text = str(content_block)

                # Check for success/failure
                if "❌" in response_text or "Failed" in response_text:
                    # Try to extract error message
                    lines = [line.strip() for line in response_text.split("\n") if line.strip()]
                    error_msg = lines[-1] if lines else "Compilation failed"
                    return {"success": False, "error": error_msg, "raw_response": response_text}

                # Extract information from success response
                lines = response_text.split("\n")
                result_data = {
                    "success": True,
                    "format": format,
                    "template": template,
                    "raw_response": response_text,
                }

                # Parse the formatted response
                for line in lines:
                    line = line.strip()
                    if "📁 Location:" in line:
                        container_path = line.split("📁 Location:", 1)[1].strip()
                        result_data["output_path"] = convert_container_path_to_host(container_path)
                    elif "📄 File:" in line:
                        result_data["filename"] = line.split("📄 File:", 1)[1].strip()
                    elif "📄 Format:" in line:
                        result_data["format"] = line.split("📄 Format:", 1)[1].strip()
                    elif "📋 Template:" in line:
                        result_data["template"] = line.split("📋 Template:", 1)[1].strip()
                    elif "📋 Log:" in line:
                        container_path = line.split("📋 Log:", 1)[1].strip()
                        result_data["log_path"] = convert_container_path_to_host(container_path)

                return result_data

    except Exception as e:
        logger.error(f"LaTeX compilation error: {e}")
        return {"success": False, "error": str(e)}


async def render_tikz_async(
    tikz_code: str,
    output_format: str = "pdf",
    server_params: StdioServerParameters | None = None,
) -> dict[str, Any]:
    """Render TikZ diagram asynchronously.

    Args:
        tikz_code: TikZ code for the diagram.
        output_format: Output format: "pdf", "png", or "svg".
        server_params: Optional StdioServerParameters. If None, uses default.

    Returns:
        Dictionary with rendering results.
    """
    if server_params is None:
        server_params = create_server_params()

    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                # Initialize the connection
                await session.initialize()

                # Call the render_tikz tool
                result = await session.call_tool(
                    "render_tikz",
                    arguments={
                        "tikz_code": tikz_code,
                        "output_format": output_format,
                    },
                )

                # Parse the result
                if not result.content:
                    return {"success": False, "error": "Empty response from server"}

                # Extract text from content block
                content_block = result.content[0]
                if isinstance(content_block, mcp.types.TextContent):
                    response_text = content_block.text
                else:
                    response_text = str(content_block)

                # Check for success/failure
                if "❌" in response_text or "Failed" in response_text:
                    lines = [line.strip() for line in response_text.split("\n") if line.strip()]
                    error_msg = lines[-1] if lines else "Rendering failed"
                    return {"success": False, "error": error_msg, "raw_response": response_text}

                lines = response_text.split("\n")
                result_data = {
                    "success": True,
                    "format": output_format,
                    "raw_response": response_text,
                }

                for line in lines:
                    line = line.strip()
                    if "📁 Location:" in line:
                        container_path = line.split("📁 Location:", 1)[1].strip()
                        result_data["output_path"] = convert_container_path_to_host(container_path)
                    elif "🎨 File:" in line:
                        result_data["filename"] = line.split("🎨 File:", 1)[1].strip()
                    elif "📄 Format:" in line:
                        result_data["format"] = line.split("📄 Format:", 1)[1].strip()
                    elif "📄 Source PDF:" in line:
                        container_path = line.split("📄 Source PDF:", 1)[1].strip()
                        result_data["source_pdf"] = convert_container_path_to_host(container_path)

                return result_data

    except Exception as e:
        logger.error(f"TikZ rendering error: {e}")
        return {"success": False, "error": str(e)}


# Synchronous versions for easier integration
def compile_latex(
    content: str,
    format: str = "pdf",
    template: str = "article",
    server_params: StdioServerParameters | None = None,
) -> dict[str, Any]:
    """Compile LaTeX document synchronously."""
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create new event loop if current one is running
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        # No event loop, create new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(compile_latex_async(content, format, template, server_params))


def render_tikz(
    tikz_code: str,
    output_format: str = "pdf",
    server_params: StdioServerParameters | None = None,
) -> dict[str, Any]:
    """Render TikZ diagram synchronously."""
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create new event loop if current one is running
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        # No event loop, create new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(render_tikz_async(tikz_code, output_format, server_params))


async def generate_diagnosis_report_async(
    data: dict[str, Any],
    template_id: str = "device_diagnosis",
    server_params: StdioServerParameters | None = None,
) -> dict[str, Any]:
    """Generate diagnosis report asynchronously.

    Args:
        data: Report data fields (32 fields)
        template_id: Template ID (default: device_diagnosis)
        server_params: Optional StdioServerParameters. If None, uses default.

    Returns:
        Dictionary with generation results.
    """
    if server_params is None:
        server_params = create_server_params()

    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                # Initialize the connection
                await session.initialize()

                # Call the generate_diagnosis_report tool
                result = await session.call_tool(
                    "generate_diagnosis_report",
                    arguments={
                        "data": data,
                        "template_id": template_id,
                    },
                )

                # Parse the result
                if not result.content:
                    return {"success": False, "error": "Empty response from server"}

                content_block = result.content[0]
                if isinstance(content_block, mcp.types.TextContent):
                    response_text = content_block.text
                else:
                    response_text = str(content_block)

                # Check for success/failure
                if "❌" in response_text or "失败" in response_text:
                    # Extract error message
                    lines = [line.strip() for line in response_text.split("\n") if line.strip()]
                    error_msg = lines[-1] if lines else "Generation failed"
                    return {"success": False, "error": error_msg, "raw_response": response_text}

                # Extract information from success response
                result_data = {
                    "success": True,
                    "template_id": template_id,
                    "raw_response": response_text,
                }

                for line in response_text.split("\n"):
                    line = line.strip()
                    if "📁 位置:" in line:
                        container_path = line.split("📁 位置:", 1)[1].strip()
                        result_data["output_path"] = convert_container_path_to_host(container_path)

                return result_data

    except Exception as e:
        logger.error(f"Diagnosis report generation error: {e}")
        return {"success": False, "error": str(e)}


def generate_diagnosis_report(
    data: dict[str, Any],
    template_id: str = "device_diagnosis",
    server_params: StdioServerParameters | None = None,
) -> dict[str, Any]:
    """Generate diagnosis report synchronously."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create new event loop in a thread
            import concurrent.futures

            def run_in_new_loop():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(
                        generate_diagnosis_report_async(data, template_id, server_params)
                    )
                finally:
                    new_loop.close()

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_new_loop)
                return future.result(timeout=120)
        else:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(generate_diagnosis_report_async(data, template_id, server_params))
            finally:
                loop.close()
    except RuntimeError:
        # No event loop, create new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(generate_diagnosis_report_async(data, template_id, server_params))
        finally:
            loop.close()
