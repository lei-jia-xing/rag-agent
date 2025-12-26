# MCP Server with LaTeX and TikZ Integration
FROM python:3.11-slim

# Metadata
LABEL maintainer="MCP LaTeX Integration"
LABEL description="MCP server with LaTeX document compilation and TikZ diagram rendering"
LABEL version="1.0.0"

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LATEX_ENGINE_DEFAULT=pdflatex \
    MCP_PROJECT_ROOT=/workspace \
    DOCUMENT_OUTPUT_DIR=documents

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git curl wget build-essential pkg-config \
    texlive-latex-base texlive-latex-extra \
    texlive-fonts-recommended texlive-fonts-extra \
    texlive-pictures texlive-lang-chinese \
    texlive-xetex texlive-luatex latexmk \
    poppler-utils pdf2svg ghostscript \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create workspace
WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    python -c "import mcp; print('MCP SDK installed')" && \
    python -c "import jinja2; print('Jinja2 installed')" && \
    pdflatex --version && xelatex --version && lualatex --version && \
    which pdftoppm && which pdf2svg && \
    rm /tmp/requirements.txt

# Create non-root user
RUN useradd --create-home --shell /bin/bash mcp && \
    mkdir -p /workspace/documents/latex /workspace/logs && \
    chown -R mcp:mcp /workspace

# Copy application files and templates
COPY --chown=mcp:mcp template_manager.py mcp_latex_tool.py run_server.py /workspace/
COPY --chown=mcp:mcp templates/ /workspace/templates/

# Switch to non-root user
USER mcp

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import mcp; print('OK')" && pdflatex --version > /dev/null || exit 1

# Expose MCP server port (for stdio reference)
EXPOSE 8000

# Start MCP server
CMD ["python3", "run_server.py"]
