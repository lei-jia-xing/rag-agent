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
    # LaTeX settings
    LATEX_ENGINE_DEFAULT=pdflatex \
    MCP_PROJECT_ROOT=/workspace \
    DOCUMENT_OUTPUT_DIR=documents

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Basic tools
    git \
    curl \
    wget \
    # Build essentials for Python packages
    build-essential \
    pkg-config \
    # LaTeX installation (using texlive-latex-extra for TikZ support)
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-pictures \
    # Additional LaTeX engines
    texlive-xetex \
    texlive-luatex \
    # LaTeX utilities
    latexmk \
    # PDF conversion tools
    poppler-utils \
    pdf2svg \
    ghostscript \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create workspace
WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    # Verify installations
    python -c "import mcp; print('MCP SDK installed')" && \
    # Verify LaTeX installation
    pdflatex --version && \
    xelatex --version && \
    lualatex --version && \
    # Verify conversion tools
    which pdftoppm && \
    which pdf2svg && \
    rm /tmp/requirements.txt

# Create non-root user
RUN useradd --create-home --shell /bin/bash mcp && \
    mkdir -p /workspace/documents/latex /workspace/logs && \
    chown -R mcp:mcp /workspace

# Copy application files
COPY --chown=mcp:mcp *.py /workspace/
RUN chmod +x /workspace/run_server.py /workspace/mcp_latex_tool.py

# Switch to non-root user
USER mcp

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import mcp; print('OK')" && pdflatex --version > /dev/null || exit 1

# Expose MCP server port (for stdio reference)
EXPOSE 8000

# Start wrapper to keep container running
# For actual MCP usage, connect your client to stdio
CMD ["python3", "run_server.py"]