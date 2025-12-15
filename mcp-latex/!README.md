# MCP Server LaTeX Integration

A complete implementation guide for integrating LaTeX document compilation with MCP (Model Context Protocol) servers, enabling AI assistants to create professional documents, academic papers, and typeset materials programmatically.

## Example Output

[UE5 Nanite System LaTeX PDF](https://raw.githubusercontent.com/AndrewAltimit/Media/main/ue5_nanite_system.pdf)  ( [source tex file](#file-ue5_nanite_system-tex) )

[DnD Mechanics PDF](https://raw.githubusercontent.com/AndrewAltimit/Media/main/dnd_mechanics_codex.pdf)  ( [source tex file](https://raw.githubusercontent.com/AndrewAltimit/Media/main/dnd_mechanics_codex.tex) )

[WASM Slideshow](https://raw.githubusercontent.com/AndrewAltimit/Media/main/wasm_wasi_wasix_presentation.pdf)  ( [source tex file](https://raw.githubusercontent.com/AndrewAltimit/Media/main/wasm_wasi_wasix_presentation.tex) )

[Slideshow with Overlays](https://raw.githubusercontent.com/AndrewAltimit/Media/main/beamer_overlay_demo.pdf)  ( [source tex file](https://raw.githubusercontent.com/AndrewAltimit/Media/main/beamer_overlay_demo.tex) )

## Usage

See the [template repository](https://github.com/AndrewAltimit/template-repo) for a complete example. LaTeX is integrated into the content-creation mcp server.

![mcp-demo](https://raw.githubusercontent.com/AndrewAltimit/template-repo/refs/heads/main/docs/mcp/architecture/demo.gif)

## Features

- **Containerized Execution**: No local LaTeX installation required
- **Multiple Engines**: Support for pdfLaTeX, XeLaTeX, LuaLaTeX
- **Multiple Output Formats**: PDF, DVI, PS support
- **TikZ Diagrams**: Render complex diagrams as standalone images (PDF, PNG, SVG)
- **Template Support**: Built-in templates for article, report, book, beamer
- **Error Extraction**: Clear error messages from compilation logs
- **Multi-pass Compilation**: Automatic reference resolution
- **MCP Protocol Integration**: Clean AI-assistant interface

## Quick Start

```bash
# Clone this gist
git clone https://gist.github.com/AndrewAltimit/99324d135251d8e80e0f130da8184d07 mcp-latex
cd mcp-latex

# Build and start the container
docker-compose up -d

# Verify installation
docker-compose exec mcp-latex-server python3 -c "
import subprocess
result = subprocess.run(['pdflatex', '--version'], capture_output=True, text=True)
print('✅ pdfLaTeX:', result.stdout.split('\\n')[0])
"
```

## Available Tools

### 1. compile_latex

Compile LaTeX documents to various formats.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `content` | string | Yes | - | LaTeX document content |
| `format` | string | No | "pdf" | Output format: pdf/dvi/ps |
| `template` | string | No | "article" | Document template: article/report/book/beamer/custom |

**Example:**
```python
{
    "tool": "compile_latex",
    "arguments": {
        "content": "\\section{Introduction}\nThis is my document.",
        "format": "pdf",
        "template": "article"
    }
}
```

### 2. render_tikz

Render TikZ diagrams as standalone images.

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `tikz_code` | string | Yes | - | TikZ code for the diagram |
| `output_format` | string | No | "pdf" | Output format: pdf/png/svg |

**Example:**
```python
{
    "tool": "render_tikz",
    "arguments": {
        "tikz_code": "\\begin{tikzpicture}\n\\draw (0,0) circle (1cm);\n\\end{tikzpicture}",
        "output_format": "png"
    }
}
```

## Example Documents

### Basic Article
```latex
\\documentclass{article}
\\usepackage[utf8]{inputenc}
\\title{My First Document}
\\author{AI Assistant}
\\date{\\today}

\\begin{document}
\\maketitle

\\section{Introduction}
This is a simple document created with MCP LaTeX integration.

\\section{Features}
\\begin{itemize}
    \\item Automatic compilation
    \\item Error handling
    \\item Multiple output formats
\\end{itemize}

\\end{document}
```

### Mathematical Document
```latex
\\documentclass{article}
\\usepackage{amsmath}
\\usepackage{amssymb}
\\usepackage{amsthm}

\\newtheorem{theorem}{Theorem}

\\begin{document}
\\title{Mathematical Proofs}
\\maketitle

\\begin{theorem}[Pythagorean Theorem]
In a right triangle with legs $a$ and $b$ and hypotenuse $c$:
\\[a^2 + b^2 = c^2\\]
\\end{theorem}

\\end{document}
```

### TikZ Diagram Examples

#### Simple Flow Chart
```latex
\\begin{tikzpicture}[node distance=2cm]
\\node[rectangle, draw] (start) {Start};
\\node[rectangle, draw, below of=start] (process) {Process};
\\node[diamond, draw, below of=process, aspect=2] (decision) {Decision?};
\\node[rectangle, draw, below left=1cm and 1cm of decision] (yes) {Yes};
\\node[rectangle, draw, below right=1cm and 1cm of decision] (no) {No};

\\draw[->] (start) -- (process);
\\draw[->] (process) -- (decision);
\\draw[->] (decision) -- node[left] {Y} (yes);
\\draw[->] (decision) -- node[right] {N} (no);
\\end{tikzpicture}
```

#### Graph Visualization
```latex
\\begin{tikzpicture}
\\node[circle, draw] (1) at (0,0) {1};
\\node[circle, draw] (2) at (2,0) {2};
\\node[circle, draw] (3) at (1,1.5) {3};
\\draw (1) -- (2);
\\draw (2) -- (3);
\\draw (3) -- (1);
\\end{tikzpicture}
```

### Beamer Presentation
```latex
\\documentclass{beamer}
\\usetheme{Madrid}
\\usecolortheme{default}

\\title{Introduction to LaTeX}
\\author{MCP Server}
\\date{\\today}

\\begin{document}

\\frame{\\titlepage}

\\begin{frame}
\\frametitle{Overview}
\\begin{itemize}
\\item What is LaTeX?
\\item Why use LaTeX?
\\item Basic concepts
\\end{itemize}
\\end{frame}

\\end{document}
```

## Architecture

The MCP LaTeX server provides two main components:

1. **LaTeXTool Class**: Core functionality for document compilation and TikZ rendering
   - Handles temporary file management
   - Executes LaTeX compilers
   - Manages output files and error extraction

2. **MCPLaTeXServer Class**: MCP protocol integration
   - Registers available tools
   - Handles async communication
   - Formats responses for AI assistants

## Installation Requirements

- Docker and Docker Compose
- At least 2GB of available disk space (for full TeX Live installation)

## Environment Variables

Configure the server behavior with these environment variables:

```bash
# .env
MCP_PORT=8000
MCP_LOG_LEVEL=INFO
MCP_PROJECT_ROOT=/workspace
DOCUMENT_OUTPUT_DIR=documents
```

## Troubleshooting

### Common Issues

1. **LaTeX package not found**
   ```
   ! LaTeX Error: File `package.sty' not found.
   ```
   Solution: Install missing package in Dockerfile:
   ```dockerfile
   RUN tlmgr install package-name
   ```

2. **Unicode errors with pdflatex**
   ```
   Package inputenc Error: Unicode character not set up
   ```
   Solution: Use XeLaTeX or LuaLaTeX for Unicode support

3. **TikZ conversion failures**
   ```
   Conversion to png failed
   ```
   Solution: Ensure pdftoppm or pdf2svg is installed

### Debugging

1. **Check container logs:**
   ```bash
   docker-compose logs -f mcp-latex-server
   ```

2. **Access container shell:**
   ```bash
   docker exec -it mcp-latex-server bash
   ```

3. **Test LaTeX directly:**
   ```bash
   echo '\\documentclass{article}\\begin{document}Test\\end{document}' > test.tex
   pdflatex test.tex
   ```

## Advanced Usage

### Custom Templates

You can extend the server to support custom templates by modifying the templates dictionary in the compile_latex method.

### Bibliography Support

For documents with citations, you may need to run bibtex or biber. This can be added as an additional compilation step.

### Performance Optimization

- Use volume mounts for TeX package cache
- Limit container resources to prevent resource exhaustion
- Consider using a lighter TeX distribution for specific use cases

## Security Considerations

- Each compilation runs in an isolated temporary directory
- Docker container provides additional security layer
- No shell commands are executed with user input
- Resource limits prevent denial of service
