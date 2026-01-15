# mcp-latex/ - LaTeX MCP Server

## OVERVIEW

Dockerized MCP (Model Context Protocol) server for LaTeX compilation. Provides template rendering + PDF generation.

## STRUCTURE

```
mcp-latex/
├── mcp_latex_tool.py    # MCP server implementation
├── template_manager.py  # Jinja2 template engine
├── run_server.py        # Entry point
├── Dockerfile           # TeX Live + Python image
├── docker-compose.yml   # Service configuration
├── templates/           # Jinja2 LaTeX templates
│   └── device_diagnosis/  # 32-field diagnosis report
└── logs/                # Runtime logs (gitignored)
```

## WHERE TO LOOK

| Task | Location |
|------|----------|
| Add LaTeX template | templates/ - create new directory |
| Modify template fields | templates/*/schema.json + template.tex.j2 |
| Add MCP tool | mcp_latex_tool.py |
| Change compilation | mcp_latex_tool.py - compile_latex() |

## CONVENTIONS

- **Template structure**: Each template = directory with `template.tex.j2` + `schema.json`
- **Jinja2 rendering**: template_manager.py handles data → LaTeX
- **MCP stdio**: Server communicates via stdin/stdout (no console logging)
- **Logging**: File-only to `/workspace/logs/`

## DOCKER

```bash
# Build
docker-compose build

# Run
docker-compose up -d

# Logs
docker-compose logs -f mcp-latex-server
```

## ANTI-PATTERNS

- **Hardcoded user**: docker-compose.yml uses `1000:1000` - may need adjustment
- **Heavy image**: Full TeX Live installation (~2GB)
