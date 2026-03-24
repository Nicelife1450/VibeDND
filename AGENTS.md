# AGENTS.md - VibeDND Repository Guide

## Repository Overview

This is a **data repository** containing power line inspection reports (巡检报告) for electrical distribution infrastructure. The repository currently contains:
- Inspection images (JPG format) organized by defect type
- Excel spreadsheets with defect summaries
- No source code, build tools, or test frameworks

## Project Structure

```
VibeDND/
└── 巡检报告/
    └── 2024/
        ├── 10kV鼓山变623福马东线缺陷/
        ├── 10kV鼓山变654下歧线缺陷/
        └── 10kV鼓山变615魁岐线缺陷/
```

Each inspection line directory contains:
- `缺陷原图/` - Original defect images organized by equipment type
- `缺陷圈图/` - Annotated defect images with circles/highlights
- Excel summary files (缺陷汇总表.xlsx)

## Build/Lint/Test Commands

**No build system exists** - this is a data-only repository.

If code is added in the future, consider:
- Python: `pytest` for tests, `ruff` for linting, `mypy` for type checking
- Node.js: `npm test`, `npm run lint`, `npm run typecheck`

## Code Style Guidelines

Since no code exists yet, the following guidelines are recommended for future development:

### Imports
- Python: Use absolute imports from project root
- Group imports: standard library, third-party, local
- Sort imports alphabetically within each group

### Formatting
- Python: Use `black` formatter, line length 88
- Use 4 spaces for indentation (no tabs)
- UTF-8 encoding for all files (Chinese characters present)

### Types
- Use type hints for all function parameters and return values
- Prefer explicit types over `Any`
- Use `Optional[T]` for nullable values

### Naming Conventions
- Python: `snake_case` for functions/variables, `PascalCase` for classes
- Constants: `UPPER_SNAKE_CASE`
- Private members: prefix with underscore `_private_var`
- Chinese directory/file names should be preserved

### Error Handling
- Use specific exception types, not bare `except:`
- Log errors with context before re-raising
- Validate input data early (fail fast)

### Documentation
- Use docstrings for public functions/classes
- Keep comments minimal - code should be self-documenting
- Document any assumptions about data formats

## Data Conventions

### Image Naming Pattern
Images follow the pattern:
```
{line_name}_{pole_id}_{defect_description}_{severity}.JPG
```
Example: `10kV华映线_福马路10-71-线路小号侧通道距树木距离不够-危急缺陷.JPG`

### Defect Severity Levels
- 危急缺陷 (Critical)
- 严重缺陷 (Serious)
- 一般缺陷 (General)

### Equipment Categories
- 架空线路 (Overhead lines)
- 柱上真空开关 (Pole-mounted vacuum switches)
- 配电变压器 (Distribution transformers)
- 电容器 (Capacitors)
- 柱上隔离开关 (Pole-mounted isolation switches)

## File Operations

When working with this repository:
- Preserve Chinese characters in filenames and paths
- Handle Excel files with `openpyxl` or `pandas`
- Image processing: use `Pillow` or `opencv-python`
- Always use UTF-8 encoding for text operations

## Future Development Suggestions

If adding code to process inspection data:
1. Create a `src/` directory for source code
2. Add `requirements.txt` or `pyproject.toml`
3. Consider using `pandas` for Excel processing
4. Add unit tests in `tests/` directory
5. Include a `README.md` with setup instructions

## Notes

- This repository contains sensitive infrastructure inspection data
- Do not commit any credentials or API keys
- Be mindful of file sizes when processing large image datasets
