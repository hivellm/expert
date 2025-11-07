# OpenSpec - Expert System

> AI agent context and workflow for the Expert System project

## What is OpenSpec?

OpenSpec provides structured context for AI agents working on the Expert System, ensuring:
- Consistent understanding of project architecture
- Standardized workflows and conventions
- Quality guidelines for changes
- Clear communication patterns

---

## Quick Start for AI Agents

### 1. Populate Project Context

**First interaction:**
```
"Please read openspec/project.md and help me fill it out
 with details about my project, tech stack, and conventions"
```

→ Agent reads `project.md` to understand the Expert System architecture, tech stack, and coding conventions.

### 2. Create Change Proposals

**For new features:**
```
"I want to add [FEATURE]. Please create an OpenSpec change 
 proposal for this feature"
```

→ Agent uses template from `AGENTS.md` to propose structured changes.

### 3. Learn the Workflow

**Understand how to work on this project:**
```
"Please explain the OpenSpec workflow from openspec/AGENTS.md
 and how I should work with you on this project"
```

→ Agent explains how it will interact with the codebase.

---

## Files

| File | Purpose |
|------|---------|
| [project.md](project.md) | Project overview, tech stack, architecture |
| [AGENTS.md](AGENTS.md) | Workflow guidelines for AI agents |
| README.md | This file - OpenSpec introduction |

---

## When to Reference OpenSpec

### Agent Should Read project.md When:
- Starting work on Expert System
- User asks "what is this project?"
- Need to understand architecture
- Checking tech stack or conventions

### Agent Should Read AGENTS.md When:
- Proposing significant changes
- Creating new experts
- Updating documentation
- Need workflow guidance

---

## Example Interactions

### Creating a New Expert

**User**: "Create the English-basic expert"

**Agent**:
1. Reads `openspec/project.md` → understands expert system
2. Reads `openspec/AGENTS.md` → follows expert creation workflow
3. Checks `expert/experts/README.md` → sees English is planned
4. Creates complete structure with manifest.json
5. Sets load_order: 3 (language expert)
6. No dependencies (standalone)
7. Updates experts/README.md

### Proposing Architecture Change

**User**: "I want to support 15 experts instead of 10"

**Agent**:
1. Reads `openspec/AGENTS.md` → uses change proposal template
2. Analyzes impact:
   - VRAM: +250MB (5 more experts)
   - Latency: +5ms loading
   - Complexity: routing logic
3. Proposes changes to:
   - PERFORMANCE.md (update VRAM budgets)
   - ARCHITECTURE.md (update constraint)
   - manifest.json schema (max_chain)
4. Lists risks and trade-offs
5. Asks user to approve

---

## Project Context Summary

**Current Phase**: Documentation complete, implementation ready  
**Tech Stack**: Rust (runtime) + Python (training)  
**Distribution**: Git-based (no NPM/PyPI)  
**Base Model**: Qwen3-0.6B (0.5GB VRAM)  
**Expert Limit**: 10 per inference  
**Target Hardware**: 8-16GB VRAM (consumer GPUs)

---

## Quick Links

- [Main README](../README.md)
- [Architecture](../docs/ARCHITECTURE.md)
- [Current Status](../STATUS.md)
- [Roadmap](../ROADMAP.md)
- [CLI Commands](../docs/CLI.md)

---

**For human contributors**: This OpenSpec directory is for AI agent context. For human-readable docs, start with [../README.md](../README.md).

---

Last Updated: 2025-11-02

