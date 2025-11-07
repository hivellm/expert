<!-- EXPERT_SYSTEM_AGENTS:START -->
# Expert System - Agent Workflow

> Guidelines for AI agents working on the Expert System project

**CRITICAL**: These guidelines are mandatory for all AI agents working on this codebase.

## Overview

This document defines how AI agents should interact with the Expert System codebase, propose changes, and maintain consistency with the architecture.

---

## Agent Responsibilities

### 1. Understanding the Architecture

Before making changes, agents should:

1. **Read core docs**: ARCHITECTURE.md, EXPERT_FORMAT.md, CLI.md
2. **Understand constraints**: Max 10 experts, no merging, KV cache isolation
3. **Follow tech stack**: Rust for runtime, Python for training
4. **Respect conventions**: No custom scripts, CLI-only operations

### 2. Proposing Changes

When user requests a feature:

1. **Clarify requirements**: Ask specific questions if ambiguous
2. **Check existing docs**: Don't duplicate, update existing
3. **Follow roadmap**: Align with P0-P6 phases
4. **Consider dependencies**: Expert dependencies, load order
5. **Validate against constraints**: VRAM limits, expert count

### 3. Making Changes

```
1. Understand request
   ↓
2. Check current state (read files)
   ↓
3. Propose plan (if complex)
   ↓
4. Implement changes
   ↓
5. Update documentation
   ↓
6. Validate (check manifests, structure)
```

---

## Documentation Guidelines

### When to Create New Docs

**Create new doc when:**
- Introducing new major concept (e.g., CLI.md, GIT_DISTRIBUTION.md)
- New phase/feature requires detailed spec
- Cross-cutting concern not fitting existing docs

**DON'T create new docs for:**
- Minor features (add to existing docs)
- Examples (use examples/ directory)
- Temporary notes (use STATUS.md)

### When to Update Existing Docs

**Always update** when:
- Architecture changes
- New expert added
- Workflow changes
- Performance targets shift

### Documentation Limits

**Keep to essential docs:**
- README.md (overview)
- STATUS.md (progress tracking)
- ROADMAP.md (implementation phases)
- QUICKSTART.md (practical guide)
- CHANGELOG (if needed, or use STATUS.md)
- Technical docs in `/docs/` (architecture, specs, guides)

**Avoid creating:**
- Multiple overlapping guides
- Per-feature documentation
- Marketing materials
- Tutorials (unless essential)

---

## Expert Creation Workflow

### For Agents Creating New Experts

```bash
# 1. Check if expert needed
# - Is it in roadmap?
# - Does similar expert exist?
# - What are dependencies?

# 2. Create structure
mkdir expert/experts/expert-<name>
cd expert/experts/expert-<name>

# 3. Create manifest.json (complete with all fields)
{
  "name": "...",
  "training": {
    "dataset": {
      "generation": { ... }  # Complete config
    },
    "config": { ... }        # Complete config
  },
  "constraints": {
    "load_order": N,
    "requires": [...]
  }
}

# 4. Create README.md (usage examples, capabilities)

# 5. Create tests/test_cases.json (validation)

# 6. Update /expert/experts/README.md (add to list)
```

### Validation Checklist

- [ ] manifest.json is valid (all required fields)
- [ ] load_order is set correctly
- [ ] Dependencies declared in `requires`
- [ ] Training config is complete
- [ ] Test cases exist
- [ ] README.md documents usage
- [ ] Added to experts/README.md list

---

## Change Proposal Template

When proposing significant changes:

```markdown
## Proposed Change: [Title]

**Type**: Architecture | Feature | Expert | Documentation  
**Phase**: P0 | P1 | P2 | P3 | P4 | P5 | P6  
**Impact**: High | Medium | Low

### Problem

What problem does this solve?

### Solution

How does this solve it?

### Changes Required

1. File: path/to/file.md
   - What: Add section on X
   - Why: Needed for Y

2. File: path/to/other.rs
   - What: Implement Z
   - Why: Required for W

### Constraints Affected

- VRAM budget: +/- X MB
- Latency: +/- X ms
- Expert limit: Still 10 or change?

### Dependencies

- Requires: [other features/experts]
- Blocks: [what this blocks]

### Testing Plan

How to validate this works?

### Documentation Updates

Which docs need updating?

### Risks

What could go wrong?
```

---

## Common Tasks

### Adding a New Expert

1. Check roadmap - is this expert planned?
2. Determine dependencies and load_order
3. Create expert directory in `/expert/experts/`
4. Write complete manifest.json
5. Create README.md with examples
6. Add test cases
7. Update experts/README.md

### Updating Manifest Schema

1. Update EXPERT_FORMAT.md with new fields
2. Update all example manifests
3. Update expert-repository-template
4. Update CLI.md if commands affected
5. Note in STATUS.md changelog

### Performance Optimization

1. Document current performance (STATUS.md)
2. Propose optimization
3. Update PERFORMANCE.md with new targets
4. Update ROADMAP.md if affects timeline
5. Validate against VRAM/latency constraints

---

## Communication with User

### User Preferences

- **Honesty over agreement**: Be critical, don't just agree
- **No flattery**: Direct, honest communication
- **Admit uncertainty**: Say "I don't know" if unsure
- **Focus on request**: Don't over-engineer
- **Realistic expectations**: Set honest timelines

### When to Ask Questions

**Ask when:**
- Requirements are ambiguous
- Multiple valid approaches exist
- Decision affects architecture significantly
- User input needed for trade-offs

**Don't ask when:**
- Can infer from context
- Standard practice exists
- Documented in existing files

---

## Git Workflow

### Commits

Agent can suggest commits but **never push**:

```bash
# Agent can do:
git add .
git commit -m "Add JSON parser expert structure"
git tag v2.0.0

# Agent returns commands for user:
echo "Run these commands to push:"
echo "  git push origin main"
echo "  git push origin v2.0.0"
```

### No Destructive Operations

**Never use:**
- `git reset --hard`
- `git push --force`
- `git rebase` (on shared branches)
- `--no-verify` flags

---

## Expert System Specific Rules

### 1. Expert Dependencies

Always check `requires` field when creating experts:

```json
{
  "constraints": {
    "requires": ["english-basic@>=1.0.0"],
    "load_order": 6
  }
}
```

### 2. Load Order Priority

```
1-2: Format parsers (JSON, XML)
3-4: Languages (English, Portuguese)
5-6: Technologies (Neo4j, Python, Rust)
7-8: Domains (Medical, Legal)
9-10: Tasks (Classification, Summarization)
```

### 3. VRAM Budgeting

Always calculate:
```
Base (0.5GB) + Experts (N * 50MB avg) + KV Cache (context-dependent) < 16GB
```

### 4. Manifest Completeness

Every manifest.json must have:
- Full `training` section (dataset.generation + config)
- Complete `constraints` (load_order, requires)
- Capabilities for router
- Routing keywords

---

## Quality Checks

Before completing work:

1. **Docs updated?**: All affected docs reflect changes
2. **Consistent?**: No contradictions across files
3. **Complete?**: All required fields populated
4. **Tested?**: Validation logic makes sense
5. **Roadmap aligned?**: Change fits current phase

---

## Examples

### Good Agent Response

```
User: "Add Python code expert"

Agent:
1. Checks roadmap - yes, it's planned
2. Checks dependencies - needs english-basic
3. Creates expert-python-code/ with complete manifest
4. Sets load_order: 6
5. Adds to experts/README.md
6. Notes in STATUS.md
```

### Bad Agent Response

```
User: "Add Python code expert"

Agent:
- Creates expert but forgets load_order
- Creates custom train.sh script (violates CLI-only rule)
- Doesn't declare english-basic dependency
- Creates new PYTHON_EXPERT.md doc (unnecessary)
```

---

## Resources

- **Main README**: [/expert/README.md](../README.md)
- **Architecture**: [/expert/docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md)
- **Status**: [/expert/STATUS.md](../STATUS.md)
- **Roadmap**: [/expert/ROADMAP.md](../ROADMAP.md)

---

## Quick Reference

```bash
# Read before starting
expert/README.md                    # Overview
expert/docs/ARCHITECTURE.md         # Core concepts
expert/STATUS.md                    # Current state

# Common operations
expert/docs/CLI.md                  # All CLI commands
expert/docs/EXPERT_FORMAT.md        # Manifest schema
expert/docs/GIT_DISTRIBUTION.md     # Distribution model

# Creating experts
expert/examples/expert-repository-template/  # Template
expert/experts/expert-json-parser/           # Real example
```

---

Last Updated: 2025-11-02

<!-- EXPERT_SYSTEM_AGENTS:END -->
