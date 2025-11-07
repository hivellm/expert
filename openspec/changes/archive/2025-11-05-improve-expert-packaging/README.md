# Improve Expert Package Completeness

**Change ID**: `improve-expert-packaging`  
**Status**: Partial (Task 1 completed)  
**Created**: 2025-11-04  

---

## Quick Summary

Expert packages (`.expert` files) were missing essential files:
- ❌ README.md (documentation)
- ❌ grammar.gbnf (validation files)

This change makes `.expert` packages **fully self-contained**.

---

## Problem

Before:
```
expert-sql.expert
├── manifest.json
├── weights/.../adapter_model.safetensors
└── LICENSE
```

Users couldn't:
- Read expert documentation without source code
- Use grammar validation (missing grammar files)
- Understand expert purpose from package alone

---

## Solution

After:
```
expert-sql.expert
├── manifest.json
├── README.md              🆕 Expert docs
├── grammar.gbnf           🆕 Validation grammar
├── LICENSE
├── weights/...
└── soft_prompts/...
```

---

## Implementation Status

✅ **Completed** (Task 1):
- README.md included in both v1.0 and v2.0 packaging
- grammar.gbnf included if exists
- Both logged during packaging

⏳ **Pending** (Tasks 2-5):
- Selective adapter file inclusion (avoid training artifacts)
- Optional tests directory
- Package contents documentation
- Validation for .expert files

---

## Files

- `proposal.md` - Detailed design
- `tasks.md` - 5 implementation tasks
- `README.md` - This file

---

## Testing

```bash
# Package expert
cd F:\Node\hivellm\expert\experts\expert-sql
F:\Node\hivellm\expert\cli\target\release\expert-cli.exe package --manifest manifest.json --weights weights

# Verify contents
tar -tzf expert-sql-0.0.1.expert
# Should now include: manifest.json, README.md, grammar.gbnf, LICENSE, adapters, soft_prompts
```

---

## Impact

- ✅ Packages fully self-documented
- ✅ Grammar validation files bundled
- ⏳ Smaller packages (pending task 2 - exclude training artifacts)
- ⏳ Testable packages (pending task 3 - optional tests/)

