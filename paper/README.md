# HiveLLM Expert System - Research Paper

Academic paper describing the Expert System architecture, experimental validation, and results.

**Status:** Experimental Preprint  
**Last Updated:** November 2025

## Files

- `expert_system.tex` - LaTeX source (arXiv format)
- `expert_system.md` - Markdown version (same content)
- `expert_system.pdf` - Compiled PDF (after compilation)

## How to Compile

### Option 1: Overleaf (Easiest)

1. Go to https://www.overleaf.com/
2. Create new project → Upload Project
3. Upload `expert_system.tex`
4. Click "Recompile"
5. Download PDF

### Option 2: Local LaTeX (Windows)

**Install MiKTeX:**
```powershell
# Download from: https://miktex.org/download
# Or with chocolatey:
choco install miktex -y
```

**Compile:**
```powershell
cd F:\Node\hivellm\expert\paper
pdflatex expert_system.tex
pdflatex expert_system.tex  # Run twice for references
```

### Option 3: Local LaTeX (Linux/WSL)

**Install TeX Live:**
```bash
sudo apt update
sudo apt install texlive-full -y
```

**Compile:**
```bash
cd /mnt/f/Node/hivellm/expert/paper
pdflatex expert_system.tex
pdflatex expert_system.tex  # Run twice for references
```

### Option 4: Pandoc (Markdown → PDF)

**Install Pandoc:**
```powershell
# Download from: https://pandoc.org/installing.html
# Or with chocolatey:
choco install pandoc -y
```

**Compile:**
```powershell
cd F:\Node\hivellm\expert\paper
pandoc expert_system.md -o expert_system.pdf `
  --pdf-engine=xelatex `
  -V geometry:margin=1in `
  -V fontsize=11pt `
  --toc
```

### Option 5: Online Markdown to PDF

1. Go to https://www.markdowntopdf.com/
2. Upload `expert_system.md`
3. Download PDF

## Paper Highlights

### Abstract Summary

Runtime-composable adapter architecture for specialized AI on consumer GPUs:
- **Architecture:** Base model (Qwen3-0.6B) + dynamic PEFT adapters
- **Validation:** Expert-SQL with 9.6/10 quality, 100% success on 30 queries
- **Memory:** 64% savings through base model sharing
- **Status:** Experimental, requires production validation

### Key Results

| Metric | Value |
|--------|-------|
| Real-world benchmark | 30/30 queries (100%) |
| Quality score | 9.6/10 |
| Training time | 3 hours (RTX 4090) |
| Training VRAM | 0.56GB (2.3% utilization) |
| Adapter size | 25.8MB |
| Memory savings | 64% (3 experts) |

### Main Sections

1. **Introduction** - Problem statement and architecture overview
2. **Related Work** - PEFT methods, MoE comparison
3. **Method** - System architecture, training protocol, base model sharing
4. **Experimental Validation** - Expert-SQL dataset, training, checkpoints
5. **Results** - Quality analysis, strengths, limitations
6. **Discussion** - Advantages, checkpoint insights, future work
7. **Experimental Nature** - Limitations and ongoing work
8. **Conclusion** - Summary and next steps

### Important Disclaimers

The paper clearly states:
- ⚠️ **Experimental nature** (mentioned 6 times)
- ⚠️ **Limited production validation** 
- ⚠️ **Synthetic benchmarks only**
- ⚠️ **Single expert fully validated** (SQL)
- ⚠️ **Ongoing training** for other experts

## Citation

```bibtex
@article{hivellm2025expert,
  title={HiveLLM Expert System: Runtime-Composable Adapters for Specialized Inference on Consumer GPUs},
  author={HiveLLM Research Team},
  journal={arXiv preprint},
  year={2025},
  note={Experimental work - Production validation pending}
}
```

## License

This paper is released under CC-BY-4.0 (Creative Commons Attribution 4.0 International).

You are free to:
- Share and redistribute
- Adapt and build upon

Under the following terms:
- Attribution: Must give appropriate credit
- No additional restrictions

## Contact

For questions or feedback:
- Email: team@hivellm.org
- Issues: https://github.com/hivellm/expert/issues
- Discussions: https://github.com/hivellm/expert/discussions

