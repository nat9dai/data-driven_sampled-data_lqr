# TikZ/PGF Plots Guide

## Summary

Your simulation now generates publication-quality TikZ/PGF plots that can be directly included in LaTeX documents. The plots are located in `/plots/tikz/`.

## What Was Created

### New Files

1. **`visualiser_tikz.py`** - A new visualizer class that exports plots to PGF format
2. **`plots/tikz/`** - Directory containing all generated TikZ plots:
   - `dd_sdlqr_states.tex`
   - `dd_sdlqr_input.tex`
   - `state_norm_comparison.tex`
   - `three_way_state_norm_comparison.tex`
   - `system_error_comparison.tex`
3. **`plots/tikz/README.md`** - Detailed usage instructions
4. **`plots/tikz/example.tex`** - A complete example LaTeX document

### Modified Files

- **`main.py`** - Now generates both PNG and TikZ/PGF versions of all plots

## Quick Start

### In Your LaTeX Document

1. Add to your preamble:
```latex
\usepackage{pgf}
\usepackage{lmodern}
```

2. Include a plot:
```latex
\begin{figure}[htbp]
  \centering
  \input{plots/tikz/three_way_state_norm_comparison.tex}
  \caption{Controller comparison}
  \label{fig:comparison}
\end{figure}
```

### Running the Simulation

Simply run:
```bash
python3 main.py
```

This will generate:
- PNG plots in `plots/` (for quick viewing)
- PGF/TikZ plots in `plots/tikz/` (for LaTeX documents)

## Advantages of PGF/TikZ Format

1. **Vector Graphics** - Perfect scaling at any resolution
2. **Consistent Typography** - Uses your document's fonts
3. **Small File Size** - More compact than PNG/PDF for line plots
4. **Editable** - Can be customized directly in LaTeX
5. **Professional** - Publication-ready quality

## Example Usage

See `plots/tikz/example.tex` for a complete working example that includes all the generated plots in a LaTeX document.

## Technical Details

- **Format**: PGF (Portable Graphics Format)
- **Backend**: Matplotlib's built-in PGF backend
- **Compatibility**: Works with pdflatex, xelatex, and lualatex
- **Dependencies**: Standard LaTeX `pgf` package

## Troubleshooting

If you encounter issues including the plots:

1. Make sure the `pgf` package is installed (it usually comes with standard LaTeX distributions)
2. Use relative paths from your main .tex file
3. If plots are in a different directory, use the `import` package:
   ```latex
   \usepackage{import}
   \import{path/to/plots/tikz/}{filename.tex}
   ```

## File Sizes

The generated PGF files are typically:
- 35-65 KB per plot
- Much smaller than equivalent high-resolution PNG files
- Scales to any size without quality loss

## Further Customization

Since these are text-based vector graphics, you can:
- Edit the .tex files directly to adjust colors, line styles, etc.
- Use PGFPlots commands to add annotations
- Combine multiple plots into subfigures
- Easily match your document's style and formatting
