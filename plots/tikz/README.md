# TikZ/PGF Plots for LaTeX

This directory contains plots exported in PGF (Portable Graphics Format) which can be directly included in LaTeX documents.

## Generated Files

- `dd_sdlqr_states.tex` - State trajectories for DD-SDLQR controller
- `dd_sdlqr_input.tex` - Control input trajectory
- `state_norm_comparison.tex` - Two-way comparison (DD-SDLQR vs SD-LQR)
- `three_way_state_norm_comparison.tex` - Three-way comparison (DD-SDLQR vs DD-LQR vs SD-LQR)
- `system_error_comparison.tex` - System estimation error comparison

## Usage in LaTeX

### Required Packages

Add these packages to your LaTeX document preamble:

```latex
\usepackage{pgf}
\usepackage{lmodern}  % Optional but recommended for better fonts
```

### Including a Single Figure

To include a plot in your document:

```latex
\begin{figure}[htbp]
  \centering
  \input{plots/tikz/three_way_state_norm_comparison.tex}
  \caption{State norm comparison among different controllers}
  \label{fig:comparison}
\end{figure}
```

### Using the Import Package (Recommended for Organized Projects)

If your plots are in a different directory from your main .tex file:

```latex
\usepackage{import}
```

Then in your document:

```latex
\begin{figure}[htbp]
  \centering
  \import{plots/tikz/}{three_way_state_norm_comparison.tex}
  \caption{State norm comparison among different controllers}
  \label{fig:comparison}
\end{figure}
```

### Adjusting Figure Size

If you need to scale the figure:

```latex
\begin{figure}[htbp]
  \centering
  \resizebox{0.8\textwidth}{!}{%
    \input{plots/tikz/three_way_state_norm_comparison.tex}
  }
  \caption{State norm comparison among different controllers}
  \label{fig:comparison}
\end{figure}
```

## Example LaTeX Document

Here's a complete minimal example:

```latex
\documentclass{article}
\usepackage{pgf}
\usepackage{lmodern}
\usepackage{graphicx}

\begin{document}

\section{Results}

Figure~\ref{fig:comparison} shows the comparison of different control strategies.

\begin{figure}[htbp]
  \centering
  \input{plots/tikz/three_way_state_norm_comparison.tex}
  \caption{State norm comparison among DD-SDLQR, DD-LQR, and SD-LQR controllers}
  \label{fig:comparison}
\end{figure}

\end{document}
```

## Advantages of PGF/TikZ Format

1. **Vector graphics** - Scales perfectly at any size
2. **Consistent fonts** - Uses the same fonts as your LaTeX document
3. **Editable** - Can be further customized in LaTeX if needed
4. **Small file size** - More compact than raster formats
5. **Professional quality** - Publication-ready figures

## Regenerating Plots

To regenerate these plots, run:

```bash
python3 main.py
```

The script will automatically generate both PNG versions (in `plots/`) and PGF/TikZ versions (in `plots/tikz/`).
