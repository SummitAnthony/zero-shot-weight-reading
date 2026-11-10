# Paper: "Circuits or Shortcuts? What Language Models Read in Raw Neural Network Weights"

The LaTeX source of the paper, and the script that computes every number and figure in it.

## Build
It needs `tmlr.sty`, `tmlr.bst`, `fancyhdr.sty`, `numbers.tex`, `references.bib` and `figures/`. Either:
- `tectonic paper.tex`, which is how it was checked: 15 pages, no errors, no undefined references; or
- `pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper`.

## Numbers and figures
Every number in the text is a macro from `numbers.tex`, and the figures are in `figures/`. `analysis.py` generates both from the saved replies in `../experiments/main` and `../experiments/pilot`, with no API calls. After changing data or analysis, rerun:

```
python analysis.py
```

Don't edit `numbers.tex` by hand. The `RFive*` macros are round 5's.

## License
The paper text is CC BY 4.0. The style files `tmlr.sty` and `tmlr.bst` are Apache-2.0 (`TMLR_STYLE_LICENSE`), and `fancyhdr.sty` is under the LaTeX Project Public License 1.3.
