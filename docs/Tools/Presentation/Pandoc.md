# Pandoc

## Basic

```bash
# HTML Export
pandoc -t html -s simple-presentation.md -o out/simple-presentation-website.html

# HTML Presentation
pandoc -t revealjs -s simple-presentation.md -o out/simple-presentation.html
# -t s5, slidy, slideous, dzslides, or revealjs

# PDF
pandoc -t beamer .\simple-presentation.md -o out/simple-presentation.pdf

# PPT
pandoc simple-presentation.md -o out/simple-presentation.pptx
```

## Advanced
```bash
pandoc -t beamer -s --bibliography bib.bib --citeproc .\presentation.md -o out/presentation.pdf
```