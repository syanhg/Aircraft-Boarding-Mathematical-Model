#!/bin/bash

# Compile all LaTeX heatmap files
echo "Compiling individual strategy heatmaps..."

pdflatex random_strategy_heatmap.tex
pdflatex back_to_front_strategy_heatmap.tex
pdflatex outside_in_strategy_heatmap.tex
pdflatex hybrid_strategy_heatmap.tex
pdflatex combined_strategy_heatmaps.tex

# Clean up auxiliary files
echo "Cleaning up auxiliary files..."
rm *.aux *.log *.out

echo "All PDF files generated successfully!"
echo "Output files:"
echo "- random_strategy_heatmap.pdf"
echo "- back_to_front_strategy_heatmap.pdf"
echo "- outside_in_strategy_heatmap.pdf"
echo "- hybrid_strategy_heatmap.pdf"
echo "- combined_strategy_heatmaps.pdf"