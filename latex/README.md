# Boeing 737-800 Boarding Strategy Heatmaps

This directory contains LaTeX files for generating heatmap visualizations of different boarding strategies for a Boeing 737-800 aircraft with 114 passengers.

## Files

1. `random_strategy_heatmap.tex` - Heatmap for the Random boarding strategy
2. `back_to_front_strategy_heatmap.tex` - Heatmap for the Back-to-Front boarding strategy
3. `outside_in_strategy_heatmap.tex` - Heatmap for the Outside-In boarding strategy
4. `hybrid_strategy_heatmap.tex` - Heatmap for the proposed Hybrid boarding strategy
5. `combined_strategy_heatmaps.tex` - Combined visualization of all strategies for comparison
6. `compile_all.sh` - Shell script to compile all LaTeX files into PDFs

## How to Compile

1. Make sure you have LaTeX installed on your system
2. Navigate to this directory
3. Run the compilation script:
   ```
   chmod +x compile_all.sh
   ./compile_all.sh
   ```
4. The script will generate PDF files for each heatmap

## Heatmap Color Coding

The heatmaps use a color gradient from yellow to orange to red:
- Yellow: Fast boarding times
- Orange: Medium boarding times
- Red: Slow boarding times

Each cell contains the estimated boarding time in minutes for that specific seat.

## Boarding Strategies

### Random Strategy
- No predetermined boarding order
- High variability across all seats
- Average boarding time: 18.7 minutes

### Back-to-Front Strategy
- Aircraft divided into 6 zones (front to back)
- Boarding proceeds from back zones to front zones
- Average boarding time: 11.2 minutes

### Outside-In Strategy
- Window seats first (A, F)
- Middle seats next (B, E)
- Aisle seats last (C, D)
- Average boarding time: 7.5 minutes

### Hybrid Strategy
- Combines zone and seat-type approaches
- Nine distinct boarding groups:
  1. Back window seats
  2. Middle window seats
  3. Front window seats
  4. Back middle seats
  5. Middle middle seats
  6. Front middle seats
  7. Back aisle seats
  8. Middle aisle seats
  9. Front aisle seats
- Average boarding time: 10.0 minutes

## Customization

You can modify the LaTeX files to:
- Change the color scheme
- Adjust the boarding time estimates
- Change the size or layout of the visualizations
- Add more details or explanations