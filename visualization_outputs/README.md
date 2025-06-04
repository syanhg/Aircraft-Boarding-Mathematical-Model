# Boarding Strategy Visualizations

This directory contains visualizations of different boarding strategies for a Boeing 737-800 aircraft with 114 passengers.

## Files

1. `boarding_strategy_comparison.png` - Comparison of all boarding strategies showing progression steps
2. `hybrid_strategy_detailed.png` - Detailed visualization of the hybrid strategy's 9 boarding steps
3. `boarding_time_heatmap.png` - Heatmap showing boarding time by seat position for each strategy

## Boarding Strategies

1. **Random** - Passengers board in random order
2. **Back-to-Front** - Aircraft divided into 6 zones, boarding from back to front
3. **Outside-In** - Window seats first, then middle seats, then aisle seats
4. **Hybrid** - Combines zone and seat type approaches with 9 distinct boarding groups:
   - Back window seats
   - Middle window seats
   - Front window seats
   - Back middle seats
   - Middle middle seats
   - Front middle seats
   - Back aisle seats
   - Middle aisle seats
   - Front aisle seats

## Visualization Color Key

- **Black** - Empty seat
- **Yellow** - Passenger queued for boarding
- **Orange** - Passenger in process of boarding
- **Green** - Passenger seated

## Boeing 737-800 Configuration

- **19 Rows** of economy class
- **6 Seats per row** (3-3 configuration)
- **114 Total seats**
- Seat layout: A, B, C | D, E, F (with aisle between C and D)