#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Boeing 737-800 Boarding Visualization
=====================================

This script creates visualizations of different boarding strategies
for a Boeing 737-800 aircraft with 114 passengers.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from matplotlib.colors import ListedColormap

# Configuration
ROWS = 19  # Number of rows in economy class
SEATS_PER_ROW = 6  # 3 seats on each side (A-F)
TOTAL_SEATS = ROWS * SEATS_PER_ROW  # 114 seats

# Define seat arrangement
# A, B, C | D, E, F
SEAT_INDICES = {
    'A': 0,  # Window (left)
    'B': 1,  # Middle (left)
    'C': 2,  # Aisle (left)
    'D': 3,  # Aisle (right)
    'E': 4,  # Middle (right)
    'F': 5,  # Window (right)
}

# Define colors for different passenger states
EMPTY = 0
QUEUED = 1
BOARDING = 2
SEATED = 3

# Color map for visualization
colors = ['#1a1a1a', '#FFD700', '#FF4500', '#4CAF50']
cmap = ListedColormap(colors)

def create_empty_aircraft():
    """Create an empty aircraft seating arrangement."""
    # Each row has 6 seats (A-F)
    # 0 = Empty, 1 = Queued, 2 = Boarding, 3 = Seated
    return np.zeros((ROWS, SEATS_PER_ROW), dtype=int)

def random_strategy(aircraft):
    """Simulate random boarding strategy."""
    # Create flat indices of all seats
    all_seats = [(r, c) for r in range(ROWS) for c in range(SEATS_PER_ROW)]
    
    # Shuffle the seats randomly
    np.random.shuffle(all_seats)
    
    # Return the boarding sequence
    return all_seats

def back_to_front_strategy(aircraft):
    """Simulate back-to-front boarding strategy with 6 zones."""
    boarding_sequence = []
    
    # Define 6 zones from back to front
    zone_size = ROWS // 6
    remainder = ROWS % 6
    
    zone_rows = []
    start_row = 0
    for i in range(6):
        size = zone_size + (1 if i < remainder else 0)
        end_row = start_row + size
        zone_rows.append((start_row, end_row))
        start_row = end_row
    
    # Process zones from back to front
    for zone_idx in range(5, -1, -1):
        start_row, end_row = zone_rows[zone_idx]
        
        # Within each zone, seats are boarded randomly
        zone_seats = [(r, c) for r in range(start_row, end_row) for c in range(SEATS_PER_ROW)]
        np.random.shuffle(zone_seats)
        boarding_sequence.extend(zone_seats)
    
    return boarding_sequence

def outside_in_strategy(aircraft):
    """Simulate outside-in (window-middle-aisle) boarding strategy."""
    boarding_sequence = []
    
    # Window seats first (A and F)
    window_seats = [(r, 0) for r in range(ROWS)] + [(r, 5) for r in range(ROWS)]
    np.random.shuffle(window_seats)
    boarding_sequence.extend(window_seats)
    
    # Middle seats next (B and E)
    middle_seats = [(r, 1) for r in range(ROWS)] + [(r, 4) for r in range(ROWS)]
    np.random.shuffle(middle_seats)
    boarding_sequence.extend(middle_seats)
    
    # Aisle seats last (C and D)
    aisle_seats = [(r, 2) for r in range(ROWS)] + [(r, 3) for r in range(ROWS)]
    np.random.shuffle(aisle_seats)
    boarding_sequence.extend(aisle_seats)
    
    return boarding_sequence

def hybrid_strategy(aircraft):
    """Simulate hybrid strategy (combines back-to-front and outside-in)."""
    boarding_sequence = []
    
    # Define 3 zones
    zone_size = ROWS // 3
    remainder = ROWS % 3
    
    zone_rows = []
    start_row = 0
    for i in range(3):
        size = zone_size + (1 if i < remainder else 0)
        end_row = start_row + size
        zone_rows.append((start_row, end_row))
        start_row = end_row
    
    # Process in the specific sequence:
    # 1. Back window, 2. Middle window, 3. Front window,
    # 4. Back middle, 5. Middle middle, 6. Front middle,
    # 7. Back aisle, 8. Middle aisle, 9. Front aisle
    
    # Window seats by zone (back to front)
    for zone_idx in range(2, -1, -1):
        start_row, end_row = zone_rows[zone_idx]
        window_seats = [(r, 0) for r in range(start_row, end_row)] + [(r, 5) for r in range(start_row, end_row)]
        np.random.shuffle(window_seats)
        boarding_sequence.extend(window_seats)
    
    # Middle seats by zone (back to front)
    for zone_idx in range(2, -1, -1):
        start_row, end_row = zone_rows[zone_idx]
        middle_seats = [(r, 1) for r in range(start_row, end_row)] + [(r, 4) for r in range(start_row, end_row)]
        np.random.shuffle(middle_seats)
        boarding_sequence.extend(middle_seats)
    
    # Aisle seats by zone (back to front)
    for zone_idx in range(2, -1, -1):
        start_row, end_row = zone_rows[zone_idx]
        aisle_seats = [(r, 2) for r in range(start_row, end_row)] + [(r, 3) for r in range(start_row, end_row)]
        np.random.shuffle(aisle_seats)
        boarding_sequence.extend(aisle_seats)
    
    return boarding_sequence

def simulate_boarding(strategy_func, num_steps=20):
    """Simulate boarding process for a given strategy."""
    aircraft = create_empty_aircraft()
    boarding_sequence = strategy_func(aircraft)
    
    # Create sequence of aircraft states at different timesteps
    states = []
    
    # Divide the boarding sequence into num_steps steps
    step_size = len(boarding_sequence) // num_steps
    if step_size == 0:
        step_size = 1
    
    for step in range(0, len(boarding_sequence), step_size):
        # Mark next batch of passengers as seated
        end_idx = min(step + step_size, len(boarding_sequence))
        for i in range(step, end_idx):
            row, seat = boarding_sequence[i]
            aircraft[row, seat] = SEATED
        
        # Save current state
        states.append(aircraft.copy())
    
    # Ensure the final state shows all passengers seated
    final_state = aircraft.copy()
    for row, seat in boarding_sequence:
        final_state[row, seat] = SEATED
    states.append(final_state)
    
    return states

def plot_seating_grid(ax, aircraft_state, title):
    """Plot a single aircraft seating grid state."""
    # Create a masked array where 0 (EMPTY) is shown as black
    masked_data = np.ma.masked_where(aircraft_state == EMPTY, aircraft_state)
    
    # Plot the seating chart
    im = ax.imshow(masked_data, cmap=cmap, vmin=0, vmax=3)
    
    # Add row and seat labels
    ax.set_yticks(range(ROWS))
    ax.set_yticklabels(range(1, ROWS+1))
    ax.set_xticks(range(SEATS_PER_ROW))
    ax.set_xticklabels(['A', 'B', 'C', 'D', 'E', 'F'])
    
    # Add title
    ax.set_title(title)
    
    # Add vertical line to separate the aisle
    ax.axvline(x=2.5, color='white', linestyle='-', linewidth=2)
    
    # Add horizontal grid lines
    for i in range(ROWS-1):
        ax.axhline(y=i+0.5, color='gray', linestyle='-', linewidth=0.5)
    
    return im

def create_progression_diagram(strategy_name, strategy_func, num_steps=6):
    """Create a diagram showing boarding progression for a strategy."""
    # Simulate boarding process
    states = simulate_boarding(strategy_func, num_steps)
    
    # Create figure
    fig, axes = plt.subplots(1, num_steps, figsize=(15, 5))
    
    # Plot each state
    for i, (ax, state) in enumerate(zip(axes, states)):
        progress = int((i / (num_steps-1)) * 100)
        title = f"{progress}% Complete" if i < num_steps-1 else "100% Complete"
        plot_seating_grid(ax, state, title)
    
    # Add overall title
    fig.suptitle(f"Boarding Progression: {strategy_name} Strategy", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    filename = f"{strategy_name.lower().replace(' ', '_')}_progression.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return filename

def create_strategy_comparison(num_steps=6):
    """Create a comparison of all boarding strategies."""
    strategies = {
        "Random": random_strategy,
        "Back-to-Front": back_to_front_strategy,
        "Outside-In": outside_in_strategy,
        "Hybrid": hybrid_strategy
    }
    
    # Simulate boarding process for each strategy
    all_states = {}
    for name, func in strategies.items():
        all_states[name] = simulate_boarding(func, num_steps)
    
    # Create figure
    fig, axes = plt.subplots(len(strategies), num_steps, figsize=(15, 12))
    
    # Plot each strategy's progression
    for i, (strategy_name, states) in enumerate(all_states.items()):
        for j, state in enumerate(states[:num_steps]):
            progress = int((j / (num_steps-1)) * 100)
            title = f"{progress}% Complete" if j < num_steps-1 else "100% Complete"
            if j == 0:  # Add strategy name to first column
                title = f"{strategy_name}\n{title}"
            plot_seating_grid(axes[i, j], state, title)
    
    # Add overall title
    fig.suptitle("Boeing 737-800 Boarding Strategy Comparison (114 passengers)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    plt.savefig("boarding_strategy_comparison.png", dpi=300, bbox_inches='tight')
    
    return "boarding_strategy_comparison.png"

def create_step_animation(strategy_name, strategy_func, num_frames=30):
    """Create an animation of the boarding process."""
    # Simulate boarding with more steps for smoother animation
    states = simulate_boarding(strategy_func, num_frames)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Initialize with empty grid
    im = plot_seating_grid(ax, states[0], f"{strategy_name} Strategy: 0% Complete")
    
    # Animation update function
    def update_frame(frame):
        ax.clear()
        progress = int((frame / (len(states)-1)) * 100)
        im = plot_seating_grid(ax, states[frame], f"{strategy_name} Strategy: {progress}% Complete")
        return [im]
    
    # Create animation
    ani = animation.FuncAnimation(fig, update_frame, frames=len(states), 
                                  interval=200, blit=False)
    
    # Save animation as GIF
    filename = f"{strategy_name.lower().replace(' ', '_')}_animation.gif"
    ani.save(filename, writer='pillow', fps=5, dpi=100)
    
    return filename

def create_detailed_visual_for_hybrid():
    """Create a detailed visualization of the hybrid strategy's 9 steps."""
    aircraft = create_empty_aircraft()
    
    # Define 3 zones
    zone_size = ROWS // 3
    zone1 = list(range(0, zone_size))  # Front
    zone2 = list(range(zone_size, 2*zone_size))  # Middle
    zone3 = list(range(2*zone_size, ROWS))  # Back
    
    # Define seat types
    window_seats = [0, 5]  # A, F
    middle_seats = [1, 4]  # B, E
    aisle_seats = [2, 3]   # C, D
    
    # Create 9 states showing the progression
    states = []
    titles = []
    
    # 1. Start with empty aircraft
    states.append(aircraft.copy())
    titles.append("Initial State")
    
    # 2. Back window seats
    aircraft_step1 = aircraft.copy()
    for row in zone3:
        for seat in window_seats:
            aircraft_step1[row, seat] = SEATED
    states.append(aircraft_step1)
    titles.append("Step 1: Back Window Seats")
    
    # 3. Middle window seats
    aircraft_step2 = aircraft_step1.copy()
    for row in zone2:
        for seat in window_seats:
            aircraft_step2[row, seat] = SEATED
    states.append(aircraft_step2)
    titles.append("Step 2: Middle Window Seats")
    
    # 4. Front window seats
    aircraft_step3 = aircraft_step2.copy()
    for row in zone1:
        for seat in window_seats:
            aircraft_step3[row, seat] = SEATED
    states.append(aircraft_step3)
    titles.append("Step 3: Front Window Seats")
    
    # 5. Back middle seats
    aircraft_step4 = aircraft_step3.copy()
    for row in zone3:
        for seat in middle_seats:
            aircraft_step4[row, seat] = SEATED
    states.append(aircraft_step4)
    titles.append("Step 4: Back Middle Seats")
    
    # 6. Middle middle seats
    aircraft_step5 = aircraft_step4.copy()
    for row in zone2:
        for seat in middle_seats:
            aircraft_step5[row, seat] = SEATED
    states.append(aircraft_step5)
    titles.append("Step 5: Middle Middle Seats")
    
    # 7. Front middle seats
    aircraft_step6 = aircraft_step5.copy()
    for row in zone1:
        for seat in middle_seats:
            aircraft_step6[row, seat] = SEATED
    states.append(aircraft_step6)
    titles.append("Step 6: Front Middle Seats")
    
    # 8. Back aisle seats
    aircraft_step7 = aircraft_step6.copy()
    for row in zone3:
        for seat in aisle_seats:
            aircraft_step7[row, seat] = SEATED
    states.append(aircraft_step7)
    titles.append("Step 7: Back Aisle Seats")
    
    # 9. Middle aisle seats
    aircraft_step8 = aircraft_step7.copy()
    for row in zone2:
        for seat in aisle_seats:
            aircraft_step8[row, seat] = SEATED
    states.append(aircraft_step8)
    titles.append("Step 8: Middle Aisle Seats")
    
    # 10. Front aisle seats (all seats filled)
    aircraft_step9 = aircraft_step8.copy()
    for row in zone1:
        for seat in aisle_seats:
            aircraft_step9[row, seat] = SEATED
    states.append(aircraft_step9)
    titles.append("Step 9: Front Aisle Seats (Complete)")
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    # Plot each state
    for i, (state, title) in enumerate(zip(states, titles)):
        plot_seating_grid(axes[i], state, title)
    
    # Add overall title
    fig.suptitle("Hybrid Strategy: Detailed Boarding Progression", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    plt.savefig("hybrid_strategy_detailed.png", dpi=300, bbox_inches='tight')
    
    return "hybrid_strategy_detailed.png"

def create_time_heatmap():
    """Create a heatmap showing boarding time by seat position."""
    # Create an array to represent boarding time for each seat
    boarding_times = np.zeros((ROWS, SEATS_PER_ROW))
    
    # Random strategy: random times
    random_times = np.random.uniform(15, 22, size=(ROWS, SEATS_PER_ROW))
    
    # Back-to-front: time increases from back to front
    btf_times = np.zeros((ROWS, SEATS_PER_ROW))
    for i in range(ROWS):
        # Time increases from back to front, but with variability in each row
        base_time = 20 * (i / ROWS)
        btf_times[i, :] = base_time + np.random.uniform(0, 3, size=SEATS_PER_ROW)
    
    # Outside-in: time increases from window to aisle
    oi_times = np.zeros((ROWS, SEATS_PER_ROW))
    # Window seats (A, F)
    oi_times[:, 0] = np.random.uniform(1, 4, size=ROWS)
    oi_times[:, 5] = np.random.uniform(1, 4, size=ROWS)
    # Middle seats (B, E)
    oi_times[:, 1] = np.random.uniform(5, 9, size=ROWS)
    oi_times[:, 4] = np.random.uniform(5, 9, size=ROWS)
    # Aisle seats (C, D)
    oi_times[:, 2] = np.random.uniform(10, 14, size=ROWS)
    oi_times[:, 3] = np.random.uniform(10, 14, size=ROWS)
    
    # Hybrid: combination of zone and seat effects
    hybrid_times = np.zeros((ROWS, SEATS_PER_ROW))
    # Window seats
    for i, row in enumerate(range(ROWS)):
        zone_factor = i // (ROWS // 3)  # 0=back, 1=middle, 2=front
        # Window seats (fastest, by zone)
        if zone_factor == 0:  # Back zone
            hybrid_times[row, 0] = np.random.uniform(1, 3, size=1)
            hybrid_times[row, 5] = np.random.uniform(1, 3, size=1)
        elif zone_factor == 1:  # Middle zone
            hybrid_times[row, 0] = np.random.uniform(3, 5, size=1)
            hybrid_times[row, 5] = np.random.uniform(3, 5, size=1)
        else:  # Front zone
            hybrid_times[row, 0] = np.random.uniform(5, 7, size=1)
            hybrid_times[row, 5] = np.random.uniform(5, 7, size=1)
        
        # Middle seats (medium, by zone)
        if zone_factor == 0:  # Back zone
            hybrid_times[row, 1] = np.random.uniform(7, 9, size=1)
            hybrid_times[row, 4] = np.random.uniform(7, 9, size=1)
        elif zone_factor == 1:  # Middle zone
            hybrid_times[row, 1] = np.random.uniform(9, 11, size=1)
            hybrid_times[row, 4] = np.random.uniform(9, 11, size=1)
        else:  # Front zone
            hybrid_times[row, 1] = np.random.uniform(11, 13, size=1)
            hybrid_times[row, 4] = np.random.uniform(11, 13, size=1)
        
        # Aisle seats (slowest, by zone)
        if zone_factor == 0:  # Back zone
            hybrid_times[row, 2] = np.random.uniform(13, 15, size=1)
            hybrid_times[row, 3] = np.random.uniform(13, 15, size=1)
        elif zone_factor == 1:  # Middle zone
            hybrid_times[row, 2] = np.random.uniform(15, 17, size=1)
            hybrid_times[row, 3] = np.random.uniform(15, 17, size=1)
        else:  # Front zone
            hybrid_times[row, 2] = np.random.uniform(17, 19, size=1)
            hybrid_times[row, 3] = np.random.uniform(17, 19, size=1)
    
    # Create a figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot each heatmap
    time_maps = [
        (random_times, "Random Strategy", axes[0, 0]),
        (btf_times, "Back-to-Front Strategy", axes[0, 1]),
        (oi_times, "Outside-In Strategy", axes[1, 0]),
        (hybrid_times, "Hybrid Strategy", axes[1, 1])
    ]
    
    for times, title, ax in time_maps:
        im = sns.heatmap(times, ax=ax, cmap="YlOrRd", annot=True, fmt=".1f",
                         xticklabels=['A', 'B', 'C', 'D', 'E', 'F'],
                         yticklabels=range(1, ROWS+1))
        ax.set_title(title)
        ax.set_xlabel("Seat")
        ax.set_ylabel("Row")
        
        # Add vertical line to separate the aisle
        ax.axvline(x=3, color='white', linestyle='-', linewidth=2)
    
    # Add overall title
    fig.suptitle("Boarding Time by Seat Position (minutes)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    plt.savefig("boarding_time_heatmap.png", dpi=300, bbox_inches='tight')
    
    return "boarding_time_heatmap.png"

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create comparison of all strategies
    print("Creating strategy comparison...")
    comparison_file = create_strategy_comparison(num_steps=6)
    print(f"Created {comparison_file}")
    
    # Create detailed visualization of hybrid strategy
    print("Creating detailed hybrid strategy visualization...")
    hybrid_file = create_detailed_visual_for_hybrid()
    print(f"Created {hybrid_file}")
    
    # Create time heatmap
    print("Creating boarding time heatmap...")
    heatmap_file = create_time_heatmap()
    print(f"Created {heatmap_file}")
    
    # Create animations for each strategy
    # print("Creating animations...")
    # for name, func in [
    #     ("Random", random_strategy),
    #     ("Back-to-Front", back_to_front_strategy),
    #     ("Outside-In", outside_in_strategy),
    #     ("Hybrid", hybrid_strategy)
    # ]:
    #     animation_file = create_step_animation(name, func)
    #     print(f"Created {animation_file}")
    
    print("All visualizations created successfully!")