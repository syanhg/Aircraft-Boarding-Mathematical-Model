#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run Boarding Visualizations
===========================

This script runs the boarding visualizations and outputs the results.
"""

import boarding_visualization as bv

if __name__ == "__main__":
    # Set random seed for reproducibility
    import numpy as np
    np.random.seed(42)
    
    # Create comparison of all strategies
    print("Creating strategy comparison...")
    comparison_file = bv.create_strategy_comparison(num_steps=6)
    print(f"Created {comparison_file}")
    
    # Create detailed visualization of hybrid strategy
    print("Creating detailed hybrid strategy visualization...")
    hybrid_file = bv.create_detailed_visual_for_hybrid()
    print(f"Created {hybrid_file}")
    
    # Create time heatmap
    print("Creating boarding time heatmap...")
    heatmap_file = bv.create_time_heatmap()
    print(f"Created {heatmap_file}")
    
    print("All visualizations created successfully!")