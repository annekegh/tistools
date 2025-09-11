#!/usr/bin/env python3
"""
Script to modify pathensemble.txt files to reflect High Acceptance (HA) weights
according to the staple method.

This script:
1. Identifies shooting paths ('sh') with LML/RMR types that need HA=2
2. Traces swapped paths ('s+', 's-') back to their original shooting paths
3. Updates the last column (weight) in pathensemble.txt files accordingly

Usage:
    python modify_pathensemble_weights.py <base_directory> [--dry-run]
    
Arguments:
    base_directory: Directory containing ensemble subdirectories (000, 001, 002, etc.)
    --dry-run: Only print what would be changed without modifying files
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
import numpy as np

def parse_pathensemble_line(line):
    """Parse a line from pathensemble.txt and return components."""
    parts = line.strip().split()
    if len(parts) < 17:  # Need at least 17 columns for weight at index 16
        return None
    
    return {
        'cycle': int(parts[0]),
        'path_num': int(parts[1]),
        'new_path_num': int(parts[2]),
        'lmr': ''.join(parts[3:6]),  # L M R
        'length': int(parts[6]),
        'flag': parts[7],
        'generation': parts[8],
        'lambmin': float(parts[9]),
        'lambmax': float(parts[10]),
        'weight': float(parts[16]),  # Weight column is at index 16
        'raw_line': line.strip(),
        'parts': parts
    }

def should_get_ha_weight(pe_i, ptype, n_ensembles):
    """
    Determine if a path type in ensemble pe_i should get high acceptance weight.
    Based on the condition from get_weights_staple.
    """
    return ((pe_i == 2 and ptype == "RMR") or
            (pe_i == n_ensembles - 1 and ptype == "LML") or
            (2 < pe_i < n_ensembles - 1 and ptype in ["LML", "RMR"]))

def find_original_shooting_path(target_length, target_lambmin, target_lambmax, 
                               orig_pe_data, all_ensemble_data, orig_ensemble_id, 
                               n_ensembles, tolerance=1e-6, length_tolerance=1, visited_swaps=None, debug=False, target_cycle=None):
    """
    Find the original shooting path, handling recursive swaps.
    
    Parameters:
    -----------
    target_length : int
        Length of the path to trace back
    target_lambmin : float
        Minimum lambda value of the path
    target_lambmax : float
        Maximum lambda value of the path
    orig_pe_data : list
        List of path data from the ensemble to search
    all_ensemble_data : dict
        Dictionary containing data from all ensembles
    orig_ensemble_id : int
        ID of the ensemble being searched
    n_ensembles : int
        Total number of ensembles
    tolerance : float
        Tolerance for lambda value comparison
    length_tolerance : int
        Tolerance for length comparison
    visited_swaps : set
        Set to track visited swaps to prevent infinite recursion
    debug : bool
        Enable debug output
    target_cycle : int
        Cycle number of the swap (for more targeted search)
        
    Returns:
    --------
    tuple: (original_path_data, chain_of_swaps) or (None, [])
    """
    if visited_swaps is None:
        visited_swaps = set()
    
    if debug:
        print(f"    DEBUG: Searching ensemble {orig_ensemble_id:03d} for path with:")
        print(f"           length={target_length}, lambmin={target_lambmin:.6f}, lambmax={target_lambmax:.6f}")
        print(f"           tolerance={tolerance}, length_tolerance={length_tolerance}")
        if target_cycle:
            print(f"           target_cycle={target_cycle}")
    
    # Smart search: look for the 20 most recent accepted paths BEFORE the target cycle
    # This is where the original shooting path should be found
    accepted_paths = []
    
    if target_cycle is not None:
        # Get all accepted paths before the target cycle, sorted by cycle number
        before_target = []
        for path_data in orig_pe_data:
            if (path_data['flag'] == 'ACC' and 
                path_data['cycle'] < target_cycle):
                before_target.append(path_data)
        
        # Sort by cycle number and take the last 100 (most recent)
        before_target.sort(key=lambda x: x['cycle'])
        accepted_paths = before_target[-100:] if len(before_target) >= 100 else before_target
        
        if debug:
            print(f"    DEBUG: Found {len(accepted_paths)} accepted paths before cycle {target_cycle}")
            if accepted_paths:
                print(f"           Cycle range: {accepted_paths[0]['cycle']} to {accepted_paths[-1]['cycle']}")
    
    # If no paths found before target cycle or no target cycle given, search all accepted paths
    if not accepted_paths:
        for path_data in orig_pe_data:
            if path_data['flag'] == 'ACC':
                accepted_paths.append(path_data)
        
        if debug:
            print(f"    DEBUG: Fallback - searching all {len(accepted_paths)} accepted paths")
    
    if debug and accepted_paths:
        # Show a sample of paths for debugging  
        sample_size = min(5, len(accepted_paths))
        print(f"    DEBUG: Showing {sample_size} paths from search set:")
        for i, path in enumerate(accepted_paths[-sample_size:]):  # Show most recent
            print(f"           [{i}] cycle={path['cycle']}, gen={path['generation']}, "
                  f"length={path['length']}, lambmin={path['lambmin']:.6f}, lambmax={path['lambmax']:.6f}")
        if len(accepted_paths) > sample_size:
            print(f"           ... and {len(accepted_paths)-sample_size} more paths")
    
    # Search for matching path
    for orig_path in accepted_paths:
        length_match = abs(orig_path['length'] - target_length) <= length_tolerance
        lambmin_match = abs(orig_path['lambmin'] - target_lambmin) < tolerance
        lambmax_match = abs(orig_path['lambmax'] - target_lambmax) < tolerance
        
        if debug and (length_match or lambmin_match or lambmax_match):
            print(f"    DEBUG: Checking path cycle {orig_path['cycle']}: length_match={length_match}, "
                  f"lambmin_match={lambmin_match}, lambmax_match={lambmax_match}")
        
        if length_match and lambmin_match and lambmax_match:
            if debug:
                print(f"    DEBUG: MATCH FOUND! cycle={orig_path['cycle']}, gen={orig_path['generation']}")
            
            # Create unique identifier for this swap to prevent cycles
            swap_id = (orig_ensemble_id, orig_path['cycle'])
            if swap_id in visited_swaps:
                if debug:
                    print(f"    DEBUG: Skipping cycle {orig_path['cycle']} - already visited")
                continue  # Skip to avoid infinite recursion
            
            # If this is a shooting path, we found the original
            if orig_path['generation'] == 'sh':
                if debug:
                    print(f"    DEBUG: Found original shooting path! cycle={orig_path['cycle']}")
                return orig_path, [(orig_ensemble_id, orig_path)]
            
            # If this is another swap, recursively find the original
            elif orig_path['generation'] in ['s+', 's-']:
                if debug:
                    print(f"    DEBUG: Found swap {orig_path['generation']}, recursing...")
                visited_swaps.add(swap_id)
                
                # Determine the next ensemble to search
                if orig_path['generation'] == 's+':
                    next_ensemble_id = orig_ensemble_id + 1
                elif orig_path['generation'] == 's-':
                    next_ensemble_id = orig_ensemble_id - 1
                
                # Check bounds and skip invalid swaps
                if next_ensemble_id < 0 or next_ensemble_id >= n_ensembles:
                    if debug:
                        print(f"    DEBUG: Next ensemble {next_ensemble_id} out of bounds")
                    continue
                
                # Skip s+ swaps in ensemble 000 and s- swaps in ensemble 001
                if (orig_path['generation'] == 's+' and orig_ensemble_id == 0) or \
                   (orig_path['generation'] == 's-' and orig_ensemble_id == 1):
                    if debug:
                        print(f"    DEBUG: Skipping {orig_path['generation']} swap in ensemble {orig_ensemble_id} (boundary)")
                    continue
                
                next_pe_data = all_ensemble_data.get(next_ensemble_id, [])
                result, chain = find_original_shooting_path(
                    orig_path['length'], orig_path['lambmin'], orig_path['lambmax'],
                    next_pe_data, all_ensemble_data, next_ensemble_id, n_ensembles,
                    tolerance, length_tolerance, visited_swaps.copy(), debug, target_cycle=orig_path['cycle']
                )
                
                if result is not None:
                    # Return the original shooting path and the full chain
                    return result, [(orig_ensemble_id, orig_path)] + chain
    
    if debug:
        print(f"    DEBUG: No matching path found in ensemble {orig_ensemble_id:03d}")
    return None, []

def process_ensemble(ensemble_dir, ensemble_id, all_ensemble_data, n_ensembles, dry_run=False):
    """
    Process a single ensemble directory and modify pathensemble.txt weights.
    
    Parameters:
    -----------
    ensemble_dir : Path
        Path to ensemble directory
    ensemble_id : int
        Ensemble ID (0, 1, 2, ...)
    all_ensemble_data : dict
        Dictionary containing data from all ensembles
    n_ensembles : int
        Total number of ensembles
    dry_run : bool
        If True, only print changes without modifying files
        
    Returns:
    --------
    list: Modified lines for the pathensemble.txt file
    """
    pathensemble_file = ensemble_dir / "pathensemble.txt"
    
    if not pathensemble_file.exists():
        print(f"Warning: {pathensemble_file} not found, skipping ensemble {ensemble_id}")
        return None
    
    print(f"Processing ensemble {ensemble_id:03d} ({ensemble_dir.name})")
    
    # Read and parse the file
    with open(pathensemble_file, 'r') as f:
        lines = f.readlines()
    
    modified_lines = []
    changes_made = 0
    
    for line_num, line in enumerate(lines):
        path_data = parse_pathensemble_line(line)
        
        if path_data is None:
            modified_lines.append(line)
            continue
        
        original_weight = path_data['weight']
        new_weight = original_weight
        
        # Check if this is an accepted path
        if path_data['flag'] == 'ACC':
            
            # Case 1: Shooting path that needs HA weight
            if (path_data['generation'] == 'sh' and 
                should_get_ha_weight(ensemble_id, path_data['lmr'], n_ensembles)):
                new_weight = 2.0
                if dry_run:
                    print(f"  Would modify shooting path cycle {path_data['cycle']}: "
                          f"{path_data['lmr']} weight {original_weight} -> {new_weight}")
                else:
                    print(f"  Modified shooting path cycle {path_data['cycle']}: "
                          f"{path_data['lmr']} weight {original_weight} -> {new_weight}")
                changes_made += 1
            
            # Case 2: Swapped path that might need HA weight
            elif path_data['generation'] in ['s+', 's-']:
                # Skip s+ swaps in ensemble 000 and s- swaps in ensemble 001
                if (path_data['generation'] == 's+' and ensemble_id == 0) or \
                   (path_data['generation'] == 's-' and ensemble_id == 1):
                    if dry_run:
                        print(f"  Skipping {path_data['generation']} swap in ensemble {ensemble_id:03d} (boundary condition)")
                    # Don't continue - we still need to write the unmodified line
                else:
                    # Determine which ensemble to look in
                    if path_data['generation'] == 's+':
                        # s+ looks in the NEXT ensemble (higher number)
                        search_ensemble_id = ensemble_id + 1
                    elif path_data['generation'] == 's-':
                        # s- looks in the PREV ensemble (lower number) 
                        search_ensemble_id = ensemble_id - 1
                    
                    # Check if target ensemble exists
                    if search_ensemble_id < 0 or search_ensemble_id >= n_ensembles:
                        if dry_run:
                            print(f"  Cannot find original for {path_data['generation']} swap: target ensemble {search_ensemble_id:03d} doesn't exist")
                    else:
                        # Find the original shooting path in the target ensemble
                        search_pe_data = all_ensemble_data.get(search_ensemble_id, [])
                        if dry_run:  # Enable debug output in dry-run mode
                            print(f"  DEBUG: Looking for original of {path_data['generation']} swap (cycle {path_data['cycle']}) in ensemble {search_ensemble_id:03d}")
                        orig_path, swap_chain = find_original_shooting_path(
                            path_data['length'], path_data['lambmin'], path_data['lambmax'], 
                            search_pe_data, all_ensemble_data, search_ensemble_id, n_ensembles,
                            tolerance=1e-6, length_tolerance=0, debug=dry_run, target_cycle=path_data['cycle']
                        )
                        
                        if orig_path is not None:
                            # Check if original path would have gotten HA weight
                            final_ensemble_id = swap_chain[-1][0] if swap_chain else search_ensemble_id
                            if should_get_ha_weight(final_ensemble_id, orig_path['lmr'], n_ensembles):
                                new_weight = 2.0
                                chain_info = " -> ".join([f"ens{eid:03d}({pdata['generation']})" 
                                                        for eid, pdata in swap_chain])
                                if dry_run:
                                    print(f"  Would modify swapped path cycle {path_data['cycle']}: "
                                          f"{path_data['generation']} {path_data['lmr']} -> "
                                          f"original {orig_path['lmr']} via {chain_info}, "
                                          f"weight {original_weight} -> {new_weight}")
                                else:
                                    print(f"  Modified swapped path cycle {path_data['cycle']}: "
                                          f"{path_data['generation']} {path_data['lmr']} -> "
                                          f"original {orig_path['lmr']} via {chain_info}, "
                                          f"weight {original_weight} -> {new_weight}")
                                changes_made += 1
                            else:
                                if dry_run:
                                    print(f"  Swapped path cycle {path_data['cycle']}: "
                                          f"original {orig_path['lmr']} in ensemble {final_ensemble_id:03d} "
                                          f"does not need HA weight")
                        else:
                            print(f"  ERROR: Could not find original shooting path for swapped path "
                                  f"cycle {path_data['cycle']} ({path_data['generation']}) "
                                  f"searching in ensemble {search_ensemble_id:03d}")
                            print(f"    Target: length={path_data['length']}, "
                                  f"lambmin={path_data['lambmin']:.6f}, "
                                  f"lambmax={path_data['lambmax']:.6f}")
        
        # Update the line with new weight if changed
        if abs(new_weight - original_weight) > 1e-10:
            # Preserve original spacing by replacing only the weight value in place
            original_line = line.rstrip('\n')
            parts = original_line.split()
            
            # Find the position of the weight column in the original line
            weight_start = 0
            for i in range(16):  # Skip first 16 columns
                weight_start = original_line.find(parts[i], weight_start) + len(parts[i])
                # Skip whitespace to get to next column
                while weight_start < len(original_line) and original_line[weight_start] == ' ':
                    weight_start += 1
            
            # Find end of weight value
            weight_end = weight_start
            while weight_end < len(original_line) and original_line[weight_end] not in [' ', '\t', '\n']:
                weight_end += 1
            
            # Replace the weight value while preserving spacing
            modified_line = (original_line[:weight_start] + 
                           f"{new_weight:.12f}" + 
                           original_line[weight_end:] + "\n")
            modified_lines.append(modified_line)
        else:
            modified_lines.append(line)
    
    print(f"  Total changes made: {changes_made}")
    return modified_lines

def load_all_ensemble_data(base_dir):
    """
    Load pathensemble.txt data from all ensemble directories.
    
    Returns:
    --------
    dict: Dictionary with ensemble_id as key and list of path data as value
    """
    all_data = {}
    ensemble_dirs = []
    
    # Find all ensemble directories (000, 001, 002, etc.)
    for item in sorted(base_dir.iterdir()):
        if item.is_dir() and item.name.isdigit() and len(item.name) == 3:
            ensemble_dirs.append(item)
    
    n_ensembles = len(ensemble_dirs)
    print(f"Found {n_ensembles} ensemble directories")
    
    # Load data from each ensemble
    for ensemble_dir in ensemble_dirs:
        ensemble_id = int(ensemble_dir.name)
        pathensemble_file = ensemble_dir / "pathensemble.txt"
        
        if pathensemble_file.exists():
            with open(pathensemble_file, 'r') as f:
                lines = f.readlines()
            
            ensemble_data = []
            for line in lines:
                path_data = parse_pathensemble_line(line)
                if path_data is not None:
                    ensemble_data.append(path_data)
            
            all_data[ensemble_id] = ensemble_data
            print(f"  Loaded {len(ensemble_data)} paths from ensemble {ensemble_id:03d}")
        else:
            print(f"  Warning: {pathensemble_file} not found")
            all_data[ensemble_id] = []
    
    return all_data, n_ensembles

def main():
    parser = argparse.ArgumentParser(description=__doc__, 
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('base_directory', 
                       help='Directory containing ensemble subdirectories (000, 001, 002, etc.)')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Only print what would be changed without modifying files')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output for troubleshooting')
    parser.add_argument('--backup', action='store_true', default=True,
                       help='Create backup files before modification (default: True)')
    parser.add_argument('--no-backup', action='store_false', dest='backup',
                       help='Do not create backup files')
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_directory)
    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist")
        sys.exit(1)
    
    if not base_dir.is_dir():
        print(f"Error: {base_dir} is not a directory")
        sys.exit(1)
    
    print(f"Processing TIS simulation in: {base_dir}")
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
    
    # Load all ensemble data first
    print("\nLoading data from all ensembles...")
    all_ensemble_data, n_ensembles = load_all_ensemble_data(base_dir)
    
    if n_ensembles == 0:
        print("Error: No ensemble directories found")
        sys.exit(1)
    
    # Process each ensemble
    print(f"\nProcessing {n_ensembles} ensembles...")
    for ensemble_id in range(n_ensembles):
        ensemble_dir = base_dir / f"{ensemble_id:03d}"
        
        if not ensemble_dir.exists():
            print(f"Warning: Ensemble directory {ensemble_dir} not found, skipping")
            continue
        
        modified_lines = process_ensemble(ensemble_dir, ensemble_id, 
                                        all_ensemble_data, n_ensembles, args.dry_run)
        
        if modified_lines is not None and not args.dry_run:
            pathensemble_file = ensemble_dir / "pathensemble.txt"
            
            # Create backup if requested
            if args.backup:
                backup_file = ensemble_dir / "pathensemble.txt.bak"
                shutil.copy2(pathensemble_file, backup_file)
                print(f"  Created backup: {backup_file}")
            
            # Write modified file
            with open(pathensemble_file, 'w') as f:
                f.writelines(modified_lines)
            print(f"  Updated: {pathensemble_file}")
        
        print()
    
    if args.dry_run:
        print("DRY RUN COMPLETED - No files were modified")
        print("Run without --dry-run to apply changes")
    else:
        print("All pathensemble.txt files have been updated with HA weights")

if __name__ == "__main__":
    main()
