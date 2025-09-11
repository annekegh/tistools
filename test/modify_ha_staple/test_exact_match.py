#!/usr/bin/env python3

# Test the exact matching logic with simulated data

def test_exact_matching():
    # Simulate a swapped path and its original
    swap_path = {
        'cycle': 294,
        'length': 88,
        'lambmin': 0.003490,
        'lambmax': 0.320727,
        'generation': 's+'
    }
    
    # Simulate ensemble data with the original shooting path somewhere in the middle
    test_paths = []
    
    # Add some early paths
    for i in range(10):
        test_paths.append({
            'cycle': i,
            'length': 50 + i,
            'lambmin': 0.1 + i*0.01,
            'lambmax': 0.2 + i*0.01,
            'generation': 'sh',
            'flag': 'ACC'
        })
    
    # Add the EXACT match for the swap (same cycle range as swap)
    test_paths.append({
        'cycle': 290,  # Close to swap cycle 294
        'length': 88,   # EXACT match
        'lambmin': 0.003490,  # EXACT match
        'lambmax': 0.320727,  # EXACT match
        'generation': 'sh',
        'flag': 'ACC'
    })
    
    # Add some later paths
    for i in range(295, 305):
        test_paths.append({
            'cycle': i,
            'length': 60 + i,
            'lambmin': 0.15 + i*0.001,
            'lambmax': 0.25 + i*0.001,
            'generation': 'sh',
            'flag': 'ACC'
        })
    
    print(f"Looking for swap path: cycle={swap_path['cycle']}, length={swap_path['length']}, lambmin={swap_path['lambmin']:.6f}, lambmax={swap_path['lambmax']:.6f}")
    print(f"Total paths to search: {len(test_paths)}")
    
    # Test exact matching
    tolerance = 1e-6
    length_tolerance = 0
    
    matches = []
    for path in test_paths:
        if path['flag'] == 'ACC':
            length_match = abs(path['length'] - swap_path['length']) <= length_tolerance
            lambmin_match = abs(path['lambmin'] - swap_path['lambmin']) < tolerance
            lambmax_match = abs(path['lambmax'] - swap_path['lambmax']) < tolerance
            
            if length_match and lambmin_match and lambmax_match:
                matches.append(path)
                print(f"MATCH: cycle={path['cycle']}, gen={path['generation']}, length={path['length']}, lambmin={path['lambmin']:.6f}, lambmax={path['lambmax']:.6f}")
    
    print(f"Found {len(matches)} exact matches")
    return len(matches) > 0

if __name__ == "__main__":
    success = test_exact_matching()
    print(f"Test {'PASSED' if success else 'FAILED'}")
