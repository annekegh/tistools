#!/usr/bin/env python3

# Test the improved search logic

def test_search_logic():
    # Simulate ensemble data
    test_data = []
    
    # Add more paths to test the 20-path limit
    cycles = list(range(5, 200, 5))  # 5, 10, 15, 20, ..., 195
    
    for cycle in cycles:
        test_data.append({
            'cycle': cycle,
            'flag': 'ACC',
            'generation': 'sh',
            'length': 100,
            'lambmin': 0.1,
            'lambmax': 0.2
        })
    
    # Test looking for paths before cycle 150 (should have > 20 paths before it)
    target_cycle = 150
    print(f"Testing search for paths before cycle {target_cycle}")
    
    # Get paths before target cycle
    before_target = []
    for path_data in test_data:
        if (path_data['flag'] == 'ACC' and 
            path_data['cycle'] < target_cycle):
            before_target.append(path_data)
    
    # Sort and take last 20
    before_target.sort(key=lambda x: x['cycle'])
    accepted_paths = before_target[-20:] if len(before_target) >= 20 else before_target
    
    print(f"Found {len(accepted_paths)} paths before cycle {target_cycle}")
    print(f"Cycle range: {accepted_paths[0]['cycle']} to {accepted_paths[-1]['cycle']}")
    print("Cycles:", [p['cycle'] for p in accepted_paths])
    
    # Verify we get exactly 20 cycles before 150
    # Should be cycles 55, 60, 65, ..., 145 (last 20 before 150)
    all_before_150 = list(range(5, 150, 5))
    expected = all_before_150[-20:]  # Last 20 before 150
    actual = [p['cycle'] for p in accepted_paths]
    
    print(f"Total paths before 150: {len(all_before_150)}")
    print(f"Expected last 20 cycles before 150: {expected}")
    print(f"Actual cycles found: {actual}")
    print(f"Found exactly 20 paths: {len(actual) == 20}")
    print(f"Test {'PASSED' if actual == expected and len(actual) == 20 else 'FAILED'}")

if __name__ == "__main__":
    test_search_logic()
