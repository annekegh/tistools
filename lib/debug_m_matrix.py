"""
Debug script to compare M matrix construction for memory order 0 vs base iSTAR.
"""

import numpy as np
import sys
# sys.path.insert(0, '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/APPTIS/tistools/lib')

# import istar_extended_memory as iem
# import istar_analysis as ia
from tistools import istar_extended_memory as iem
from tistools import istar_analysis as ia

# Test with a simple system: 5 interfaces
n_interfaces = 5

print("=" * 70)
print("DEBUGGING M MATRIX CONSTRUCTION - Memory Order 0 vs Base iSTAR")
print("=" * 70)
print(f"\nNumber of interfaces: {n_interfaces}")

# Create extended state space for memory order 0
print("\n--- Extended Memory (Order 0) State Space ---")
state_space_0 = iem.ExtendedStateSpace(n_interfaces, memory_order=0)
state_space_0.print_states()

print(f"\nNumber of states: {state_space_0.n_states}")
print(f"Absorbing states: {state_space_0.absorbing_states}")
print(f"Expected for base iSTAR: 2*N = 2*{n_interfaces} = {2*n_interfaces}")

# For base iSTAR:
# NS = 2*N, states = [0-], [0←], [0→], [1⊂], [2⊂], ..., [N-2⊂], [1⊃], [2⊃], ..., [N-1⊃=B]
# State indices:
#   0: [0-]
#   1: [0←]
#   2: [0→]
#   3 to N: [i⊂] for i=1 to N-2  (N-2 states)
#   N+1 to 2N-1: [i⊃] for i=1 to N-1 (N-1 states)
# Total: 3 + (N-2) + (N-1) = 3 + 2N - 3 = 2N

print("\n--- State Mapping Comparison ---")
print("Base iSTAR states:")
print("  0: [0-]")
print("  1: [0←]")
print("  2: [0→]")
for i in range(1, n_interfaces-1):
    print(f"  {2+i}: [{i}⊂]")
for i in range(1, n_interfaces):
    print(f"  {n_interfaces+i}: [{i}⊃]")
print(f"  {2*n_interfaces-1}: [B] (last state, should go to [0-])")

print("\nExtended memory (order 0) states:")
for state in state_space_0.states:
    print(f"  {state.state_id}: {state.label} (absorbing={state.is_absorbing})")

# Create a dummy q matrix (all zeros except for some transitions)
print("\n--- Creating Test q Matrix ---")
q_test = np.zeros((state_space_0.n_states, state_space_0.n_states))

# For testing, create simple probabilities
# [0→] should go to some forward states
# Forward states should go to backward states
# Backward states should go back to [0←]

# Set up some test probabilities
if state_space_0.n_states >= 3:
    # [0→] goes forward to state 3 with prob 0.3, state 4 with prob 0.5, etc.
    if state_space_0.n_states > 3:
        q_test[2, 3:min(6, state_space_0.n_states)] = 0.2
        q_test[2, 1] = max(0, 1.0 - np.sum(q_test[2, :]))  # Rest goes to [0←]

print(f"q_test shape: {q_test.shape}")
print(f"q_test[2, :] (from [0→]): {q_test[2, :]}")

# Construct M using extended memory function
print("\n--- Constructing M Matrix (Extended Memory, Order 0) ---")
M_extended = iem.construct_M_extended(q_test, state_space_0)

print("\nM_extended matrix:")
print(f"Shape: {M_extended.shape}")
print(f"Row sums: {np.sum(M_extended, axis=1)}")
print(f"\nM_extended[0, :] ([0-] transitions): {M_extended[0, :]}")
print(f"M_extended[1, :] ([0←] transitions): {M_extended[1, :]}")
print(f"M_extended[2, :] ([0→] transitions): {M_extended[2, :]}")
print(f"M_extended[-1, :] ([B] transitions): {M_extended[-1, :]}")

# For base iSTAR, construct P matrix (interface-to-interface)
print("\n--- Base iSTAR M Matrix Construction ---")
P_test = np.zeros((n_interfaces, n_interfaces))
# Simple test: P[0, 0] = 0.2, P[0, i] = 0.1 for i > 0
P_test[0, 0] = 0.2
P_test[0, 1:] = 0.1
# Normalize
for i in range(n_interfaces):
    row_sum = np.sum(P_test[i, :])
    if row_sum > 0:
        P_test[i, :] /= row_sum

print(f"\nP_test (interface-to-interface) shape: {P_test.shape}")
print(f"P_test:\n{P_test}")

# Construct M using base iSTAR function
NS = 2 * n_interfaces
M_base = ia.construct_M_istar(P_test, NS, n_interfaces)

print(f"\nM_base (base iSTAR) shape: {M_base.shape}")
print(f"Row sums: {np.sum(M_base, axis=1)}")
print(f"\nM_base[0, :] ([0-] transitions): {M_base[0, :]}")
print(f"M_base[1, :] ([0←] transitions): {M_base[1, :]}")
print(f"M_base[2, :] ([0→] transitions): {M_base[2, :]}")
print(f"M_base[-1, :] ([B] transitions): {M_base[-1, :]}")

print("\n--- Key Differences ---")
print(f"Extended [B] row: M_extended[-1, :] = {M_extended[-1, :]}")
print(f"Base [B] row:     M_base[-1, :] = {M_base[-1, :]}")

print(f"\nExtended [B] → [0-]: M_extended[-1, 0] = {M_extended[-1, 0]}")
print(f"Base [B] → [0-]:     M_base[-1, 0] = {M_base[-1, 0]}")

print(f"\nExtended [B] → [B]: M_extended[-1, -1] = {M_extended[-1, -1]}")
print(f"Base [B] → [B]:     M_base[-1, -1] = {M_base[-1, -1]}")

if np.allclose(M_extended[-1, :], M_base[-1, :]):
    print("\n✓ [B] row matches between extended (order 0) and base iSTAR")
else:
    print("\n✗ [B] row DOES NOT match between extended (order 0) and base iSTAR")
    print("   This is the bug!")
