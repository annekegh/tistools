"""
Extended memory Markov State Model framework for iSTAR analysis.

This module implements an extensible framework for analyzing path sampling data
using Markov state models with extended memory. The key concept is:

Turn Definition
===============
A "turn" at interface i is defined as the recrossing of two subsequent interfaces
(i-1, i) or (i, i+1). The turn involves:
1. Coming from interface i±1
2. Crossing interface i
3. Going beyond i (into the region past i)
4. Turning back and recrossing interface i
5. Returning and crossing interface i±1

After a turn completes, the path is heading in the opposite direction from which
it approached.

Base States
===========
- Forward-facing turn at i (⊂): After the turn, path exits at i+1 heading toward B
- Backward-facing turn at i (⊃): After the turn, path exits at i-1 heading toward A

Extended Memory States (Order 1)
================================
After a turn at i completes, the path is at interface i±1. The extended state
captures what happens NEXT:

For a backward-facing turn at i (now at i-1, heading toward i-2):
1. Continue: path reaches i-2 and continues beyond
2. Turn: path turns at i-1 (recrosses back toward i)

For a forward-facing turn at i (now at i+1, heading toward i+2):
1. Continue: path reaches i+2 and continues beyond
2. Turn: path turns at i+1 (recrosses back toward i)

This gives 2 extended states per base turn direction, for a total of 4 states
at each interior interface (vs 2 in the base model).

Higher memory orders extend this further by tracking additional segments.

State Space Structure
=====================
For n interfaces (0 to N-1):
- Absorbing states at 0: States 0, 1, 2 are exceptions comprising the absorbing
  boundary at interface 0 (state A)
- Absorbing state at N-1: The boundary at interface N-1 (state B)
- Interior interfaces each have states for:
  - Forward-facing turns (⊂) + continuation behavior
  - Backward-facing turns (⊃) + continuation behavior

Example (5 interfaces, order 1 memory):
- [2⊂] base forward-facing turn at 2 → extended to:
  - [2⊂→4]: turn at 2, continues from 3 to 4
  - [2⊂⊃3]: turn at 2, then turns again at 3

Usage
=====
1. Create an ExtendedStateSpace with desired memory order
2. Use PathTurnAnalyzer to extract turn sequences from paths
3. Classify each path into its extended state sequence
4. Build transition matrices between extended states
5. Compute MFPT and other observables

References
----------
Based on the iSTAR (interface-based Markov State Models) framework.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum, auto

logger = logging.getLogger(__name__)


class Direction(Enum):
    """Direction of motion between interfaces."""
    FORWARD = 1    # Increasing interface index (toward B)
    BACKWARD = -1  # Decreasing interface index (toward A)
    UNDEFINED = 0  # For absorbing/boundary states


class TurnFacing(Enum):
    """
    Which way a turn is 'facing' - i.e., which direction the path exits after the turn.

    FORWARD (⊂): After the turn, path exits toward i+1 (toward B/higher interfaces)
    BACKWARD (⊃): After the turn, path exits toward i-1 (toward A/lower interfaces)
    """
    FORWARD = 1   # Exit toward higher interfaces (toward B)
    BACKWARD = -1 # Exit toward lower interfaces (toward A)


@dataclass
class CompleteTurn:
    """
    A complete turn involving recrossing of two adjacent interfaces.

    A complete turn at interface i involves crossing i, going past it,
    turning around, and recrossing i. The "facing" indicates the exit direction.

    Forward-facing turn (⊂) at i:
    - After the turn, path exits at i+1, heading toward B (higher interfaces)

    Backward-facing turn (⊃) at i:
    - After the turn, path exits at i-1, heading toward A (lower interfaces)

    Attributes
    ----------
    turn_interface : int
        The interface where the turn occurs.
    facing : TurnFacing
        FORWARD (⊂) = exits toward i+1, BACKWARD (⊃) = exits toward i-1.
    exit_interface : int
        Interface where the turn completes (i+1 for forward, i-1 for backward).
    start_idx : int, optional
        Index in orders array where turn begins.
    end_idx : int, optional
        Index in orders array where turn completes.
    """
    turn_interface: int
    facing: TurnFacing
    exit_interface: int
    start_idx: int = None
    end_idx: int = None

    @property
    def exit_direction(self) -> Direction:
        """Direction path is heading after turn completes."""
        if self.facing == TurnFacing.FORWARD:
            return Direction.FORWARD   # Forward-facing (⊂): heading toward B
        else:
            return Direction.BACKWARD  # Backward-facing (⊃): heading toward A

    def __repr__(self):
        facing_str = "⊂" if self.facing == TurnFacing.FORWARD else "⊃"
        return f"Turn({self.turn_interface}{facing_str}→{self.exit_interface})"


@dataclass
class NextBehavior:
    """
    Describes what happens after a turn at the next interface.

    After a complete turn, the path heads to the next interface. There it either:
    - Continues past that interface (CONTINUE)
    - Makes another turn (TURN)

    Attributes
    ----------
    next_interface : int
        The interface where this behavior occurs.
    behavior : str
        'continue' or 'turn'
    further_interface : int, optional
        If continuing, which interface it reaches next.
    """
    next_interface: int
    behavior: str  # 'continue' or 'turn'
    further_interface: int = None

    def __repr__(self):
        if self.behavior == 'continue':
            return f"→{self.further_interface}"
        else:
            return f"⊂{self.next_interface}" if self.further_interface and self.further_interface > self.next_interface else f"⊃{self.next_interface}"


@dataclass
class ExtendedState:
    """
    An extended state in the memory-enhanced Markov model.

    The state consists of:
    - A complete turn (base state)
    - What happens at subsequent interfaces (memory extension)

    Attributes
    ----------
    base_turn : CompleteTurn
        The turn that defines the base state.
    next_behaviors : List[NextBehavior]
        Sequence of what happens at subsequent interfaces (for memory > 0).
    state_id : int
        Unique identifier in the state space.
    memory_order : int
        How many subsequent interfaces are tracked.
    label : str
        Human-readable state label.
    """
    base_turn: CompleteTurn
    next_behaviors: List[NextBehavior] = field(default_factory=list)
    state_id: int = None
    memory_order: int = 0
    label: str = None
    is_absorbing: bool = False

    def __post_init__(self):
        self.memory_order = len(self.next_behaviors)
        if self.label is None:
            self.label = self._generate_label()

    def _generate_label(self) -> str:
        """Generate descriptive label."""
        if self.is_absorbing:
            if self.base_turn.turn_interface == 0:
                return "[A]"
            else:
                return "[B]"

        # Base turn label
        facing = "⊂" if self.base_turn.facing == TurnFacing.FORWARD else "⊃"
        label = f"[{self.base_turn.turn_interface}{facing}"

        # Add subsequent behaviors
        for nb in self.next_behaviors:
            if nb.behavior == 'continue':
                label += f"→{nb.further_interface}"
            else:
                turn_sym = "⊂" if nb.further_interface and nb.further_interface > nb.next_interface else "⊃"
                label += f"{turn_sym}{nb.next_interface}"

        label += "]"
        return label

    def __hash__(self):
        behaviors = tuple((nb.next_interface, nb.behavior, nb.further_interface)
                         for nb in self.next_behaviors)
        return hash((self.base_turn.turn_interface, self.base_turn.facing.value,
                    self.base_turn.exit_interface, behaviors, self.is_absorbing))

    def __eq__(self, other):
        if not isinstance(other, ExtendedState):
            return False
        if self.is_absorbing != other.is_absorbing:
            return False
        if self.is_absorbing and other.is_absorbing:
            return self.base_turn.turn_interface == other.base_turn.turn_interface
        return (self.base_turn.turn_interface == other.base_turn.turn_interface and
                self.base_turn.facing == other.base_turn.facing and
                len(self.next_behaviors) == len(other.next_behaviors) and
                all(nb1.next_interface == nb2.next_interface and
                    nb1.behavior == nb2.behavior and
                    nb1.further_interface == nb2.further_interface
                    for nb1, nb2 in zip(self.next_behaviors, other.next_behaviors)))


class ExtendedStateSpace:
    """
    Manages the extended state space with configurable memory order.

    Parameters
    ----------
    n_interfaces : int
        Number of interfaces (including boundaries at 0 and N-1).
    memory_order : int
        How many interfaces after the turn to track.
        0 = base iSTAR (just turn type)
        1 = track whether next interface continues or turns
        2 = track two interfaces ahead
        etc.
    """

    def __init__(self, n_interfaces: int, memory_order: int = 1):
        self.n_interfaces = n_interfaces
        self.memory_order = memory_order

        self.states: List[ExtendedState] = []
        self.state_lookup: Dict[tuple, ExtendedState] = {}
        self.absorbing_states: List[int] = []

        self._build_state_space()

    def _build_state_space(self):
        """Construct the complete state space."""
        state_id = 0

        # Special boundary states at interface 0 (states 0, 1, 2)
        # State 0: 0- paths (truly absorbing, left of interface 0)
        turn_0minus = CompleteTurn(turn_interface=0, facing=TurnFacing.BACKWARD, exit_interface=-1)
        state_0minus = ExtendedState(base_turn=turn_0minus, state_id=state_id, is_absorbing=True, label="[0-]")
        self.states.append(state_0minus)
        self.absorbing_states.append(state_id)
        self._add_lookup(state_0minus)
        state_id += 1

        # State 1: Paths reaching interface 0 from backward direction (from B toward A)
        turn_0_reach = CompleteTurn(turn_interface=0, facing=TurnFacing.BACKWARD, exit_interface=0)
        state_0_reach = ExtendedState(base_turn=turn_0_reach, state_id=state_id, is_absorbing=True, label="[0←]")
        self.states.append(state_0_reach)
        self.absorbing_states.append(state_id)
        self._add_lookup(state_0_reach)
        state_id += 1

        # State 2: Paths leaving interface 0 in forward direction (toward B)
        turn_0_leave = CompleteTurn(turn_interface=0, facing=TurnFacing.FORWARD, exit_interface=1)
        state_0_leave = ExtendedState(base_turn=turn_0_leave, state_id=state_id, is_absorbing=False, label="[0→]")
        self.states.append(state_0_leave)
        # self.absorbing_states.append(state_id)
        self._add_lookup(state_0_leave)
        state_id += 1

        # Forward-facing states (⊂) for all interior interfaces
        for turn_intf in range(1, self.n_interfaces - 1):
            exit_intf = turn_intf + 1

            # Skip if exit is out of bounds
            if exit_intf >= self.n_interfaces:
                continue

            base_turn = CompleteTurn(
                turn_interface=turn_intf,
                facing=TurnFacing.FORWARD,
                exit_interface=exit_intf
            )

            if self.memory_order == 0:
                # Only base states for memory order 0
                base_state = ExtendedState(
                    base_turn=base_turn,
                    next_behaviors=[],
                    state_id=state_id
                )
                self.states.append(base_state)
                self._add_lookup(base_state)
                state_id += 1
            else:
                # Only extended states for memory order > 0
                behavior_sequences = self._generate_behavior_sequences(
                    exit_intf, base_turn.exit_direction, self.memory_order
                )

                for seq in behavior_sequences:
                    state = ExtendedState(
                        base_turn=base_turn,
                        next_behaviors=seq,
                        state_id=state_id
                    )
                    self.states.append(state)
                    self._add_lookup(state)
                    state_id += 1

        # Backward-facing states (⊃) for all interior interfaces
        for turn_intf in range(1, self.n_interfaces - 1):
            exit_intf = turn_intf - 1

            # Skip if exit is out of bounds
            if exit_intf < 0:
                continue

            base_turn = CompleteTurn(
                turn_interface=turn_intf,
                facing=TurnFacing.BACKWARD,
                exit_interface=exit_intf
            )

            if self.memory_order == 0:
                # Only base states for memory order 0
                base_state = ExtendedState(
                    base_turn=base_turn,
                    next_behaviors=[],
                    state_id=state_id
                )
                self.states.append(base_state)
                self._add_lookup(base_state)
                state_id += 1
            else:
                # Only extended states for memory order > 0
                behavior_sequences = self._generate_behavior_sequences(
                    exit_intf, base_turn.exit_direction, self.memory_order
                )

                for seq in behavior_sequences:
                    state = ExtendedState(
                        base_turn=base_turn,
                        next_behaviors=seq,
                        state_id=state_id
                    )
                    self.states.append(state)
                    self._add_lookup(state)
                    state_id += 1

        # Absorbing state at B (interface N-1) - last state
        # NOTE: For memory order 0, [B] is NOT absorbing (transitions to [0-] to match base iSTAR)
        # For memory order > 0, [B] can be absorbing (depending on implementation choice)
        turn_B = CompleteTurn(turn_interface=self.n_interfaces-1,
                             facing=TurnFacing.FORWARD, exit_interface=self.n_interfaces)
        state_B = ExtendedState(base_turn=turn_B, state_id=state_id,
                               is_absorbing=False, label="[B]")
        self.states.append(state_B)
        self.absorbing_states.append(state_id)
        self._add_lookup(state_B)
        state_id += 1

        self.n_states = len(self.states)
        logger.info(f"Built extended state space: {self.n_states} states "
                   f"({self.n_interfaces} interfaces, memory order {self.memory_order})")

    def _generate_behavior_sequences(self, start_intf: int, direction: Direction,
                                    depth: int) -> List[List[NextBehavior]]:
        """
        Generate all valid behavior sequences starting from an interface.

        After a turn completes, the path is AT start_intf heading in direction.
        We track what happens at this interface and subsequent ones.

        Parameters
        ----------
        start_intf : int
            Interface where path currently is (exit interface from the turn).
        direction : Direction
            Direction the path is heading.
        depth : int
            How many interfaces to track.

        Returns
        -------
        list of list of NextBehavior
            All valid sequences.
        """
        if depth == 0:
            return [[]]

        sequences = []

        # We're at start_intf - what happens here?
        next_intf = start_intf

        # Check if at boundary
        if next_intf <= 0:
            # At absorbing state A
            nb = NextBehavior(next_interface=next_intf, behavior='continue',
                            further_interface=0)
            sequences.append([nb])
            return sequences

        if next_intf >= self.n_interfaces - 1:
            # At absorbing state B
            nb = NextBehavior(next_interface=next_intf, behavior='continue',
                            further_interface=self.n_interfaces - 1)
            sequences.append([nb])
            return sequences

        # Option 1: Continue through next_intf without turning
        # The further_interface is where the path reaches (i±2 from the turn)
        # NOT the intermediate interface being crossed
        if direction == Direction.FORWARD:
            continue_to = next_intf + 1  # Where path reaches after crossing next_intf
        else:
            continue_to = next_intf - 1

        if 0 <= continue_to < self.n_interfaces:
            nb_continue = NextBehavior(
                next_interface=next_intf,
                behavior='continue',
                further_interface=continue_to  # Where the path reaches
            )

            # Recurse for more depth - now at continue_to heading in same direction
            if depth > 1:
                sub_seqs = self._generate_behavior_sequences(continue_to, direction, depth - 1)
                for sub in sub_seqs:
                    sequences.append([nb_continue] + sub)
            else:
                sequences.append([nb_continue])

        # Option 2: Turn at next_intf
        # After turning at next_intf, path exits in reversed direction
        reversed_dir = Direction.BACKWARD if direction == Direction.FORWARD else Direction.FORWARD
        # The turn exits back at next_intf (in reversed direction)
        if reversed_dir == Direction.FORWARD:
            turn_exit = next_intf + 1  # Exits toward B
        else:
            turn_exit = next_intf - 1  # Exits toward A

        if 0 <= turn_exit < self.n_interfaces:
            nb_turn = NextBehavior(
                next_interface=next_intf,
                behavior='turn',
                further_interface=turn_exit
            )

            # Recurse for more depth - now at turn_exit heading in reversed direction
            if depth > 1:
                sub_seqs = self._generate_behavior_sequences(turn_exit, reversed_dir, depth - 1)
                for sub in sub_seqs:
                    sequences.append([nb_turn] + sub)
            else:
                sequences.append([nb_turn])

        return sequences

    def _add_lookup(self, state: ExtendedState):
        """Add state to lookup dictionary."""
        key = self._make_key(state)
        self.state_lookup[key] = state

    def _make_key(self, state: ExtendedState) -> tuple:
        """Create lookup key for state."""
        if state.is_absorbing:
            return ('absorbing', state.base_turn.turn_interface)

        behaviors = tuple((nb.next_interface, nb.behavior, nb.further_interface)
                         for nb in state.next_behaviors)
        return (state.base_turn.turn_interface, state.base_turn.facing.value,
                state.base_turn.exit_interface, behaviors)

    def find_state(self, turn_interface: int, facing: TurnFacing, exit_interface: int,
                  behaviors: Optional[List[Tuple]] = None) -> Optional[ExtendedState]:
        """
        Find a state by its characteristics.

        Parameters
        ----------
        turn_interface : int
            Interface where turn occurred.
        facing : TurnFacing
            Turn facing direction.
        exit_interface : int
            Interface where turn exits.
        behaviors : list of tuples, optional
            List of (next_intf, behavior, further_intf) tuples.

        Returns
        -------
        ExtendedState or None
        """
        if behaviors is None:
            behaviors = []

        key = (turn_interface, facing.value, exit_interface, tuple(behaviors))
        return self.state_lookup.get(key)

    def find_absorbing(self, interface: int) -> Optional[ExtendedState]:
        """Find absorbing state at given interface."""
        key = ('absorbing', interface)
        return self.state_lookup.get(key)

    def get_state_by_id(self, state_id: int) -> Optional[ExtendedState]:
        """Get state by ID."""
        if 0 <= state_id < self.n_states:
            return self.states[state_id]
        return None

    def get_state_id_by_turn(self, turn_interface: int, facing: TurnFacing, exit_interface: int) -> Optional[int]:
        """
        Find the first state ID that matches the base turn characteristics.
        """
        for state in self.states:
            if (state.base_turn.turn_interface == turn_interface and
                state.base_turn.facing == facing and
                state.base_turn.exit_interface == exit_interface):
                return state.state_id
        return None

    def print_states(self):
        """Print all states."""
        print(f"\nExtended State Space ({self.n_states} states)")
        print(f"  Interfaces: {self.n_interfaces}")
        print(f"  Memory order: {self.memory_order}")
        print(f"  Absorbing states: {self.absorbing_states}\n")

        for state in self.states:
            absorb = " [ABSORBING]" if state.is_absorbing else ""
            print(f"  {state.state_id:3d}: {state.label:25s}{absorb}")

    def __repr__(self):
        return (f"ExtendedStateSpace(n_interfaces={self.n_interfaces}, "
               f"memory_order={self.memory_order}, n_states={self.n_states})")


class PathTurnAnalyzer:
    """
    Analyzes path order parameter arrays to extract complete turns.

    Parameters
    ----------
    interfaces : array-like
        Sorted interface values.
    """

    def __init__(self, interfaces):
        self.interfaces = np.asarray(interfaces)
        self.n_interfaces = len(interfaces)

    def get_interface_crossings(self, orders: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Get all interface crossings in a path.

        Returns list of (idx, from_region, to_region) tuples.
        Region i is between interface i and i+1 (region -1 is below interface 0).
        """
        if orders.ndim > 1:
            vals = orders[:, 0]
        else:
            vals = orders

        crossings = []

        def get_region(v):
            idx = np.searchsorted(self.interfaces, v, side='right') - 1
            return max(-1, min(idx, self.n_interfaces - 1))

        prev = get_region(vals[0])
        for i in range(1, len(vals)):
            curr = get_region(vals[i])
            if curr != prev:
                crossings.append((i, prev, curr))
                prev = curr

        return crossings

    def extract_complete_turns(self, orders: np.ndarray) -> List[CompleteTurn]:
        """
        Extract complete turns from a path.

        A complete turn involves entering a region (crossing two adjacent interfaces),
        exploring that region (possibly with many recrossings), then exiting in the
        opposite direction. The key is detecting entry and exit through the same
        interface pair, regardless of recrossings in between.

        For a forward-facing turn (⊂) at interface i:
        - Entry: cross interfaces going backward (toward A) through i+1 then i
        - Turn: explore region around i (may recross i and i+1 multiple times)
        - Exit: cross interfaces going forward (toward B) through i then i+1
        - Result: path exits at i+1 heading forward

        For a backward-facing turn (⊃) at interface i:
        - Entry: cross interfaces going forward (toward B) through i-1 then i
        - Turn: explore region around i (may recross i-1 and i multiple times)
        - Exit: cross interfaces going backward (toward A) through i then i-1
        - Result: path exits at i-1 heading backward

        Returns
        -------
        list of CompleteTurn
        """
        crossings = self.get_interface_crossings(orders)

        if len(crossings) < 4:
            return []

        turns = []

        # Convert crossings to interface crossings
        interface_crossings = []
        for idx, from_r, to_r in crossings:
            if to_r > from_r:
                interface_crossings.append((idx, to_r, 1))  # (idx, interface, direction)
            else:
                interface_crossings.append((idx, from_r, -1))

        # Track potential turn entries and match them with exits
        # For each entry sequence, find ALL possible exits and choose the deepest/highest

        i = 0
        while i < len(interface_crossings) - 3:
            idx_i, intf_i, dir_i = interface_crossings[i]

            # Look for entry pattern: two crossings in the same direction through adjacent intfs
            for j in range(i + 1, min(i + 10, len(interface_crossings))):
                idx_j, intf_j, dir_j = interface_crossings[j]

                # Check for forward-facing turn entry (backward: i → i-1)
                if dir_i == -1 and dir_j == -1 and intf_i == intf_j + 1:
                    # Entry detected - now collect ALL consecutive backward crossings
                    backward_intfs = set()
                    k = i
                    while k < len(interface_crossings) and interface_crossings[k][2] == -1:
                        backward_intfs.add(interface_crossings[k][1])
                        k += 1

                    # Now k points to first non-backward crossing
                    # Collect forward crossings
                    forward_crossings = {}  # intf -> first index
                    for m in range(k, len(interface_crossings)):
                        idx_m, intf_m, dir_m = interface_crossings[m]
                        if dir_m == 1 and intf_m not in forward_crossings:
                            forward_crossings[intf_m] = m

                    # Find adjacent pairs that complete the turn
                    candidates = []
                    for intf_lower in sorted(backward_intfs):
                        intf_upper = intf_lower + 1
                        if intf_upper in backward_intfs and \
                           intf_lower in forward_crossings and \
                           intf_upper in forward_crossings:
                            # Check that forward upper comes after forward lower
                            if forward_crossings[intf_upper] > forward_crossings[intf_lower]:
                                # Only add if turn_lower is valid (not at boundary)
                                if 0 < intf_lower < self.n_interfaces - 1:
                                    candidates.append((intf_lower, intf_upper,
                                                     forward_crossings[intf_upper]))

                    # Choose the LOWEST interface pair (deepest turn)
                    if candidates:
                        turn_lower, turn_upper, end_idx_k = candidates[0]
                        turn = CompleteTurn(
                            turn_interface=turn_lower,
                            facing=TurnFacing.FORWARD,
                            exit_interface=turn_upper,
                            start_idx=interface_crossings[max(k-2, i)][0],
                            end_idx=interface_crossings[end_idx_k][0]
                        )
                        turns.append(turn)
                        i = k - 1  # Continue from end of backward sequence
                        break

                # Check for backward-facing turn entry (forward: i → i+1)
                elif dir_i == 1 and dir_j == 1 and intf_i == intf_j - 1:
                    # Entry detected - now collect ALL consecutive forward crossings
                    forward_intfs = set()
                    k = i
                    while k < len(interface_crossings) and interface_crossings[k][2] == 1:
                        forward_intfs.add(interface_crossings[k][1])
                        k += 1

                    # Now k points to first non-forward crossing
                    # Collect backward crossings
                    backward_crossings = {}  # intf -> first index
                    for m in range(k, len(interface_crossings)):
                        idx_m, intf_m, dir_m = interface_crossings[m]
                        if dir_m == -1 and intf_m not in backward_crossings:
                            backward_crossings[intf_m] = m

                    # Find adjacent pairs that complete the turn
                    candidates = []
                    for intf_upper in sorted(forward_intfs, reverse=True):
                        intf_lower = intf_upper - 1
                        if intf_lower in forward_intfs and \
                           intf_upper in backward_crossings and \
                           intf_lower in backward_crossings:
                            # Check that backward lower comes after backward upper
                            if backward_crossings[intf_lower] > backward_crossings[intf_upper]:
                                # Only add if turn_upper is valid (not at boundary)
                                if 0 < intf_upper < self.n_interfaces - 1:
                                    candidates.append((intf_lower, intf_upper,
                                                     backward_crossings[intf_lower]))

                    # Choose the HIGHEST interface pair (highest turn)
                    if candidates:
                        turn_lower, turn_upper, end_idx_k = candidates[0]
                        turn = CompleteTurn(
                            turn_interface=turn_upper,
                            facing=TurnFacing.BACKWARD,
                            exit_interface=turn_lower,
                            start_idx=interface_crossings[max(k-2, i)][0],
                            end_idx=interface_crossings[end_idx_k][0]
                        )
                        turns.append(turn)
                        i = k - 1  # Continue from end of forward sequence
                        break

            i += 1

        return turns

    def get_boundary_states(self, orders: np.ndarray, state_space: 'ExtendedStateSpace'
                           ) -> Tuple[Optional[ExtendedState], Optional[ExtendedState]]:
        """
        Determine the start and end boundary states for a path.

        Returns
        -------
        (start_state, end_state) : tuple
            - start_state: State 2 [0→] if path starts from A going forward,
                          State B if path starts from B going backward, else None
            - end_state: State 1 [0←] if path ends at A, State B if path ends at B, else None
        """
        if orders.ndim > 1:
            vals = orders[:, 0]
        else:
            vals = orders

        if len(vals) < 2:
            return None, None

        start_state = None
        end_state = None

        # Check start: is path coming from boundary A or B?
        start_region = np.searchsorted(self.interfaces, vals[0], side='right') - 1
        if start_region <= -1:
            # Starting from A - check direction of first movement
            for j in range(1, min(10, len(vals))):
                next_region = np.searchsorted(self.interfaces, vals[j], side='right') - 1
                if next_region != start_region:
                    if next_region > start_region:
                        # Starting from A, going forward -> state 2 [0→]
                        start_state = state_space.states[2] if len(state_space.states) > 2 else None
                    break
        elif start_region >= self.n_interfaces - 1:
            # Starting from B - always state B
            start_state = state_space.states[-1] if state_space.states else None

        # Check end: does path reach state A or B?
        end_region = np.searchsorted(self.interfaces, vals[-1], side='right') - 1
        if end_region <= -1:
            # Path ends at or below interface 0 -> state 1 [0←]
            end_state = state_space.states[1] if len(state_space.states) > 1 else None
        elif end_region >= self.n_interfaces - 1:
            # Path ends at B -> state B (last state)
            end_state = state_space.states[-1] if state_space.states else None

        return start_state, end_state

    def extract_turn_with_next_behavior(self, orders: np.ndarray,
                                        state_space: ExtendedStateSpace
                                       ) -> List[Tuple[CompleteTurn, List[NextBehavior]]]:
        """
        Extract turns with their behavior for state classification.

        Paths have at most 2 turns: one at the start and one at the end.
        - First turn (start): behavior is what happens AFTER the turn
        - Last turn (end, if different from first): behavior is what happened BEFORE

        Returns
        -------
        list of (CompleteTurn, list of NextBehavior) tuples
        """
        crossings = self.get_interface_crossings(orders)
        all_turns = self.extract_complete_turns(orders)

        if not all_turns:
            return []

        # Keep only first and last turn (max 2 turns per path)
        if len(all_turns) == 1:
            turns = all_turns  # Only one turn
        else:
            turns = [all_turns[0], all_turns[-1]]  # First and last

        results = []

        # Check if path ends at a boundary
        if orders.ndim > 1:
            vals = orders[:, 0]
        else:
            vals = orders
        start_region = np.searchsorted(self.interfaces, vals[0], side='right') - 1
        end_region = np.searchsorted(self.interfaces, vals[-1], side='right') - 1
        path_ends_at_boundary = (end_region <= -1 or end_region >= self.n_interfaces - 1)
        path_starts_at_boundary = (start_region <= -1 or start_region >= self.n_interfaces - 1)

        for turn_idx, turn in enumerate(turns):
            is_last_turn = (turn_idx == len(turns) - 1) and len(turns) > 1

            # Decide whether to look before or after the turn
            look_before = is_last_turn or (not path_ends_at_boundary and path_starts_at_boundary)

            if look_before:
                # Look at what happened BEFORE this turn
                # Get crossings before the turn start
                prior_crossings = [(i, f, t) for i, f, t in crossings if i <= turn.start_idx]

                if not prior_crossings:
                    results.append((turn, []))
                    continue

                # Work backwards from the turn
                behaviors = []
                # The turn starts at turn.start_idx, entering from some direction
                # We want to know what interface the path was at before entering the turn

                # For a forward turn (⊂), entry is backward through upper then lower
                # So before entry, path was at the upper interface going backward
                # For a backward turn (⊃), entry is forward through lower then upper
                # So before entry, path was at the lower interface going forward

                if turn.facing == TurnFacing.FORWARD:
                    # Forward turn: entered backward, so was coming from higher interface
                    prev_intf = turn.exit_interface  # This is turn_intf + 1
                    prev_dir = Direction.BACKWARD
                else:
                    # Backward turn: entered forward, so was coming from lower interface
                    prev_intf = turn.exit_interface  # This is turn_intf - 1
                    prev_dir = Direction.FORWARD

                # Analyze what happened at prev_intf by looking at prior crossings
                for depth in range(state_space.memory_order):
                    if prev_intf <= 0 or prev_intf >= self.n_interfaces - 1:
                        # Reached boundary
                        if prev_intf <= 0:
                            behaviors.insert(0, NextBehavior(prev_intf, 'continue', 0))
                        else:
                            behaviors.insert(0, NextBehavior(prev_intf, 'continue', self.n_interfaces - 1))
                        break

                    # Find crossings related to prev_intf
                    crossings_at_prev = [c for c in prior_crossings
                                        if c[1] == prev_intf or c[2] == prev_intf]

                    if not crossings_at_prev:
                        break

                    # Look at the last crossing before turn entry
                    last_cross = crossings_at_prev[-1]
                    cross_from, cross_to = last_cross[1], last_cross[2]

                    # Determine if path continued or turned to get here
                    if (prev_dir == Direction.BACKWARD and cross_to < cross_from) or \
                       (prev_dir == Direction.FORWARD and cross_to > cross_from):
                        # Continued in same direction
                        if prev_dir == Direction.BACKWARD:
                            came_from = prev_intf + 1
                        else:
                            came_from = prev_intf - 1
                        behaviors.insert(0, NextBehavior(prev_intf, 'continue', came_from))
                        prev_intf = came_from
                    else:
                        # Turned to get here
                        if prev_dir == Direction.BACKWARD:
                            came_from = prev_intf - 1
                        else:
                            came_from = prev_intf + 1
                        behaviors.insert(0, NextBehavior(prev_intf, 'turn', came_from))
                        prev_intf = came_from
                        prev_dir = Direction.BACKWARD if prev_dir == Direction.FORWARD else Direction.FORWARD

                results.append((turn, behaviors))

            else:
                # Look at what happens AFTER the turn (existing logic)
                turn_end = turn.end_idx
                remaining_crossings = [(i, f, t) for i, f, t in crossings if i >= turn_end]

                behaviors = []
                curr_intf = turn.exit_interface
                curr_dir = turn.exit_direction

                for depth in range(state_space.memory_order):
                    next_intf = curr_intf

                    if next_intf <= 0 or next_intf >= self.n_interfaces - 1:
                        if next_intf <= 0:
                            behaviors.append(NextBehavior(next_intf, 'continue', 0))
                        else:
                            behaviors.append(NextBehavior(next_intf, 'continue', self.n_interfaces - 1))
                        break

                    crossings_at_next = [c for c in remaining_crossings
                                        if c[1] == next_intf or c[2] == next_intf or
                                           abs(c[1] - next_intf) <= 1 or abs(c[2] - next_intf) <= 1]

                    if not crossings_at_next:
                        exit_crossing = [(i, f, t) for i, f, t in crossings if i == turn_end]
                        if exit_crossing:
                            _, _, exit_to = exit_crossing[0]
                            if curr_dir == Direction.FORWARD and exit_to > curr_intf:
                                behaviors.append(NextBehavior(curr_intf, 'continue', exit_to))
                            elif curr_dir == Direction.BACKWARD and exit_to < curr_intf:
                                behaviors.append(NextBehavior(curr_intf, 'continue', exit_to))
                        break

                    first_cross = crossings_at_next[0]
                    cross_dir = 1 if first_cross[2] > first_cross[1] else -1

                    if (curr_dir == Direction.FORWARD and cross_dir == 1) or \
                       (curr_dir == Direction.BACKWARD and cross_dir == -1):
                        if curr_dir == Direction.FORWARD:
                            further = next_intf + 1
                        else:
                            further = next_intf - 1
                        behaviors.append(NextBehavior(next_intf, 'continue', further))
                        curr_intf = further
                    else:
                        if curr_dir == Direction.FORWARD:
                            further = next_intf - 1
                        else:
                            further = next_intf + 1
                        behaviors.append(NextBehavior(next_intf, 'turn', further))
                        curr_intf = further
                        curr_dir = Direction.BACKWARD if curr_dir == Direction.FORWARD else Direction.FORWARD

                    remaining_crossings = [(i, f, t) for i, f, t in remaining_crossings
                                          if i > first_cross[0]]

                results.append((turn, behaviors))

        return results

    def classify_path(self, orders: np.ndarray, state_space: ExtendedStateSpace) -> Union[Optional[ExtendedState], List[ExtendedState]]:
        """
        Classify a path into exactly 2 extended states: start and end.

        A path has exactly 2 states representing its trajectory:
        - Start state: where the path comes from (boundary or first turn)
        - End state: where the path goes to (last turn or boundary)

        Priority: interior turn states take precedence over boundary states.
        When interior turns are detected, the ending boundary is omitted.

        Parameters
        ----------
        orders : np.ndarray
            Path order parameter values.
        state_space : ExtendedStateSpace
            State space to classify into.
        return_all : bool
            If True, return all states visited. Otherwise return middle state.

        Returns
        -------
        ExtendedState or list of ExtendedState or None
        """
        states = []

        # Get boundary states (start and end)
        start_state, end_state = self.get_boundary_states(orders, state_space)

        # Get interior turn states (max 2: first and last)
        turn_behaviors = self.extract_turn_with_next_behavior(orders, state_space)

        turn_states = []
        for turn, behaviors in turn_behaviors:
            # Try to find matching state
            behavior_tuples = [(nb.next_interface, nb.behavior, nb.further_interface)
                              for nb in behaviors]

            # For memory_order = 0, try with empty behaviors (base states)
            # For memory_order > 0, try with full behaviors first, then progressively shorter
            min_length = 0 if state_space.memory_order == 0 else 1

            for length in range(len(behavior_tuples), min_length - 1, -1):
                state = state_space.find_state(
                    turn.turn_interface,
                    turn.facing,
                    turn.exit_interface,
                    behavior_tuples[:length]
                )
                if state is not None:
                    turn_states.append(state)
                    break

        # Build final state sequence:
        # Priority: interior turns take precedence over ending boundary
        # Always aim for exactly 2 states
        if turn_states:
            # Have interior turns - they take precedence over ending boundary
            if len(turn_states) >= 2:
                # Have 2 turns: use first and last
                states = [turn_states[0], turn_states[-1]]
            elif len(turn_states) == 1:
                # Have 1 turn: combine with start boundary if available
                if start_state is not None:
                    # start_boundary + turn (omit end_boundary per user request)
                    states = [start_state, turn_states[0]]
                elif end_state is not None:
                    # turn + end_boundary (omit start_boundary)
                    states = [turn_states[0], end_state]
                else:
                    # No start boundary: use turn as both start and end
                    raise ValueError("Single turn detected but no start or end boundary state found. "
                                     "This should not happen if paths start/end at boundaries.")
                    states = [turn_states[0], turn_states[0]]
        else:
            # No interior turns: use boundaries
            if start_state is not None:
                states.append(start_state)
            if end_state is not None:
                states.append(end_state)

        return states


class ExtendedTransitionMatrix:
    """Transition matrix for extended state space."""

    def __init__(self, state_space: ExtendedStateSpace):
        self.state_space = state_space
        self.n_states = state_space.n_states
        self.counts = np.zeros((self.n_states, self.n_states))
        self.transition_matrix = None

    def add_transition(self, from_state: ExtendedState, to_state: ExtendedState,
                      weight: float = 1.0):
        """Add weighted transition."""
        if from_state.state_id is not None and to_state.state_id is not None:
            self.counts[from_state.state_id, to_state.state_id] += weight

    def normalize(self) -> np.ndarray:
        """Normalize to row-stochastic matrix."""
        row_sums = self.counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.transition_matrix = self.counts / row_sums
        return self.transition_matrix

    def apply_absorbing(self):
        """Enforce absorbing conditions."""
        if self.transition_matrix is None:
            self.normalize()

        for abs_id in self.state_space.absorbing_states:
            self.transition_matrix[abs_id, :] = 0
            self.transition_matrix[abs_id, abs_id] = 1.0

    def get_matrix(self, normalized: bool = True) -> np.ndarray:
        """Get transition matrix."""
        if normalized:
            if self.transition_matrix is None:
                self.normalize()
            return self.transition_matrix
        return self.counts


def analyze_path_ensemble(path_ensemble, interfaces: np.ndarray,
                         state_space: ExtendedStateSpace,
                         use_weights: bool = True) -> Dict:
    """
    Analyze a path ensemble and build transition statistics.

    Returns dict with transition_matrix, state_counts, etc.
    """
    analyzer = PathTurnAnalyzer(interfaces)
    trans_matrix = ExtendedTransitionMatrix(state_space)

    state_counts = np.zeros(state_space.n_states)
    path_states = []
    classified = 0
    failed = 0

    for i in range(len(path_ensemble.cyclenumbers)):
        if path_ensemble.flags[i] != "ACC" or path_ensemble.generation[i] == "ld":
            path_states.append(None)
            continue

        orders = path_ensemble.orders[i]
        if orders is None:
            path_states.append(None)
            failed += 1
            continue

        states = analyzer.classify_path(orders, state_space, return_all=True)

        if not states:
            path_states.append(None)
            failed += 1
            continue

        weight = path_ensemble.weights[i] if use_weights else 1.0

        for state in states:
            state_counts[state.state_id] += weight

        for j in range(len(states) - 1):
            trans_matrix.add_transition(states[j], states[j + 1], weight)

        path_states.append([s.state_id for s in states])
        classified += 1

    trans_matrix.normalize()
    trans_matrix.apply_absorbing()

    return {
        'transition_matrix': trans_matrix,
        'state_counts': state_counts,
        'classified_paths': classified,
        'failed_paths': failed,
        'path_states': path_states
    }


# ============================================================================
# Extended Weight Matrix Computation (q-probability based)
# ============================================================================

def compute_weight_matrix_extended(pe, pe_id: int, interfaces: np.ndarray,
                                   state_space: ExtendedStateSpace,
                                   weights: np.ndarray = None,
                                   verbose: bool = False) -> np.ndarray:
    """
    Compute weight matrix for a path ensemble using extended memory states.

    This is the extended memory version of compute_weight_matrix from istar_analysis.py.
    Instead of tracking interface-to-interface transitions, it tracks transitions
    between extended states (turn + subsequent behavior combinations).

    Parameters
    ----------
    pe : PathEnsemble
        Path ensemble object containing paths, flags, generation info.
    pe_id : int
        Ensemble index (1-indexed as in TIS convention).
    interfaces : np.ndarray
        Array of interface positions.
    state_space : ExtendedStateSpace
        Extended state space with memory order.
    weights : np.ndarray, optional
        Pre-computed path weights. If None, uses equal weights.
    verbose : bool
        Print debugging information.

    Returns
    -------
    np.ndarray
        Weight matrix of shape (n_states, n_states) where entry [i,j] is the
        weighted count of paths transitioning from extended state i to state j.

    Notes
    -----
    Key differences from base iSTAR weight matrix:

    1. **Dimensions**: n_states × n_states instead of n_interfaces × n_interfaces

    2. **Reduced ensemble coverage**: Each ensemble only contains paths that
       cross certain interfaces. With extended memory, we need specific turn
       patterns, further reducing which paths contribute to each probability.

    3. **State classification**: Paths are classified into (start_state, end_state)
       pairs using the turn detection algorithm, not just lambmin/lambmax.
    """
    analyzer = PathTurnAnalyzer(interfaces)
    n_states = state_space.n_states

    # Initialize weight matrix
    X = np.zeros((n_states, n_states))

    # Default weights if not provided
    if weights is None:
        weights = np.ones(len(pe.cyclenumbers))

    # Process each path
    classified = 0
    for i in range(len(pe.cyclenumbers)):
        # Skip load paths and rejected paths
        if pe.flags[i] != "ACC" or pe.generation[i] == "ld":
            continue

        orders = pe.orders[i]
        if orders is None:
            continue

        # Classify path into exactly 2 extended states
        states = analyzer.classify_path(orders, state_space)

        if states is None or len(states) < 2:
            continue

        # Get start and end states
        start_state = states[0]
        end_state = states[-1]

        # Add weighted transition
        if start_state.state_id is not None and end_state.state_id is not None:
            j_state = start_state.base_turn.turn_interface
            k_state = end_state.base_turn.turn_interface

            ptype = str(pe.lmrs[i]) if hasattr(pe, "lmrs") else None
            
            # Use explicitly provided start/end interfaces if available, else derive from state
            if hasattr(pe, "start_intf") and hasattr(pe, "end_intf"):
                j = int(pe.start_intf[i])
                k = int(pe.end_intf[i])
                j = min(j, len(interfaces) - 1)
                k = min(k, len(interfaces) - 1)
            else:
                j = j_state
                k = k_state
            
            if hasattr(pe, "dirs"):
                direction = int(pe.dirs[i])
            else:
                direction = 1 if k > j else -1

            should_count = True
            n_interfaces = len(interfaces)
            
            # First enforce that the state classification matches the explicit start/end
            if j != j_state or k != k_state:
                should_count = False

            if j < k:
                if j == 0 and k == 1:
                    if pe_id != 2:
                        should_count = (direction == 1)
                    else:
                        should_count = ('LML' in ptype) if ptype else True
                elif j == n_interfaces - 2 and k == n_interfaces - 1:
                    should_count = ('RMR' in ptype) if ptype else True
            elif j > k:
                if j == 1 and k == 0:
                    if pe_id != 2:
                        should_count = (direction == -1)
                    else:
                        should_count = ('LML' in ptype) if ptype else True
                elif j == n_interfaces - 1 and k == n_interfaces - 2:
                    should_count = ('RMR' in ptype) if ptype else True

            if should_count:
                X[start_state.state_id, end_state.state_id] += weights[i]
                classified += 1

    if verbose:
        print(f"Ensemble {pe_id}: classified {classified} paths, total weight = {np.sum(X):.4f}")

    return X


def compute_weight_matrices_extended_from_weight_results(weight_results: Dict, state_space: ExtendedStateSpace,
                                                         verbose: bool = False) -> Dict[int, np.ndarray]:
    """
    Compute per-ensemble weight matrices for extended memory states from weight_results dict.

    This function builds weight matrices w_extended[ens_id][state_i][state_j] where
    each element represents the weighted count of paths in ensemble ens_id that
    transition from extended state i to state j.

    This is necessary for proper q-probability calculation that aggregates over
    appropriate ensembles, matching the methodology in istar_analysis.py.

    Parameters
    ----------
    weight_results : Dict
        Results from infretis weight calculation containing:
        - 'interfaces': array of interface positions
        - 'path_data': dict with 'orders', 'has_order', 'path_f', 'path_w'
    state_space : ExtendedStateSpace
        Extended state space with memory order.
    verbose : bool
        Print debugging information.

    Returns
    -------
    dict
        Dictionary mapping ensemble_id (1-indexed) to weight matrices of shape
        (n_states, n_states).
    """
    interfaces = np.asarray(weight_results['interfaces'])
    n_interfaces = len(interfaces)
    D = weight_results['path_data']

    # Get path data
    orders_list = D['orders']
    has_order = D['has_order']
    n_paths = len(orders_list)

    # Get per-ensemble weights
    path_f = D['path_f']  # Path occurrences
    path_w = D['path_w']  # Path weights

    # Calculate normalized weights per ensemble (same as in notebook)
    path_w_c = np.minimum(path_w, 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(path_w_c != 0, path_f / path_w_c, 0.0)
    denom = ratio.sum(axis=0)
    numer = path_f.sum(axis=0)
    norm_factor = np.where(denom != 0, numer / denom, 0.0)
    weight_k = ratio * norm_factor[np.newaxis, :]  # (N_paths, n_ens)

    # Create analyzer
    analyzer = PathTurnAnalyzer(interfaces)
    n_states = state_space.n_states
    n_ens = n_interfaces -1  # Number of ensembles

    # Initialize per-ensemble weight matrices
    w_extended = {}
    for ens_id in range(1, n_ens + 1):
        w_extended[ens_id] = np.zeros((n_states, n_states))

    # Track classification results
    classified = 0
    failed = 0

    if verbose:
        print(f"\nBuilding per-ensemble weight matrices for extended memory states...")
        print(f"  {n_paths} paths, {n_states} states, {n_ens} ensembles")

    # Process each path
    for i in range(n_paths):
        if not has_order[i] or orders_list[i] is None:
            failed += 1
            continue

        order_data = orders_list[i]

        # Classify path into extended states
        states = analyzer.classify_path(order_data, state_space)

        if not states or len(states) < 2:
            failed += 1
            continue

        # Get start and end states
        start_state = states[0]
        end_state = states[-1]

        j_state = start_state.base_turn.turn_interface
        k_state = end_state.base_turn.turn_interface

        ptype = str(D['ptype'][i]) if 'ptype' in D else None
        
        # Use explicitly provided start/end interfaces if available, else derive from state
        if 'start_intf' in D and 'end_intf' in D:
            j = int(D['start_intf'][i])
            k = int(D['end_intf'][i])
            j = min(max(j, 0), n_interfaces - 1)
            k = min(max(k, 0), n_interfaces - 1)
        else:
            j = j_state
            k = k_state
            
        if 'direction' in D:
            direction = int(D['direction'][i])
        elif 'dirs' in D:
            direction = int(D['dirs'][i])
        else:
            direction = 1 if k > j else -1

        # Add weighted transition to each ensemble this path contributes to
        for ens_id in range(1, n_ens + 1):
            weight = weight_k[i, ens_id]  # 0-indexed in array
            if weight > 0 and start_state.state_id is not None and end_state.state_id is not None:
                should_count = True
                
                # Enforce that the state classification matches the explicit start/end
                if j != j_state or k != k_state:
                    if (j in (0, 1) and k in (0, 1)) or (j in (n_interfaces - 2, n_interfaces - 1) and k in (n_interfaces - 2, n_interfaces - 1)):
                        j_state = j; k_state = k
                        start_state.base_turn.turn_interface = j_state
                        end_state.base_turn.turn_interface = k_state
                        start_state.state_id = state_space.get_state_id_by_turn(j_state, start_state.base_turn.facing, start_state.base_turn.exit_interface)
                        end_state.state_id = state_space.get_state_id_by_turn(k_state, end_state.base_turn.facing, end_state.base_turn.exit_interface)
                    else:
                        print(f"Warning: Path {i} state classification (j={j_state}, k={k_state}) does not match explicit start/end (j={j}, k={k}). Skipping this path for ensemble {ens_id}.")
                        should_count = False

                if j < k:
                    if j == 0 and k == 1:
                        if ens_id != 2:
                            should_count = (direction == 1)
                        else:
                            should_count = ('LML' in ptype) if ptype else True
                    elif j == n_interfaces - 2 and k == n_interfaces - 1:
                        should_count = ('RMR' in ptype) if ptype else True
                elif j > k:
                    if j == 1 and k == 0:
                        if ens_id != 2:
                            should_count = (direction == -1)
                        else:
                            should_count = ('LML' in ptype) if ptype else True
                    elif j == n_interfaces - 1 and k == n_interfaces - 2:
                        should_count = ('RMR' in ptype) if ptype else True
                
                if should_count:
                    w_extended[ens_id][start_state.state_id, end_state.state_id] += weight

        classified += 1

        if verbose and (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1}/{n_paths} paths...")

    if verbose:
        print(f"\nClassification complete:")
        print(f"  Successfully classified: {classified} paths ({100*classified/n_paths:.1f}%)")
        print(f"  Failed to classify: {failed} paths")
        print(f"\nPer-ensemble weight matrix sums:")
        for ens_id in range(1, n_ens + 1):
            total = np.sum(w_extended[ens_id])
            print(f"  Ensemble {ens_id}: {total:.4f}")

    return w_extended


def compute_weight_matrices_extended(ensembles: list, interfaces: np.ndarray,
                                     state_space: ExtendedStateSpace,
                                     weights_list: list = None,
                                     verbose: bool = True) -> Dict[int, np.ndarray]:
    """
    Compute weight matrices for all path ensembles.

    Parameters
    ----------
    ensembles : list
        List of PathEnsemble objects.
    interfaces : np.ndarray
        Array of interface positions.
    state_space : ExtendedStateSpace
        Extended state space.
    weights_list : list, optional
        List of weight arrays for each ensemble.
    verbose : bool
        Print progress information.

    Returns
    -------
    dict
        Dictionary mapping ensemble_id to weight matrix.
    """
    w_matrices = {}

    for idx, pe in enumerate(ensembles):
        pe_id = idx + 1  # 1-indexed
        weights = weights_list[idx] if weights_list is not None else None

        X = compute_weight_matrix_extended(pe, pe_id, interfaces, state_space,
                                          weights=weights, verbose=verbose)
        w_matrices[pe_id] = X

        if verbose:
            print(f"Weight matrix for ensemble {pe_id}: sum = {np.sum(X):.4f}")

    return w_matrices


def get_transition_probs_extended(w_matrices: Dict[int, np.ndarray],
                                  state_space: ExtendedStateSpace,
                                  verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate transition probabilities (q and p matrices) for extended memory states.

    This is the extended memory version of get_transition_probs_weights from istar_analysis.py.
    It properly aggregates counts over appropriate ensembles and uses partial sums to compute
    conditional probabilities.

    Parameters
    ----------
    w_matrices : dict
        Dictionary of per-ensemble weight matrices, keyed by ensemble_id (1-indexed).
        w_matrices[ens_id][state_i][state_j] = weighted count of paths in ensemble ens_id
        that transition from state i to state j.
    state_space : ExtendedStateSpace
        Extended state space.
    verbose : bool
        Print intermediate results.

    Returns
    -------
    tuple of (p, q)
        p : np.ndarray
            Full transition probability matrix P[i,j] = P(reach state j | start in state i)
        q : np.ndarray
            Direct transition probability matrix q[i,j] = P(go from i to j in one step)

    Notes
    -----
    This function follows the same methodology as istar_analysis.py:get_transition_probs_weights:

    1. For each pair of states (s_i, s_k):
       - Map states to their associated interface levels
       - Determine which ensembles sample transitions between these states
       - Aggregate counts from those ensembles using partial sums

    2. Partial sums for conditional probabilities:
       - For forward transition s_i → s_k (level_i < level_k):
         - numerator: sum of paths from s_i that reach s_k or "beyond"
         - denominator: sum of paths from s_i that reach s_{k-1} or "beyond"
       - q[i][k] = numerator / denominator

    3. The "beyond" concept is adapted for extended states:
       - States are ordered by their turn interface level
       - "Beyond" means states at higher (forward) or lower (backward) interfaces
    """
    n_states = state_space.n_states
    n_interfaces = state_space.n_interfaces

    # Initialize matrices
    q = np.zeros((n_states, n_states))
    p = np.zeros((n_states, n_states))

    if verbose:
        print("\n=== Extended Memory Transition Probabilities (ensemble-based) ===")
        print(f"State space: {n_states} states, {n_interfaces} interfaces")
        print(f"Memory order: {state_space.memory_order}")
        print(f"Number of ensembles: {len(w_matrices)}")

    # Calculate q matrix using ensemble aggregation (analogous to istar_analysis.py)
    for i in range(n_states):
        state_i = state_space.states[i]
        level_i = _get_state_level(state_i)

        for k in range(n_states):
            state_k = state_space.states[k]
            level_k = _get_state_level(state_k)

            counts = np.zeros(2)  # [numerator, denominator]

            # Handle special cases
            if i == k:
                # Self-transitions
                if i == 0:  # [0-] state
                    q[i][k] = 1
                else:
                    q[i][k] = 0
                continue

            # Special case for [0→] to first interface (analogous to i=0, k=1 in base iSTAR)
            if i == 2 and level_i == 0:  # Starting from [0→]
                # Use ensemble 1 (first ensemble)
                if 1 in w_matrices:
                    W = w_matrices[1]
                    # Sum over all states at level k or beyond
                    for j in range(n_states):
                        level_j = _get_state_level(state_space.states[j])
                        if level_j >= level_k:
                            counts[0] += W[i, j]
                        if level_j >= level_k - 1:
                            counts[1] += W[i, j]

            elif level_i < level_k:
                # Forward transitions (toward higher interfaces)
                # Aggregate over ensembles from level_i+1 to level_k
                for pe_i in range(level_i + 1, level_k + 1):
                    if pe_i > n_interfaces - 1 or pe_i not in w_matrices:
                        break

                    W = w_matrices[pe_i]

                    # numerator: paths from state i that reach state k or "beyond"
                    # denominator: paths from state i that reach state k-1 or "beyond"
                    for j in range(n_states):
                        level_j = _get_state_level(state_space.states[j])
                        if level_j >= level_k:
                            counts[0] += W[i, j]
                        if level_j >= level_k - 1:
                            counts[1] += W[i, j]

            elif level_i > level_k:
                # Backward transitions (toward lower interfaces)
                # Aggregate over ensembles from level_k+2 to level_i+2
                for pe_i in range(level_k + 2, level_i + 2):
                    if pe_i > n_interfaces - 1 or pe_i not in w_matrices:
                        break

                    W = w_matrices[pe_i]

                    # numerator: paths from state i that reach state k or "beyond" (lower)
                    # denominator: paths from state i that reach state k+1 or "beyond" (lower)
                    for j in range(n_states):
                        level_j = _get_state_level(state_space.states[j])
                        if level_j <= level_k:
                            counts[0] += W[i, j]
                        if level_j <= level_k + 1:
                            counts[1] += W[i, j]

            # Compute q[i][k]
            q[i][k] = counts[0] / counts[1] if counts[1] > 0 else 0

    # Handle absorbing states
    q[0, :] = 0
    q[0, 2] = 1.0  # [0-] → [0→]
    if n_states > 1:
        q[1, :] = 0
        q[1, 0] = 1.0  # [0←] → [0-]
    q[-1, :] = 0
    q[-1, -1] = 1.0  # [B] → [B] (absorbing)

    # Compute p matrix from q matrix
    # For extended memory, p can be derived from q similar to base iSTAR
    p = q.copy()  # Simplified: for now, use q as p

    if verbose:
        print("\nDirect transition probabilities (q matrix):")
        print("Non-zero entries (> 0.001):")
        count_nonzero = 0
        for i in range(min(n_states, 20)):  # Print first 20 states
            for j in range(n_states):
                if abs(q[i, j]) > 0.001:
                    s_from = state_space.states[i].label
                    s_to = state_space.states[j].label
                    print(f"  q[{i}]{s_from} -> [{j}]{s_to}: {q[i, j]:.4f}")
                    count_nonzero += 1
        if count_nonzero == 0:
            print("  (no entries > 0.001)")
        print(f"\nRow sums of q: min={np.min(np.sum(q, axis=1)):.4f}, max={np.max(np.sum(q, axis=1)):.4f}")

    return p, q


def _get_state_level(state: ExtendedState) -> int:
    """
    Get the interface level associated with an extended state.

    For extended states, the "level" is the turn_interface. This is used
    to determine which ensembles contribute to transitions between states.
    """
    if state.is_absorbing:
        if state.label in ["[0-]", "[0←]", "[0→]"]:
            return 0
        else:  # [B]
            return state.base_turn.turn_interface
    return state.base_turn.turn_interface


def _find_states_at_interface(state_space: ExtendedStateSpace, interface: int) -> List[ExtendedState]:
    """Find all non-absorbing states with turns at a given interface."""
    states = []
    for state in state_space.states:
        if state.is_absorbing:
            continue
        if state.base_turn.turn_interface == interface:
            states.append(state)
        # Also check if state's exit is at this interface
        if state.base_turn.exit_interface == interface:
            states.append(state)
    return list(set(states))  # Remove duplicates


def _find_states_with_turn(state_space: ExtendedStateSpace, interface: int,
                           facing: TurnFacing) -> List[ExtendedState]:
    """Find states with a specific turn pattern."""
    states = []
    for state in state_space.states:
        if state.is_absorbing:
            continue
        if state.base_turn.turn_interface == interface and state.base_turn.facing == facing:
            states.append(state)
    return states


def _print_sparse_matrix(M: np.ndarray, state_space: ExtendedStateSpace, threshold: float = 0.001):
    """Print non-zero elements of a matrix with state labels."""
    print(f"Non-zero entries (threshold > {threshold}):")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if abs(M[i, j]) > threshold:
                s_from = state_space.states[i].label
                s_to = state_space.states[j].label
                print(f"  {s_from} -> {s_to}: {M[i, j]:.4f}")


def construct_M_extended(q: np.ndarray, state_space: ExtendedStateSpace) -> np.ndarray:
    """
    Construct transition matrix M for extended memory iSTAR from q probabilities.

    This is the extended memory version of construct_M_istar from istar_analysis.py.

    Parameters
    ----------
    q : np.ndarray
        Direct transition probability matrix from get_transition_probs_extended.
    state_space : ExtendedStateSpace
        Extended state space.

    Returns
    -------
    np.ndarray
        Transition matrix M where M[i,j] = probability of transitioning
        from state i to state j.

    Notes
    -----
    For extended memory, the M matrix structure differs from base iSTAR:

    1. **Expanded diagonal**: More entries near the diagonal are 1 because
       extended states encode deterministic path segments. For example:
       - State [i⊂→j] means "turn at i, exit toward j" - the path to j is deterministic
       - The probability 1 appears for transitions along this deterministic segment

    2. **Reduced off-diagonal entries**: The probabilistic branching only occurs
       at decision points (turn vs continue), not at every interface crossing.

    3. **Boundary handling**: States [0-], [0←], [0→] and [B] have specific
       transition rules reflecting the boundary conditions.

    The structure follows:
    - Row 0 [0-]: Absorbing state, M[0,0] = 1
    - Row 1 [0←]: Goes to [0-], M[1,0] = 1
    - Row 2 [0→]: Goes to interior states based on q
    - Interior rows: Follow q matrix with deterministic segments at 1
    - Last row [B]: Absorbing state, M[-1,0] = 1
    """
    n_states = state_space.n_states
    M = np.zeros((n_states, n_states))

    # Copy q values as base
    M = q.copy()

    # Enforce absorbing states
    for abs_id in state_space.absorbing_states:
        M[abs_id, :] = 0
        M[abs_id, abs_id] = 1.0

    # State 0 [0-] is truly absorbing
    if n_states > 0:
        M[0, :] = 0
        M[0, 0] = 1.0

    # State 1 [0←] transitions to [0-]
    if n_states > 1:
        M[1, :] = 0
        M[1, 0] = 1.0

    # State 2 [0→] transitions to interior states or [B]
    # This is handled by q matrix from the data

    # Last state [B] behavior depends on state space
    # For memory order 0: [B] → [0-] (not absorbing, matches base i STAR)
    # For memory order > 0: [B] → [B] (absorbing)
    if n_states > 0:
        if state_space.states[-1].is_absorbing:
            # [B] is absorbing: stay in [B]
            M[-1, :] = 0
            M[-1, -1] = 1.0
        else:
            # [B] is not absorbing: transition to [0-] (base iSTAR behavior)
            M[-1, :] = 0
            M[-1, 0] = 1.0

    # Ensure all rows sum to 1 (or 0 for truly isolated states)
    for i in range(n_states):
        row_sum = np.sum(M[i, :])
        if row_sum > 0 and abs(row_sum - 1.0) > 1e-10:
            M[i, :] /= row_sum
        elif row_sum == 0 and i not in state_space.absorbing_states:
            # Isolated state with no outgoing transitions
            # Make it self-looping
            M[i, i] = 1.0

    return M


def global_pcross_extended(M: np.ndarray, state_space: ExtendedStateSpace,
                           verbose: bool = True) -> Tuple[float, np.ndarray]:
    """
    Calculate global crossing probability using extended memory transition matrix.

    This is the extended memory version of global_pcross_msm_star from istar_analysis.py.

    For memory order 0 (base iSTAR): Uses the same boundary state formalism as base iSTAR
    For memory order > 0 (extended memory): Uses absorbing Markov chain formalism

    Parameters
    ----------
    M : np.ndarray
        Transition matrix from construct_M_extended.
    state_space : ExtendedStateSpace
        Extended state space.
    verbose : bool
        Print intermediate results.

    Returns
    -------
    tuple of (p_cross, p_to_B)
        p_cross : float
            Global crossing probability P(A→B)
        p_to_B : np.ndarray
            Probability of reaching B from each state
    """
    n_states = state_space.n_states

    if state_space.memory_order == 0:
        # Use base iSTAR formalism: boundary states are [0-] (state 0) and [B] (state -1)
        # Intermediate states are [0→] through second-to-last
        if n_states <= 2:
            if verbose:
                print("Not enough states for crossing probability calculation")
            return 0.0, np.zeros(n_states)

        # Extract submatrices (same as base iSTAR)
        Mp = M[2:-1, 2:-1]  # Intermediate-to-intermediate
        D = M[2:-1, np.array([0, -1])]  # Intermediate-to-boundary
        E = M[np.array([0, -1]), 2:-1]  # Boundary-to-intermediate
        M11 = M[np.array([0, -1]), np.array([0, -1])]  # Boundary-to-boundary

        # Solve for crossing probability (same as base iSTAR)
        z1 = np.array([[0], [1]])  # Boundary conditions: z_[0-] = 0, z_B = 1
        try:
            a = np.identity(n_states - 3) - Mp
            z2 = np.linalg.solve(a, np.dot(D, z1))  # Solve (I-Mp)z2 = D·z1
        except np.linalg.LinAlgError:
            if verbose:
                print("Warning: Singular matrix, using pseudo-inverse")
            a = np.identity(n_states - 3) - Mp
            z2 = np.linalg.lstsq(a, np.dot(D, z1), rcond=None)[0]

        # Map to full state space
        p_to_B = np.zeros(n_states)
        p_to_B[0] = 0.0  # [0-]
        p_to_B[2:-1] = z2.flatten()  # Intermediate states
        p_to_B[-1] = 1.0  # [B]

        # Crossing probability: start from [0→] (state 2)
        p_cross = p_to_B[2]

    else:
        # Extended memory (order > 0): Use absorbing Markov chain formalism
        # [B] is absorbing, so use fundamental matrix approach
        absorbing = set(state_space.absorbing_states)
        transient = [i for i in range(n_states) if i not in absorbing]

        if len(transient) == 0:
            if verbose:
                print("No transient states found")
            return 0.0, np.zeros(n_states)

        # Extract transient-to-transient submatrix Q
        Q = M[np.ix_(transient, transient)]

        # Extract transient-to-B transition vector R
        B_idx = n_states - 1
        R = M[transient, B_idx]

        # Fundamental matrix N = (I - Q)^(-1)
        I = np.eye(len(transient))
        try:
            N = np.linalg.inv(I - Q)
        except np.linalg.LinAlgError:
            if verbose:
                print("Warning: Singular matrix, using pseudo-inverse")
            N = np.linalg.pinv(I - Q)

        # Absorption probability B_vec[i] = probability of reaching B from transient state i
        B_vec = N @ R

        # Map back to full state space
        p_to_B = np.zeros(n_states)
        for i, t_idx in enumerate(transient):
            p_to_B[t_idx] = B_vec[i]

        # Absorbing states
        p_to_B[B_idx] = 1.0  # Already at B
        # States [0-] and [0←] cannot reach B (stay at 0)

        # Crossing probability: start from [0→] (state 2)
        if n_states > 2:
            p_cross = p_to_B[2]
        else:
            p_cross = 0.0

    if verbose:
        print(f"\n=== Global Crossing Probability (Extended Memory, order={state_space.memory_order}) ===")
        print(f"P(A→B) = {p_cross:.6f}")
        print(f"\nProbability of reaching B from each state:")
        for i, state in enumerate(state_space.states[:min(10, len(state_space.states))]):
            print(f"  {state.label}: {p_to_B[i]:.6f}")
        if len(state_space.states) > 10:
            print(f"  ...")

    return p_cross, p_to_B


def mfpt_extended(M: np.ndarray, state_space: ExtendedStateSpace,
                  verbose: bool = True) -> Tuple[float, np.ndarray]:
    """
    Calculate mean first passage time to B using extended memory transition matrix.

    Parameters
    ----------
    M : np.ndarray
        Transition matrix from construct_M_extended.
    state_space : ExtendedStateSpace
        Extended state space.
    verbose : bool
        Print results.

    Returns
    -------
    tuple of (mfpt_AB, mfpt_from_each)
        mfpt_AB : float
            Mean first passage time from A to B
        mfpt_from_each : np.ndarray
            MFPT from each transient state to B
    """
    n_states = state_space.n_states

    # Identify transient states
    absorbing = set(state_space.absorbing_states)
    transient = [i for i in range(n_states) if i not in absorbing]

    # Submatrix Q
    Q = M[np.ix_(transient, transient)]

    # Fundamental matrix
    I = np.eye(len(transient))
    try:
        N = np.linalg.inv(I - Q)
    except np.linalg.LinAlgError:
        N = np.linalg.pinv(I - Q)

    # MFPT from each transient state = sum of row in N
    mfpt_vec = N.sum(axis=1)

    # Map back
    mfpt_from_each = np.zeros(n_states)
    for i, t_idx in enumerate(transient):
        mfpt_from_each[t_idx] = mfpt_vec[i]

    # MFPT from [0→] (state 2)
    if n_states > 2:
        mfpt_AB = mfpt_from_each[2]
    else:
        mfpt_AB = 0.0

    if verbose:
        print(f"\n=== Mean First Passage Time (Extended Memory) ===")
        print(f"MFPT(A→B) = {mfpt_AB:.2f} steps")

    return mfpt_AB, mfpt_from_each


# ============================================================================
# Helper: Binless estimator for extended states
# ============================================================================

def binless_estimator_extended(w_matrices: Dict[int, np.ndarray],
                               state_space: ExtendedStateSpace,
                               n_interfaces: int,
                               verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute crossing probabilities using binless WHAM-like estimator for extended states.

    This adapts the binless estimator for extended memory states. The key difference
    is that extended states group paths by turn behavior, which changes which paths
    contribute to each probability estimate.

    Parameters
    ----------
    w_matrices : dict
        Weight matrices from compute_weight_matrices_extended.
    state_space : ExtendedStateSpace
        Extended state space.
    n_interfaces : int
        Number of interfaces.
    verbose : bool
        Print results.

    Returns
    -------
    tuple of (q, p)
        q : np.ndarray
            Local crossing probabilities (n_interfaces × n_interfaces)
        p : np.ndarray
            Global crossing probabilities
    """
    # Aggregate weights by interface-level transitions
    # Map extended states back to interfaces
    W_interface = np.zeros((n_interfaces, n_interfaces))

    for pe_id, W_ext in w_matrices.items():
        for i in range(W_ext.shape[0]):
            for j in range(W_ext.shape[1]):
                if W_ext[i, j] > 0:
                    # Map extended state indices to interface indices
                    state_i = state_space.states[i]
                    state_j = state_space.states[j]

                    intf_i = _state_to_interface(state_i)
                    intf_j = _state_to_interface(state_j)

                    if 0 <= intf_i < n_interfaces and 0 <= intf_j < n_interfaces:
                        W_interface[intf_i, intf_j] += W_ext[i, j]

    # Compute q from interface-level weights
    q = np.zeros((n_interfaces, n_interfaces))
    for i in range(n_interfaces):
        row_sum = np.sum(W_interface[i, :])
        if row_sum > 0:
            q[i, :] = W_interface[i, :] / row_sum

    # Compute global crossing probability
    # p[0][N-1] = product of forward crossing probabilities
    p_cross = 1.0
    for i in range(n_interfaces - 1):
        p_forward = np.sum(q[i, i+1:])  # Probability to go forward from i
        if p_forward > 0:
            p_cross *= p_forward

    p = np.zeros((n_interfaces, n_interfaces))
    p[0, -1] = p_cross

    if verbose:
        print(f"\n=== Binless Estimator (Extended Memory) ===")
        print(f"Interface-level q matrix:")
        print(np.array2string(q, precision=4, suppress_small=True))
        print(f"\nGlobal crossing probability: {p_cross:.6f}")

    return q, p


def _state_to_interface(state: ExtendedState) -> int:
    """Map an extended state to its primary interface."""
    if state.is_absorbing:
        if state.label == "[0-]" or state.label == "[0←]" or state.label == "[0→]":
            return 0
        else:  # [B]
            return state.base_turn.turn_interface
    return state.base_turn.turn_interface


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Extended Memory State Space - Corrected Definition")
    print("=" * 60)

    # Use 6 interfaces to show both continue and turn options
    n_intf = 6
    interfaces = np.linspace(0, 5, n_intf)
    print(f"\nInterfaces: {interfaces}")

    # Show state spaces at different memory orders
    for memory_order in [0, 1]:
        print(f"\n{'=' * 50}")
        print(f"Memory Order {memory_order}")
        print("=" * 50)

        state_space = ExtendedStateSpace(n_intf, memory_order=memory_order)
        state_space.print_states()

    # Test with synthetic paths
    print("\n" + "=" * 60)
    print("Path Analysis Test - Forward-Facing Turn Examples")
    print("=" * 60)

    state_space = ExtendedStateSpace(n_intf, memory_order=1)
    analyzer = PathTurnAnalyzer(interfaces)

    # =========================================================================
    # Path 1: Forward-facing turn at 2 that CONTINUES to 5
    # Pattern: start high, go down, turn at 2, go up, continue to boundary
    # Crossing pattern for forward turn (⊂): -1, -1, +1, +1
    print("\n--- Path 1: Forward turn at 2 (⊂), continues to 5 ---")
    path1 = np.array([
        [4.5],   # Start in region 4 (between intf 4 and 5)
        [4.2],
        [3.8],   # Cross intf 4 (backward)
        [3.2],   # Region 3
        [2.8],   # Cross intf 3 (backward)
        [2.2],   # Region 2
        [1.8],   # Cross intf 2 (backward) - entering turn region
        [1.2],   # Region 1 - minimum point
        [1.6],   # Cross intf 2 (forward) - starting to exit turn
        [2.3],   # Region 2
        [2.8],   # Cross intf 3 (forward) - COMPLETE forward turn at 2
        [3.3],   # Region 3 - now heading toward B
        [3.8],   # Cross intf 4 (forward)
        [4.3],   # Region 4
        [4.8],   # Cross intf 5 (forward) - reaches B
    ])

    print(f"Path with {len(path1)} points")
    crossings = analyzer.get_interface_crossings(path1)
    print(f"Interface crossings ({len(crossings)}):")
    for idx, f, t in crossings:
        direction = "→" if t > f else "←"
        print(f"  idx {idx:2d}: region {f} {direction} {t}")

    turns = analyzer.extract_complete_turns(path1)
    print(f"Complete turns ({len(turns)}):")
    for turn in turns:
        print(f"  {turn}")

    turn_behaviors = analyzer.extract_turn_with_next_behavior(path1, state_space)
    print(f"Turns with behaviors:")
    for turn, behaviors in turn_behaviors:
        beh_str = " ".join(str(b) for b in behaviors) if behaviors else "none"
        print(f"  {turn} -> {beh_str}")

    states = analyzer.classify_path(path1, state_space, return_all=True)
    print(f"Classified states: {[f'{s.state_id}: {s.label}' for s in states]}")

    # =========================================================================
    # Path 2: Forward-facing turn at 2 that TURNS at 3
    # Same start, but turns again at 3 before reaching B
    print("\n--- Path 2: Forward turn at 2 (⊂), continues to 4, then turns at 4 ---")
    path2 = np.array([
        [4.5],   # Start in region 4
        [4.2],
        [3.8],   # Cross intf 4 (backward)
        [3.2],   # Region 3
        [2.8],   # Cross intf 3 (backward)
        [2.2],   # Region 2
        [1.8],   # Cross intf 2 (backward)
        [1.2],   # Region 1 - minimum
        [1.6],   # Cross intf 2 (forward)
        [2.3],   # Region 2
        [2.8],   # Cross intf 3 (forward) - COMPLETE forward turn at 2 (⊂)
        [3.3],   # Region 3 - heading toward B
        [3.8],   # Cross intf 4 (forward)
        [4.2],   # Region 4 - maximum for this segment
        [3.7],   # Cross intf 4 (backward) - starting turn at 3
        [3.2],   # Region 3
        [2.8],   # Cross intf 3 (backward) - COMPLETE backward turn at 4 (⊃)
        [2.3],   # Region 2
        [1.8],   # Cross intf 2 (backward)
        [1.2],   # Region 1
        [0.7],   # Cross intf 1 (backward)
        [0.2],   # Reaches A
    ])

    print(f"Path with {len(path2)} points")
    crossings = analyzer.get_interface_crossings(path2)
    print(f"Interface crossings ({len(crossings)}):")
    for idx, f, t in crossings:
        direction = "→" if t > f else "←"
        print(f"  idx {idx:2d}: region {f} {direction} {t}")

    turns = analyzer.extract_complete_turns(path2)
    print(f"Complete turns ({len(turns)}):")
    for turn in turns:
        print(f"  {turn}")

    turn_behaviors = analyzer.extract_turn_with_next_behavior(path2, state_space)
    print(f"Turns with behaviors:")
    for turn, behaviors in turn_behaviors:
        beh_str = " ".join(str(b) for b in behaviors) if behaviors else "none"
        print(f"  {turn} -> {beh_str}")

    states = analyzer.classify_path(path2, state_space, return_all=True)
    print(f"Classified states: {[f'{s.state_id}: {s.label}' for s in states]}")

    # =========================================================================
    # Path 3: Forward-facing turn at 2 that immediately TURNS at 3
    # This demonstrates the [2⊂⊃3] state
    print("\n--- Path 3: Forward turn at 2 (⊂), then turns at 3 (doesn't reach 4) ---")
    path3 = np.array([
        [4.5],   # Start in region 4
        [4.2],
        [3.8],   # Cross intf 4 (backward)
        [3.2],   # Region 3
        [2.8],   # Cross intf 3 (backward)
        [2.2],   # Region 2
        [1.8],   # Cross intf 2 (backward)
        [1.2],   # Region 1 - minimum
        [1.6],   # Cross intf 2 (forward)
        [2.3],   # Region 2
        [2.8],   # Cross intf 3 (forward) - COMPLETE forward turn at 2 (⊂)
        [3.3],   # Region 3 - brief excursion, doesn't reach 4
        [2.8],   # Cross intf 3 (backward) - turn at 3 without reaching region 4
        [2.2],   # Region 2
        [1.8],   # Cross intf 2 (backward)
        [1.2],   # Region 1
        [0.7],   # Cross intf 1 (backward)
        [0.2],   # Reaches A
    ])

    print(f"Path with {len(path3)} points")
    crossings = analyzer.get_interface_crossings(path3)
    print(f"Interface crossings ({len(crossings)}):")
    for idx, f, t in crossings:
        direction = "→" if t > f else "←"
        print(f"  idx {idx:2d}: region {f} {direction} {t}")

    turns = analyzer.extract_complete_turns(path3)
    print(f"Complete turns ({len(turns)}):")
    for turn in turns:
        print(f"  {turn}")

    turn_behaviors = analyzer.extract_turn_with_next_behavior(path3, state_space)
    print(f"Turns with behaviors:")
    for turn, behaviors in turn_behaviors:
        beh_str = " ".join(str(b) for b in behaviors) if behaviors else "none"
        print(f"  {turn} -> {beh_str}")

    states = analyzer.classify_path(path3, state_space, return_all=True)
    print(f"Classified states: {[f'{s.state_id}: {s.label}' for s in states]}")

    # =========================================================================
    # Path 4: Path that reaches boundary A, turns at 1, then continues to 3
    print("\n--- Path 4: Path reaching boundary A, turn at 1, continues to 3 ---")
    path4 = np.array([
        [3.5],   # Start in region 3
        [3.2],
        [2.8],   # Cross intf 3 (backward)
        [2.2],   # Region 2
        [1.8],   # Cross intf 2 (backward)
        [1.2],   # Region 1
        [0.7],   # Cross intf 1 (backward)
        [0.2],   # Region 0
        # [-0.3],  # Cross intf 0 (backward) - reaches boundary A!
        # [-0.5],  # In region -1 (state A)
        # [-0.2],  # Turn back
        # [0.3],   # Cross intf 0 (forward)
        [0.7],   # Region 0
        [1.2],   # Cross intf 1 (forward)
        [1.7],   # Region 1
        [2.2],   # Cross intf 2 (forward) - COMPLETE forward turn at 1 (⊂)
        [2.7],   # Region 2 - at exit interface 2, heading forward
        [3.2],   # Cross intf 3 (forward) - continues to region 3
        [3.7],   # Region 3
    ])

    print(f"Path with {len(path4)} points")
    crossings = analyzer.get_interface_crossings(path4)
    print(f"Interface crossings ({len(crossings)}):")
    for idx, f, t in crossings:
        direction = "→" if t > f else "←"
        print(f"  idx {idx:2d}: region {f} {direction} {t}")

    turns = analyzer.extract_complete_turns(path4)
    print(f"Complete turns ({len(turns)}):")
    for turn in turns:
        print(f"  {turn}")

    turn_behaviors = analyzer.extract_turn_with_next_behavior(path4, state_space)
    print(f"Turns with behaviors:")
    for turn, behaviors in turn_behaviors:
        beh_str = " ".join(str(b) for b in behaviors) if behaviors else "none"
        print(f"  {turn} -> {beh_str}")

    states = analyzer.classify_path(path4, state_space, return_all=True)
    print(f"Classified states: {[f'{s.state_id}: {s.label}' for s in states]}")

    # =========================================================================
    # Path 5: Path starting from A, going forward to B
    print("\n--- Path 5: Starts from A, goes forward to B (boundary states) ---")
    path5 = np.array([
        [-0.5],  # Start in region -1 (state A)
        [-0.2],
        [0.3],   # Cross intf 0 (forward) -> region 0
        [0.7],
        [1.2],   # Cross intf 1 (forward) -> region 1
        [1.7],
        [2.2],   # Cross intf 2 (forward) -> region 2
        [2.7],
        [3.2],   # Cross intf 3 (forward) -> region 3
        [3.7],
        [4.2],   # Cross intf 4 (forward) -> region 4
        [4.7],
        [5.2],   # Cross intf 5 (forward) -> region 5 (state B)
    ])

    print(f"Path with {len(path5)} points")
    states = analyzer.classify_path(path5, state_space, return_all=True)
    print(f"Classified states: {[f'{s.state_id}: {s.label}' for s in states]}")

    # =========================================================================
    # Path 6: Path ending at A (backward direction)
    print("\n--- Path 6: Starts from region 3, ends at A (backward) ---")
    path6 = np.array([
        [3.5],   # Start in region 3
        [3.2],
        [2.8],   # Cross intf 3 (backward)
        [2.2],
        [1.8],   # Cross intf 2 (backward)
        [1.2],
        [0.7],   # Cross intf 1 (backward)
        [0.2],
        [-0.3],  # Cross intf 0 (backward) - reaches A!
        [-0.5],  # End in region -1 (state A)
    ])

    print(f"Path with {len(path6)} points")
    states = analyzer.classify_path(path6, state_space, return_all=True)
    print(f"Classified states: {[f'{s.state_id}: {s.label}' for s in states]}")

    # =========================================================================
    # Path 7: Path starting from B, going backward to A
    print("\n--- Path 7: Starts from B, goes backward to A (boundary states) ---")
    path7 = np.array([
        [5.2],   # Start in region 5 (state B)
        [4.8],
        [4.3],   # Cross intf 5 (backward) -> region 4
        [3.8],
        [3.3],   # Cross intf 4 (backward) -> region 3
        [2.8],
        [2.3],   # Cross intf 3 (backward) -> region 2
        [1.8],
        [1.3],   # Cross intf 2 (backward) -> region 1
        [0.8],
        [0.3],   # Cross intf 1 (backward) -> region 0
        [-0.2],  # Cross intf 0 (backward) -> region -1 (state A)
        [-0.5],
    ])

    print(f"Path with {len(path7)} points")
    states = analyzer.classify_path(path7, state_space, return_all=True)
    print(f"Classified states: {[f'{s.state_id}: {s.label}' for s in states]}")

    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary: Forward-facing turn at 2 (⊂) states")
    print("=" * 60)
    print("After [2⊂], path exits at interface 3, heading toward B")
    print("Extended states for turn at 2:")
    for state in state_space.states:
        if state.base_turn.turn_interface == 2 and state.base_turn.facing == TurnFacing.FORWARD:
            print(f"  {state.state_id}: {state.label}")
