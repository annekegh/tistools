#!/usr/bin/env python3
"""
Stitch PPTIS path fragments into single XTC files.

Each path directory (e.g., 144/) contains:
  - accepted/: folder with XTC/TRR files
  - traj.txt: file describing frame order
    - trajF files: forward trajectories (use frames in order)
    - trajB files: backward trajectories (use frames in REVERSE order, reverse velocities)

This script reads traj.txt to determine the order of trajectory segments,
reverses trajB files, and concatenates everything into a single XTC file
named {path_id}.xtc.

Uses GROMACS gmx trjcat and gmx trjconv for fastest performance.
"""

import argparse
import os
import subprocess
import tempfile
import time
import shutil
from pathlib import Path
from collections import OrderedDict


def robust_copy(src, dst):
    """Copy with one retry on failure (for transient IO errors over SSH/GVFS)."""
    try:
        shutil.copy(src, dst)
    except (FileNotFoundError, IOError, OSError) as e:
        print(f"      Copy failed: {str(e)[:100]}, retrying...")
        time.sleep(1)
        shutil.copy(src, dst)  # Let it raise if it fails again


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, 
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--top", required=True, 
                  help="Topology file (GRO, PDB, TPR) for GROMACS")
    p.add_argument("--base-dir", required=True,
                  help="Base directory containing path folders (e.g., load/)")
    p.add_argument("--path-ids", nargs="+", type=int,
                  help="List of path IDs to process. If not provided, auto-detects all paths with XTC files in accepted/")
    p.add_argument("--accepted-dir", default="accepted",
                  help="Name of subdirectory containing trajectory files")
    p.add_argument("--traj-file", default="traj.txt",
                  help="Name of trajectory description file")
    p.add_argument("--skip-existing", action="store_true",
                  help="Skip paths that already have output files")
    p.add_argument("--skip-overlap", action="store_true",
                  help="Skip first frame of each segment after the first (removes potential overlap)")
    return p.parse_args()


def parse_traj_txt(traj_file):
    """
    Parse traj.txt file to get trajectory segment ordering.
    
    The traj.txt file lists frames in order, but frame indices are just sequential.
    What matters is:
    - The order of unique trajectory files
    - Whether each file is trajB (backward) or trajF (forward)
    
    Returns:
        list of tuples: (filename, is_backward)
        where is_backward = True for trajB files, False for trajF files
    """
    print(f"  Reading trajectory order from: {traj_file.name}")
    seen_files = OrderedDict()  # Preserve order
    
    with open(traj_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 2:
                continue
            
            filename = parts[1]
            
            # Determine if backward or forward based on filename
            is_backward = 'trajB' in filename
            
            # Store unique files in order
            if filename not in seen_files:
                seen_files[filename] = is_backward
    
    result = [(fname, is_back) for fname, is_back in seen_files.items()]
    print(f"  Found {len(result)} unique trajectory segments")
    return result


def reverse_trajectory(input_xtc, output_xtc, topology):
    """
    Reverse a trajectory file using GROMACS by extracting frames in chunks.
    Uses gmx trjconv with -b/-e flags to extract individual frames.
    
    Args:
        input_xtc: Input trajectory file path
        output_xtc: Output reversed trajectory file path
        topology: Topology file for structure information
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # First get list of all time values
        print(f"    Analyzing trajectory to extract time points...")
        cmd_dump = ['/usr/local/gromacs/bin/gmx', 'dump', '-f', str(input_xtc)]
        result = subprocess.run(cmd_dump, capture_output=True, text=True)
        
        times = []
        import re
        for line in result.stdout.split('\n'):
            # Look for lines like: natoms=33749  step=0  time=0.0000000e+00
            match = re.search(r'time=([0-9.eE+-]+)', line)
            if match:
                times.append(float(match.group(1)))
        
        if not times:
            print(f"    ERROR: Could not extract time values")
            return False
        
        print(f"    Reversing {len(times)} frames (times: {times[0]:.1f} to {times[-1]:.1f} ps)...")
        
        # Extract each frame individually in reverse order
        tmpdir = output_xtc.parent
        frame_files = []
        
        for i, time_val in enumerate(reversed(times)):
            if i == 0 or (i + 1) % max(1, len(times) // 5) == 0 or i == len(times) - 1:
                print(f"    Extracting frame {i+1}/{len(times)} (time={time_val:.1f} ps)...")
            frame_file = tmpdir / f"rev_frame_{i:06d}.xtc"
            # Use -b and -e to extract single time point
            cmd = [
                '/usr/local/gromacs/bin/gmx', 'trjconv',
                '-f', str(input_xtc),
                '-s', str(topology),
                '-o', str(frame_file),
                '-b', str(time_val),
                '-e', str(time_val),
                '-pbc', 'none'
            ]
            result = subprocess.run(cmd, input=b'0\n', capture_output=True)
            if result.returncode != 0:
                print(f"    ERROR extracting time {time_val}: {result.stderr.decode()[:100]}")
                for f in frame_files:
                    if f.exists():
                        f.unlink()
                return False
            frame_files.append(frame_file)
        
        # Concatenate all frames
        print(f"    Concatenating {len(frame_files)} reversed frames...")
        cmd_cat = [
            '/usr/local/gromacs/bin/gmx', 'trjcat',
            '-f', *[str(f) for f in frame_files],
            '-o', str(output_xtc),
            '-cat'
        ]
        result = subprocess.run(cmd_cat, capture_output=True)
        
        # Clean up frame files
        for f in frame_files:
            if f.exists():
                f.unlink()
        
        if result.returncode != 0:
            print(f"    ERROR concatenating: {result.stderr.decode()[:100]}")
            return False
        
        print(f"    Successfully reversed to: {output_xtc.name}")
        return True
        
    except Exception as e:
        print(f"    ERROR during reversal: {str(e)[:200]}")
        return False


def get_trajectory_times(traj_file):
    """
    Extract all time values from a trajectory file.
    
    Args:
        traj_file: Path to trajectory file
    
    Returns:
        list of floats: Time values in ps, or empty list on error
    """
    try:
        cmd_dump = ['/usr/local/gromacs/bin/gmx', 'dump', '-f', str(traj_file)]
        result = subprocess.run(cmd_dump, capture_output=True, text=True)
        
        times = []
        import re
        for line in result.stdout.split('\n'):
            match = re.search(r'time=([0-9.eE+-]+)', line)
            if match:
                times.append(float(match.group(1)))
        
        return times
    except Exception:
        return []


def check_frame_overlap(traj_file1, traj_file2, topology, n_atoms_to_check=10):
    """
    Check if the last frame of traj_file1 matches the first frame of traj_file2
    by comparing coordinates of a few atoms.
    
    Args:
        traj_file1: First trajectory file
        traj_file2: Second trajectory file  
        topology: Topology file
        n_atoms_to_check: Number of atoms to compare (default 10)
    
    Returns:
        bool: True if frames overlap (coordinates match within tolerance)
    """
    try:
        import numpy as np
        
        # Extract last frame of first trajectory
        times1 = get_trajectory_times(traj_file1)
        if not times1:
            return False
        
        tmpdir = Path(traj_file1).parent
        last_frame = tmpdir / "check_last.gro"
        
        cmd = [
            '/usr/local/gromacs/bin/gmx', 'trjconv',
            '-f', str(traj_file1),
            '-s', str(topology),
            '-o', str(last_frame),
            '-dump', str(times1[-1]),
            '-pbc', 'none'
        ]
        result = subprocess.run(cmd, input=b'0\n', capture_output=True)
        if result.returncode != 0:
            return False
        
        # Extract first frame of second trajectory
        times2 = get_trajectory_times(traj_file2)
        if not times2:
            if last_frame.exists():
                last_frame.unlink()
            return False
        
        first_frame = tmpdir / "check_first.gro"
        
        cmd = [
            '/usr/local/gromacs/bin/gmx', 'trjconv',
            '-f', str(traj_file2),
            '-s', str(topology),
            '-o', str(first_frame),
            '-dump', str(times2[0]),
            '-pbc', 'none'
        ]
        result = subprocess.run(cmd, input=b'0\n', capture_output=True)
        if result.returncode != 0:
            if last_frame.exists():
                last_frame.unlink()
            return False
        
        # Read coordinates from GRO files (simple parsing)
        def read_gro_coords(gro_file, n_atoms):
            """Read first n_atoms coordinates from GRO file"""
            coords = []
            with open(gro_file, 'r') as f:
                lines = f.readlines()
                # Skip title and atom count lines
                for i in range(2, min(2 + n_atoms, len(lines) - 1)):
                    line = lines[i]
                    # GRO format: positions at columns 20-28, 28-36, 36-44 (in nm)
                    try:
                        x = float(line[20:28])
                        y = float(line[28:36])
                        z = float(line[36:44])
                        coords.append([x, y, z])
                    except:
                        break
            return np.array(coords)
        
        coords1 = read_gro_coords(last_frame, n_atoms_to_check)
        coords2 = read_gro_coords(first_frame, n_atoms_to_check)
        
        # Clean up
        if last_frame.exists():
            last_frame.unlink()
        if first_frame.exists():
            first_frame.unlink()
        
        # Compare coordinates
        if len(coords1) != len(coords2) or len(coords1) == 0:
            return False
        
        # Calculate RMSD between the atom positions
        diff = coords1 - coords2
        rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
        
        # Frames are considered identical if RMSD < 0.001 nm (1 pm)
        return rmsd < 0.001
        
    except Exception as e:
        print(f"      Warning: Could not check overlap: {str(e)[:100]}")
        return False


def skip_first_frame(traj_file, output_file, topology):
    """
    Skip the first frame from a trajectory file (to remove potential overlap).
    
    Args:
        traj_file: Input trajectory file
        output_file: Output trajectory file
        topology: Topology file
    
    Returns:
        bool: True if successful, False if failed
        Note: If only 1 frame exists, returns True but does NOT create output file
              (caller should skip this segment entirely)
    """
    try:
        times = get_trajectory_times(traj_file)
        if len(times) == 0:
            print(f"      ERROR: Could not read trajectory")
            return False
        
        if len(times) == 1:
            print(f"      Segment has only 1 frame - skipping entire segment (overlap frame)")
            # Don't create output file - signal to skip this segment
            return True
        
        # Extract all frames except the first
        print(f"      Skipping first frame (keeping {len(times)-1} frames)...")
        cmd = [
            '/usr/local/gromacs/bin/gmx', 'trjconv',
            '-f', str(traj_file),
            '-s', str(topology),
            '-o', str(output_file),
            '-b', str(times[1]),  # Start from second frame
            '-pbc', 'none'
        ]
        result = subprocess.run(cmd, input=b'0\n', capture_output=True)
        
        return result.returncode == 0
    except Exception as e:
        print(f"      ERROR: {str(e)[:100]}")
        return False


def extract_and_stitch(topology, accepted_dir, traj_segments, output_file, skip_overlap=False):
    """
    Process trajectory segments and stitch them into a single XTC file.
    
    Args:
        topology: Topology file for GROMACS
        accepted_dir: Directory containing trajectory files
        traj_segments: List of (filename, is_backward) tuples
        output_file: Output XTC file path
        skip_overlap: If True, skip first frame of each segment after the first
    
    Uses GROMACS tools for speed.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        processed_files = []
        
        # Copy topology to temp location if it's a GVFS path (GROMACS can't access it)
        local_topology = topology
        if str(topology).startswith('/run/user') and 'gvfs' in str(topology):
            print(f"  Copying topology to temporary location...")
            local_topology = tmpdir / Path(topology).name
            robust_copy(topology, local_topology)
            print(f"  Using local topology: {local_topology.name}")
        
        print(f"  Processing {len(traj_segments)} trajectory segment(s)...")
        
        for i, (filename, is_backward) in enumerate(traj_segments):
            # Find the actual file (handle extension variations)
            base_name = filename.rsplit('.', 1)[0]  # Remove extension from traj.txt
            traj_path = None
            
            print(f"    [{i+1}/{len(traj_segments)}] Looking for: {base_name}.*")
            
            for ext in ['.xtc', '.trr', '.XTC', '.TRR']:
                candidate = Path(accepted_dir) / (base_name + ext)
                if candidate.exists():
                    traj_path = candidate
                    print(f"        Found: {traj_path.name}")
                    break
            
            if traj_path is None:
                print(f"    WARNING: File not found for {filename}, skipping")
                continue
            
            print(f"    [{i+1}/{len(traj_segments)}] Processing: {traj_path.name} ({'BACKWARD' if is_backward else 'FORWARD'})")
            
            if is_backward:
                # Reverse the trajectory
                print(f"        Reversing trajectory (backward direction)...")
                output_seg = tmpdir / f"seg_{i:04d}_reversed.xtc"
                if not reverse_trajectory(traj_path, output_seg, local_topology):
                    print(f"    WARNING: Failed to reverse {filename}")
                    continue
                processed_files.append(output_seg)
            else:
                # Forward trajectory - use as is, just convert to XTC if needed
                if traj_path.suffix.lower() == '.xtc':
                    # Already XTC, can use directly or copy
                    print(f"        Copying forward trajectory...")
                    output_seg = tmpdir / f"seg_{i:04d}.xtc"
                    robust_copy(traj_path, output_seg)
                    processed_files.append(output_seg)
                else:
                    # Convert TRR to XTC
                    print(f"        Converting TRR to XTC...")
                    output_seg = tmpdir / f"seg_{i:04d}.xtc"
                    cmd = [
                        '/usr/local/gromacs/bin/gmx', 'trjconv',
                        '-f', str(traj_path),
                        '-s', str(topology),
                        '-o', str(output_seg)
                    ]
                    result = subprocess.run(cmd, input=b'0\n', capture_output=True)
                    if result.returncode != 0:
                        print(f"    WARNING: Conversion failed for {filename}")
                        continue
                    processed_files.append(output_seg)
        
        if not processed_files:
            print(f"  ERROR: No trajectory segments processed!")
            return False
        
        # Check for overlapping frames between segments
        if skip_overlap and len(processed_files) > 1:
            print(f"  Checking for overlapping frames between segments...")
            overlaps_found = []
            
            for i in range(len(processed_files) - 1):
                print(f"    Checking overlap between segments {i+1} and {i+2}...")
                if check_frame_overlap(processed_files[i], processed_files[i+1], local_topology):
                    print(f"      ✓ Overlap detected - will remove first frame from segment {i+2}")
                    overlaps_found.append(i+1)  # Index of segment to fix
                else:
                    print(f"      No overlap detected")
            
            if overlaps_found:
                print(f"  Removing overlapping frames from {len(overlaps_found)} segment(s)...")
                segments_to_remove = []
                for idx in overlaps_found:
                    print(f"    Processing segment {idx+1}...")
                    fixed_file = tmpdir / f"seg_{idx:04d}_no_overlap.xtc"
                    if skip_first_frame(processed_files[idx], fixed_file, local_topology):
                        # Check if output file was created (won't exist if only 1 frame)
                        if fixed_file.exists():
                            processed_files[idx] = fixed_file
                        else:
                            print(f"      Segment {idx+1} completely removed (was only overlap frame)")
                            segments_to_remove.append(idx)
                    else:
                        print(f"      WARNING: Could not skip first frame, keeping original")
                
                # Remove segments that were entirely overlap frames
                for idx in reversed(segments_to_remove):
                    processed_files.pop(idx)
                
                if segments_to_remove:
                    print(f"  Removed {len(segments_to_remove)} segment(s) that were only overlap frames")
            else:
                print(f"  ✓ No overlapping frames detected")
        
        # Concatenate all segments
        print(f"  Concatenating {len(processed_files)} segment(s) into final trajectory...")
        print(f"  Output file: {output_file}")
        
        if len(processed_files) == 1:
            # Just copy the single file
            print(f"  Single segment - copying directly...")
            robust_copy(processed_files[0], output_file)
        else:
            print(f"  Running gmx trjcat to merge {len(processed_files)} segments...")
            cmd = [
                '/usr/local/gromacs/bin/gmx', 'trjcat',
                '-f', *[str(f) for f in processed_files],
                '-o', str(output_file),
                '-cat'
            ]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                print(f"  ERROR: Final concatenation failed")
                stderr = result.stderr.decode() if result.stderr else "No error message"
                print(f"  stderr: {stderr}")
                return False
        
        print(f"  Successfully created: {output_file}")
        return True


def main():
    args = parse_args()
    
    base_dir = Path(args.base_dir)
    
    print(f"\n{'='*60}")
    print(f"PPTIS Path Stitching Script")
    print(f"{'='*60}")
    print(f"Base directory: {base_dir}")
    print(f"Topology file:  {args.top}")
    
    if not base_dir.exists():
        print(f"ERROR: Base directory does not exist: {base_dir}")
        return
    
    # Get list of path IDs
    if args.path_ids:
        path_ids = args.path_ids
        print(f"Processing specified {len(path_ids)} path(s): {path_ids}")
    else:
        # Auto-detect all numeric subdirectories that have XTC files in accepted/
        print(f"\nScanning for paths with trajectory files...")
        path_ids = []
        for item in base_dir.iterdir():
            if item.is_dir() and item.name.isdigit():
                accepted_dir = item / args.accepted_dir
                if accepted_dir.exists():
                    # Check if there are any XTC/TRR files
                    has_traj = any(accepted_dir.glob('*.xtc'))
                    if has_traj:
                        path_ids.append(int(item.name))
                    if len(path_ids) % 1000 == 0 and len(path_ids) > 0:
                        print(f"  Detected {len(path_ids)} paths so far...")
        path_ids.sort()
        print(f"Auto-detected {len(path_ids)} paths with trajectory files")
    
    print(f"\n{'='*60}")
    print(f"Processing {len(path_ids)} paths...")
    print(f"{'='*60}\n")
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for path_id in path_ids:
        path_dir = base_dir / str(path_id)
        accepted_dir = path_dir / args.accepted_dir
        traj_txt = path_dir / args.traj_file
        output_file = path_dir / f"{path_id}.xtc"
        
        print(f"\n{'='*60}")
        print(f"Path {path_id}: {path_dir}")
        print(f"{'='*60}")
        
        # Check if output exists and skip if requested
        if args.skip_existing and output_file.exists():
            print(f"  ✓ Skipping (output already exists: {output_file.name})")
            skip_count += 1
            continue
        
        # Check if required files exist
        if not path_dir.exists():
            print(f"  ✗ WARNING: Path directory does not exist")
            fail_count += 1
            continue
        
        if not accepted_dir.exists():
            print(f"  ✗ WARNING: Accepted directory does not exist: {accepted_dir}")
            fail_count += 1
            continue
        
        if not traj_txt.exists():
            print(f"  ✗ WARNING: traj.txt not found")
            fail_count += 1
            continue
        
        # Parse traj.txt to get trajectory segments
        traj_segments = parse_traj_txt(traj_txt)
        if not traj_segments:
            print(f"  ✗ WARNING: No trajectory segments found in traj.txt")
            fail_count += 1
            continue
        
        # Extract and stitch
        if extract_and_stitch(args.top, accepted_dir, traj_segments, output_file, args.skip_overlap):
            print(f"  ✓ SUCCESS: Created {output_file.name}")
            success_count += 1
        else:
            print(f"  ✗ FAILED: Could not create output file")
            fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Successful: {success_count}")
    print(f"  Skipped:    {skip_count}")
    print(f"  Failed:     {fail_count}")
    print(f"  Total:      {len(path_ids)}")
    
    if success_count > 0:
        print(f"\nOutput files saved in individual path directories as: <path_id>.xtc")


if __name__ == "__main__":
    main()
