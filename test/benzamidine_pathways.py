#!/usr/bin/env python3
"""
Cluster PPTIS (or TIS) pathway fragments of ligand unbinding from a protein.

This script is designed for PPTIS workflows where each XTC/TRR trajectory file
represents a complete path fragment (not a long MD trajectory with multiple events).

Approach
--------
1) Load topology and trajectory fragments (XTC, TRR, etc.) - each file is one pathway sample.
2) Align each fragment to the protein backbone.
3) Define binding-site center from the topology reference frame (reused for all fragments).
4) Track ligand COM for each fragment and compute vectors from site center.
5) Compute a fragment-level directional descriptor (mean unit vector) over the entire fragment.
6) Cluster fragments using DBSCAN (cosine distance) to identify dominant pathway directions.
7) Export per-fragment COM paths and unit vectors for visualization.

Notes
-----
- Each input trajectory file should be one PPTIS path sample.
- Adjust --ligand selection to match your ligand residue name.
- TRR and XTC formats are both supported automatically via MDAnalysis.
- Visualization: use the output CSVs in any 3D plotting tool or PyMOL.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import align
except ImportError as e:
    raise SystemExit("MDAnalysis is required. Install with `pip install MDAnalysis`." )

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plots will be skipped.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--top", required=True, help="Topology file (PDB, GRO, PSF, PRMTOP, etc.)")
    p.add_argument("--trajs", nargs="+", required=True, help="One or more XTC/TRR trajectory files (each file = one PPTIS path)")
    p.add_argument("--ligand", default="resname BEN", help="MDAnalysis selection for the ligand (default: 'resname BEN')")
    p.add_argument("--protein", default="protein and name N CA C O and not altloc B", help="Protein selection for alignment")
    p.add_argument("--site-cut", type=float, default=5.0, help="Å cutoff to define binding site around ligand in ref frame")
    p.add_argument("--stride", type=int, default=1, help="Stride frames when reading trajectories")
    p.add_argument("--dbscan-eps", type=float, default=0.25, help="DBSCAN epsilon for cosine distance clustering")
    p.add_argument("--dbscan-min-samples", type=int, default=2, help="DBSCAN min_samples")
    p.add_argument("--outdir", default="pptis_pathways", help="Output directory")
    p.add_argument("--no-plot", action="store_true", help="Disable plot generation")
    return p.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def unit_vectors(v):
    """Return unit vectors for an array of shape (n, 3). Zeros stay zeros."""
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    with np.errstate(invalid='ignore'):
        u = np.divide(v, norms, out=np.zeros_like(v), where=norms>0)
    return u


def compute_site_center(u, lig_sel: str, site_cut: float) -> np.ndarray:
    """Compute binding-site center from frame 0: centroid of protein heavy atoms within site_cut Å of ligand heavy atoms."""
    protein = u.select_atoms("protein and not name H*")
    ligand = u.select_atoms(lig_sel + " and not name H*")
    if ligand.n_atoms == 0:
        raise ValueError(f"Ligand selection found no atoms: {lig_sel}")
    u.trajectory[0]
    
    # Find protein atoms within cutoff of any ligand atom using distance calculation
    from MDAnalysis.lib.distances import distance_array
    dist_matrix = distance_array(ligand.positions, protein.positions, box=u.dimensions)
    # Get protein atoms within cutoff of ANY ligand atom
    close_prot_indices = np.where(np.any(dist_matrix <= site_cut, axis=0))[0]
    
    if len(close_prot_indices) == 0:
        raise ValueError("No protein atoms found within site-cut of ligand in frame 0; consider increasing --site-cut.")
    
    prot_near = protein[close_prot_indices]
    center = prot_near.positions.mean(axis=0)
    return center


def compute_pathway_descriptor(coms, site_center):
    """Compute mean directional unit vector for an entire pathway fragment.
    
    Returns:
        mean_dir: normalized mean direction vector (3,)
        path_length: total path length in Angstroms
        mean_dist: mean distance from site center
        max_dist: maximum distance from site center
    """
    vecs = coms - site_center
    dists = np.linalg.norm(vecs, axis=1)
    uvecs = unit_vectors(vecs)
    
    # Mean unit vector as pathway descriptor
    mean_dir = uvecs.mean(axis=0)
    mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-12)
    
    # Path statistics
    path_length = np.sum(np.linalg.norm(np.diff(coms, axis=0), axis=1))
    mean_dist = dists.mean()
    max_dist = dists.max()
    
    return mean_dir, path_length, mean_dist, max_dist


def plot_pathways_3d(summary_df, paths_dir, site_center, outdir):
    """Create 3D visualization of clustered pathways."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    clusters = sorted(summary_df['cluster'].unique())
    n_clusters = len([c for c in clusters if c != -1])
    
    # Create colormap
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_clusters + 1)))
    cluster_colors = {}
    for i, cl in enumerate(clusters):
        if cl == -1:
            cluster_colors[cl] = 'gray'
        else:
            cluster_colors[cl] = colors[i % len(colors)]
    
    # Plot all pathways in 3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for _, row in summary_df.iterrows():
        path_file = paths_dir / f"path_{row['path_id']:04d}.csv"
        df_path = pd.read_csv(path_file)
        
        color = cluster_colors[row['cluster']]
        label = f"Cluster {row['cluster']}" if row['cluster'] != -1 else "Noise"
        
        # Plot pathway with transparency
        ax.plot(df_path['x'], df_path['y'], df_path['z'], 
               color=color, alpha=0.6, linewidth=1.5)
        
        # Mark start and end points
        ax.scatter(df_path['x'].iloc[0], df_path['y'].iloc[0], df_path['z'].iloc[0],
                  color=color, marker='o', s=50, edgecolors='black', linewidths=0.5)
        ax.scatter(df_path['x'].iloc[-1], df_path['y'].iloc[-1], df_path['z'].iloc[-1],
                  color=color, marker='s', s=50, edgecolors='black', linewidths=0.5)
    
    # Mark binding site center
    ax.scatter(*site_center, color='red', marker='*', s=300, 
              edgecolors='black', linewidths=2, label='Binding Site')
    
    ax.set_xlabel('X (Å)', fontsize=12)
    ax.set_ylabel('Y (Å)', fontsize=12)
    ax.set_zlabel('Z (Å)', fontsize=12)
    ax.set_title('PPTIS Pathway Clustering (3D)', fontsize=14, fontweight='bold')
    
    # Create legend with unique entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(outdir / 'pathways_3d.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved 3D pathway plot: {outdir / 'pathways_3d.png'}")


def plot_cluster_summary(summary_df, outdir):
    """Create summary plots for pathway statistics."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    clusters = sorted(summary_df['cluster'].unique())
    cluster_labels = ['Noise' if c == -1 else f'C{c}' for c in clusters]
    
    # 1. Cluster size bar plot
    ax = axes[0, 0]
    cluster_counts = summary_df['cluster'].value_counts().sort_index()
    colors = ['gray' if c == -1 else f'C{i}' for i, c in enumerate(cluster_counts.index)]
    ax.bar(range(len(cluster_counts)), cluster_counts.values, color=colors, edgecolor='black')
    ax.set_xticks(range(len(cluster_counts)))
    ax.set_xticklabels([f'Noise' if c == -1 else f'C{c}' for c in cluster_counts.index])
    ax.set_ylabel('Number of Pathways', fontsize=11)
    ax.set_title('Pathways per Cluster', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Path length distribution
    ax = axes[0, 1]
    for cl in clusters:
        data = summary_df[summary_df['cluster'] == cl]['path_length_A']
        label = 'Noise' if cl == -1 else f'Cluster {cl}'
        color = 'gray' if cl == -1 else f'C{cl % 10}'
        ax.hist(data, alpha=0.6, label=label, bins=15, color=color, edgecolor='black')
    ax.set_xlabel('Path Length (Å)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Path Length Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Mean distance from binding site
    ax = axes[1, 0]
    for cl in clusters:
        data = summary_df[summary_df['cluster'] == cl]['mean_dist_A']
        label = 'Noise' if cl == -1 else f'Cluster {cl}'
        color = 'gray' if cl == -1 else f'C{cl % 10}'
        ax.hist(data, alpha=0.6, label=label, bins=15, color=color, edgecolor='black')
    ax.set_xlabel('Mean Distance from Site (Å)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Mean Distance Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Duration vs path length scatter
    ax = axes[1, 1]
    for cl in clusters:
        data = summary_df[summary_df['cluster'] == cl]
        label = 'Noise' if cl == -1 else f'Cluster {cl}'
        color = 'gray' if cl == -1 else f'C{cl % 10}'
        ax.scatter(data['duration_ps'], data['path_length_A'], 
                  alpha=0.7, label=label, s=50, color=color, edgecolors='black', linewidths=0.5)
    ax.set_xlabel('Duration (ps)', fontsize=11)
    ax.set_ylabel('Path Length (Å)', fontsize=11)
    ax.set_title('Duration vs Path Length', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outdir / 'pathway_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved statistics plot: {outdir / 'pathway_statistics.png'}")


def plot_directional_vectors(summary_df, outdir):
    """Create 3D quiver plot showing mean directional vectors for each cluster."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    clusters = sorted(summary_df['cluster'].unique())
    
    # Plot unit sphere as reference
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='lightgray')
    
    # Plot mean direction vectors for each cluster
    for cl in clusters:
        data = summary_df[summary_df['cluster'] == cl]
        color = 'gray' if cl == -1 else f'C{cl % 10}'
        label = 'Noise' if cl == -1 else f'Cluster {cl}'
        
        # Plot individual pathway vectors
        for _, row in data.iterrows():
            ax.quiver(0, 0, 0, 
                     row['mean_dir_x'], row['mean_dir_y'], row['mean_dir_z'],
                     color=color, alpha=0.4, arrow_length_ratio=0.15, linewidth=1.5)
        
        # Plot cluster mean direction
        mean_x = data['mean_dir_x'].mean()
        mean_y = data['mean_dir_y'].mean()
        mean_z = data['mean_dir_z'].mean()
        norm = np.sqrt(mean_x**2 + mean_y**2 + mean_z**2)
        if norm > 0:
            mean_x, mean_y, mean_z = mean_x/norm, mean_y/norm, mean_z/norm
            ax.quiver(0, 0, 0, mean_x, mean_y, mean_z,
                     color=color, alpha=1.0, arrow_length_ratio=0.2, 
                     linewidth=3, label=f'{label} (mean)')
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title('Pathway Directional Vectors', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    
    # Set equal aspect ratio
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    
    plt.tight_layout()
    plt.savefig(outdir / 'directional_vectors.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved directional vectors plot: {outdir / 'directional_vectors.png'}")


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    ensure_dir(outdir)
    paths_dir = outdir / "paths"
    ensure_dir(paths_dir)

    # Load topology once to get binding site center
    print("Loading topology to determine binding site center...")
    u_ref = mda.Universe(args.top)
    ligand_ref = u_ref.select_atoms(args.ligand)
    if ligand_ref.n_atoms == 0:
        raise SystemExit(f"No ligand atoms for selection: {args.ligand}")
    
    # Compute site center from topology
    site_center = compute_site_center(u_ref, args.ligand, args.site_cut)
    print(f"Binding site center: ({site_center[0]:.2f}, {site_center[1]:.2f}, {site_center[2]:.2f}) Å\n")

    all_pathways = []  # rows for summary
    path_id = 0

    for traj_file in args.trajs:
        print(f"Processing pathway {path_id}: {Path(traj_file).name}")
        u = mda.Universe(args.top, traj_file)
        protBB = u.select_atoms(args.protein)
        ligand = u.select_atoms(args.ligand)
        if ligand.n_atoms == 0:
            raise SystemExit(f"No ligand atoms for selection: {args.ligand}")

        # Align entire trajectory to protein backbone (first frame as reference)
        aligner = align.AlignTraj(u, u, select=args.protein, in_memory=False)
        aligner.run(step=args.stride)

        # Extract ligand COM for entire pathway
        frames = []
        coms = []
        times = []
        dists = []

        for ts in u.trajectory[::args.stride]:
            frames.append(ts.frame)
            times.append(ts.time)
            com = ligand.center_of_mass()
            coms.append(com)
            dists.append(np.linalg.norm(com - site_center))

        frames = np.array(frames)
        times = np.array(times)
        coms = np.array(coms)  # shape (n,3)
        dists = np.array(dists)

        print(f"  Frames: {len(frames)}, Duration: {times[-1]-times[0]:.1f} ps")

        # Compute pathway descriptor for entire fragment
        mean_dir, path_length, mean_dist, max_dist = compute_pathway_descriptor(coms, site_center)

        # Save full pathway trace
        df_path = pd.DataFrame({
            'frame': frames,
            'time_ps': times,
            'x': coms[:,0], 'y': coms[:,1], 'z': coms[:,2],
            'dist_A': dists
        })
        path_csv = paths_dir / f"path_{path_id:04d}.csv"
        df_path.to_csv(path_csv, index=False)

        # Save unit vectors
        vecs = coms - site_center
        uvecs = unit_vectors(vecs)
        df_uvec = pd.DataFrame({'ux': uvecs[:,0], 'uy': uvecs[:,1], 'uz': uvecs[:,2]})
        df_uvec.to_csv(paths_dir / f"path_{path_id:04d}_unitvec.csv", index=False)

        all_pathways.append({
            'path_id': path_id,
            'trajectory_file': str(traj_file),
            'n_frames': len(frames),
            'duration_ps': times[-1] - times[0],
            'path_length_A': path_length,
            'mean_dist_A': mean_dist,
            'max_dist_A': max_dist,
            'mean_dir_x': mean_dir[0],
            'mean_dir_y': mean_dir[1],
            'mean_dir_z': mean_dir[2]
        })
        path_id += 1
        print(f"  Path length: {path_length:.2f} Å, Mean dist: {mean_dist:.2f} Å, Max dist: {max_dist:.2f} Å\n")

    if len(all_pathways) == 0:
        print("No pathways processed.")
        return

    # Cluster pathways by mean direction using cosine distance
    print(f"Clustering {len(all_pathways)} pathways...")
    summary_df = pd.DataFrame(all_pathways)
    dirs = summary_df[["mean_dir_x","mean_dir_y","mean_dir_z"]].to_numpy()
    # Normalize just in case
    dirs = normalize(dirs)
    # DBSCAN with cosine metric groups similar pathway directions
    clustering = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples, metric='cosine')
    labels = clustering.fit_predict(dirs)
    summary_df['cluster'] = labels

    # Sort by cluster then by path_id
    summary_df.sort_values(["cluster","path_id"], inplace=True)
    summary_csv = outdir / "pathway_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # Write text report
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    
    with open(outdir / "REPORT.txt", "w") as f:
        f.write("PPTIS Pathway Clustering Report\n")
        f.write("================================\n\n")
        f.write(f"Total pathways: {len(summary_df)}\n")
        f.write(f"DBSCAN clusters: {n_clusters}\n")
        f.write(f"Noise pathways (cluster -1): {n_noise}\n")
        f.write(f"DBSCAN parameters: eps={args.dbscan_eps}, min_samples={args.dbscan_min_samples}\n\n")
        
        for cl in sorted(summary_df['cluster'].unique()):
            dfc = summary_df[summary_df['cluster']==cl]
            cluster_name = "NOISE" if cl == -1 else f"Cluster {cl}"
            f.write(f"{cluster_name}: {len(dfc)} pathway(s)\n")
            f.write("-" * 80 + "\n")
            for _, row in dfc.iterrows():
                f.write(f"  Path {row.path_id:4d} | {Path(row.trajectory_file).name:40s} | "
                       f"{row.n_frames:5d} frames | {row.duration_ps:8.1f} ps | "
                       f"length {row.path_length_A:7.1f} Å\n")
                f.write(f"           mean_dir: ({row.mean_dir_x:6.3f}, {row.mean_dir_y:6.3f}, {row.mean_dir_z:6.3f}) | "
                       f"mean_dist: {row.mean_dist_A:6.2f} Å | max_dist: {row.max_dist_A:6.2f} Å\n")
            f.write("\n")

    print(f"\nWrote summary: {summary_csv}")
    print(f"Per-pathway traces in: {paths_dir}")
    print(f"Report: {outdir / 'REPORT.txt'}")
    print("\nClustering summary:")
    print(f"  {n_clusters} cluster(s) found")
    print(f"  {n_noise} noise pathway(s)")
    
    # Generate plots
    if not args.no_plot and MATPLOTLIB_AVAILABLE:
        print("\nGenerating visualizations...")
        plot_pathways_3d(summary_df, paths_dir, site_center, outdir)
        plot_cluster_summary(summary_df, outdir)
        plot_directional_vectors(summary_df, outdir)
        print("All plots saved!")
    elif args.no_plot:
        print("\nPlot generation disabled (--no-plot)")
    elif not MATPLOTLIB_AVAILABLE:
        print("\nMatplotlib not available - skipping plots")
    
    print("\nDone.")


if __name__ == "__main__":
    main()
