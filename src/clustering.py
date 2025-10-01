"""
Clustering module with cluster manipulation and subclustering utilities
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from typing import List, Union
import matplotlib.pyplot as plt


def compute_multiple_resolutions(adata: ad.AnnData,
                                 resolutions: List[float] = [0.25, 0.5, 1.0, 1.5, 2.0],
                                 random_state: int = 42) -> ad.AnnData:
    """
    Compute Leiden clustering at multiple resolutions for comparison
    """
    print(f"Computing Leiden clustering at {len(resolutions)} resolutions...")

    for res in resolutions:
        key = f'leiden_res{res}'
        sc.tl.leiden(adata, resolution=res, key_added=key, random_state=random_state)
        n_clusters = adata.obs[key].nunique()
        print(f"  Resolution {res}: {n_clusters} clusters")

    return adata


def merge_clusters(adata: ad.AnnData,
                  clusters_to_merge: List[Union[str, int]],
                  clustering_key: str = 'leiden',
                  new_cluster_name: Union[str, int] = None) -> ad.AnnData:
    """
    Merge multiple clusters into one
    """
    if new_cluster_name is None:
        new_cluster_name = clusters_to_merge[0]

    # Convert to strings for comparison
    clusters_to_merge = [str(c) for c in clusters_to_merge]
    new_cluster_name = str(new_cluster_name)

    # Create new column with merged clusters
    merged_key = f'{clustering_key}_merged'
    adata.obs[merged_key] = adata.obs[clustering_key].astype(str)

    # Merge clusters
    mask = adata.obs[merged_key].isin(clusters_to_merge)
    adata.obs.loc[mask, merged_key] = new_cluster_name

    # Convert back to categorical
    adata.obs[merged_key] = pd.Categorical(adata.obs[merged_key])

    n_original = adata.obs[clustering_key].nunique()
    n_merged = adata.obs[merged_key].nunique()

    print(f"Merged clusters {clusters_to_merge} into '{new_cluster_name}'")
    print(f"Clusters: {n_original} → {n_merged}")
    return adata


def split_cluster(adata: ad.AnnData,
                 cluster_id: Union[str, int],
                 clustering_key: str = 'leiden',
                 resolution: float = 1.5,
                 random_state: int = 42) -> ad.AnnData:
    """
    Split a cluster by re-clustering at higher resolution
    """
    cluster_id = str(cluster_id)

    # Subset to cluster
    mask = adata.obs[clustering_key].astype(str) == cluster_id
    adata_subset = adata[mask].copy()

    print(f"Splitting cluster {cluster_id} ({adata_subset.n_obs} cells)...")

    # Re-compute neighbors and clustering
    sc.pp.neighbors(adata_subset, random_state=random_state)
    sc.tl.leiden(adata_subset, resolution=resolution, random_state=random_state)

    n_subclusters = adata_subset.obs['leiden'].nunique()

    # Update original adata
    split_key = f'{clustering_key}_split'
    adata.obs[split_key] = adata.obs[clustering_key].astype(str)

    # Assign new cluster IDs
    for i, subcluster in enumerate(adata_subset.obs['leiden'].cat.categories):
        subcluster_mask = adata_subset.obs['leiden'] == subcluster
        new_cluster_name = f"{cluster_id}.{i}"

        # Map back to original adata
        cell_indices = adata_subset.obs_names[subcluster_mask]
        adata.obs.loc[cell_indices, split_key] = new_cluster_name

    # Convert to categorical
    adata.obs[split_key] = pd.Categorical(adata.obs[split_key])

    print(f"Cluster {cluster_id} split into {n_subclusters} subclusters")
    return adata


def subcluster_cells(adata: ad.AnnData,
                    cluster_ids: Union[str, int, List[Union[str, int]]],
                    clustering_key: str = 'leiden',
                    resolution: float = 1.0,
                    n_neighbors: int = 15,
                    n_pcs: int = 30,
                    random_state: int = 42) -> ad.AnnData:
    """
    Perform detailed subclustering on specific clusters

    Returns a new AnnData object with full re-analysis (HVG, PCA, UMAP, clustering)
    of the selected cluster(s).
    """
    if not isinstance(cluster_ids, list):
        cluster_ids = [cluster_ids]

    cluster_ids = [str(c) for c in cluster_ids]

    # Subset to selected clusters
    mask = adata.obs[clustering_key].astype(str).isin(cluster_ids)
    adata_subset = adata[mask].copy()

    print(f"Subclustering {len(cluster_ids)} cluster(s) ({adata_subset.n_obs} cells)...")

    # Full re-analysis pipeline
    # HVGs
    if 'highly_variable' in adata_subset.var.columns:
        hvgs = adata_subset.var_names[adata_subset.var['highly_variable']]
        adata_subset = adata_subset[:, hvgs].copy()

    # PCA
    sc.tl.pca(adata_subset, n_comps=min(n_pcs, adata_subset.n_obs - 1), random_state=random_state)

    # Neighbors
    sc.pp.neighbors(adata_subset, n_neighbors=n_neighbors, n_pcs=n_pcs, random_state=random_state)

    # UMAP
    sc.tl.umap(adata_subset, random_state=random_state)

    # Leiden
    sc.tl.leiden(adata_subset, resolution=resolution, key_added='leiden_sub', random_state=random_state)

    n_subclusters = adata_subset.obs['leiden_sub'].nunique()
    print(f"Identified {n_subclusters} subclusters")

    return adata_subset


def plot_cluster_sizes(adata: ad.AnnData,
                      clustering_key: str = 'leiden',
                      figsize: tuple = (10, 5)) -> plt.Figure:
    """
    Plot cluster size distribution as bar plot and pie chart
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    cluster_counts = adata.obs[clustering_key].value_counts().sort_index()

    # Bar plot
    ax = axes[0]
    cluster_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Cells')
    ax.set_title('Cluster Sizes')
    ax.grid(axis='y', alpha=0.3)

    # Pie chart
    ax = axes[1]
    ax.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=90)
    ax.set_title('Cluster Proportions')

    plt.tight_layout()
    return fig


def compute_cluster_statistics(adata: ad.AnnData,
                               clustering_key: str = 'leiden') -> pd.DataFrame:
    """
    Compute statistics for each cluster (size, QC metrics)
    """
    clusters = adata.obs[clustering_key].cat.categories

    stats_list = []
    for cluster in clusters:
        mask = adata.obs[clustering_key] == cluster
        cluster_data = adata.obs[mask]

        stats = {
            'cluster': cluster,
            'n_cells': mask.sum(),
            'pct_cells': mask.sum() / adata.n_obs * 100,
            'mean_counts': cluster_data['total_counts'].mean(),
            'mean_genes': cluster_data['n_genes_by_counts'].mean(),
            'mean_mito_pct': cluster_data['pct_counts_mt'].mean()
        }
        stats_list.append(stats)

    stats_df = pd.DataFrame(stats_list)
    return stats_df