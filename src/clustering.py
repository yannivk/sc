"""
Clustering module with Leiden algorithm, cluster manipulation, and subclustering
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from typing import Optional, List, Union
import matplotlib.pyplot as plt


def compute_leiden_clustering(adata: ad.AnnData,
                              resolution: float = 1.0,
                              key_added: Optional[str] = None,
                              random_state: int = 42) -> ad.AnnData:
    """
    Compute Leiden clustering

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with neighbors
    resolution : float, default=1.0
        Resolution parameter (higher = more clusters)
    key_added : str, optional
        Key to store clustering results (default: 'leiden' or 'leiden_res{resolution}')
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    adata : AnnData
        Updated AnnData with clustering in .obs
    """
    if key_added is None:
        key_added = f'leiden_res{resolution}'

    print(f"Computing Leiden clustering (resolution={resolution})...")

    sc.tl.leiden(
        adata,
        resolution=resolution,
        key_added=key_added,
        random_state=random_state
    )

    n_clusters = adata.obs[key_added].nunique()
    print(f"Identified {n_clusters} clusters")

    return adata


def compute_multiple_resolutions(adata: ad.AnnData,
                                 resolutions: List[float] = [0.25, 0.5, 1.0, 1.5, 2.0],
                                 random_state: int = 42) -> ad.AnnData:
    """
    Compute clustering at multiple resolutions

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with neighbors
    resolutions : list, default=[0.25, 0.5, 1.0, 1.5, 2.0]
        List of resolution values to test
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    adata : AnnData
        Updated AnnData with multiple clustering results
    """
    print(f"Computing Leiden clustering at {len(resolutions)} resolutions...")

    for res in resolutions:
        key = f'leiden_res{res}'
        sc.tl.leiden(adata, resolution=res, key_added=key, random_state=random_state)
        n_clusters = adata.obs[key].nunique()
        print(f"  Resolution {res}: {n_clusters} clusters")

    return adata


def plot_clustering_resolutions(adata: ad.AnnData,
                                resolutions: List[float] = [0.25, 0.5, 1.0, 1.5],
                                basis: str = 'umap',
                                ncols: int = 2,
                                figsize: Optional[tuple] = None) -> plt.Figure:
    """
    Plot clustering results for multiple resolutions

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with clustering results
    resolutions : list
        List of resolutions to plot
    basis : str, default='umap'
        Embedding to use for plotting
    ncols : int, default=2
        Number of columns in subplot grid
    figsize : tuple, optional
        Figure size (auto-calculated if None)

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    keys = [f'leiden_res{res}' for res in resolutions]

    # Filter keys that exist
    keys = [k for k in keys if k in adata.obs.columns]

    if not keys:
        raise ValueError("No clustering results found for specified resolutions")

    fig = sc.pl.embedding(
        adata,
        basis=basis,
        color=keys,
        ncols=ncols,
        show=False,
        return_fig=True,
        frameon=False,
        legend_loc='on data',
        legend_fontsize='x-small'
    )

    return fig


def merge_clusters(adata: ad.AnnData,
                  clusters_to_merge: List[Union[str, int]],
                  clustering_key: str = 'leiden',
                  new_cluster_name: Optional[Union[str, int]] = None,
                  copy: bool = False) -> Optional[ad.AnnData]:
    """
    Merge multiple clusters into one

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with clustering
    clusters_to_merge : list
        List of cluster IDs to merge
    clustering_key : str, default='leiden'
        Key in adata.obs containing cluster assignments
    new_cluster_name : str or int, optional
        Name for merged cluster (default: use first cluster ID)
    copy : bool, default=False
        Return a copy instead of modifying in place

    Returns
    -------
    adata : AnnData or None
        Updated AnnData (if copy=True)
    """
    if copy:
        adata = adata.copy()

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

    if copy:
        return adata


def split_cluster(adata: ad.AnnData,
                 cluster_id: Union[str, int],
                 clustering_key: str = 'leiden',
                 resolution: float = 1.5,
                 random_state: int = 42,
                 copy: bool = False) -> Optional[ad.AnnData]:
    """
    Split a cluster by re-clustering at higher resolution

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with clustering
    cluster_id : str or int
        Cluster ID to split
    clustering_key : str, default='leiden'
        Key in adata.obs containing cluster assignments
    resolution : float, default=1.5
        Resolution for sub-clustering
    random_state : int, default=42
        Random seed for reproducibility
    copy : bool, default=False
        Return a copy instead of modifying in place

    Returns
    -------
    adata : AnnData or None
        Updated AnnData (if copy=True)
    """
    if copy:
        adata = adata.copy()

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

    if copy:
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

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    cluster_ids : str, int, or list
        Cluster ID(s) to subcluster
    clustering_key : str, default='leiden'
        Key in adata.obs containing cluster assignments
    resolution : float, default=1.0
        Resolution for subclustering
    n_neighbors : int, default=15
        Number of neighbors for graph construction
    n_pcs : int, default=30
        Number of PCs to use
    random_state : int, default=42
        Random seed

    Returns
    -------
    adata_subset : AnnData
        Subsetted and reclustered AnnData object
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
    Plot cluster size distribution

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with clustering
    clustering_key : str, default='leiden'
        Key in adata.obs containing cluster assignments
    figsize : tuple, default=(10, 5)
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
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
    Compute statistics for each cluster

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with clustering
    clustering_key : str, default='leiden'
        Key in adata.obs containing cluster assignments

    Returns
    -------
    stats : DataFrame
        Statistics table with cluster metrics
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