"""
Annotation module for marker gene identification and manual cell type annotation
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from typing import Optional, List, Dict, Union
import matplotlib.pyplot as plt


def find_marker_genes(adata: ad.AnnData,
                     clustering_key: str = 'leiden',
                     method: str = 'wilcoxon',
                     n_genes: int = 100,
                     use_raw: bool = False) -> ad.AnnData:
    """
    Find marker genes for each cluster

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with clustering
    clustering_key : str, default='leiden'
        Key in adata.obs containing cluster assignments
    method : str, default='wilcoxon'
        Statistical test to use ('wilcoxon', 't-test', 'logreg')
    n_genes : int, default=100
        Number of genes to test
    use_raw : bool, default=False
        Use raw data for calculation

    Returns
    -------
    adata : AnnData
        Updated AnnData with marker gene results in .uns['rank_genes_groups']
    """
    print(f"Finding marker genes using {method} test...")

    sc.tl.rank_genes_groups(
        adata,
        groupby=clustering_key,
        method=method,
        use_raw=use_raw,
        n_genes=n_genes,
        key_added='rank_genes_groups'
    )

    print("Marker gene analysis complete")
    return adata


def get_top_markers(adata: ad.AnnData,
                   cluster: Optional[Union[str, int]] = None,
                   n_genes: int = 10,
                   key: str = 'rank_genes_groups') -> pd.DataFrame:
    """
    Get top marker genes for a cluster or all clusters

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with marker gene results
    cluster : str or int, optional
        Specific cluster to get markers for (if None, returns all)
    n_genes : int, default=10
        Number of top genes to return
    key : str, default='rank_genes_groups'
        Key in adata.uns containing marker gene results

    Returns
    -------
    markers : DataFrame
        Top marker genes with statistics
    """
    if key not in adata.uns:
        raise ValueError(f"No marker gene results found. Run find_marker_genes() first.")

    result = adata.uns[key]

    if cluster is not None:
        cluster = str(cluster)
        groups = [cluster]
    else:
        groups = result['names'].dtype.names

    markers_list = []
    for group in groups:
        for i in range(min(n_genes, len(result['names'][group]))):
            marker = {
                'cluster': group,
                'gene': result['names'][group][i],
                'score': result['scores'][group][i],
                'logfoldchange': result['logfoldchanges'][group][i],
                'pval': result['pvals'][group][i],
                'pval_adj': result['pvals_adj'][group][i]
            }
            markers_list.append(marker)

    markers_df = pd.DataFrame(markers_list)
    return markers_df


def plot_marker_genes_heatmap(adata: ad.AnnData,
                              n_genes: int = 10,
                              clustering_key: str = 'leiden',
                              key: str = 'rank_genes_groups',
                              figsize: Optional[tuple] = None,
                              **kwargs) -> plt.Figure:
    """
    Plot heatmap of top marker genes

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with marker gene results
    n_genes : int, default=10
        Number of top genes per cluster to plot
    clustering_key : str, default='leiden'
        Key in adata.obs for grouping
    key : str, default='rank_genes_groups'
        Key in adata.uns containing marker gene results
    figsize : tuple, optional
        Figure size
    **kwargs
        Additional arguments passed to sc.pl.rank_genes_groups_heatmap

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig = sc.pl.rank_genes_groups_heatmap(
        adata,
        n_genes=n_genes,
        groupby=clustering_key,
        key=key,
        show=False,
        **kwargs
    )
    return fig


def plot_marker_genes_dotplot(adata: ad.AnnData,
                              n_genes: int = 5,
                              clustering_key: str = 'leiden',
                              key: str = 'rank_genes_groups',
                              figsize: Optional[tuple] = None,
                              **kwargs) -> plt.Figure:
    """
    Plot dotplot of top marker genes

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with marker gene results
    n_genes : int, default=5
        Number of top genes per cluster to plot
    clustering_key : str, default='leiden'
        Key in adata.obs for grouping
    key : str, default='rank_genes_groups'
        Key in adata.uns containing marker gene results
    figsize : tuple, optional
        Figure size
    **kwargs
        Additional arguments passed to sc.pl.rank_genes_groups_dotplot

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig = sc.pl.rank_genes_groups_dotplot(
        adata,
        n_genes=n_genes,
        groupby=clustering_key,
        key=key,
        show=False,
        **kwargs
    )
    return fig


def plot_marker_genes_violin(adata: ad.AnnData,
                             genes: List[str],
                             clustering_key: str = 'leiden',
                             ncols: int = 4,
                             figsize: Optional[tuple] = None,
                             **kwargs) -> plt.Figure:
    """
    Plot violin plots for specified marker genes

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    genes : list
        List of gene names to plot
    clustering_key : str, default='leiden'
        Key in adata.obs for grouping
    ncols : int, default=4
        Number of columns in subplot grid
    figsize : tuple, optional
        Figure size
    **kwargs
        Additional arguments passed to sc.pl.violin

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Filter genes that exist
    genes = [g for g in genes if g in adata.var_names]

    if not genes:
        raise ValueError("None of the specified genes found in dataset")

    fig = sc.pl.violin(
        adata,
        keys=genes,
        groupby=clustering_key,
        multi_panel=True,
        show=False,
        **kwargs
    )
    return fig


def plot_genes_on_umap(adata: ad.AnnData,
                      genes: List[str],
                      ncols: int = 3,
                      cmap: str = 'viridis',
                      figsize: Optional[tuple] = None,
                      **kwargs) -> plt.Figure:
    """
    Plot gene expression on UMAP

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with UMAP
    genes : list
        List of gene names to plot
    ncols : int, default=3
        Number of columns in subplot grid
    cmap : str, default='viridis'
        Colormap for expression
    figsize : tuple, optional
        Figure size
    **kwargs
        Additional arguments passed to sc.pl.umap

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Filter genes that exist
    genes = [g for g in genes if g in adata.var_names]

    if not genes:
        raise ValueError("None of the specified genes found in dataset")

    fig = sc.pl.umap(
        adata,
        color=genes,
        ncols=ncols,
        cmap=cmap,
        show=False,
        frameon=False,
        **kwargs
    )
    return fig


def annotate_clusters(adata: ad.AnnData,
                     annotations: Dict[Union[str, int], str],
                     clustering_key: str = 'leiden',
                     annotation_key: str = 'cell_type') -> ad.AnnData:
    """
    Manually annotate clusters with cell type labels

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with clustering
    annotations : dict
        Dictionary mapping cluster IDs to cell type names
        Example: {'0': 'T cells', '1': 'B cells', '2': 'Monocytes'}
    clustering_key : str, default='leiden'
        Key in adata.obs containing cluster assignments
    annotation_key : str, default='cell_type'
        Key to store cell type annotations

    Returns
    -------
    adata : AnnData
        Updated AnnData with cell type annotations in .obs
    """
    print("Annotating clusters...")

    # Convert cluster IDs to strings for mapping
    annotations = {str(k): v for k, v in annotations.items()}

    # Create cell type column
    adata.obs[annotation_key] = adata.obs[clustering_key].astype(str).map(annotations)

    # Fill unannotated clusters with 'Unknown'
    adata.obs[annotation_key] = adata.obs[annotation_key].fillna('Unknown')

    # Convert to categorical
    adata.obs[annotation_key] = pd.Categorical(adata.obs[annotation_key])

    # Summary
    annotated = (adata.obs[annotation_key] != 'Unknown').sum()
    print(f"Annotated {annotated}/{adata.n_obs} cells")

    annotation_counts = adata.obs[annotation_key].value_counts()
    print("\nCell type distribution:")
    for cell_type, count in annotation_counts.items():
        pct = count / adata.n_obs * 100
        print(f"  {cell_type}: {count} ({pct:.1f}%)")

    return adata


def update_cluster_annotation(adata: ad.AnnData,
                              cluster_id: Union[str, int],
                              new_annotation: str,
                              clustering_key: str = 'leiden',
                              annotation_key: str = 'cell_type') -> ad.AnnData:
    """
    Update annotation for a specific cluster

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    cluster_id : str or int
        Cluster ID to update
    new_annotation : str
        New cell type annotation
    clustering_key : str, default='leiden'
        Key in adata.obs containing cluster assignments
    annotation_key : str, default='cell_type'
        Key containing cell type annotations

    Returns
    -------
    adata : AnnData
        Updated AnnData
    """
    cluster_id = str(cluster_id)

    if annotation_key not in adata.obs.columns:
        adata.obs[annotation_key] = 'Unknown'

    mask = adata.obs[clustering_key].astype(str) == cluster_id
    n_cells = mask.sum()

    adata.obs.loc[mask, annotation_key] = new_annotation

    print(f"Updated cluster {cluster_id} → '{new_annotation}' ({n_cells} cells)")

    return adata


def get_cluster_marker_summary(adata: ad.AnnData,
                               cluster: Union[str, int],
                               n_genes: int = 20,
                               key: str = 'rank_genes_groups') -> pd.DataFrame:
    """
    Get a comprehensive summary of marker genes for a cluster

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with marker gene results
    cluster : str or int
        Cluster ID
    n_genes : int, default=20
        Number of top genes to return
    key : str, default='rank_genes_groups'
        Key in adata.uns containing marker gene results

    Returns
    -------
    summary : DataFrame
        Summary table with marker genes and their statistics
    """
    markers = get_top_markers(adata, cluster=cluster, n_genes=n_genes, key=key)

    # Add mean expression in cluster vs others
    cluster = str(cluster)
    cluster_mask = adata.obs['leiden'].astype(str) == cluster

    expr_summary = []
    for gene in markers['gene']:
        if gene in adata.var_names:
            expr_cluster = adata[cluster_mask, gene].X.mean()
            expr_other = adata[~cluster_mask, gene].X.mean()
            expr_summary.append({
                'mean_expr_cluster': expr_cluster,
                'mean_expr_other': expr_other,
                'fold_change': expr_cluster / (expr_other + 1e-10)
            })
        else:
            expr_summary.append({
                'mean_expr_cluster': np.nan,
                'mean_expr_other': np.nan,
                'fold_change': np.nan
            })

    expr_df = pd.DataFrame(expr_summary)
    summary = pd.concat([markers, expr_df], axis=1)

    return summary


def export_annotations(adata: ad.AnnData,
                      annotation_key: str = 'cell_type',
                      output_file: str = '../outputs/cell_annotations.csv') -> None:
    """
    Export cell type annotations to CSV file

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    annotation_key : str, default='cell_type'
        Key in adata.obs containing annotations
    output_file : str
        Output file path
    """
    if annotation_key not in adata.obs.columns:
        raise ValueError(f"Annotation key '{annotation_key}' not found in adata.obs")

    annotations = adata.obs[[annotation_key]].copy()
    annotations.to_csv(output_file)

    print(f"Annotations exported to: {output_file}")


def plot_annotated_umap(adata: ad.AnnData,
                       annotation_key: str = 'cell_type',
                       figsize: tuple = (10, 8),
                       legend_loc: str = 'right margin',
                       **kwargs) -> plt.Figure:
    """
    Plot UMAP colored by cell type annotations

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    annotation_key : str, default='cell_type'
        Key in adata.obs containing annotations
    figsize : tuple, default=(10, 8)
        Figure size
    legend_loc : str, default='right margin'
        Legend location
    **kwargs
        Additional arguments passed to sc.pl.umap

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    if annotation_key not in adata.obs.columns:
        raise ValueError(f"Annotation key '{annotation_key}' not found in adata.obs")

    fig = sc.pl.umap(
        adata,
        color=annotation_key,
        legend_loc=legend_loc,
        frameon=False,
        show=False,
        **kwargs
    )
    return fig