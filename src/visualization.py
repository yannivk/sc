"""
Visualization utilities for scRNA-seq analysis
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from typing import Optional, List, Union
import matplotlib.pyplot as plt
import seaborn as sns


def plot_qc_overview(adata: ad.AnnData,
                    groupby: Optional[str] = None,
                    figsize: tuple = (15, 5)) -> plt.Figure:
    """
    Plot overview of QC metrics

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with QC metrics
    groupby : str, optional
        Key in adata.obs to group by
    figsize : tuple, default=(15, 5)
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    if groupby is None:
        # Simple violin plots
        axes[0].violinplot([adata.obs['total_counts']])
        axes[0].set_ylabel('Total counts')
        axes[0].set_title('Total Counts per Cell')

        axes[1].violinplot([adata.obs['n_genes_by_counts']])
        axes[1].set_ylabel('Number of genes')
        axes[1].set_title('Genes per Cell')

        axes[2].violinplot([adata.obs['pct_counts_mt']])
        axes[2].set_ylabel('Mitochondrial %')
        axes[2].set_title('Mitochondrial Percentage')
    else:
        # Grouped violin plots
        sc.pl.violin(adata, ['total_counts', 'n_genes_by_counts', 'pct_counts_mt'],
                    groupby=groupby, multi_panel=True, show=False, ax=axes)

    plt.tight_layout()
    return fig


def plot_cell_cycle(adata: ad.AnnData,
                   basis: str = 'umap',
                   figsize: tuple = (12, 4)) -> plt.Figure:
    """
    Plot cell cycle phase distribution

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with cell cycle scores
    basis : str, default='umap'
        Embedding to use
    figsize : tuple, default=(12, 4)
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    if 'phase' not in adata.obs.columns:
        print("Cell cycle scoring not performed. Run sc.tl.score_genes_cell_cycle() first.")
        return None

    fig = sc.pl.embedding(
        adata,
        basis=basis,
        color=['phase', 'S_score', 'G2M_score'],
        ncols=3,
        show=False,
        return_fig=True
    )
    return fig


def plot_gene_expression_comparison(adata: ad.AnnData,
                                   genes: List[str],
                                   groupby: str,
                                   plot_type: str = 'violin',
                                   figsize: Optional[tuple] = None) -> plt.Figure:
    """
    Compare gene expression across groups

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    genes : list
        List of genes to plot
    groupby : str
        Key in adata.obs to group by
    plot_type : {'violin', 'dotplot', 'matrixplot'}
        Type of plot
    figsize : tuple, optional
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    genes = [g for g in genes if g in adata.var_names]

    if not genes:
        raise ValueError("None of the specified genes found in dataset")

    if plot_type == 'violin':
        fig = sc.pl.violin(adata, keys=genes, groupby=groupby,
                          multi_panel=True, show=False)
    elif plot_type == 'dotplot':
        fig = sc.pl.dotplot(adata, var_names=genes, groupby=groupby,
                           show=False, return_fig=True)
    elif plot_type == 'matrixplot':
        fig = sc.pl.matrixplot(adata, var_names=genes, groupby=groupby,
                              show=False, return_fig=True)
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")

    return fig


def plot_embedding_grid(adata: ad.AnnData,
                       color_keys: List[str],
                       basis: str = 'umap',
                       ncols: int = 3,
                       figsize: Optional[tuple] = None,
                       **kwargs) -> plt.Figure:
    """
    Plot multiple embedding visualizations in a grid

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    color_keys : list
        List of keys in adata.obs or var_names to color by
    basis : str, default='umap'
        Embedding to use
    ncols : int, default=3
        Number of columns in grid
    figsize : tuple, optional
        Figure size
    **kwargs
        Additional arguments passed to sc.pl.embedding

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig = sc.pl.embedding(
        adata,
        basis=basis,
        color=color_keys,
        ncols=ncols,
        show=False,
        return_fig=True,
        **kwargs
    )
    return fig


def plot_cluster_dendrogram(adata: ad.AnnData,
                           clustering_key: str = 'leiden',
                           figsize: tuple = (10, 5)) -> plt.Figure:
    """
    Plot dendrogram of cluster relationships

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
    sc.tl.dendrogram(adata, groupby=clustering_key)

    fig = sc.pl.dendrogram(adata, groupby=clustering_key, show=False)
    return fig


def plot_gene_ranking(adata: ad.AnnData,
                     groups: Optional[List[str]] = None,
                     n_genes: int = 20,
                     key: str = 'rank_genes_groups',
                     figsize: tuple = (12, 8)) -> plt.Figure:
    """
    Plot ranked marker genes

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with marker gene results
    groups : list, optional
        Specific groups to plot
    n_genes : int, default=20
        Number of genes to show
    key : str, default='rank_genes_groups'
        Key in adata.uns containing results
    figsize : tuple, default=(12, 8)
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig = sc.pl.rank_genes_groups(
        adata,
        groups=groups,
        n_genes=n_genes,
        key=key,
        show=False
    )
    return fig


def create_summary_figure(adata: ad.AnnData,
                         clustering_key: str = 'leiden',
                         annotation_key: Optional[str] = None,
                         qc_metrics: bool = True,
                         figsize: tuple = (20, 12)) -> plt.Figure:
    """
    Create comprehensive summary figure

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    clustering_key : str, default='leiden'
        Key in adata.obs containing clusters
    annotation_key : str, optional
        Key in adata.obs containing cell type annotations
    qc_metrics : bool, default=True
        Include QC metrics in visualization
    figsize : tuple, default=(20, 12)
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    n_rows = 3 if qc_metrics else 2
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n_rows, 4, hspace=0.3, wspace=0.3)

    # Row 1: UMAP visualizations
    ax1 = fig.add_subplot(gs[0, 0])
    sc.pl.umap(adata, color=clustering_key, ax=ax1, show=False, frameon=False,
              legend_loc='on data', legend_fontsize='small', title='Clusters')

    if annotation_key and annotation_key in adata.obs.columns:
        ax2 = fig.add_subplot(gs[0, 1])
        sc.pl.umap(adata, color=annotation_key, ax=ax2, show=False, frameon=False,
                  legend_loc='right margin', legend_fontsize='small', title='Cell Types')

    ax3 = fig.add_subplot(gs[0, 2])
    sc.pl.umap(adata, color='total_counts', ax=ax3, show=False, frameon=False,
              title='Total Counts')

    ax4 = fig.add_subplot(gs[0, 3])
    sc.pl.umap(adata, color='n_genes_by_counts', ax=ax4, show=False, frameon=False,
              title='Genes Detected')

    # Row 2: Cluster statistics
    ax5 = fig.add_subplot(gs[1, :2])
    cluster_counts = adata.obs[clustering_key].value_counts().sort_index()
    cluster_counts.plot(kind='bar', ax=ax5, color='steelblue', edgecolor='black')
    ax5.set_xlabel('Cluster')
    ax5.set_ylabel('Number of Cells')
    ax5.set_title('Cluster Sizes')
    ax5.grid(axis='y', alpha=0.3)

    if qc_metrics:
        # Row 3: QC metrics
        ax6 = fig.add_subplot(gs[2, 0])
        ax6.hist(adata.obs['total_counts'], bins=50, edgecolor='black', alpha=0.7)
        ax6.set_xlabel('Total counts')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Total Counts Distribution')

        ax7 = fig.add_subplot(gs[2, 1])
        ax7.hist(adata.obs['n_genes_by_counts'], bins=50, edgecolor='black', alpha=0.7)
        ax7.set_xlabel('Number of genes')
        ax7.set_ylabel('Frequency')
        ax7.set_title('Genes Detected Distribution')

        ax8 = fig.add_subplot(gs[2, 2])
        ax8.hist(adata.obs['pct_counts_mt'], bins=50, edgecolor='black', alpha=0.7)
        ax8.set_xlabel('Mitochondrial %')
        ax8.set_ylabel('Frequency')
        ax8.set_title('Mitochondrial % Distribution')

    return fig