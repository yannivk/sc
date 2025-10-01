"""
Visualization utilities for scRNA-seq analysis
"""

import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
from typing import Optional


def plot_qc_overview(adata: ad.AnnData,
                    groupby: Optional[str] = None,
                    figsize: tuple = (15, 5)) -> plt.Figure:
    """
    Plot overview of QC metrics as violin plots, optionally grouped
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    if groupby is None:
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
        sc.pl.violin(adata, ['total_counts', 'n_genes_by_counts', 'pct_counts_mt'],
                    groupby=groupby, multi_panel=True, show=False, ax=axes)

    plt.tight_layout()
    return fig


def create_summary_figure(adata: ad.AnnData,
                         clustering_key: str = 'leiden',
                         annotation_key: Optional[str] = None,
                         qc_metrics: bool = True,
                         figsize: tuple = (20, 12)) -> plt.Figure:
    """
    Create comprehensive summary figure with UMAP, cluster stats, and QC metrics
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
