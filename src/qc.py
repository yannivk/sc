"""
Quality control module for scRNA-seq data
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from typing import Optional, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_qc_metrics(adata: ad.AnnData,
                         mito_prefix: str = 'MT-',
                         ribo_prefix: str = 'RPS,RPL',
                         hb_prefix: str = 'HB') -> ad.AnnData:
    """
    Calculate quality control metrics

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    mito_prefix : str, default='MT-'
        Prefix for mitochondrial genes
    ribo_prefix : str, default='RPS,RPL'
        Prefixes for ribosomal genes (comma-separated)
    hb_prefix : str, default='HB'
        Prefix for hemoglobin genes

    Returns
    -------
    adata : AnnData
        Updated AnnData with QC metrics in .obs and .var
    """
    print("Calculating QC metrics...")

    # Identify mitochondrial genes
    adata.var['mt'] = adata.var_names.str.startswith(mito_prefix)

    # Identify ribosomal genes
    ribo_prefixes = [p.strip() for p in ribo_prefix.split(',')]
    adata.var['ribo'] = adata.var_names.str.startswith(tuple(ribo_prefixes))

    # Identify hemoglobin genes
    adata.var['hb'] = adata.var_names.str.startswith(hb_prefix)

    # Calculate QC metrics
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=['mt', 'ribo', 'hb'],
        percent_top=None,
        log1p=False,
        inplace=True
    )

    # Add log-transformed values for easier threshold selection
    adata.obs['log_total_counts'] = np.log10(adata.obs['total_counts'] + 1)
    adata.obs['log_n_genes_by_counts'] = np.log10(adata.obs['n_genes_by_counts'] + 1)

    print(f"QC metrics calculated:")
    print(f"  - Mitochondrial genes: {adata.var['mt'].sum()}")
    print(f"  - Ribosomal genes: {adata.var['ribo'].sum()}")
    print(f"  - Hemoglobin genes: {adata.var['hb'].sum()}")

    return adata


def calculate_mad_thresholds(adata: ad.AnnData,
                             n_mads: float = 5.0,
                             n_mads_mito: float = 3.0) -> Dict[str, Tuple[float, float]]:
    """
    Calculate MAD-based filtering thresholds

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with QC metrics
    n_mads : float, default=5.0
        Number of MADs for counts and genes
    n_mads_mito : float, default=3.0
        Number of MADs for mitochondrial percentage

    Returns
    -------
    thresholds : dict
        Dictionary with thresholds for each metric
    """
    from scipy.stats import median_abs_deviation

    def mad_threshold(values, n_mads, direction='both'):
        """Calculate MAD-based thresholds"""
        median = np.median(values)
        mad = median_abs_deviation(values)

        if direction == 'lower':
            return (median - n_mads * mad, np.inf)
        elif direction == 'upper':
            return (-np.inf, median + n_mads * mad)
        else:  # both
            return (median - n_mads * mad, median + n_mads * mad)

    thresholds = {
        'total_counts': mad_threshold(adata.obs['log_total_counts'], n_mads),
        'n_genes': mad_threshold(adata.obs['log_n_genes_by_counts'], n_mads),
        'pct_counts_mt': mad_threshold(adata.obs['pct_counts_mt'], n_mads_mito, direction='upper')
    }

    # Convert log thresholds back to linear scale
    thresholds['total_counts'] = tuple(10**x - 1 for x in thresholds['total_counts'])
    thresholds['n_genes'] = tuple(10**x - 1 for x in thresholds['n_genes'])

    return thresholds


def plot_qc_metrics(adata: ad.AnnData,
                   thresholds: Optional[Dict] = None,
                   figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Generate comprehensive QC plots

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with QC metrics
    thresholds : dict, optional
        Dictionary with filtering thresholds
    figsize : tuple, default=(15, 10)
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object with QC plots
    """
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    fig.suptitle('Quality Control Metrics', fontsize=16, fontweight='bold')

    # 1. Total counts distribution
    ax = axes[0, 0]
    ax.hist(adata.obs['total_counts'], bins=100, edgecolor='black', alpha=0.7)
    if thresholds and 'total_counts' in thresholds:
        ax.axvline(thresholds['total_counts'][0], color='red', linestyle='--', label='Lower threshold')
        ax.axvline(thresholds['total_counts'][1], color='red', linestyle='--', label='Upper threshold')
        ax.legend()
    ax.set_xlabel('Total counts')
    ax.set_ylabel('Number of cells')
    ax.set_title('Total Counts per Cell')

    # 2. Number of genes distribution
    ax = axes[0, 1]
    ax.hist(adata.obs['n_genes_by_counts'], bins=100, edgecolor='black', alpha=0.7)
    if thresholds and 'n_genes' in thresholds:
        ax.axvline(thresholds['n_genes'][0], color='red', linestyle='--', label='Lower threshold')
        ax.axvline(thresholds['n_genes'][1], color='red', linestyle='--', label='Upper threshold')
        ax.legend()
    ax.set_xlabel('Number of genes')
    ax.set_ylabel('Number of cells')
    ax.set_title('Genes Detected per Cell')

    # 3. Mitochondrial percentage distribution
    ax = axes[0, 2]
    ax.hist(adata.obs['pct_counts_mt'], bins=100, edgecolor='black', alpha=0.7)
    if thresholds and 'pct_counts_mt' in thresholds:
        ax.axvline(thresholds['pct_counts_mt'][1], color='red', linestyle='--', label='Upper threshold')
        ax.legend()
    ax.set_xlabel('Mitochondrial %')
    ax.set_ylabel('Number of cells')
    ax.set_title('Mitochondrial Percentage')

    # 4. Log-scale total counts
    ax = axes[1, 0]
    ax.hist(adata.obs['log_total_counts'], bins=100, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Log10(Total counts)')
    ax.set_ylabel('Number of cells')
    ax.set_title('Total Counts (Log Scale)')

    # 5. Log-scale number of genes
    ax = axes[1, 1]
    ax.hist(adata.obs['log_n_genes_by_counts'], bins=100, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Log10(Number of genes)')
    ax.set_ylabel('Number of cells')
    ax.set_title('Genes Detected (Log Scale)')

    # 6. Counts vs genes scatter
    ax = axes[1, 2]
    ax.scatter(adata.obs['total_counts'], adata.obs['n_genes_by_counts'],
              alpha=0.3, s=1)
    ax.set_xlabel('Total counts')
    ax.set_ylabel('Number of genes')
    ax.set_title('Counts vs Genes')

    # 7. Counts vs mitochondrial percentage
    ax = axes[2, 0]
    ax.scatter(adata.obs['total_counts'], adata.obs['pct_counts_mt'],
              alpha=0.3, s=1)
    ax.set_xlabel('Total counts')
    ax.set_ylabel('Mitochondrial %')
    ax.set_title('Counts vs Mitochondrial %')

    # 8. Genes vs mitochondrial percentage
    ax = axes[2, 1]
    ax.scatter(adata.obs['n_genes_by_counts'], adata.obs['pct_counts_mt'],
              alpha=0.3, s=1)
    ax.set_xlabel('Number of genes')
    ax.set_ylabel('Mitochondrial %')
    ax.set_title('Genes vs Mitochondrial %')

    # 9. Ribosomal percentage distribution
    ax = axes[2, 2]
    if 'pct_counts_ribo' in adata.obs.columns:
        ax.hist(adata.obs['pct_counts_ribo'], bins=100, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Ribosomal %')
        ax.set_ylabel('Number of cells')
        ax.set_title('Ribosomal Percentage')
    else:
        ax.text(0.5, 0.5, 'No ribosomal data', ha='center', va='center',
               transform=ax.transAxes)
        ax.set_title('Ribosomal Percentage')

    plt.tight_layout()
    return fig


def filter_cells(adata: ad.AnnData,
                min_counts: Optional[float] = None,
                max_counts: Optional[float] = None,
                min_genes: Optional[float] = None,
                max_genes: Optional[float] = None,
                max_mito: Optional[float] = None,
                copy: bool = False) -> Optional[ad.AnnData]:
    """
    Filter cells based on QC metrics

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    min_counts : float, optional
        Minimum total counts per cell
    max_counts : float, optional
        Maximum total counts per cell
    min_genes : float, optional
        Minimum number of genes per cell
    max_genes : float, optional
        Maximum number of genes per cell
    max_mito : float, optional
        Maximum mitochondrial percentage
    copy : bool, default=False
        Return a copy instead of modifying in place

    Returns
    -------
    adata : AnnData or None
        Filtered AnnData object (if copy=True)
    """
    if copy:
        adata = adata.copy()

    n_cells_before = adata.n_obs
    print(f"Cells before filtering: {n_cells_before}")

    # Create filter mask
    filter_mask = np.ones(adata.n_obs, dtype=bool)

    if min_counts is not None:
        filter_mask &= adata.obs['total_counts'] >= min_counts
    if max_counts is not None:
        filter_mask &= adata.obs['total_counts'] <= max_counts
    if min_genes is not None:
        filter_mask &= adata.obs['n_genes_by_counts'] >= min_genes
    if max_genes is not None:
        filter_mask &= adata.obs['n_genes_by_counts'] <= max_genes
    if max_mito is not None:
        filter_mask &= adata.obs['pct_counts_mt'] <= max_mito

    # Apply filter
    adata._inplace_subset_obs(filter_mask)

    n_cells_after = adata.n_obs
    n_cells_removed = n_cells_before - n_cells_after

    print(f"Cells after filtering: {n_cells_after}")
    print(f"Cells removed: {n_cells_removed} ({n_cells_removed/n_cells_before*100:.2f}%)")

    # Filter genes (remove genes not detected in any cell)
    n_genes_before = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=1)
    n_genes_after = adata.n_vars

    if n_genes_after < n_genes_before:
        print(f"Genes removed (not detected): {n_genes_before - n_genes_after}")

    if copy:
        return adata


def filter_genes_by_counts(adata: ad.AnnData,
                          min_cells: int = 20,
                          copy: bool = False) -> Optional[ad.AnnData]:
    """
    Filter genes based on minimum number of cells expressing them

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    min_cells : int, default=20
        Minimum number of cells expressing a gene
    copy : bool, default=False
        Return a copy instead of modifying in place

    Returns
    -------
    adata : AnnData or None
        Filtered AnnData object (if copy=True)
    """
    if copy:
        adata = adata.copy()

    n_genes_before = adata.n_vars
    print(f"Genes before filtering: {n_genes_before}")

    sc.pp.filter_genes(adata, min_cells=min_cells)

    n_genes_after = adata.n_vars
    n_genes_removed = n_genes_before - n_genes_after

    print(f"Genes after filtering: {n_genes_after}")
    print(f"Genes removed: {n_genes_removed} ({n_genes_removed/n_genes_before*100:.2f}%)")

    if copy:
        return adata


def get_qc_summary(adata: ad.AnnData) -> pd.DataFrame:
    """
    Generate summary statistics of QC metrics

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with QC metrics

    Returns
    -------
    summary : DataFrame
        Summary statistics table
    """
    metrics = ['total_counts', 'n_genes_by_counts', 'pct_counts_mt']

    if 'pct_counts_ribo' in adata.obs.columns:
        metrics.append('pct_counts_ribo')

    summary = adata.obs[metrics].describe().T

    return summary