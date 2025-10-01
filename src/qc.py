"""
Quality control module for scRNA-seq data
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from typing import Optional, Tuple, Dict
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation


def calculate_qc_metrics(adata: ad.AnnData,
                         mito_prefix: str = 'MT-',
                         ribo_prefix: str = 'RPS,RPL',
                         hb_prefix: str = 'HB') -> ad.AnnData:
    """
    Calculate quality control metrics with gene type identification

    Identifies mitochondrial, ribosomal, and hemoglobin genes, then calculates
    QC metrics using scanpy. Adds log-transformed values for threshold selection.
    """
    print("Calculating QC metrics...")

    # Identify gene types
    adata.var['mt'] = adata.var_names.str.startswith(mito_prefix)
    ribo_prefixes = [p.strip() for p in ribo_prefix.split(',')]
    adata.var['ribo'] = adata.var_names.str.startswith(tuple(ribo_prefixes))
    adata.var['hb'] = adata.var_names.str.startswith(hb_prefix)

    # Calculate QC metrics
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=['mt', 'ribo', 'hb'],
        percent_top=None,
        log1p=False,
        inplace=True
    )

    # Add log-transformed values for MAD threshold calculation
    adata.obs['log_total_counts'] = np.log10(adata.obs['total_counts'] + 1)
    adata.obs['log_n_genes_by_counts'] = np.log10(adata.obs['n_genes_by_counts'] + 1)

    print(f"  MT genes: {adata.var['mt'].sum()}, Ribo genes: {adata.var['ribo'].sum()}, Hb genes: {adata.var['hb'].sum()}")
    return adata


def calculate_mad_thresholds(adata: ad.AnnData,
                             n_mads: float = 5.0,
                             n_mads_mito: float = 3.0) -> Dict[str, Tuple[float, float]]:
    """
    Calculate MAD-based filtering thresholds

    Uses median absolute deviation to compute robust thresholds for QC filtering.
    Works on log-transformed counts/genes for better threshold estimation.
    """
    def mad_threshold(values, n_mads, direction='both'):
        median = np.median(values)
        mad = median_abs_deviation(values)
        if direction == 'lower':
            return (median - n_mads * mad, np.inf)
        elif direction == 'upper':
            return (-np.inf, median + n_mads * mad)
        else:
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
    Generate comprehensive 9-panel QC plot with optional threshold lines
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