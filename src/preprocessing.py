"""
Preprocessing module for normalization, scaling, and feature selection
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from typing import Optional, Literal
import matplotlib.pyplot as plt


def detect_doublets(adata: ad.AnnData,
                   expected_doublet_rate: float = 0.06,
                   sim_doublet_ratio: float = 2.0,
                   random_state: int = 42) -> ad.AnnData:
    """
    Detect doublets using Scrublet

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    expected_doublet_rate : float, default=0.06
        Expected doublet rate (typically 0.05-0.10)
    sim_doublet_ratio : float, default=2.0
        Number of doublets to simulate relative to observed cells
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    adata : AnnData
        Updated AnnData with doublet scores and predictions in .obs
    """
    try:
        import scrublet as scr
    except ImportError:
        raise ImportError(
            "scrublet is required for doublet detection. "
            "Install with: pip install scrublet"
        )

    print("Detecting doublets with Scrublet...")

    # Initialize Scrublet
    scrub = scr.Scrublet(
        adata.X,
        expected_doublet_rate=expected_doublet_rate,
        sim_doublet_ratio=sim_doublet_ratio,
        random_state=random_state
    )

    # Run doublet detection
    doublet_scores, predicted_doublets = scrub.scrub_doublets(
        min_counts=2,
        min_cells=3,
        min_gene_variability_pctl=85,
        n_prin_comps=30
    )

    # Store results
    adata.obs['doublet_score'] = doublet_scores
    adata.obs['predicted_doublet'] = predicted_doublets

    # Summary
    n_doublets = predicted_doublets.sum()
    doublet_rate = n_doublets / adata.n_obs * 100

    print(f"Detected {n_doublets} doublets ({doublet_rate:.2f}%)")
    print(f"Doublet score range: {doublet_scores.min():.3f} - {doublet_scores.max():.3f}")

    return adata


def plot_doublet_scores(adata: ad.AnnData,
                       threshold: Optional[float] = None,
                       figsize: tuple = (12, 4)) -> plt.Figure:
    """
    Plot doublet score distribution

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with doublet scores
    threshold : float, optional
        Threshold for doublet classification
    figsize : tuple, default=(12, 4)
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    ax = axes[0]
    ax.hist(adata.obs['doublet_score'], bins=50, edgecolor='black', alpha=0.7)
    if threshold is not None:
        ax.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.3f}')
        ax.legend()
    ax.set_xlabel('Doublet Score')
    ax.set_ylabel('Number of Cells')
    ax.set_title('Doublet Score Distribution')

    # Box plot by prediction
    ax = axes[1]
    if 'predicted_doublet' in adata.obs.columns:
        adata.obs.boxplot(column='doublet_score', by='predicted_doublet', ax=ax)
        ax.set_xlabel('Predicted Doublet')
        ax.set_ylabel('Doublet Score')
        ax.set_title('Doublet Scores by Prediction')
        plt.suptitle('')  # Remove default title

    plt.tight_layout()
    return fig


def filter_doublets(adata: ad.AnnData,
                   threshold: Optional[float] = None,
                   copy: bool = False) -> Optional[ad.AnnData]:
    """
    Filter doublets from dataset

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with doublet predictions
    threshold : float, optional
        Custom doublet score threshold (overrides predicted_doublet column)
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

    if threshold is not None:
        # Use custom threshold
        filter_mask = adata.obs['doublet_score'] < threshold
    else:
        # Use predicted doublet column
        filter_mask = ~adata.obs['predicted_doublet']

    adata._inplace_subset_obs(filter_mask)

    n_cells_after = adata.n_obs
    n_doublets_removed = n_cells_before - n_cells_after

    print(f"Cells before doublet filtering: {n_cells_before}")
    print(f"Cells after doublet filtering: {n_cells_after}")
    print(f"Doublets removed: {n_doublets_removed} ({n_doublets_removed/n_cells_before*100:.2f}%)")

    if copy:
        return adata


def normalize_data(adata: ad.AnnData,
                  method: Literal['log1p', 'scran', 'pearson'] = 'log1p',
                  target_sum: Optional[float] = 1e4,
                  copy: bool = False) -> Optional[ad.AnnData]:
    """
    Normalize count data

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    method : {'log1p', 'scran', 'pearson'}
        Normalization method:
        - 'log1p': Shifted log transformation (standard)
        - 'scran': Scran normalization (for batch correction)
        - 'pearson': Pearson residuals (for rare cell types)
    target_sum : float, optional, default=1e4
        Target sum for log1p normalization
    copy : bool, default=False
        Return a copy instead of modifying in place

    Returns
    -------
    adata : AnnData or None
        Normalized AnnData object (if copy=True)
    """
    if copy:
        adata = adata.copy()

    print(f"Normalizing data with method: {method}")

    # Store raw counts
    adata.layers['counts'] = adata.X.copy()

    if method == 'log1p':
        # Standard log normalization
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
        print(f"Applied log1p normalization (target_sum={target_sum})")

    elif method == 'scran':
        # Scran normalization (requires scran package)
        print("Note: Scran normalization requires R and the scran package")
        print("Falling back to log1p normalization")
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)

    elif method == 'pearson':
        # Pearson residuals
        try:
            from scanpy.experimental.pp import normalize_pearson_residuals
            normalize_pearson_residuals(adata)
            print("Applied Pearson residuals normalization")
        except ImportError:
            print("Pearson residuals not available, falling back to log1p")
            sc.pp.normalize_total(adata, target_sum=target_sum)
            sc.pp.log1p(adata)

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    if copy:
        return adata


def select_highly_variable_genes(adata: ad.AnnData,
                                 method: Literal['seurat', 'cell_ranger', 'seurat_v3'] = 'seurat',
                                 n_top_genes: int = 2000,
                                 layer: Optional[str] = None,
                                 batch_key: Optional[str] = None) -> ad.AnnData:
    """
    Select highly variable genes

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    method : {'seurat', 'cell_ranger', 'seurat_v3'}
        Method for identifying highly variable genes
    n_top_genes : int, default=2000
        Number of highly variable genes to select
    layer : str, optional
        Layer to use (if None, uses .X)
    batch_key : str, optional
        Batch key for batch-aware HVG selection

    Returns
    -------
    adata : AnnData
        Updated AnnData with HVG information in .var
    """
    print(f"Selecting {n_top_genes} highly variable genes using {method} method...")

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor=method,
        layer=layer,
        batch_key=batch_key
    )

    n_hvgs = adata.var['highly_variable'].sum()
    print(f"Selected {n_hvgs} highly variable genes")

    return adata


def plot_highly_variable_genes(adata: ad.AnnData,
                               figsize: tuple = (10, 5)) -> plt.Figure:
    """
    Plot highly variable genes

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with HVG information
    figsize : tuple, default=(10, 5)
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig = sc.pl.highly_variable_genes(adata, show=False)
    return fig


def regress_out(adata: ad.AnnData,
               keys: list,
               n_jobs: Optional[int] = None) -> ad.AnnData:
    """
    Regress out unwanted sources of variation

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    keys : list
        Keys in adata.obs to regress out (e.g., ['total_counts', 'pct_counts_mt'])
    n_jobs : int, optional
        Number of parallel jobs

    Returns
    -------
    adata : AnnData
        Updated AnnData with regressed data
    """
    print(f"Regressing out: {', '.join(keys)}")

    sc.pp.regress_out(adata, keys=keys, n_jobs=n_jobs)

    print("Regression complete")
    return adata


def scale_data(adata: ad.AnnData,
              max_value: Optional[float] = 10.0,
              zero_center: bool = True) -> ad.AnnData:
    """
    Scale data to unit variance and zero mean

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    max_value : float, optional, default=10.0
        Clip values exceeding this after scaling
    zero_center : bool, default=True
        Zero-center the data

    Returns
    -------
    adata : AnnData
        Updated AnnData with scaled data
    """
    print("Scaling data...")

    sc.pp.scale(adata, max_value=max_value, zero_center=zero_center)

    print("Data scaled")
    return adata