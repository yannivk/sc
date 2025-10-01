"""
Preprocessing module for doublet detection
"""

import numpy as np
import matplotlib.pyplot as plt
import anndata as ad
from typing import Optional


def detect_doublets(adata: ad.AnnData,
                   expected_doublet_rate: float = 0.06,
                   random_state: int = 42) -> ad.AnnData:
    """
    Detect doublets using Scrublet

    Wrapper around Scrublet library to detect potential doublet cells.
    Stores doublet scores and predictions in adata.obs.
    """
    import scrublet as scr

    print("Detecting doublets with Scrublet...")

    scrub = scr.Scrublet(
        adata.X,
        expected_doublet_rate=expected_doublet_rate,
        random_state=random_state
    )

    doublet_scores, predicted_doublets = scrub.scrub_doublets(
        min_counts=2,
        min_cells=3,
        min_gene_variability_pctl=85,
        n_prin_comps=30
    )

    adata.obs['doublet_score'] = doublet_scores
    adata.obs['predicted_doublet'] = predicted_doublets

    n_doublets = predicted_doublets.sum()
    doublet_rate = n_doublets / adata.n_obs * 100

    print(f"Detected {n_doublets} doublets ({doublet_rate:.2f}%)")
    print(f"Doublet score range: {doublet_scores.min():.3f} - {doublet_scores.max():.3f}")

    return adata


def plot_doublet_scores(adata: ad.AnnData,
                       threshold: Optional[float] = None,
                       figsize: tuple = (12, 4)) -> plt.Figure:
    """
    Plot doublet score distribution with histogram and box plot
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
        plt.suptitle('')

    plt.tight_layout()
    return fig
