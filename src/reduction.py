"""
Dimensionality reduction utilities
"""

import numpy as np
import matplotlib.pyplot as plt
import anndata as ad


def plot_pca_variance(adata: ad.AnnData,
                     n_pcs: int = 50,
                     figsize: tuple = (12, 4)) -> plt.Figure:
    """
    Plot PCA variance explained with individual and cumulative plots
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    variance_ratio = adata.uns['pca']['variance_ratio'][:n_pcs]

    # Variance ratio
    ax = axes[0]
    ax.plot(range(1, len(variance_ratio) + 1), variance_ratio, 'o-')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Ratio')
    ax.set_title('Variance Explained by Each PC')
    ax.grid(True, alpha=0.3)

    # Cumulative variance
    ax = axes[1]
    cumsum_variance = np.cumsum(variance_ratio)
    ax.plot(range(1, len(cumsum_variance) + 1), cumsum_variance, 'o-')
    ax.axhline(y=0.9, color='r', linestyle='--', label='90% variance')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Cumulative Variance Ratio')
    ax.set_title('Cumulative Variance Explained')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_embedding_comparison(adata: ad.AnnData,
                             basis: list = ['pca', 'umap', 'tsne'],
                             color: str = None,
                             figsize: tuple = (15, 5)) -> plt.Figure:
    """
    Compare different embeddings side by side
    """
    n_plots = len(basis)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 1:
        axes = [axes]

    for ax, b in zip(axes, basis):
        embedding_key = f'X_{b}'
        if embedding_key not in adata.obsm:
            ax.text(0.5, 0.5, f'{b.upper()} not computed',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(b.upper())
            continue

        coords = adata.obsm[embedding_key][:, :2]

        if color is not None and color in adata.obs:
            c = adata.obs[color]
            scatter = ax.scatter(coords[:, 0], coords[:, 1], c=c, s=1, alpha=0.6)
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], s=1, alpha=0.6)

        ax.set_xlabel(f'{b.upper()}1')
        ax.set_ylabel(f'{b.upper()}2')
        ax.set_title(b.upper())
        ax.axis('equal')

    plt.tight_layout()
    return fig
