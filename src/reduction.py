"""
Dimensionality reduction module (PCA, UMAP, neighbors)
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from typing import Optional, Literal
import matplotlib.pyplot as plt


def compute_pca(adata: ad.AnnData,
               n_comps: int = 50,
               use_highly_variable: bool = True,
               random_state: int = 42) -> ad.AnnData:
    """
    Compute PCA

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    n_comps : int, default=50
        Number of principal components to compute
    use_highly_variable : bool, default=True
        Use only highly variable genes
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    adata : AnnData
        Updated AnnData with PCA results in .obsm['X_pca'] and .varm['PCs']
    """
    print(f"Computing PCA with {n_comps} components...")

    sc.tl.pca(
        adata,
        n_comps=n_comps,
        use_highly_variable=use_highly_variable,
        random_state=random_state
    )

    print(f"PCA computed: {adata.obsm['X_pca'].shape}")
    return adata


def plot_pca_variance(adata: ad.AnnData,
                     n_pcs: int = 50,
                     figsize: tuple = (12, 4)) -> plt.Figure:
    """
    Plot PCA variance explained

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with PCA results
    n_pcs : int, default=50
        Number of PCs to show
    figsize : tuple, default=(12, 4)
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Variance ratio
    ax = axes[0]
    variance_ratio = adata.uns['pca']['variance_ratio'][:n_pcs]
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


def plot_pca_scatter(adata: ad.AnnData,
                    color: Optional[str] = None,
                    components: list = ['1,2', '3,4'],
                    figsize: tuple = (12, 5)) -> plt.Figure:
    """
    Plot PCA scatter plots

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with PCA results
    color : str, optional
        Key in adata.obs or adata.var_names to color by
    components : list, default=['1,2', '3,4']
        PC pairs to plot
    figsize : tuple, default=(12, 5)
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig = sc.pl.pca(
        adata,
        color=color,
        components=components,
        show=False,
        return_fig=True
    )
    return fig


def compute_neighbors(adata: ad.AnnData,
                     n_neighbors: int = 15,
                     n_pcs: int = 30,
                     metric: Literal['euclidean', 'cosine'] = 'euclidean',
                     random_state: int = 42) -> ad.AnnData:
    """
    Compute neighborhood graph

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with PCA
    n_neighbors : int, default=15
        Number of neighbors to use
    n_pcs : int, default=30
        Number of PCs to use
    metric : {'euclidean', 'cosine'}, default='euclidean'
        Distance metric
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    adata : AnnData
        Updated AnnData with neighbors in .obsp and .uns
    """
    print(f"Computing neighbors (k={n_neighbors}, n_pcs={n_pcs})...")

    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        metric=metric,
        random_state=random_state
    )

    print("Neighbor graph computed")
    return adata


def compute_umap(adata: ad.AnnData,
                min_dist: float = 0.5,
                spread: float = 1.0,
                n_components: int = 2,
                random_state: int = 42) -> ad.AnnData:
    """
    Compute UMAP embedding

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with neighbors
    min_dist : float, default=0.5
        Minimum distance between points in low-dimensional representation
    spread : float, default=1.0
        Effective scale of embedded points
    n_components : int, default=2
        Number of dimensions for UMAP
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    adata : AnnData
        Updated AnnData with UMAP in .obsm['X_umap']
    """
    print(f"Computing UMAP (min_dist={min_dist}, spread={spread})...")

    sc.tl.umap(
        adata,
        min_dist=min_dist,
        spread=spread,
        n_components=n_components,
        random_state=random_state
    )

    print("UMAP computed")
    return adata


def compute_tsne(adata: ad.AnnData,
                n_pcs: int = 30,
                perplexity: float = 30.0,
                n_components: int = 2,
                random_state: int = 42) -> ad.AnnData:
    """
    Compute t-SNE embedding

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with PCA
    n_pcs : int, default=30
        Number of PCs to use
    perplexity : float, default=30.0
        t-SNE perplexity parameter
    n_components : int, default=2
        Number of dimensions for t-SNE
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    adata : AnnData
        Updated AnnData with t-SNE in .obsm['X_tsne']
    """
    print(f"Computing t-SNE (perplexity={perplexity})...")

    sc.tl.tsne(
        adata,
        n_pcs=n_pcs,
        perplexity=perplexity,
        n_jobs=-1,
        random_state=random_state
    )

    print("t-SNE computed")
    return adata


def plot_umap(adata: ad.AnnData,
             color: Optional[list] = None,
             figsize: Optional[tuple] = None,
             ncols: int = 3,
             **kwargs) -> plt.Figure:
    """
    Plot UMAP embedding

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with UMAP
    color : list, optional
        Keys in adata.obs or adata.var_names to color by
    figsize : tuple, optional
        Figure size
    ncols : int, default=3
        Number of columns for subplot grid
    **kwargs
        Additional arguments passed to sc.pl.umap

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig = sc.pl.umap(
        adata,
        color=color,
        ncols=ncols,
        show=False,
        return_fig=True,
        **kwargs
    )
    return fig


def plot_embedding_comparison(adata: ad.AnnData,
                             basis: list = ['pca', 'umap', 'tsne'],
                             color: Optional[str] = None,
                             figsize: tuple = (15, 5)) -> plt.Figure:
    """
    Compare different embeddings side by side

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with multiple embeddings
    basis : list, default=['pca', 'umap', 'tsne']
        Embeddings to compare
    color : str, optional
        Key in adata.obs to color by
    figsize : tuple, default=(15, 5)
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
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