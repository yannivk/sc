"""
Annotation module for marker gene analysis and manual cell type annotation
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from typing import Union, Dict
import matplotlib.pyplot as plt


def get_top_markers(adata: ad.AnnData,
                   cluster: Union[str, int] = None,
                   n_genes: int = 10,
                   key: str = 'rank_genes_groups') -> pd.DataFrame:
    """
    Extract top marker genes from scanpy rank_genes_groups results

    Returns DataFrame with gene names, scores, log fold changes, and p-values.
    """
    if key not in adata.uns:
        raise ValueError(f"No marker gene results found. Run sc.tl.rank_genes_groups() first.")

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

    return pd.DataFrame(markers_list)


def annotate_clusters(adata: ad.AnnData,
                     annotations: Dict[Union[str, int], str],
                     clustering_key: str = 'leiden',
                     annotation_key: str = 'cell_type') -> ad.AnnData:
    """
    Manually annotate clusters with cell type labels

    Maps cluster IDs to cell type names. Unannotated clusters are labeled 'Unknown'.
    """
    print("Annotating clusters...")

    annotations = {str(k): v for k, v in annotations.items()}

    adata.obs[annotation_key] = adata.obs[clustering_key].astype(str).map(annotations)
    adata.obs[annotation_key] = adata.obs[annotation_key].fillna('Unknown')
    adata.obs[annotation_key] = pd.Categorical(adata.obs[annotation_key])

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
    Get comprehensive marker summary with mean expression in cluster vs others
    """
    markers = get_top_markers(adata, cluster=cluster, n_genes=n_genes, key=key)

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
    return pd.concat([markers, expr_df], axis=1)
