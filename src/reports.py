"""
PDF report generation module
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import anndata as ad
from typing import Optional, List
from pathlib import Path

from . import qc, visualization, annotation


def create_qc_report(adata: ad.AnnData,
                    output_file: str = '../outputs/reports/qc_report.pdf',
                    thresholds: Optional[dict] = None) -> None:
    """
    Generate comprehensive QC report as PDF

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with QC metrics
    output_file : str
        Output PDF file path
    thresholds : dict, optional
        Dictionary with filtering thresholds used
    """
    # Create output directory if needed
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating QC report: {output_file}")

    with PdfPages(output_file) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.7, 'Single-Cell RNA-seq', ha='center', fontsize=24, fontweight='bold')
        fig.text(0.5, 0.6, 'Quality Control Report', ha='center', fontsize=20)
        fig.text(0.5, 0.4, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                ha='center', fontsize=12)
        fig.text(0.5, 0.35, f'Dataset: {adata.n_obs} cells × {adata.n_vars} genes',
                ha='center', fontsize=12)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # QC summary statistics
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')

        summary = qc.get_qc_summary(adata)
        summary_text = summary.to_string()

        ax.text(0.1, 0.9, 'QC Metrics Summary', fontsize=16, fontweight='bold',
               transform=ax.transAxes)
        ax.text(0.1, 0.05, summary_text, fontsize=10, family='monospace',
               transform=ax.transAxes, verticalalignment='bottom')

        if thresholds:
            threshold_text = "Filtering Thresholds Applied:\n"
            for key, val in thresholds.items():
                threshold_text += f"  {key}: {val}\n"
            ax.text(0.1, 0.45, threshold_text, fontsize=10,
                   transform=ax.transAxes)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # QC plots
        fig = qc.plot_qc_metrics(adata, thresholds=thresholds, figsize=(15, 10))
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # QC overview
        if 'leiden' in adata.obs.columns:
            fig = visualization.plot_qc_overview(adata, groupby='leiden', figsize=(15, 5))
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

        # Doublet scores if available
        if 'doublet_score' in adata.obs.columns:
            from . import preprocessing
            fig = preprocessing.plot_doublet_scores(adata)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    print(f"QC report saved: {output_file}")


def create_analysis_report(adata: ad.AnnData,
                          output_file: str = '../outputs/reports/analysis_report.pdf',
                          clustering_key: str = 'leiden',
                          annotation_key: Optional[str] = 'cell_type',
                          marker_genes: Optional[List[str]] = None) -> None:
    """
    Generate comprehensive analysis report as PDF

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with complete analysis
    output_file : str
        Output PDF file path
    clustering_key : str, default='leiden'
        Key in adata.obs containing clusters
    annotation_key : str, optional
        Key in adata.obs containing cell type annotations
    marker_genes : list, optional
        Specific marker genes to highlight
    """
    # Create output directory if needed
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating analysis report: {output_file}")

    with PdfPages(output_file) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.7, 'Single-Cell RNA-seq', ha='center', fontsize=24, fontweight='bold')
        fig.text(0.5, 0.6, 'Analysis Report', ha='center', fontsize=20)
        fig.text(0.5, 0.4, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                ha='center', fontsize=12)
        fig.text(0.5, 0.35, f'Dataset: {adata.n_obs} cells × {adata.n_vars} genes',
                ha='center', fontsize=12)

        n_clusters = adata.obs[clustering_key].nunique()
        fig.text(0.5, 0.25, f'Clusters: {n_clusters}', ha='center', fontsize=12)

        if annotation_key and annotation_key in adata.obs.columns:
            n_types = adata.obs[annotation_key].nunique()
            fig.text(0.5, 0.2, f'Cell types: {n_types}', ha='center', fontsize=12)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Summary figure
        fig = visualization.create_summary_figure(
            adata,
            clustering_key=clustering_key,
            annotation_key=annotation_key,
            qc_metrics=True,
            figsize=(20, 12)
        )
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # PCA variance
        if 'pca' in adata.uns:
            from . import reduction
            fig = reduction.plot_pca_variance(adata, n_pcs=50)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

        # Cluster statistics
        from . import clustering
        fig = clustering.plot_cluster_sizes(adata, clustering_key=clustering_key)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Cluster statistics table
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')

        stats = clustering.compute_cluster_statistics(adata, clustering_key=clustering_key)
        stats_text = stats.to_string(index=False)

        ax.text(0.1, 0.95, 'Cluster Statistics', fontsize=16, fontweight='bold',
               transform=ax.transAxes)
        ax.text(0.1, 0.05, stats_text, fontsize=9, family='monospace',
               transform=ax.transAxes, verticalalignment='bottom')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Marker genes heatmap
        if 'rank_genes_groups' in adata.uns:
            try:
                fig = annotation.plot_marker_genes_heatmap(
                    adata,
                    n_genes=10,
                    clustering_key=clustering_key
                )
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Could not generate marker genes heatmap: {e}")

            # Marker genes dotplot
            try:
                fig = annotation.plot_marker_genes_dotplot(
                    adata,
                    n_genes=5,
                    clustering_key=clustering_key
                )
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Could not generate marker genes dotplot: {e}")

        # Custom marker genes if provided
        if marker_genes:
            try:
                fig = annotation.plot_genes_on_umap(adata, genes=marker_genes, ncols=3)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Could not plot marker genes: {e}")

        # Annotation summary if available
        if annotation_key and annotation_key in adata.obs.columns:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('off')

            annotation_counts = adata.obs[annotation_key].value_counts()
            annotation_text = "Cell Type Distribution:\n\n"
            for cell_type, count in annotation_counts.items():
                pct = count / adata.n_obs * 100
                annotation_text += f"{cell_type}: {count} cells ({pct:.1f}%)\n"

            ax.text(0.1, 0.9, 'Cell Type Annotation Summary', fontsize=16, fontweight='bold',
                   transform=ax.transAxes)
            ax.text(0.1, 0.1, annotation_text, fontsize=12,
                   transform=ax.transAxes, verticalalignment='bottom')

            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

            # Annotated UMAP
            fig = annotation.plot_annotated_umap(
                adata,
                annotation_key=annotation_key,
                figsize=(12, 10)
            )
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    print(f"Analysis report saved: {output_file}")


def create_full_report(adata: ad.AnnData,
                      output_file: str = '../outputs/reports/full_report.pdf',
                      **kwargs) -> None:
    """
    Generate complete report combining QC and analysis

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    output_file : str
        Output PDF file path
    **kwargs
        Additional arguments for report sections
    """
    print(f"Generating full report: {output_file}")

    # Create output directory
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Generate separate reports temporarily
    qc_file = output_file.replace('.pdf', '_qc_temp.pdf')
    analysis_file = output_file.replace('.pdf', '_analysis_temp.pdf')

    create_qc_report(adata, output_file=qc_file, **kwargs)
    create_analysis_report(adata, output_file=analysis_file, **kwargs)

    # Merge PDFs (simple approach: just use analysis report as full report)
    import shutil
    shutil.move(analysis_file, output_file)

    # Clean up temp files
    Path(qc_file).unlink(missing_ok=True)

    print(f"Full report saved: {output_file}")