"""
Interactive widgets for Jupyter notebook pipeline
"""

import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import anndata as ad
from typing import Optional, Callable, Dict, List
import pandas as pd

from . import qc, preprocessing, reduction, clustering, annotation, visualization
import scanpy as sc


class QCFilterWidget:
    """Interactive widget for QC filtering"""

    def __init__(self, adata: ad.AnnData):
        self.adata = adata
        self.filtered_adata = None

        # Calculate MAD thresholds
        self.thresholds = qc.calculate_mad_thresholds(adata)

        # Create widgets
        self.min_counts = widgets.FloatSlider(
            value=max(0, self.thresholds['total_counts'][0]),
            min=0,
            max=adata.obs['total_counts'].max(),
            step=100,
            description='Min counts:',
            style={'description_width': '150px'}
        )

        self.max_counts = widgets.FloatSlider(
            value=min(adata.obs['total_counts'].max(), self.thresholds['total_counts'][1]),
            min=0,
            max=adata.obs['total_counts'].max(),
            step=100,
            description='Max counts:',
            style={'description_width': '150px'}
        )

        self.min_genes = widgets.IntSlider(
            value=max(0, int(self.thresholds['n_genes'][0])),
            min=0,
            max=int(adata.obs['n_genes_by_counts'].max()),
            step=10,
            description='Min genes:',
            style={'description_width': '150px'}
        )

        self.max_genes = widgets.IntSlider(
            value=min(int(adata.obs['n_genes_by_counts'].max()), int(self.thresholds['n_genes'][1])),
            min=0,
            max=int(adata.obs['n_genes_by_counts'].max()),
            step=10,
            description='Max genes:',
            style={'description_width': '150px'}
        )

        self.max_mito = widgets.FloatSlider(
            value=min(20.0, self.thresholds['pct_counts_mt'][1]),
            min=0,
            max=100,
            step=0.5,
            description='Max mito %:',
            style={'description_width': '150px'}
        )

        self.output = widgets.Output()
        self.button = widgets.Button(description='Apply Filters', button_style='success')
        self.button.on_click(self._on_button_click)

        # Live update
        widgets.interactive(self._update_plot,
                          min_counts=self.min_counts,
                          max_counts=self.max_counts,
                          min_genes=self.min_genes,
                          max_genes=self.max_genes,
                          max_mito=self.max_mito)

    def display(self):
        """Display the widget"""
        display(widgets.VBox([
            widgets.HTML("<h3>Quality Control Filtering</h3>"),
            self.min_counts,
            self.max_counts,
            self.min_genes,
            self.max_genes,
            self.max_mito,
            self.button,
            self.output
        ]))

        # Initial plot
        with self.output:
            self._update_plot(
                self.min_counts.value,
                self.max_counts.value,
                self.min_genes.value,
                self.max_genes.value,
                self.max_mito.value
            )

    def _update_plot(self, min_counts, max_counts, min_genes, max_genes, max_mito):
        """Update plot with current thresholds"""
        with self.output:
            clear_output(wait=True)

            # Count cells that pass filters
            mask = (
                (self.adata.obs['total_counts'] >= min_counts) &
                (self.adata.obs['total_counts'] <= max_counts) &
                (self.adata.obs['n_genes_by_counts'] >= min_genes) &
                (self.adata.obs['n_genes_by_counts'] <= max_genes) &
                (self.adata.obs['pct_counts_mt'] <= max_mito)
            )

            n_pass = mask.sum()
            n_total = len(mask)
            pct_pass = n_pass / n_total * 100

            print(f"Cells passing filters: {n_pass} / {n_total} ({pct_pass:.1f}%)")

            # Plot with thresholds
            thresholds = {
                'total_counts': (min_counts, max_counts),
                'n_genes': (min_genes, max_genes),
                'pct_counts_mt': (0, max_mito)
            }

            fig = qc.plot_qc_metrics(self.adata, thresholds=thresholds, figsize=(15, 10))
            plt.show()

    def _on_button_click(self, b):
        """Apply filters"""
        with self.output:
            clear_output(wait=True)
            print("Applying filters...")

            # Create filter mask
            n_cells_before = self.adata.n_obs
            mask = (
                (self.adata.obs['total_counts'] >= self.min_counts.value) &
                (self.adata.obs['total_counts'] <= self.max_counts.value) &
                (self.adata.obs['n_genes_by_counts'] >= self.min_genes.value) &
                (self.adata.obs['n_genes_by_counts'] <= self.max_genes.value) &
                (self.adata.obs['pct_counts_mt'] <= self.max_mito.value)
            )

            self.filtered_adata = self.adata[mask].copy()
            n_cells_after = self.filtered_adata.n_obs

            print(f"Cells before: {n_cells_before}, after: {n_cells_after}, removed: {n_cells_before - n_cells_after}")

            # Remove genes not detected in any cell
            sc.pp.filter_genes(self.filtered_adata, min_cells=1)

            print("✓ Filters applied successfully!")

    def get_filtered_data(self) -> Optional[ad.AnnData]:
        """Get filtered AnnData object"""
        return self.filtered_adata


class DoubletFilterWidget:
    """Interactive widget for doublet filtering"""

    def __init__(self, adata: ad.AnnData):
        self.adata = adata
        self.filtered_adata = None

        # Auto threshold from predicted doublets
        if 'predicted_doublet' in adata.obs.columns:
            doublets = adata.obs[adata.obs['predicted_doublet']]['doublet_score']
            if len(doublets) > 0:
                auto_threshold = doublets.min()
            else:
                auto_threshold = adata.obs['doublet_score'].quantile(0.95)
        else:
            auto_threshold = adata.obs['doublet_score'].quantile(0.95)

        self.threshold = widgets.FloatSlider(
            value=auto_threshold,
            min=adata.obs['doublet_score'].min(),
            max=adata.obs['doublet_score'].max(),
            step=0.01,
            description='Threshold:',
            style={'description_width': '150px'}
        )

        self.output = widgets.Output()
        self.button = widgets.Button(description='Remove Doublets', button_style='warning')
        self.button.on_click(self._on_button_click)

        widgets.interactive(self._update_plot, threshold=self.threshold)

    def display(self):
        """Display the widget"""
        display(widgets.VBox([
            widgets.HTML("<h3>Doublet Detection</h3>"),
            self.threshold,
            self.button,
            self.output
        ]))

        with self.output:
            self._update_plot(self.threshold.value)

    def _update_plot(self, threshold):
        """Update plot with current threshold"""
        with self.output:
            clear_output(wait=True)

            n_doublets = (self.adata.obs['doublet_score'] >= threshold).sum()
            n_total = self.adata.n_obs
            pct_doublets = n_doublets / n_total * 100

            print(f"Doublets above threshold: {n_doublets} / {n_total} ({pct_doublets:.1f}%)")

            fig = preprocessing.plot_doublet_scores(self.adata, threshold=threshold)
            plt.show()

    def _on_button_click(self, b):
        """Remove doublets"""
        with self.output:
            clear_output(wait=True)
            print("Removing doublets...")

            n_cells_before = self.adata.n_obs
            mask = self.adata.obs['doublet_score'] < self.threshold.value
            self.filtered_adata = self.adata[mask].copy()
            n_cells_after = self.filtered_adata.n_obs

            print(f"Cells before: {n_cells_before}, after: {n_cells_after}, doublets removed: {n_cells_before - n_cells_after}")
            print("✓ Doublets removed successfully!")

    def get_filtered_data(self) -> Optional[ad.AnnData]:
        """Get filtered AnnData object"""
        return self.filtered_adata


class PCAWidget:
    """Interactive widget for PCA component selection"""

    def __init__(self, adata: ad.AnnData):
        self.adata = adata
        self.selected_n_pcs = 30

        self.n_pcs_slider = widgets.IntSlider(
            value=30,
            min=5,
            max=min(50, adata.obsm['X_pca'].shape[1]),
            step=5,
            description='N PCs:',
            style={'description_width': '150px'}
        )

        self.output = widgets.Output()
        self.button = widgets.Button(description='Confirm Selection', button_style='success')
        self.button.on_click(self._on_button_click)

        widgets.interactive(self._update_plot, n_pcs=self.n_pcs_slider)

    def display(self):
        """Display the widget"""
        display(widgets.VBox([
            widgets.HTML("<h3>PCA Component Selection</h3>"),
            self.n_pcs_slider,
            self.button,
            self.output
        ]))

        with self.output:
            self._update_plot(self.n_pcs_slider.value)

    def _update_plot(self, n_pcs):
        """Update plot"""
        with self.output:
            clear_output(wait=True)

            variance_ratio = self.adata.uns['pca']['variance_ratio'][:n_pcs]
            cumsum_variance = variance_ratio.sum()

            print(f"PCs 1-{n_pcs} explain {cumsum_variance*100:.2f}% of variance")

            fig = reduction.plot_pca_variance(self.adata, n_pcs=50)
            # Add vertical line at selected n_pcs
            for ax in fig.axes:
                ax.axvline(n_pcs, color='red', linestyle='--', label=f'Selected: {n_pcs}')
                ax.legend()
            plt.show()

    def _on_button_click(self, b):
        """Confirm selection"""
        self.selected_n_pcs = self.n_pcs_slider.value
        with self.output:
            clear_output(wait=True)
            print(f"✓ Selected {self.selected_n_pcs} principal components")

    def get_n_pcs(self) -> int:
        """Get selected number of PCs"""
        return self.selected_n_pcs


class ClusteringResolutionWidget:
    """Interactive widget for clustering resolution selection"""

    def __init__(self, adata: ad.AnnData, resolutions: List[float] = [0.25, 0.5, 1.0, 1.5]):
        self.adata = adata
        self.resolutions = resolutions
        self.selected_resolution = 1.0

        # Compute clustering at all resolutions if not already done
        for res in resolutions:
            key = f'leiden_res{res}'
            if key not in adata.obs.columns:
                sc.tl.leiden(adata, resolution=res, key_added=key)

        self.resolution_selector = widgets.Dropdown(
            options=resolutions,
            value=1.0,
            description='Resolution:',
            style={'description_width': '150px'}
        )

        self.output = widgets.Output()
        self.button = widgets.Button(description='Select Resolution', button_style='success')
        self.button.on_click(self._on_button_click)

        widgets.interactive(self._update_plot, resolution=self.resolution_selector)

    def display(self):
        """Display the widget"""
        display(widgets.VBox([
            widgets.HTML("<h3>Clustering Resolution Selection</h3>"),
            self.resolution_selector,
            self.button,
            self.output
        ]))

        with self.output:
            self._update_plot(self.resolution_selector.value)

    def _update_plot(self, resolution):
        """Update plot"""
        with self.output:
            clear_output(wait=True)

            key = f'leiden_res{resolution}'
            n_clusters = self.adata.obs[key].nunique()

            print(f"Resolution {resolution}: {n_clusters} clusters")

            # Plot all resolutions
            keys = [f'leiden_res{r}' for r in self.resolutions if f'leiden_res{r}' in self.adata.obs.columns]
            if keys:
                sc.pl.umap(self.adata, color=keys, ncols=2, frameon=False,
                          legend_loc='on data', legend_fontsize='x-small', show=True)

    def _on_button_click(self, b):
        """Confirm selection"""
        self.selected_resolution = self.resolution_selector.value
        key = f'leiden_res{self.selected_resolution}'

        # Copy to 'leiden' column
        self.adata.obs['leiden'] = self.adata.obs[key].copy()

        with self.output:
            clear_output(wait=True)
            print(f"✓ Selected resolution: {self.selected_resolution}")

    def get_resolution(self) -> float:
        """Get selected resolution"""
        return self.selected_resolution


class GeneVisualizationWidget:
    """Interactive widget for gene visualization"""

    def __init__(self, adata: ad.AnnData):
        self.adata = adata

        self.gene_input = widgets.Textarea(
            value='',
            placeholder='Enter gene names (one per line)',
            description='Genes:',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='50%', height='100px')
        )

        self.plot_button = widgets.Button(description='Plot Genes', button_style='info')
        self.plot_button.on_click(self._plot_genes)

        self.output = widgets.Output()

    def display(self):
        """Display the widget"""
        display(widgets.VBox([
            widgets.HTML("<h3>Gene Expression Visualization</h3>"),
            self.gene_input,
            self.plot_button,
            self.output
        ]))

    def _plot_genes(self, b):
        """Plot specified genes"""
        genes = [g.strip() for g in self.gene_input.value.split('\n') if g.strip()]

        if not genes:
            with self.output:
                clear_output(wait=True)
                print("Please enter at least one gene name")
            return

        with self.output:
            clear_output(wait=True)

            # Filter genes that exist
            valid_genes = [g for g in genes if g in self.adata.var_names]
            invalid_genes = [g for g in genes if g not in self.adata.var_names]

            if invalid_genes:
                print(f"Warning: Genes not found: {', '.join(invalid_genes)}")

            if not valid_genes:
                print("No valid genes to plot")
                return

            print(f"Plotting {len(valid_genes)} genes...")

            sc.pl.umap(self.adata, color=valid_genes, ncols=3, cmap='viridis',
                      frameon=False, show=True)


class AnnotationWidget:
    """Interactive widget for manual cluster annotation"""

    def __init__(self, adata: ad.AnnData, clustering_key: str = 'leiden'):
        self.adata = adata
        self.clustering_key = clustering_key
        self.annotations = {}

        clusters = sorted(adata.obs[clustering_key].unique().astype(str))

        self.cluster_selector = widgets.Dropdown(
            options=clusters,
            description='Cluster:',
            style={'description_width': '150px'}
        )

        self.annotation_input = widgets.Text(
            value='',
            placeholder='Enter cell type',
            description='Cell type:',
            style={'description_width': '150px'}
        )

        self.n_genes_slider = widgets.IntSlider(
            value=10,
            min=5,
            max=50,
            step=5,
            description='N markers:',
            style={'description_width': '150px'}
        )

        self.show_markers_button = widgets.Button(description='Show Markers', button_style='info')
        self.show_markers_button.on_click(self._show_markers)

        self.annotate_button = widgets.Button(description='Annotate Cluster', button_style='success')
        self.annotate_button.on_click(self._annotate_cluster)

        self.finish_button = widgets.Button(description='Finish Annotation', button_style='warning')
        self.finish_button.on_click(self._finish_annotation)

        self.output = widgets.Output()

    def display(self):
        """Display the widget"""
        display(widgets.VBox([
            widgets.HTML("<h3>Manual Cluster Annotation</h3>"),
            self.cluster_selector,
            self.n_genes_slider,
            self.show_markers_button,
            self.annotation_input,
            self.annotate_button,
            self.finish_button,
            self.output
        ]))

    def _show_markers(self, b):
        """Show marker genes for selected cluster"""
        cluster = self.cluster_selector.value

        with self.output:
            clear_output(wait=True)

            if 'rank_genes_groups' not in self.adata.uns:
                print("No marker genes found. Run find_marker_genes() first.")
                return

            print(f"Top {self.n_genes_slider.value} marker genes for cluster {cluster}:")

            markers = annotation.get_top_markers(
                self.adata,
                cluster=cluster,
                n_genes=self.n_genes_slider.value
            )

            print(markers[['gene', 'logfoldchange', 'pval_adj']].to_string(index=False))

    def _annotate_cluster(self, b):
        """Annotate selected cluster"""
        cluster = self.cluster_selector.value
        cell_type = self.annotation_input.value.strip()

        if not cell_type:
            with self.output:
                print("Please enter a cell type name")
            return

        self.annotations[cluster] = cell_type

        with self.output:
            clear_output(wait=True)
            print(f"✓ Cluster {cluster} → {cell_type}")
            print(f"\nCurrent annotations: {len(self.annotations)}/{len(self.adata.obs[self.clustering_key].unique())}")
            for c, ct in sorted(self.annotations.items()):
                print(f"  Cluster {c}: {ct}")

    def _finish_annotation(self, b):
        """Apply all annotations"""
        if not self.annotations:
            with self.output:
                print("No annotations to apply")
            return

        annotation.annotate_clusters(
            self.adata,
            annotations=self.annotations,
            clustering_key=self.clustering_key,
            annotation_key='cell_type'
        )

        with self.output:
            clear_output(wait=True)
            print("✓ Annotations applied successfully!")

            # Plot annotated UMAP
            sc.pl.umap(self.adata, color='cell_type', legend_loc='right margin',
                      frameon=False, show=True)

    def get_annotations(self) -> Dict[str, str]:
        """Get annotation dictionary"""
        return self.annotations