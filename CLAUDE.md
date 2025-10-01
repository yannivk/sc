# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Interactive single-cell RNA-seq analysis pipeline built on Scanpy. Provides Jupyter notebook-based workflow with interactive widgets for QC filtering, clustering, and manual cell type annotation.

## Environment Setup

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate sc_pipeline

# Enable Jupyter widgets (required for interactive features)
jupyter nbextension enable --py widgetsnbextension --sys-prefix

# Verify installation
python -c "import scanpy, anndata, scrublet, ipywidgets; print('✓ All packages installed')"
```

## Running the Pipeline

### Launch Jupyter Notebook
```bash
cd notebooks
jupyter notebook
# Open pipeline.ipynb and run cells sequentially
```

### Test with Example Data
```bash
cd data
wget http://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz
tar -xzf pbmc3k_filtered_gene_bc_matrices.tar.gz
cd ../notebooks
# In notebook: adata = io.load_10x_mtx('../data/filtered_gene_bc_matrices/hg19/')
```

## Architecture

### Design Philosophy

**Minimal wrappers, maximal clarity**: This pipeline uses Scanpy functions directly in the notebook. The `src/` modules only contain functions that add real value beyond Scanpy's API.

### Module Structure

The `src/` directory contains **essential utilities only**:

- **io.py**: Data loading (h5ad, rds/Seurat, 10X mtx) and checkpoint saving
- **qc.py**:
  - `calculate_qc_metrics()` - Gene type identification (MT/ribo/hb) + QC calculation
  - `calculate_mad_thresholds()` - MAD-based threshold computation
  - `plot_qc_metrics()` - Custom 9-panel QC visualization
- **preprocessing.py**:
  - `detect_doublets()` - Scrublet integration
  - `plot_doublet_scores()` - Doublet visualization
- **reduction.py**:
  - `plot_pca_variance()` - Custom variance plots
  - `plot_embedding_comparison()` - Side-by-side embedding comparison
- **clustering.py**:
  - `compute_multiple_resolutions()` - Batch Leiden at multiple resolutions
  - `merge_clusters()` / `split_cluster()` - Cluster manipulation
  - `subcluster_cells()` - Full subclustering pipeline
  - `plot_cluster_sizes()` / `compute_cluster_statistics()` - Cluster analysis
- **annotation.py**:
  - `get_top_markers()` - Extract markers from Scanpy results
  - `annotate_clusters()` / `update_cluster_annotation()` - Annotation mapping
  - `get_cluster_marker_summary()` - Comprehensive marker info
- **visualization.py**:
  - `plot_qc_overview()` - Custom QC violin plots
  - `create_summary_figure()` - Complex multi-panel summary
- **interactive.py**: IPython widget classes for notebook interaction
- **reports.py**: PDF report generation

### What to Use Directly from Scanpy

**Use these Scanpy functions directly in the notebook instead of wrappers:**

- **Filtering**: `adata = adata[mask].copy()` + `sc.pp.filter_genes()`
- **Normalization**: `sc.pp.normalize_total()` + `sc.pp.log1p()`
- **HVG selection**: `sc.pp.highly_variable_genes()`
- **Scaling**: `sc.pp.scale()`
- **Regress out**: `sc.pp.regress_out()`
- **Dimensionality reduction**: `sc.tl.pca()`, `sc.tl.umap()`, `sc.pp.neighbors()`
- **Clustering**: `sc.tl.leiden()`
- **Marker genes**: `sc.tl.rank_genes_groups()`
- **Plotting**: `sc.pl.umap()`, `sc.pl.violin()`, `sc.pl.rank_genes_groups_heatmap()`, etc.

### Data Flow

1. **Load** → Raw AnnData object (cells × genes matrix)
2. **QC** → Filter cells/genes based on quality metrics
3. **Preprocess** → Normalize, log-transform, select HVGs
4. **Reduce** → PCA → UMAP embedding
5. **Cluster** → Leiden clustering on KNN graph
6. **Annotate** → Identify markers → Manual cell type assignment
7. **Export** → Save annotated h5ad + PDF report

### Key Design Patterns

- **AnnData-centric**: All functions operate on `anndata.AnnData` objects in-place or return modified copies
- **Interactive widgets**: Use `ipywidgets` for live parameter adjustment (QC thresholds, clustering resolution)
- **Checkpoints**: Save intermediate states (`outputs/checkpoints/`) to resume analysis
- **Method selection**: Multiple options for normalization and HVG selection based on dataset characteristics

## Common Development Commands

### Running Tests
```bash
# Currently no formal test suite
# Manual testing via running pipeline.ipynb with test dataset
```

### Code Style
```bash
# No automated linting configured
# Follow existing code style in src/ modules
```

### Adding New Features

**When to add a new function to `src/`:**
- It orchestrates multiple Scanpy calls
- It adds custom logic not available in Scanpy
- It wraps complex external libraries (like Scrublet)
- It provides custom visualization

**When NOT to add to `src/`:**
- Simple Scanpy wrapper with just print statements → Use Scanpy directly in notebook
- One-line helper → Use inline in notebook
- Already exists in Scanpy API → Use Scanpy

**Development workflow:**
1. **Use Scanpy directly in notebook first**
2. **Only extract to module** if function is reused or adds complexity
3. **Keep functions simple** - minimal error handling, clear purpose
4. **Test in notebook** before integrating
5. **Document briefly** - what it does, not how (code should be self-explanatory)

### Common Workflows in Notebook

**Cell filtering:**
```python
# Create boolean mask
mask = (
    (adata.obs['total_counts'] >= min_counts) &
    (adata.obs['total_counts'] <= max_counts) &
    (adata.obs['pct_counts_mt'] <= max_mito)
)
adata = adata[mask].copy()
print(f"Filtered to {adata.n_obs} cells")

# Remove unexpressed genes
sc.pp.filter_genes(adata, min_cells=1)
```

**Normalization & HVG:**
```python
# Store raw counts
adata.layers['counts'] = adata.X.copy()

# Normalize and log-transform
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print("Normalized and log-transformed")

# Select HVGs
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat')
print(f"Selected {adata.var['highly_variable'].sum()} HVGs")
```

**Dimensionality reduction:**
```python
# PCA
sc.tl.pca(adata, n_comps=50, use_highly_variable=True)
print(f"PCA computed: {adata.obsm['X_pca'].shape}")

# Neighbors + UMAP
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
sc.tl.umap(adata, min_dist=0.5)
print("UMAP computed")
```

**Clustering:**
```python
sc.tl.leiden(adata, resolution=1.0)
n_clusters = adata.obs['leiden'].nunique()
print(f"Identified {n_clusters} clusters")

# Visualize
sc.pl.umap(adata, color='leiden', legend_loc='on data')
```

**Marker genes & visualization:**
```python
sc.tl.rank_genes_groups(adata, groupby='leiden', method='wilcoxon')
sc.pl.rank_genes_groups_heatmap(adata, n_genes=10)
sc.pl.umap(adata, color=['CD3D', 'CD79A', 'CD14'], ncols=3)
```

## File Locations

- **Input data**: Place in `data/` (gitignored)
- **Checkpoints**: Saved to `outputs/checkpoints/` (gitignored)
- **Reports**: Generated in `outputs/reports/` (gitignored)
- **Main workflow**: `notebooks/pipeline.ipynb`

## Scanpy API Reference

This pipeline is built on **Scanpy** - a scalable Python toolkit for single-cell analysis that efficiently handles datasets with >1 million cells.

### Key Scanpy Modules Used

- **`scanpy.pp`** (preprocessing): Normalization, filtering, QC metrics, HVG selection, log transformation
- **`scanpy.tl`** (tools): PCA, UMAP, t-SNE, Leiden/Louvain clustering, differential expression, marker genes
- **`scanpy.pl`** (plotting): Scatter plots, heatmaps, violin plots, dot plots, UMAP visualizations

### Important Scanpy Patterns

- **In-place operations**: Most `scanpy.pp.*` functions modify AnnData in-place by default
- **Copy parameter**: Use `copy=True` to return modified copy instead of modifying in-place
- **Key storage conventions**:
  - `adata.X`: Main expression matrix (cells × genes)
  - `adata.obs`: Cell-level metadata (QC metrics, cluster assignments, annotations)
  - `adata.var`: Gene-level metadata (highly variable flags, marker statistics)
  - `adata.obsm`: Multi-dimensional arrays (PCA coordinates in `'X_pca'`, UMAP in `'X_umap'`)
  - `adata.uns`: Unstructured metadata (parameters, neighbor graphs)
  - `adata.layers`: Alternative data representations (raw counts, normalized, scaled)

### Scanpy Documentation & Community

- **Official docs**: https://scanpy.readthedocs.io/en/stable/
- **API reference**: https://scanpy.readthedocs.io/en/stable/api.html
- **Community forum**: discourse.scverse.org
- **GitHub**: https://github.com/scverse/scanpy

## Notes

- **Input**: Pipeline assumes raw count data (not normalized)
- **Raw counts preservation**: Store in `adata.layers['counts']` before normalization
- **HVG flag**: `adata.var['highly_variable']` boolean column
- **Clustering results**: `adata.obs['leiden']` or `adata.obs['leiden_res_{resolution}']`
- **Annotations**: `adata.obs['cell_type']`
- **Embeddings**: `adata.obsm['X_pca']`, `adata.obsm['X_umap']`
- **Neighbor graph**: `adata.obsp['connectivities']`, `adata.obsp['distances']`
- **Marker genes**: `adata.uns['rank_genes_groups']`

## Code Style

- **Concise functions**: Keep functions under 30 lines when possible
- **Minimal error handling**: Let Python/Scanpy errors bubble up naturally
- **Direct over wrapped**: Use `sc.pl.umap()` instead of creating `plot_umap()` wrapper
- **Print for feedback**: Simple print statements for user feedback, not logging
- **Type hints**: Use for function signatures (helps IDE autocomplete)
