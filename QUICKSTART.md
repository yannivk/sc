# Quick Start Guide

Get started with the pipeline in 5 minutes.

## 1. Setup (One-time)

```bash
# Navigate to pipeline directory
cd /path/to/sc_pipeline

# Create environment
conda env create -f environment.yml

# Activate environment
conda activate sc_pipeline

# Enable Jupyter widgets
jupyter nbextension enable --py widgetsnbextension
```

## 2. Prepare Your Data

Place your data file in the `data/` directory:
```bash
cp /path/to/your_data.h5ad data/
```

Supported formats: `.h5ad`, `.rds`, or 10X `matrix.mtx` folder

## 3. Launch Pipeline

```bash
cd notebooks
jupyter notebook
```

Open `pipeline.ipynb` in the browser that opens.

## 4. Run Analysis

Execute cells sequentially (Shift + Enter):

### Core Workflow:

1. **Load Data** (Cell 2)
   ```python
   adata = io.load_data('../data/your_data.h5ad')
   ```

2. **QC Filtering** (Cell 6)
   - Adjust sliders interactively
   - Click "Apply Filters"

3. **Doublet Removal** (Cell 10)
   - Adjust threshold
   - Click "Remove Doublets"

4. **Normalization** (Cell 12)
   ```python
   preprocessing.normalize_data(adata, method='log1p')
   ```

5. **Feature Selection** (Cell 14)
   ```python
   preprocessing.select_highly_variable_genes(adata, n_top_genes=2000)
   ```

6. **Dimensionality Reduction** (Cells 18-20)
   - PCA → Select components → UMAP

7. **Clustering** (Cell 25)
   - View multiple resolutions
   - Select optimal resolution

8. **Gene Visualization** (Cell 30)
   - Enter gene names
   - Click "Plot Genes"

9. **Marker Genes** (Cells 32-35)
   - Automatic identification
   - View heatmaps & dotplots

10. **Annotation** (Cell 37)
    - For each cluster:
      - Select cluster
      - Click "Show Markers"
      - Enter cell type
      - Click "Annotate Cluster"
    - Click "Finish Annotation"

11. **Save & Report** (Cells 42-44)
    ```python
    io.save_checkpoint(adata, 'annotated')
    reports.create_analysis_report(adata)
    ```

## 5. View Results

Your outputs are in:
- `outputs/checkpoints/annotated.h5ad` - Annotated data
- `outputs/reports/analysis_report.pdf` - Full report
- `outputs/cell_annotations.csv` - Cell type labels

## Key Interactive Widgets

### QC Filter Widget
```python
qc_widget = interactive.QCFilterWidget(adata)
qc_widget.display()
# Adjust → Click "Apply Filters"
adata = qc_widget.get_filtered_data()
```

### Gene Visualization Widget
```python
gene_widget = interactive.GeneVisualizationWidget(adata)
gene_widget.display()
# Enter genes → Click "Plot Genes"
```

### Annotation Widget
```python
annotation_widget = interactive.AnnotationWidget(adata)
annotation_widget.display()
# Select cluster → Show markers → Enter type → Annotate
```

## Example Dataset

Test with PBMC 3k dataset:
```bash
cd data
wget http://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz
tar -xzf pbmc3k_filtered_gene_bc_matrices.tar.gz
cd ..
```

In notebook:
```python
adata = io.load_10x_mtx('data/filtered_gene_bc_matrices/hg19/')
```

## Typical Runtime

For a dataset with 10,000 cells:
- QC & Filtering: 1-2 min
- Normalization: 30 sec
- PCA: 30 sec
- UMAP: 1-2 min
- Clustering: 30 sec
- Marker genes: 2-5 min
- Manual annotation: 5-15 min (depends on you!)

**Total: ~15-30 minutes**

## Tips

1. **Start with defaults** - They're based on best practices
2. **Save checkpoints** - Use `io.save_checkpoint()` frequently
3. **Explore interactively** - Adjust parameters, see live updates
4. **Document decisions** - Add markdown cells explaining choices
5. **Generate reports** - PDF reports are great for sharing

## Common Parameters

### QC Filtering
- **Min/Max counts**: 500 - 20,000 (typical)
- **Min/Max genes**: 200 - 5,000 (typical)
- **Max mito %**: 5-20% (depends on tissue)

### HVG Selection
- **n_top_genes**: 2000 (standard), 3000-4000 (complex datasets)

### PCA
- **n_pcs**: 30-50 (usually 30 is sufficient)

### UMAP
- **min_dist**: 0.3-0.5 (0.5 is good default)
- **n_neighbors**: 10-30 (15 is good default)

### Clustering
- **Resolution**:
  - 0.25-0.5: Broad cell types
  - 1.0: Standard granularity
  - 1.5-2.0: Fine-grained subtypes

## Troubleshooting

**Widget not showing?**
```bash
jupyter nbextension enable --py widgetsnbextension
```

**Import errors?**
```bash
conda activate sc_pipeline
```

**Slow performance?**
- Reduce dataset size for testing
- Lower n_top_genes (e.g., 1000)
- Use fewer PCA components (e.g., 20)

## Next Steps

- See `SETUP.md` for detailed installation
- See `README.md` for full documentation
- Customize pipeline for your specific needs
- Share results with `reports.create_analysis_report()`

---

**Questions?** Check the full documentation or create an issue.

Happy analyzing! 🧬