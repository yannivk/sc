# Single-Cell RNA-seq Analysis Pipeline

An interactive, best-practices-based pipeline for single-cell RNA-seq preprocessing, dimensionality reduction, clustering, and annotation.

## Features

- **Multiple input formats**: h5ad, rds (Seurat), mtx (10X)
- **Interactive decision points**: QC filtering, normalization, clustering resolution, annotation
- **Best practices**: Based on sc-best-practices.org and cutting-edge methods (2024-2025)
- **Manual annotation**: Cluster-based with marker gene visualization
- **Cluster manipulation**: Split and merge clusters interactively
- **Quality control**: Comprehensive QC metrics with PDF reports
- **Checkpoints**: Save at key stages (raw, preprocessed, annotated)

## Setup

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate sc_pipeline
```

Alternatively, with mamba (faster):

```bash
mamba env create -f environment.yml
mamba activate sc_pipeline
```

### 2. Install Jupyter Extensions

```bash
jupyter nbextension enable --py widgetsnbextension
```

### 3. Launch Pipeline

```bash
cd notebooks
jupyter notebook pipeline.ipynb
```

## Usage

1. **Load data**: Select input format (h5ad/rds/mtx) and load your dataset
2. **QC & Filtering**: View QC plots and interactively set filtering thresholds
3. **Doublet Detection**: Identify and remove doublets
4. **Normalization**: Choose normalization method
5. **Feature Selection**: Select highly variable genes
6. **Dimensionality Reduction**: PCA and UMAP with parameter tuning
7. **Clustering**: Test multiple resolutions and select optimal clustering
8. **Gene Visualization**: View expression of specific genes on UMAP
9. **Cluster Manipulation**: Split/merge clusters as needed
10. **Annotation**: Manually annotate cell types based on marker genes
11. **Subclustering**: Optionally subcluster specific populations
12. **Export**: Save annotated dataset and generate PDF report

## Pipeline Methods

### Quality Control
- MAD-based filtering (median absolute deviation)
- Mitochondrial content filtering
- Doublet detection (Scrublet)
- Ambient RNA awareness

### Normalization
- Log transformation (default)
- Scran normalization (batch correction)
- Pearson residuals (rare cell types)

### Feature Selection
- Seurat method (standard)
- Deviance-based (cutting-edge, recommended)

### Dimensionality Reduction
- PCA for initial reduction
- UMAP for visualization
- Leiden clustering on KNN graph

### Annotation
- Top N marker genes per cluster
- Custom gene visualization
- Manual cell type assignment

## Outputs

### Checkpoints (outputs/checkpoints/)
- `raw.h5ad`: Initial loaded data
- `preprocessed.h5ad`: After QC and preprocessing
- `annotated.h5ad`: Final annotated dataset

### Reports (outputs/reports/)
- `qc_report.pdf`: Quality control plots
- `analysis_report.pdf`: Final UMAP with annotations

## Project Structure

```
sc_pipeline/
├── environment.yml          # Conda environment
├── requirements.txt         # Pip packages
├── README.md               # This file
├── notebooks/
│   └── pipeline.ipynb      # Main interactive pipeline
├── src/
│   ├── io.py               # Data loading
│   ├── qc.py               # Quality control
│   ├── preprocessing.py    # Normalization, scaling
│   ├── reduction.py        # PCA, UMAP
│   ├── clustering.py       # Clustering & manipulation
│   ├── annotation.py       # Marker genes & annotation
│   ├── visualization.py    # Plotting functions
│   ├── interactive.py      # Widget interfaces
│   └── reports.py          # PDF generation
├── data/                   # Place your input data here
└── outputs/                # Pipeline outputs
```

## Citation

If you use this pipeline, please cite:

- **Scanpy**: Wolf, F. A., et al. (2018). SCANPY: large-scale single-cell gene expression data analysis. Genome Biology, 19(1), 15.
- **Best Practices**: Heumos, L., et al. (2023). Best practices for single-cell analysis across modalities. Nature Reviews Genetics, 24(8), 550-572.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT License