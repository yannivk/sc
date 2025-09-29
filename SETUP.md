# Setup Guide

Complete setup instructions for the single-cell RNA-seq analysis pipeline.

## Prerequisites

- macOS, Linux, or Windows (with WSL)
- Conda or Mamba package manager
- At least 16GB RAM (100GB recommended for large datasets)
- 10GB free disk space

## Installation

### 1. Clone or Download Repository

If using git:
```bash
cd /path/to/your/directory
git clone <repository-url> sc_pipeline
cd sc_pipeline
```

Or if you already have the files, navigate to the directory:
```bash
cd /path/to/sc_pipeline
```

### 2. Create Conda Environment

Using conda (slower):
```bash
conda env create -f environment.yml
```

Using mamba (faster, recommended):
```bash
# Install mamba if you don't have it
conda install -n base -c conda-forge mamba

# Create environment
mamba env create -f environment.yml
```

This will create an environment named `sc_pipeline` with all required packages.

### 3. Activate Environment

```bash
conda activate sc_pipeline
```

### 4. Enable Jupyter Widgets

For interactive widgets to work properly:
```bash
jupyter nbextension enable --py widgetsnbextension --sys-prefix
```

### 5. Verify Installation

Test that key packages are installed:
```bash
python -c "import scanpy, anndata, scrublet, ipywidgets; print('✓ All packages installed')"
```

## Directory Structure

After setup, your directory should look like:
```
sc_pipeline/
├── environment.yml          # Conda environment specification
├── requirements.txt         # Pip requirements (alternative)
├── README.md               # Main documentation
├── SETUP.md               # This file
├── .gitignore             # Git ignore rules
├── notebooks/
│   └── pipeline.ipynb      # Main interactive pipeline
├── src/                    # Python modules
│   ├── __init__.py
│   ├── io.py               # Data loading
│   ├── qc.py               # Quality control
│   ├── preprocessing.py    # Normalization & scaling
│   ├── reduction.py        # PCA, UMAP
│   ├── clustering.py       # Clustering
│   ├── annotation.py       # Marker genes & annotation
│   ├── visualization.py    # Plotting
│   ├── interactive.py      # Jupyter widgets
│   └── reports.py          # PDF generation
├── data/                   # Place your data here (gitignored)
└── outputs/                # Pipeline outputs (gitignored)
    ├── checkpoints/        # Saved .h5ad files
    └── reports/            # PDF reports
```

## Preparing Your Data

### Supported Input Formats

1. **AnnData (.h5ad)**: Already in correct format
2. **Seurat (.rds)**: R Seurat objects (requires rpy2)
3. **10X Genomics (matrix.mtx)**: Standard 10X output

### Data Location

Place your data in the `data/` directory:
```bash
mkdir -p data
cp /path/to/your/data.h5ad data/
```

### Example Data

If you don't have data, you can download a test dataset:
```bash
# Example: PBMC 3k dataset from 10X Genomics
mkdir -p data
cd data
wget http://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz
tar -xzf pbmc3k_filtered_gene_bc_matrices.tar.gz
cd ..
```

Then load in pipeline with:
```python
adata = io.load_10x_mtx('data/filtered_gene_bc_matrices/hg19/')
```

## Running the Pipeline

### 1. Launch Jupyter Notebook

```bash
cd notebooks
jupyter notebook
```

This will open a browser window with Jupyter.

### 2. Open Pipeline Notebook

Click on `pipeline.ipynb` to open the main pipeline.

### 3. Execute Cells

Run cells sequentially:
- Click on a cell
- Press `Shift + Enter` to execute
- Follow instructions in markdown cells

### 4. Interactive Widgets

When you encounter interactive widgets:
- Adjust sliders/inputs
- View live updates
- Click confirmation buttons
- Continue to next cell

## Common Issues & Solutions

### Issue: Jupyter widgets not displaying

**Solution:**
```bash
conda activate sc_pipeline
jupyter nbextension enable --py widgetsnbextension --sys-prefix
jupyter notebook
```

### Issue: "Module not found" errors

**Solution:**
Make sure you're in the correct directory and environment:
```bash
conda activate sc_pipeline
cd /path/to/sc_pipeline/notebooks
jupyter notebook
```

### Issue: RDS files not loading

**Solution:**
R integration requires additional setup:
```bash
# Install R (if not already installed)
conda install -c conda-forge r-base r-seurat

# Test R integration
python -c "import rpy2; print('✓ rpy2 working')"
```

### Issue: Out of memory errors

**Solution:**
- Use a smaller subset of data for testing
- Reduce `n_top_genes` in HVG selection
- Reduce `n_comps` in PCA
- Process on a machine with more RAM

### Issue: Slow performance

**Solution:**
- Use mamba instead of conda for faster package management
- Reduce dataset size for testing
- Use fewer PCA components
- Set `n_jobs` parameter in functions that support it

## Git Setup (Optional)

If you want to version control your analysis:

### 1. Initialize Git Repository

```bash
cd /path/to/sc_pipeline
git init
```

### 2. Important: Don't Commit Data

The `.gitignore` file is already configured to exclude:
- `data/` directory (raw data)
- `outputs/` directory (results)
- `.h5ad`, `.rds`, `.mtx` files
- Jupyter checkpoints

### 3. Commit Your Work

```bash
git add notebooks/pipeline.ipynb
git add src/
git commit -m "Initial pipeline setup"
```

### 4. Connect to Remote (Optional)

```bash
git remote add origin <your-repo-url>
git push -u origin main
```

## Sharing with Colleagues

### Option 1: Share Full Directory

Zip the entire directory (excluding data):
```bash
# Create archive without data
tar -czf sc_pipeline.tar.gz \
  --exclude='data' \
  --exclude='outputs' \
  --exclude='.git' \
  sc_pipeline/
```

Send `sc_pipeline.tar.gz` to colleagues.

### Option 2: Git Repository

1. Push to GitHub/GitLab
2. Colleagues clone:
```bash
git clone <repository-url>
cd sc_pipeline
conda env create -f environment.yml
conda activate sc_pipeline
```

### Option 3: Share Environment Only

If colleagues have the code, they just need the environment:
```bash
# They run:
conda env create -f environment.yml
conda activate sc_pipeline
```

## Updating the Pipeline

To update packages:
```bash
conda activate sc_pipeline
conda update --all
```

To update specific package:
```bash
conda update scanpy
```

## Uninstalling

To remove the environment:
```bash
conda deactivate
conda env remove -n sc_pipeline
```

## Getting Help

1. Check error messages carefully
2. Refer to documentation in README.md
3. Check package documentation:
   - Scanpy: https://scanpy.readthedocs.io/
   - AnnData: https://anndata.readthedocs.io/
4. Create an issue in the repository (if using git)

## Next Steps

After successful setup:
1. Read the README.md for pipeline overview
2. Open `notebooks/pipeline.ipynb`
3. Follow the step-by-step workflow
4. Customize parameters for your data
5. Save results and generate reports

Happy analyzing! 🔬