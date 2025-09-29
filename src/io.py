"""
Data I/O module for loading different scRNA-seq file formats
"""

import os
from pathlib import Path
from typing import Union, Optional
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np


def load_h5ad(file_path: Union[str, Path]) -> ad.AnnData:
    """
    Load AnnData object from h5ad file

    Parameters
    ----------
    file_path : str or Path
        Path to .h5ad file

    Returns
    -------
    adata : AnnData
        Loaded AnnData object
    """
    print(f"Loading h5ad file: {file_path}")
    adata = sc.read_h5ad(file_path)
    print(f"Loaded {adata.n_obs} cells × {adata.n_vars} genes")
    return adata


def load_rds(file_path: Union[str, Path],
             assay: str = "RNA",
             slot: str = "counts") -> ad.AnnData:
    """
    Load Seurat object from RDS file and convert to AnnData

    Parameters
    ----------
    file_path : str or Path
        Path to .rds file containing Seurat object
    assay : str, default="RNA"
        Seurat assay to extract
    slot : str, default="counts"
        Seurat slot to extract (counts, data, or scale.data)

    Returns
    -------
    adata : AnnData
        Converted AnnData object
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter

        print(f"Loading RDS file: {file_path}")

        # Activate pandas conversion
        pandas2ri.activate()

        # Load R libraries and RDS file
        ro.r(f'''
        library(Seurat)
        library(SeuratObject)
        seurat_obj <- readRDS("{file_path}")
        ''')

        # Extract count matrix
        count_matrix = ro.r(f'''
        as.matrix(GetAssayData(seurat_obj, assay = "{assay}", slot = "{slot}"))
        ''')

        # Extract metadata
        metadata = ro.r('seurat_obj@meta.data')

        # Convert to pandas/numpy
        with localconverter(ro.default_converter + pandas2ri.converter):
            meta_df = ro.conversion.rpy2py(metadata)
            counts = np.array(count_matrix)
            gene_names = list(ro.r('rownames(seurat_obj)'))
            cell_names = list(ro.r('colnames(seurat_obj)'))

        # Create AnnData object
        adata = ad.AnnData(
            X=counts.T,  # Transpose: AnnData expects cells × genes
            obs=meta_df,
            var=pd.DataFrame(index=gene_names)
        )

        print(f"Loaded {adata.n_obs} cells × {adata.n_vars} genes")
        pandas2ri.deactivate()

        return adata

    except ImportError:
        raise ImportError(
            "rpy2 is required to load RDS files. "
            "Install with: conda install -c conda-forge rpy2"
        )
    except Exception as e:
        raise RuntimeError(f"Error loading RDS file: {str(e)}")


def load_10x_mtx(data_dir: Union[str, Path],
                 var_names: str = 'gene_symbols',
                 make_unique: bool = True) -> ad.AnnData:
    """
    Load 10X Genomics data from MTX format

    Expected directory structure:
    data_dir/
        matrix.mtx (or matrix.mtx.gz)
        barcodes.tsv (or barcodes.tsv.gz)
        genes.tsv or features.tsv (or .gz versions)

    Parameters
    ----------
    data_dir : str or Path
        Directory containing matrix.mtx, barcodes.tsv, and genes/features.tsv
    var_names : str, default='gene_symbols'
        Column to use as variable names ('gene_symbols' or 'gene_ids')
    make_unique : bool, default=True
        Make variable names unique by appending gene IDs

    Returns
    -------
    adata : AnnData
        Loaded AnnData object
    """
    print(f"Loading 10X MTX data from: {data_dir}")

    adata = sc.read_10x_mtx(
        data_dir,
        var_names=var_names,
        make_unique=make_unique
    )

    print(f"Loaded {adata.n_obs} cells × {adata.n_vars} genes")
    return adata


def load_data(file_path: Union[str, Path],
              file_type: Optional[str] = None,
              **kwargs) -> ad.AnnData:
    """
    Auto-detect and load scRNA-seq data from various formats

    Parameters
    ----------
    file_path : str or Path
        Path to file or directory
    file_type : str, optional
        Force specific file type: 'h5ad', 'rds', or 'mtx'
        If None, auto-detect from file extension
    **kwargs
        Additional arguments passed to specific loader functions

    Returns
    -------
    adata : AnnData
        Loaded AnnData object
    """
    file_path = Path(file_path)

    # Auto-detect file type if not specified
    if file_type is None:
        if file_path.suffix == '.h5ad':
            file_type = 'h5ad'
        elif file_path.suffix == '.rds':
            file_type = 'rds'
        elif file_path.is_dir():
            # Check if directory contains MTX files
            mtx_files = list(file_path.glob('*.mtx*'))
            if mtx_files:
                file_type = 'mtx'
            else:
                raise ValueError(f"No MTX files found in directory: {file_path}")
        else:
            raise ValueError(
                f"Cannot auto-detect file type for: {file_path}. "
                "Please specify file_type parameter."
            )

    # Load based on file type
    if file_type == 'h5ad':
        return load_h5ad(file_path, **kwargs)
    elif file_type == 'rds':
        return load_rds(file_path, **kwargs)
    elif file_type == 'mtx':
        return load_10x_mtx(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def save_checkpoint(adata: ad.AnnData,
                   name: str,
                   output_dir: Union[str, Path] = "../outputs/checkpoints") -> None:
    """
    Save AnnData object as checkpoint

    Parameters
    ----------
    adata : AnnData
        AnnData object to save
    name : str
        Checkpoint name (e.g., 'raw', 'preprocessed', 'annotated')
    output_dir : str or Path, default="../outputs/checkpoints"
        Directory to save checkpoint
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{name}.h5ad"
    print(f"Saving checkpoint: {output_path}")
    adata.write_h5ad(output_path)
    print(f"Checkpoint saved: {name}.h5ad")


def load_checkpoint(name: str,
                   checkpoint_dir: Union[str, Path] = "../outputs/checkpoints") -> ad.AnnData:
    """
    Load AnnData object from checkpoint

    Parameters
    ----------
    name : str
        Checkpoint name (e.g., 'raw', 'preprocessed', 'annotated')
    checkpoint_dir : str or Path, default="../outputs/checkpoints"
        Directory containing checkpoints

    Returns
    -------
    adata : AnnData
        Loaded AnnData object
    """
    checkpoint_path = Path(checkpoint_dir) / f"{name}.h5ad"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    adata = sc.read_h5ad(checkpoint_path)
    print(f"Loaded {adata.n_obs} cells × {adata.n_vars} genes")
    return adata