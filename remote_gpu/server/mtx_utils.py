import os
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
import hashlib
import json


class MTXConverter:
    """Convert MTX format to CSR format for SpMV execution"""

    def __init__(self, storage_dir: str = "./mtx_cache"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.csr_cache_dir = self.storage_dir / "csr_cache"
        self.csr_cache_dir.mkdir(exist_ok=True)

    def get_mtx_path(self, filename: str) -> Path:
        """Get path for MTX file"""
        return self.storage_dir / filename

    def get_csr_cache_path(self, filename: str) -> Path:
        """Get path for cached CSR data"""
        base_name = Path(filename).stem
        return self.csr_cache_dir / f"{base_name}.npz"

    def mtx_exists(self, filename: str) -> bool:
        """Check if MTX file exists in cache"""
        return self.get_mtx_path(filename).exists()

    def save_mtx_file(self, filename: str, content: bytes) -> Path:
        """Save MTX file to cache"""
        path = self.get_mtx_path(filename)
        with open(path, 'wb') as f:
            f.write(content)
        return path

    def read_mtx_file(self, filename: str) -> Tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray]:
        """
        Read MTX file and return COO format data

        Returns:
            num_rows, num_cols, nnz, row_indices, col_indices, values
        """
        mtx_path = self.get_mtx_path(filename)

        if not mtx_path.exists():
            raise FileNotFoundError(f"MTX file not found: {filename}")

        # Read MTX file
        with open(mtx_path, 'r') as f:
            # Skip comments
            for line in f:
                if not line.startswith('%'):
                    # First non-comment line contains dimensions
                    parts = line.strip().split()
                    num_rows = int(parts[0])
                    num_cols = int(parts[1])
                    nnz = int(parts[2])
                    break

            # Read matrix entries
            row_indices = np.zeros(nnz, dtype=np.int32)
            col_indices = np.zeros(nnz, dtype=np.int32)
            values = np.zeros(nnz, dtype=np.float32)

            for i, line in enumerate(f):
                if i >= nnz:
                    break
                parts = line.strip().split()
                # MTX format is 1-indexed, convert to 0-indexed
                row_indices[i] = int(parts[0]) - 1
                col_indices[i] = int(parts[1]) - 1
                values[i] = float(parts[2])

        return num_rows, num_cols, nnz, row_indices, col_indices, values

    def coo_to_csr(self, num_rows: int, num_cols: int, nnz: int,
                   row_indices: np.ndarray, col_indices: np.ndarray,
                   values: np.ndarray) -> Dict[str, Any]:
        """
        Convert COO format to CSR format

        Returns:
            Dictionary with CSR format data
        """
        # Sort by row then column
        indices = np.lexsort((col_indices, row_indices))
        row_indices = row_indices[indices]
        col_indices = col_indices[indices]
        values = values[indices]

        # Build CSR row pointer
        row_ptr = np.zeros(num_rows + 1, dtype=np.int32)
        for i in range(nnz):
            row_ptr[row_indices[i] + 1] += 1
        row_ptr = np.cumsum(row_ptr)

        return {
            'num_rows': int(num_rows),
            'num_cols': int(num_cols),
            'nnz': int(nnz),
            'values': values,
            'col_indices': col_indices.astype(np.int32),
            'row_ptr': row_ptr
        }

    def load_csr_from_cache(self, filename: str) -> Dict[str, Any]:
        """Load CSR data from cache"""
        cache_path = self.get_csr_cache_path(filename)

        if not cache_path.exists():
            return None

        data = np.load(cache_path)
        return {
            'num_rows': int(data['num_rows']),
            'num_cols': int(data['num_cols']),
            'nnz': int(data['nnz']),
            'values': data['values'],
            'col_indices': data['col_indices'],
            'row_ptr': data['row_ptr']
        }

    def save_csr_to_cache(self, filename: str, csr_data: Dict[str, Any]):
        """Save CSR data to cache"""
        cache_path = self.get_csr_cache_path(filename)

        np.savez_compressed(
            cache_path,
            num_rows=csr_data['num_rows'],
            num_cols=csr_data['num_cols'],
            nnz=csr_data['nnz'],
            values=csr_data['values'],
            col_indices=csr_data['col_indices'],
            row_ptr=csr_data['row_ptr']
        )

    def convert_mtx_to_csr(self, filename: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Convert MTX file to CSR format

        Args:
            filename: MTX file name
            use_cache: Whether to use cached CSR data if available

        Returns:
            Dictionary with CSR format data
        """
        # Try to load from cache first
        if use_cache:
            csr_data = self.load_csr_from_cache(filename)
            if csr_data is not None:
                return csr_data

        # Read MTX file
        num_rows, num_cols, nnz, row_indices, col_indices, values = self.read_mtx_file(filename)

        # Convert to CSR
        csr_data = self.coo_to_csr(num_rows, num_cols, nnz, row_indices, col_indices, values)

        # Save to cache
        if use_cache:
            self.save_csr_to_cache(filename, csr_data)

        return csr_data

    def get_matrix_info(self, filename: str) -> Dict[str, Any]:
        """Get basic information about MTX file"""
        if not self.mtx_exists(filename):
            return None

        mtx_path = self.get_mtx_path(filename)

        # Read just the header
        with open(mtx_path, 'r') as f:
            for line in f:
                if not line.startswith('%'):
                    parts = line.strip().split()
                    return {
                        'filename': filename,
                        'num_rows': int(parts[0]),
                        'num_cols': int(parts[1]),
                        'nnz': int(parts[2]),
                        'exists': True
                    }

        return None

    def list_cached_matrices(self) -> list:
        """List all cached MTX files"""
        matrices = []
        for mtx_file in self.storage_dir.glob("*.mtx"):
            info = self.get_matrix_info(mtx_file.name)
            if info:
                matrices.append(info)
        return matrices
