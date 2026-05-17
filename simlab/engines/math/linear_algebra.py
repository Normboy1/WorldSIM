"""Linear algebra engine using NumPy.

Provides system solving, eigenvalues, SVD, determinant, inverse,
matrix multiplication, and rank computation.
"""

from __future__ import annotations

import numpy as np


class LinearAlgebraEngine:
    """NumPy-backed linear algebra operations."""

    @staticmethod
    def _to_array(M: list | np.ndarray) -> np.ndarray:
        return np.array(M, dtype=float)

    def solve_linear_system(self, A: list, b: list) -> dict:
        """Solve the linear system Ax = b.

        Returns the solution vector and condition number.
        """
        A_arr = self._to_array(A)
        b_arr = self._to_array(b)
        try:
            x = np.linalg.solve(A_arr, b_arr)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Linear system has no unique solution (singular or incompatible A and b)."
            ) from exc
        cond = float(np.linalg.cond(A_arr))
        residual = float(np.linalg.norm(A_arr @ x - b_arr))
        return {
            "solution": x.tolist(),
            "condition_number": cond,
            "residual": residual,
        }

    def compute_eigenvalues(self, A: list) -> dict:
        """Compute eigenvalues and eigenvectors of a square matrix."""
        A_arr = self._to_array(A)
        eigenvalues, eigenvectors = np.linalg.eig(A_arr)
        # Sort by real part of eigenvalue
        idx = np.argsort(eigenvalues.real)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        return {
            "eigenvalues": [complex(v) if v.imag != 0 else float(v.real) for v in eigenvalues],
            "eigenvalues_real": eigenvalues.real.tolist(),
            "eigenvalues_imag": eigenvalues.imag.tolist(),
            "eigenvectors": eigenvectors.tolist(),
            "is_symmetric": bool(np.allclose(A_arr, A_arr.T)),
            "spectral_radius": float(np.max(np.abs(eigenvalues))),
        }

    def compute_svd(self, A: list) -> dict:
        """Compute the Singular Value Decomposition of A = U @ diag(S) @ Vt."""
        A_arr = self._to_array(A)
        U, S, Vt = np.linalg.svd(A_arr, full_matrices=True)
        rank = int(np.sum(S > 1e-10))
        return {
            "U": U.tolist(),
            "singular_values": S.tolist(),
            "Vt": Vt.tolist(),
            "rank": rank,
            "condition_number": float(S[0] / S[-1]) if S[-1] > 1e-15 else float("inf"),
        }

    def compute_determinant(self, A: list) -> float:
        """Return the determinant of a square matrix."""
        A_arr = self._to_array(A)
        return float(np.linalg.det(A_arr))

    def compute_inverse(self, A: list) -> dict:
        """Compute the matrix inverse (raises if singular)."""
        A_arr = self._to_array(A)
        det = float(np.linalg.det(A_arr))
        if abs(det) < 1e-15:
            raise ValueError(f"Matrix is singular (det ≈ {det:.3e}); inverse does not exist.")
        inv = np.linalg.inv(A_arr)
        verification = float(np.linalg.norm(A_arr @ inv - np.eye(A_arr.shape[0])))
        return {
            "inverse": inv.tolist(),
            "determinant": det,
            "verification_error": verification,
        }

    def matrix_multiply(self, A: list, B: list) -> dict:
        """Multiply two matrices A @ B."""
        A_arr = self._to_array(A)
        B_arr = self._to_array(B)
        if A_arr.shape[1] != B_arr.shape[0]:
            raise ValueError(
                f"Shape mismatch: ({A_arr.shape}) @ ({B_arr.shape}) is invalid."
            )
        C = A_arr @ B_arr
        return {
            "result": C.tolist(),
            "shape": list(C.shape),
            "frobenius_norm": float(np.linalg.norm(C, "fro")),
        }

    def compute_rank(self, A: list) -> int:
        """Return the rank of matrix A."""
        A_arr = self._to_array(A)
        return int(np.linalg.matrix_rank(A_arr))

    def qr_decomposition(self, A: list) -> dict:
        """Compute QR decomposition of A = Q @ R."""
        A_arr = self._to_array(A)
        Q, R = np.linalg.qr(A_arr)
        return {
            "Q": Q.tolist(),
            "R": R.tolist(),
            "verification_error": float(np.linalg.norm(Q @ R - A_arr)),
        }

    def compute_trace(self, A: list) -> float:
        """Return the trace (sum of diagonal) of a square matrix."""
        A_arr = self._to_array(A)
        return float(np.trace(A_arr))

    def compute_norm(self, A: list, order: str = "frobenius") -> float:
        """Return the matrix norm."""
        A_arr = self._to_array(A)
        norm_map = {
            "frobenius": "fro",
            "fro": "fro",
            "nuclear": "nuc",
            "2": 2,
            "1": 1,
            "inf": np.inf,
        }
        ord_val = norm_map.get(str(order), "fro")
        return float(np.linalg.norm(A_arr, ord_val))
