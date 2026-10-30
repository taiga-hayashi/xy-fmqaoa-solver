"""Optical multilayer thin-film evaluator and exhaustive truth-table builder.

Supports both a zero-dependency vectorized NumPy transfer-matrix method (TMM)
and the vendored `tmm_fast` implementation under `data/inputs/mlfilter/v1-public-code`.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import PROJECT_ROOT, STUDY_DIR, atomic_write_json, sha256_file, utc_now

BUNDLED_INPUT_DIR = STUDY_DIR / "inputs"
PROJECT_INPUT_DIR = PROJECT_ROOT / "data/inputs/mlfilter/v1-public-code"
DEFAULT_INPUT_DIR = BUNDLED_INPUT_DIR if (BUNDLED_INPUT_DIR / "manifest.json").exists() else PROJECT_INPUT_DIR
MAX_LAYERS = 10


def verify_optical_inputs(input_dir: Path = DEFAULT_INPUT_DIR) -> str:
    manifest_path = input_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for item in manifest["files"]:
        actual = sha256_file(input_dir / item["path"])
        if actual != item["sha256"]:
            raise ValueError(f"Checksum mismatch for {item['path']}: {actual} != {item['sha256']}")
    return sha256_file(manifest_path)


def _coh_tmm_normal_numpy(
    n_stack: np.ndarray,
    thickness_m: float,
    wavelength_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact normal-incidence coherent TMM for lossless dispersive stacks.

    Args:
        n_stack: shape (batch, L + 2, n_wl), real refractive indices including
            semi-infinite ambient n[:, 0, :] and substrate n[:, -1, :].
        thickness_m: physical thickness (meters) of each interior film layer.
        wavelength_m: shape (n_wl,), vacuum wavelengths in meters.

    Returns:
        (T, R) power transmittance and reflectance of shape (batch, n_wl).
    """
    batch, total_layers, n_wl = n_stack.shape
    layers = total_layers - 2
    # Interface Fresnel coefficients at normal incidence: r_{j,j+1}, t_{j,j+1}
    n_left = n_stack[:, :-1, :]
    n_right = n_stack[:, 1:, :]
    denom = n_left + n_right
    r_int = (n_left - n_right) / denom
    inv_t = denom / (2.0 * n_left)

    # Start with interface 0 -> 1 matrix M = (1 / t_01) [[1, r_01], [r_01, 1]]
    m00 = inv_t[:, 0, :].astype(np.complex128)
    m01 = (inv_t[:, 0, :] * r_int[:, 0, :]).astype(np.complex128)
    m10 = m01.copy()
    m11 = m00.copy()

    two_pi_d_over_wl = (2.0 * np.pi * thickness_m) / wavelength_m[None, :]
    for k in range(1, layers + 1):
        delta = n_stack[:, k, :] * two_pi_d_over_wl
        p_neg = np.exp(-1j * delta)
        p_pos = np.conj(p_neg)
        it = inv_t[:, k, :]
        ri = r_int[:, k, :]
        # Multiply M @ diag(p_neg, p_pos) @ (it * [[1, ri], [ri, 1]])
        a00 = p_neg * it
        a01 = a00 * ri
        a11 = p_pos * it
        a10 = a11 * ri

        n00 = m00 * a00 + m01 * a10
        n01 = m00 * a01 + m01 * a11
        n10 = m10 * a00 + m11 * a10
        n11 = m10 * a01 + m11 * a11
        m00, m01, m10, m11 = n00, n01, n10, n11

    t_amp = 1.0 / m00
    r_amp = m10 / m00
    trans = (n_stack[:, -1, :] / n_stack[:, 0, :]) * (np.abs(t_amp) ** 2)
    refl = np.abs(r_amp) ** 2
    return trans, refl


class FilterEvaluator:
    """Evaluates solar-weighted figure of merit (FOM) for 4-material optical filters."""

    materials = ("SiO2", "Si3N4", "Al2O3", "TiO2")

    def __init__(
        self,
        layers: int,
        *,
        input_dir: Path = DEFAULT_INPUT_DIR,
        tmm_backend: str = "numpy",
    ) -> None:
        if isinstance(layers, bool) or not isinstance(layers, (int, np.integer)) or not (1 <= layers <= MAX_LAYERS):
            raise ValueError(f"layers must be an integer in 1..{MAX_LAYERS}")
        if tmm_backend not in ("numpy", "tmm_fast"):
            raise ValueError("tmm_backend must be 'numpy' or 'tmm_fast'")
        self.layers = int(layers)
        self.input_dir = Path(input_dir)
        self.tmm_backend = tmm_backend
        self.input_manifest_sha256 = verify_optical_inputs(self.input_dir)

        ref = self.input_dir / "reference"
        # Match public-code float32 parsing followed by float64 promotion.
        self.index = np.loadtxt(ref / "dielectric_ref.txt", delimiter=",").astype(np.float32).astype(float)
        self.solar = np.loadtxt(ref / "solar_spectrum.txt", delimiter=",").astype(np.float32).astype(float)
        if self.index.shape != (4, 2201) or self.solar.shape != (2201,):
            raise ValueError("Unexpected reference spectrum dimensions")
        self.wavelength_nm = np.arange(300, 2501, dtype=float)
        self.wavelength = np.linspace(300, 2500, 2201) * 1e-9
        self.ideal = ((self.wavelength_nm >= 400) & (self.wavelength_nm <= 750)).astype(float)
        self.solar_norm = float(np.sum(self.solar.astype(np.float32) ** 2))

    def validate_labels(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values)
        if arr.ndim not in (1, 2) or arr.shape[-1] != self.layers:
            raise ValueError("One material index per layer required")
        if arr.dtype.kind not in "iu" or np.any((arr < 0) | (arr > 3)):
            raise ValueError("Material indices must be integers in 0..3")
        return arr.reshape(-1, self.layers).astype(np.int64)

    def spectra(self, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        arr = self.validate_labels(labels)
        n_stack = np.empty((len(arr), self.layers + 2, 2201), dtype=float)
        n_stack[:, 0, :] = 1.4
        n_stack[:, -1, :] = 1.45
        n_stack[:, 1:-1, :] = self.index[arr]

        if self.tmm_backend == "numpy":
            trans, refl = _coh_tmm_normal_numpy(n_stack, 50e-9, self.wavelength)
        else:
            vendor_candidates = [
                self.input_dir / "vendor/site-packages",
                PROJECT_INPUT_DIR / "vendor/site-packages",
            ]
            for vc in vendor_candidates:
                if vc.exists() and str(vc) not in sys.path:
                    sys.path.insert(0, str(vc))
            from tmm_fast import coh_tmm

            thickness = np.full((len(arr), self.layers + 2), 50e-9)
            thickness[:, [0, -1]] = np.inf
            res = coh_tmm("p", n_stack, thickness, np.array([0.0]), self.wavelength, device="cpu")
            trans, refl = res["T"][:, 0, :], res["R"][:, 0, :]

        if not np.isfinite(trans).all() or not np.isfinite(refl).all():
            raise ArithmeticError("Non-finite optical spectrum")
        if max(float(np.max(np.abs(trans + refl - 1.0))), float(max(0.0, -trans.min(), -refl.min()))) > 1e-10:
            raise ArithmeticError("Lossless power conservation violated")
        return trans, refl

    def fom(self, transmission: np.ndarray) -> np.ndarray:
        t = np.asarray(transmission)
        if t.ndim != 2 or t.shape[1] != 2201 or not np.isfinite(t).all():
            raise ValueError("Finite batch of 2201 transmission samples required")
        return 10.0 * np.sum(((t - self.ideal) * self.solar) ** 2, axis=1) / self.solar_norm

    def evaluate(self, labels: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        arr = self.validate_labels(labels)
        out = np.empty(len(arr), dtype=np.float64)
        for start in range(0, len(arr), batch_size):
            trans, _ = self.spectra(arr[start : start + batch_size])
            out[start : start + batch_size] = self.fom(trans)
        return out


def labels_for(layers: int) -> np.ndarray:
    return np.array(list(itertools.product(range(4), repeat=layers)), dtype=np.int8)


def onehot(labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(labels)
    return np.eye(4, dtype=np.int8)[arr].reshape(len(arr), 4 * arr.shape[1])


def ensure_truth_table(
    layers: int,
    out_dir: Path,
    *,
    tmm_backend: str = "numpy",
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Load or compute the exhaustive `4^L` FOM table and true ranks (1 = best)."""
    truth_dir = Path(out_dir) / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)
    npz_path = truth_dir / f"L{layers}_truth.npz"
    meta_path = truth_dir / f"L{layers}_truth.json"

    if npz_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if sha256_file(npz_path) == meta.get("truth_table_sha256"):
            fom = np.load(npz_path)["fom"]
            order = np.argsort(fom, kind="stable")
            ranks = np.empty(len(fom), dtype=np.int64)
            ranks[order] = np.arange(1, len(fom) + 1)
            return fom, ranks

    t0 = time.perf_counter()
    ev = FilterEvaluator(layers, tmm_backend=tmm_backend)
    all_labels = labels_for(layers)
    fom = ev.evaluate(all_labels, batch_size=batch_size)
    elapsed = time.perf_counter() - t0

    tmp_npz = npz_path.with_name(npz_path.stem + "_tmp.npz")
    np.savez_compressed(tmp_npz, fom=fom)
    tmp_npz.replace(npz_path)

    order = np.argsort(fom, kind="stable")
    ranks = np.empty(len(fom), dtype=np.int64)
    ranks[order] = np.arange(1, len(fom) + 1)

    atomic_write_json(
        meta_path,
        {
            "layers": layers,
            "candidates": int(4**layers),
            "tmm_backend": tmm_backend,
            "input_manifest_sha256": ev.input_manifest_sha256,
            "truth_table_sha256": sha256_file(npz_path),
            "global_min_fom": float(fom[order[0]]),
            "global_min_id": int(order[0]),
            "global_min_labels": all_labels[order[0]].tolist(),
            "elapsed_seconds": elapsed,
            "created_utc": utc_now(),
        },
    )
    return fom, ranks

