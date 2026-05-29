"""
fix-install.py
==============
Installs all dependencies required by FedGATSage.
Handles PyTorch Geometric correctly by matching the installed PyTorch version.

Usage:
    python fix-install.py
    python fix-install.py --cpu-only      # force CPU-only torch
    python fix-install.py --verify-only   # only check what is/isn't installed
"""

import subprocess
import sys
import argparse
import importlib

# ── Packages installable with plain pip ──────────────────────────────────────
STANDARD_PACKAGES = [
    ("numpy",        "numpy>=1.21.0"),
    ("pandas",       "pandas>=1.3.0"),
    ("scipy",        "scipy>=1.7.0"),
    ("sklearn",      "scikit-learn>=1.0.0"),
    ("matplotlib",   "matplotlib>=3.3.0"),
    ("seaborn",      "seaborn>=0.11.0"),
    ("plotly",       "plotly>=5.0.0"),
    ("networkx",     "networkx>=2.6.0"),
    ("community",    "python-louvain>=0.15"),
    ("tqdm",         "tqdm>=4.62.0"),
    ("yaml",         "pyyaml>=5.4.0"),
    ("gdown",        "gdown"),           # Kaggle / Google Drive downloads
]

OPTIONAL_PACKAGES = [
    ("numba",  "numba>=0.54.0"),         # optional speed-up
]


def run(cmd: list, label: str = ""):
    print(f"\n  >> {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  [WARN] '{label}' may have failed (exit {result.returncode})")
    return result.returncode == 0


def is_installed(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def get_torch_version() -> tuple:
    """Return (major, minor, patch, cuda_tag) e.g. (2, 1, 0, 'cu118')"""
    try:
        import torch
        v = torch.__version__          # e.g. "2.1.0+cu118" or "2.1.0+cpu"
        tag = v.split("+")[1] if "+" in v else "cpu"
        parts = v.split("+")[0].split(".")
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2].split("a")[0].split("b")[0].split("rc")[0])
        return major, minor, patch, tag
    except Exception:
        return None


def install_torch(cpu_only: bool):
    """Install PyTorch if not present."""
    if is_installed("torch"):
        import torch
        print(f"  [OK] torch {torch.__version__} already installed.")
        return

    print("\n── Installing PyTorch ──────────────────────────────────────────")
    if cpu_only:
        run([sys.executable, "-m", "pip", "install",
             "torch", "torchvision", "torchaudio",
             "--index-url", "https://download.pytorch.org/whl/cpu"],
            "torch cpu")
    else:
        # Let pip pick the best CUDA-compatible wheel automatically
        run([sys.executable, "-m", "pip", "install", "torch"], "torch")


def install_torch_geometric():
    """Install torch-geometric (and optional scatter/sparse) matching installed torch."""
    if is_installed("torch_geometric"):
        import torch_geometric
        print(f"  [OK] torch_geometric {torch_geometric.__version__} already installed.")
        return

    print("\n── Installing PyTorch Geometric ────────────────────────────────")

    ver = get_torch_version()
    if ver is None:
        print("  [WARN] Could not detect torch version — installing torch-geometric without extras.")
        run([sys.executable, "-m", "pip", "install", "torch-geometric"], "torch-geometric")
        return

    major, minor, patch, cuda_tag = ver
    torch_str = f"{major}.{minor}.{patch}"  # e.g. "2.1.0"

    # torch-geometric core (always works)
    run([sys.executable, "-m", "pip", "install", "torch-geometric"], "torch-geometric")

    # torch-scatter and torch-sparse from the official wheel index
    # These are optional but improve performance for some ops
    wheel_url = f"https://data.pyg.org/whl/torch-{torch_str}+{cuda_tag}.html"
    print(f"  Trying scatter/sparse wheels from: {wheel_url}")
    ok = run([sys.executable, "-m", "pip", "install",
              "torch-scatter", "torch-sparse",
              "-f", wheel_url, "-q"],
             "torch-scatter/sparse")
    if not ok:
        print("  [INFO] torch-scatter/sparse wheels not found for this torch version — skipping.")
        print("         Core torch-geometric still works without them.")


def install_standard(verify_only: bool):
    print("\n── Standard packages ───────────────────────────────────────────")
    for module, pip_spec in STANDARD_PACKAGES:
        if is_installed(module):
            print(f"  [OK] {module}")
        elif not verify_only:
            print(f"  [INSTALL] {pip_spec}")
            run([sys.executable, "-m", "pip", "install", pip_spec, "-q"], pip_spec)
        else:
            print(f"  [MISSING] {module}")


def install_optional(verify_only: bool):
    print("\n── Optional packages ───────────────────────────────────────────")
    for module, pip_spec in OPTIONAL_PACKAGES:
        if is_installed(module):
            print(f"  [OK] {module}")
        elif not verify_only:
            print(f"  [INSTALL] {pip_spec}")
            run([sys.executable, "-m", "pip", "install", pip_spec, "-q"], pip_spec)
        else:
            print(f"  [MISSING] {module}  (optional)")


def verify_all():
    print("\n── Final verification ──────────────────────────────────────────")
    all_modules = [m for m, _ in STANDARD_PACKAGES] + ["torch", "torch_geometric"]
    all_ok = True
    for module in all_modules:
        if is_installed(module):
            try:
                m = importlib.import_module(module)
                ver = getattr(m, "__version__", "?")
            except Exception:
                ver = "?"
            print(f"  [OK]      {module:<20} {ver}")
        else:
            print(f"  [MISSING] {module}")
            all_ok = False

    print()
    if all_ok:
        print("All required packages are installed. You can run the experiment.")
    else:
        print("Some packages are missing — re-run without --verify-only to install them.")


def main():
    parser = argparse.ArgumentParser(description="FedGATSage dependency installer")
    parser.add_argument("--cpu-only",     action="store_true",
                        help="Install CPU-only PyTorch (no CUDA)")
    parser.add_argument("--verify-only",  action="store_true",
                        help="Only check what is installed, do not install")
    args = parser.parse_args()

    print("=" * 60)
    print("  FedGATSage — Dependency Installer")
    print("=" * 60)
    print(f"  Python : {sys.version.split()[0]}")

    if not args.verify_only:
        install_torch(cpu_only=args.cpu_only)
        install_torch_geometric()

    install_standard(verify_only=args.verify_only)
    install_optional(verify_only=args.verify_only)
    verify_all()

    if not args.verify_only:
        print("\nDone. Next steps:")
        print("  1. python fix-NF-TON-IoT-dataset.py")
        print("  2. python preprocess_data.py --input_file data/CIC-ToN-IoT.csv --output_dir data/cic --num_clients 5")
        print("  3. python experiments/fedgatsage_experiment.py --data_dir data/cic --num_clients 5 --num_rounds 5 --demo_mode")


if __name__ == "__main__":
    main()
