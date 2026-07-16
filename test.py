from pathlib import Path
path_name = "data/phase1/private_set1/HCM0249/train/sparse/0/points3D.bin"
path = Path(path_name)
if path.exists():
    print(f"Found COLMAP points3D file: {path}")
