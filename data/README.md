# 📂 BBB-MolGraph Benchmark Datasets

To keep the repository lightweight and adhere to open-source data distribution standards, large benchmark datasets are not tracked directly in this Git repository.

---

## 📥 Dataset Download Links

The Blood-Brain Barrier Penetration (BBBP) benchmark dataset can be obtained from the following official sources:

| Source | Link / Reference | Description |
| :--- | :--- | :--- |
| **MoleculeNet Official** | [MoleculeNet BBBP Dataset](https://moleculenet.org/datasets-1) | Standard benchmark collection for molecular machine learning |
| **DeepChem S3 Direct Download** | [`https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv`](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv) | Direct CSV download link (2,050 raw compounds) |
| **Martins et al. Primary Literature** | [ACS J. Chem. Inf. Model. 2012, 52, 6, 1686–1697](https://doi.org/10.1021/ci300124c) | Original scientific reference and experimental curation protocol |

---

## ⚡ Quick Download Command

You can automatically download and place the benchmark dataset into `data/raw/` using `curl` or Python:

### Option A: Using curl (Terminal / PowerShell)

```bash
mkdir -p data/raw
curl -L https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv -o data/raw/BBBP_combined.csv
```

### Option B: Using Python

```python
import urllib.request
from pathlib import Path

out_dir = Path("data/raw")
out_dir.mkdir(parents=True, exist_ok=True)

url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
urllib.request.urlretrieve(url, out_dir / "BBBP_combined.csv")
print("Dataset successfully downloaded to data/raw/BBBP_combined.csv")
```

---

## 🧪 Quickstart Toy Sample Included

For instant verification, unit testing, and inference demonstration without downloading the full dataset, a lightweight 5-compound sample is provided at:
- [`data/sample_molecules.csv`](sample_molecules.csv)

You can run prediction directly on this sample:
```bash
python -m bbb_molgraph.cli predict --input-csv data/sample_molecules.csv --output-csv results/predictions.csv --smiles-col smiles
```

---

## 📁 Expected Directory Structure

```text
data/
├── README.md                 # Dataset overview & download links (this file)
├── sample_molecules.csv      # Lightweight 5-compound sample for quick testing
├── raw/                      # Ignored by Git (place downloaded BBBP_combined.csv here)
│   └── BBBP_combined.csv
├── splits/                   # Ignored by Git (precomputed scaffold cross-validation folds)
│   └── scaffold_folds.json
└── processed/                # Ignored by Git (cached graph tensors)
```
