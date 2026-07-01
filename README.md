# Smartphone Demand DSS — Pipeline Bab III

Sistem Pendukung Keputusan Pengadaan Stok Smartphone berbasis perbandingan
**SARIMAX, Random Forest, dan LSTM** untuk 4 gerai ritel independen di Indonesia,
dilanjutkan optimasi inventori dan purwarupa DSS berbasis dashboard.

Implementasi mengikuti `IMPLEMENTATION_PLAN_Bab3.md` (Data Contract §1 & keputusan
desain §2 bersifat kontrak — tidak diubah tanpa alasan berbasis data).

## Struktur

```
data/{raw,interim,processed}   src/   models/   reports/{figures,results}   tests/   app/
```

## Setup

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Menjalankan

```bash
# Pipeline penuh (Tahap 1->8), reproducible (seed=42):
python -m src.run_all --config config.yaml [--no-trends] [--from-stage N] [--to-stage M]

# Per tahap, contoh Tahap 1 (clean):
python -m src.data.clean --config config.yaml

# DSS dashboard:
streamlit run app/dashboard.py

# Uji:
pytest -q
```

## Status tahapan

| Tahap | Modul | Status |
|---|---|---|
| 1. Clean | `src/data/clean.py` | ✅ implemented + tests hijau |
| 2. Aggregate (mingguan gerai×merek) | `src/data/aggregate.py` | ⬜ |
| 3. Google Trends (eksogen) | `src/data/google_trends.py` | ⬜ |
| 4. EDA | `src/eda/explore.py` | ⬜ |
| 5. Features | `src/features/build.py` | ⬜ |
| 6. Model (SARIMAX/RF/LSTM) | `src/models/` | ⬜ |
| 7. Evaluasi + Diebold-Mariano | `src/evaluation/` | ⬜ |
| 8. Optimasi inventori | `src/inventory/optimize.py` | ⬜ |
| 9. DSS dashboard | `app/dashboard.py` | ⬜ |

## Catatan data (diverifikasi pada data riil)

- 6320 baris mentah → 76 Void + 19 duplikat penuh dibuang → **6225 transaksi bersih**,
  Σqty=6907, 5 merek, 15 SKU, 4 gerai, rentang 2022-01-02..2025-06-30.
- Grid mingguan penuh = **184 minggu**; 1 minggu (2022-05-02..05-08) nol penjualan di
  semua gerai — **libur Lebaran 2022**, nilai nol yang valid (bukan data hilang).
- Fraksi minggu-nol level gerai×merek = **0.258 ≤ 0.30** → memvalidasi keputusan D1.
