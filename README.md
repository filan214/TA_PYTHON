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
| 2. Aggregate (mingguan gerai×merek) | `src/data/aggregate.py` | ✅ implemented + tests hijau |
| 3. Google Trends (eksogen) | `src/data/google_trends.py` | ✅ implemented + tests hijau |
| 4. EDA | `src/eda/explore.py` | ✅ implemented + tests hijau |
| 5. Features (2 varian: baseline & gt) | `src/features/build.py` | ✅ implemented + tests hijau |
| 6a. SARIMA/SARIMAX (baseline & gt) | `src/models/sarimax.py` | ✅ implemented + tests hijau |
| 6b. Random Forest (baseline & gt) | `src/models/random_forest.py` | ✅ implemented + tests hijau |
| 6c. LSTM (baseline & gt) | `src/models/lstm.py` | ✅ implemented + tests hijau |
| 7. Evaluasi + Diebold-Mariano | `src/evaluation/` | ✅ implemented + tests hijau |
| 8. Optimasi inventori | `src/inventory/optimize.py` | ⬜ |
| 9. DSS dashboard | `app/dashboard.py` | ⬜ |

## Catatan data (diverifikasi pada data riil)

- 6320 baris mentah → 76 Void + 19 duplikat penuh dibuang → **6225 transaksi bersih**,
  Σqty=6907, 5 merek, 15 SKU, 4 gerai, rentang 2022-01-02..2025-06-30.
- Grid mingguan penuh = **184 minggu**; 1 minggu (2022-05-02..05-08) nol penjualan di
  semua gerai — **libur Lebaran 2022**, nilai nol yang valid (bukan data hilang).
- Fraksi minggu-nol level gerai×merek = **0.258 ≤ 0.30** → memvalidasi keputusan D1.
- Google Trends (`geo=ID`) berhasil di-fetch & di-cache: **920 baris** (5 merek × 184 minggu),
  `gt_index ∈ [25,100]`, cakupan **100%** grid, tanpa NaN. Puncak minat pencarian jatuh di
  **minggu Lebaran** (2022-05-02, 2023-04-24) → mendukung rasional eksogen D5 (sinyal musiman).
- EDA (Tahap 4): **17/20 deret stasioner** di level (ADF p<0.05) → `d_mode=0`; **3 non-stasioner**
  (Toko_A/B×Realme, Toko_C×Vivo) → `d=1`. **Kekuatan musiman Fs<0.64 untuk SEMUA deret**
  (rentang 0.36–0.59) → `D_mode=0`: musiman tahunan m=52 bersifat *lemah/spike-event*
  (puncak minggu Lebaran & Harbolnas woy~44/48), bukan siklus halus. Perlu keputusan
  penanganan musiman SARIMAX (lihat catatan D5/D6).
- SARIMAX (Tahap 6a, data-driven D6): dari 40 model (20 deret × 2 varian, 0 fallback),
  hanya **12 memilih suku musiman non-nol** (6 baseline + 6 gt) — 14/20 deret per varian
  memilih `(P,D,Q)=0`, dan **tak ada** yang memilih diferensiasi musiman (semua D=0).
  Diferensiasi biasa: 34 model d=0, 6 model d=1 (selaras ADF). Ini mengonfirmasi D6.
- Rezim evaluasi **one-step-ahead walk-forward (D9)** seragam untuk ketiga algoritma:
  SARIMAX pakai `statsmodels append()` (orde tetap dari Tahap 6a, tanpa re-fit); RF/LSTM
  alami one-step. Prasyarat validitas uji DM (Tahap 7) & σ galat one-step (Tahap 8).
- Pratinjau ablation GT (walk-forward; final di Tahap 7): MAE keseluruhan SARIMAX
  baseline 1.642 vs gt 1.595 — pada rezim seragam GT sedikit membantu; uji DM per-deret
  memutuskan di Tahap 7.
- Random Forest (Tahap 6b, one-step D9): MAE baseline 1.407 vs gt 1.400 — GT nyaris tak
  mengubah akurasi. Namun **`gt_index` menempati importance rank #2 dari 22 fitur**
  (mean 0.075, setelah `year_trend`). Peringatan: importance impurity bias ke fitur
  kontinu (year_trend, gt_index) — Δ MAE ~0 lebih dapat dipercaya; uji DM (Tahap 7)
  jadi penentu. RF (MAE ~1.40) mengungguli SARIMAX walk-forward (~1.6) sejauh ini.
- Pratinjau perbandingan 3 algoritma × 2 varian (740 baris uji, one-step D9; final di Tahap 7):
  peringkat MAE — **RF gt 1.400 < RF baseline 1.407 < LSTM gt 1.540 < LSTM baseline 1.572
  < SARIMAX gt 1.595 < SARIMAX baseline 1.642**. RF terbaik; LSTM ~setara tebak-rata-rata
  (data pendek/sparse); SARIMAX sMAPE ~97% vs RF/LSTM ~84%. GT konsisten sedikit membantu
  (ΔMAE −0.007 s/d −0.047) tapi kecil → signifikansi diputuskan uji DM (Tahap 7).
- **Evaluasi final (Tahap 7, uji Diebold-Mariano, squared-error, koreksi HLN, α=0.05):**
  - **Model terbaik = Random Forest + Google Trends** (MAE_mean 1.400, RMSE 1.747). MAPE
    dihitung hanya pada minggu non-nol; sMAPE dilaporkan sebagai pendamping (data ber-nol).
  - **Antar-algoritma (varian gt):** RF unggul **signifikan** atas SARIMAX (DM=6.02, p<0.001)
    dan LSTM (DM=4.48, p<0.001); LSTM unggul atas SARIMAX (DM=3.59, p<0.001). Peringkat
    signifikan: **RF > LSTM > SARIMAX** — tree model mengalahkan model klasik & deep learning
    pada data ritel hyper-lokal yang pendek/sparse (temuan sejalan ref [21]).
  - **Ablation GT (jawaban D5, berbasis bukti):** GT terbukti membantu **signifikan hanya untuk RF**
    (baseline→gt, DM=2.49, p=0.013). Untuk SARIMAX (p=0.114) dan LSTM (p=0.102) **tidak signifikan**.
    Per deret pun mayoritas tak signifikan (SARIMAX 17/20, RF 17/20, LSTM 15/20). Jadi klaim
    "GT membantu" **tidak** berlaku umum di data toko riil ini — hanya untuk RF.
  - Artefak: `metrics_summary.csv`, `metrics_per_series.csv`, `dm_tests.csv`,
    `gt_ablation_comparison.csv`, `reports/figures/actual_vs_pred_*.png` (6 deret × 2 varian).
