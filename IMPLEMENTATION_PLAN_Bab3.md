# Implementation Plan — Pipeline Bab III (Demand Forecasting & DSS Smartphone)

> **Untuk:** Claude Code
> **Konteks:** Implementasi metodologi Bab III skripsi Filan — "Sistem Pendukung Keputusan Pengadaan Stok Smartphone Berbasis Perbandingan LSTM, Random Forest, dan SARIMA di Indonesia".
> **Prinsip utama:** Setiap keputusan teknis di plan ini sudah disesuaikan dengan **karakteristik data riil** (`pos_transactions_raw.csv`, 6.320 baris, 4 gerai). Jangan mengganti keputusan desain kunci (Bagian 2) tanpa alasan berbasis data.

---

## 0. Ringkasan & Definisi of Done keseluruhan

Bangun pipeline end-to-end yang, dari satu file CSV transaksi POS mentah + Google Trends, menghasilkan:

1. Data mingguan bersih level **gerai × merek** (20 deret, 184 minggu).
2. Tiga model peramalan terlatih (**SARIMAX, Random Forest, LSTM**) dengan evaluasi komparatif.
3. Uji signifikansi **Diebold-Mariano** antar-model.
4. Parameter optimasi inventori (**safety stock, reorder point, order-up-to level**) dari model terbaik.
5. Purwarupa **DSS dashboard (Streamlit)** untuk 4 pemilik gerai.

**Pipeline harus reproducible**: `python -m src.run_all --config config.yaml` menjalankan seluruh tahap 1→8, dan `streamlit run app/dashboard.py` menjalankan DSS. Seed acak difiksasi. Semua artefak (data interim, model, hasil) tersimpan ke disk.

---

## 1. Data Contract (WAJIB dibaca sebelum coding)

File: `data/raw/pos_transactions_raw.csv` — encoding UTF-8 dengan BOM (`utf-8-sig`).

### 1.1 Skema kolom (23 kolom)

| Kolom | Tipe | Catatan penting |
|---|---|---|
| `sales_no` | str | **BUKAN unique** — ada 24 duplikat sales_no, 19 di antaranya baris identik penuh |
| `created_at` | datetime | timestamp transaksi — **ini yang dipakai untuk resampling waktu** |
| `paid_at` | datetime | null pada baris Void |
| `status` | str | `Paid` (6.244) / `Void` (76) |
| `order_type` | str | `Offline` (5.889) / `Online` (431) |
| `customer_name` | str | 84% null — **abaikan untuk forecasting** |
| `product_name` | str | mis. "Xiaomi Redmi Note 12" — sumber inferensi merek |
| `variant_name` | str | warna, 13 varian/SKU — abaikan untuk forecasting |
| `sku` | str | 15 SKU unik (mis. `XIA-RN12`) |
| `category` | str | `Handphone` (homogen 100%) |
| `qty` | int | 0 hanya pada Void; mayoritas 1 (rata-rata 1,1) |
| `price`,`discount`,`subtotal`,`tax`,`service_charge`,`rounding`,`grand_total` | int (Rupiah) | `grand_total`=0 hanya pada Void |
| `payment_method` | str | Tunai/QRIS/EDC/E-Wallet/Cicilan 0% |
| `staff_name` | str | 6 staf |
| `printer_id`,`note` | str | banyak null — abaikan |
| `store` | str | `Toko_A`/`Toko_B`/`Toko_C`/`Toko_D` |

### 1.2 Fakta data yang sudah diverifikasi (jadikan **assertion test**, bukan asumsi)

```
raw rows                = 6320
Void rows               = 76      (status=='Void', qty==0, grand_total==0)
exact duplicate rows    = 19      (df[df.status=='Paid'].duplicated())
clean transactions      = 6225    (Paid, dedup)
clean units (sum qty)   = 6907
stores                  = 4       {Toko_A:2103, Toko_B:1777, Toko_C:1324, Toko_D:1116}  # raw counts
SKUs                    = 15
brands                  = 5       {Xiaomi(+Redmi), OPPO, Samsung, Realme, Vivo}
date range              = 2022-01-02 .. 2025-06-30
ISO-week buckets        = 184
```

### 1.3 Mapping merek (dari `product_name`, case-insensitive)

```python
BRAND_MAP = lambda n: next((b for key,b in [
    ('xiaomi','Xiaomi'),('redmi','Xiaomi'),   # Redmi = sub-brand Xiaomi
    ('oppo','OPPO'),('samsung','Samsung'),
    ('realme','Realme'),('vivo','Vivo')
] if key in n.lower()), 'Other')
```
Setelah mapping, tidak boleh ada 'Other'. Tambahkan assert.

---

## 2. Keputusan Desain KUNCI (jangan diubah tanpa data)

| # | Keputusan | Alasan berbasis data |
|---|---|---|
| D1 | **Unit peramalan = gerai × merek** (20 deret), BUKAN per-SKU | Per-SKU-gerai: 64% minggu nol, 49/60 seri >50% nol, banyak SKU lifecycle pendek → intermittent, tak layak SARIMA/LSTM. Gerai×merek: ~26% nol, layak. |
| D2 | **Granularitas waktu = mingguan** (resample `W`) | Selaras siklus keputusan pengadaan; harian terlalu sparse |
| D3 | **Split temporal 80:20** = 147 minggu latih / 37 uji, tanpa shuffle | Meniru peramalan ke depan; cegah leakage |
| D4 | **Cross-validation = `TimeSeriesSplit`** (expanding window) | Standar deret waktu; train selalu mendahului val |
| D5 | **Google Trends `geo='ID'`** sebagai eksogen; SARIMA → **SARIMAX** | Penajam sinyal musiman (Lebaran/Harbolnas) |
| D6 | **Periode musiman s = 52** (mingguan tahunan) | Data menunjukkan tren tahunan yang jelas |
| D7 | **Metrik = MAPE, RMSE, MAE**; target MAPE < 15%; uji **Diebold-Mariano** | Sesuai Bab II/III |
| D8 | **Safety stock berbasis σ galat peramalan** (bukan σ permintaan historis) | Pendekatan Prak dkk. [23] di Bab II |

**Catatan intermittency:** meski di level gerai×merek jauh lebih sehat, tetap ada minggu nol. Untuk model, minggu nol adalah nilai valid (0 unit), bukan missing. Jangan diinterpolasi.

---

## 3. Struktur Proyek

```
smartphone-demand-dss/
├── README.md
├── requirements.txt
├── config.yaml                 # semua parameter terpusat
├── data/
│   ├── raw/pos_transactions_raw.csv
│   ├── interim/                # hasil antara (clean, aggregated)
│   └── processed/              # feature matrix per deret, siap model
├── src/
│   ├── __init__.py
│   ├── config.py               # loader config.yaml → dataclass
│   ├── run_all.py              # orkestrator tahap 1→8
│   ├── data/
│   │   ├── clean.py            # Tahap 1
│   │   ├── aggregate.py        # Tahap 2
│   │   └── google_trends.py    # Tahap 3
│   ├── eda/explore.py          # Tahap 4
│   ├── features/build.py       # Tahap 5
│   ├── models/
│   │   ├── base.py             # ABC Forecaster
│   │   ├── sarimax.py          # Tahap 6a
│   │   ├── random_forest.py    # Tahap 6b
│   │   └── lstm.py             # Tahap 6c
│   ├── evaluation/
│   │   ├── metrics.py          # Tahap 7
│   │   └── diebold_mariano.py  # Tahap 7
│   ├── inventory/optimize.py   # Tahap 8
│   └── utils/
│       ├── splits.py           # split temporal + TimeSeriesSplit helper
│       └── io.py               # simpan/muat artefak, seed
├── models/                     # artefak model terlatih (.pkl/.keras)
├── reports/
│   ├── figures/                # plot EDA & hasil
│   └── results/                # tabel metrik, hasil DM, params inventori (CSV/JSON)
├── tests/                      # pytest — assertion data + unit test tiap modul
└── app/dashboard.py            # Streamlit DSS
```

---

## 4. Environment & Dependencies

`requirements.txt` (samakan dengan Tabel 3.2 skripsi; pin versi minor):

```
python>=3.10
pandas>=2.0
numpy>=1.24
statsmodels>=0.14        # SARIMAX, ADF, ACF/PACF, seasonal_decompose
scikit-learn>=1.3        # RandomForest, MinMaxScaler, TimeSeriesSplit, GridSearchCV
tensorflow>=2.13         # LSTM (keras)
pmdarima>=2.0            # auto_arima untuk orde SARIMA awal (opsional, boleh manual)
matplotlib>=3.7
seaborn>=0.12
pytrends>=4.9            # Google Trends (unofficial API)
streamlit>=1.28
pyyaml>=6.0
pytest>=7.4
joblib>=1.3
```

**Gotcha pytrends:** API tak resmi, rentan rate-limit / berubah. WAJIB:
- Cache hasil Google Trends ke `data/interim/google_trends.csv`. Jika file ada, jangan re-fetch.
- Sediakan **fallback**: jika pytrends gagal, pipeline tetap jalan dengan flag `--no-trends` (model tanpa fitur GT), agar tidak memblok pekerjaan. Log warning jelas.

---

## 5. Tahapan Implementasi (map 1:1 ke Bab III)

Setiap tahap: **Objective → Input → Output → Steps → Definition of Done (DoD)**. Kerjakan berurutan; jangan lanjut sebelum DoD tahap sebelumnya hijau.

### Tahap 1 — `src/data/clean.py` (Bab III §3.1.4 pembersihan)

**Objective:** ubah CSV mentah → DataFrame transaksi bersih.
**Input:** `data/raw/pos_transactions_raw.csv`
**Output:** `data/interim/transactions_clean.parquet`

**Steps:**
1. Baca CSV (`encoding='utf-8-sig'`), parse `created_at`,`paid_at` ke datetime.
2. Filter `status == 'Paid'`.
3. Drop baris duplikat identik penuh (`drop_duplicates()`).
4. Tambah kolom `brand` via `BRAND_MAP`; assert tak ada 'Other'.
5. Tambah kolom `week` = `created_at.dt.to_period('W')`.
6. Simpan parquet.

**DoD (pytest `tests/test_clean.py`):**
- `len(raw) == 6320`
- baris Void terhapus = 76; duplikat terhapus = 19
- `len(clean) == 6225`, `clean.qty.sum() == 6907`
- `clean.brand.nunique() == 5`; `clean.sku.nunique() == 15`; `clean.store.nunique() == 4`

### Tahap 2 — `src/data/aggregate.py` (Bab III §3.1.4 agregasi)

**Objective:** deret mingguan gerai × merek pada grid waktu penuh (isi minggu tanpa penjualan = 0).
**Output:** `data/interim/weekly_store_brand.parquet` — kolom: `store, brand, week_start(date), units(int)`.

**Steps:**
1. Group `['store','brand','week']`, sum `qty`.
2. Bangun grid penuh: produk kartesian {4 store × 5 brand} × {semua 184 minggu} → reindex, `fillna(0)`.
3. Konversi `week` (Period) → `week_start` (Timestamp, Senin) agar ramah model & plot.
4. Simpan.

**DoD:**
- 20 kombinasi store×brand; tiap kombinasi 184 baris → total 3.680 baris.
- Tak ada NaN; `units` integer ≥ 0.
- Sanity: `df.units.sum() == 6907`.
- Cetak fraksi minggu-nol per deret; assert **mean ≤ 0.30** (validasi keputusan D1). Log deret terburuk.

### Tahap 3 — `src/data/google_trends.py` (Bab III §3.1.3 data eksogen)

**Objective:** ambil indeks Google Trends mingguan `geo='ID'` untuk tiap merek, selaraskan ke grid minggu.
**Output:** `data/interim/google_trends.csv` — kolom: `week_start, brand, gt_index(0..100)`.

**Steps:**
1. Untuk tiap merek, query pytrends kata kunci relevan (mis. `["Xiaomi Redmi","Xiaomi HP"]`, `["OPPO HP"]`, dst — definisikan `KEYWORDS_BY_BRAND` di `config.yaml`), `timeframe='2022-01-01 2025-06-30'`, `geo='ID'`.
2. Resample/align ke `week_start` grid (Google Trends mingguan biasanya sudah mingguan; selaraskan tanggal Senin, forward-fill maksimal 1 minggu jika perlu).
3. Simpan; jika file sudah ada → skip fetch (idempoten).
4. Fallback `--no-trends`: hasilkan `gt_index = NaN`/kolom kosong; downstream harus toleran.

**DoD:**
- File tercache; run kedua tidak memanggil network.
- Nilai `gt_index` dalam [0,100]; cakupan minggu ≥ 95% grid (sisanya boleh di-ffill).
- Jika fallback aktif, pipeline tetap lulus sampai akhir.

### Tahap 4 — `src/eda/explore.py` (Bab III §3.1.5 EDA)

**Objective:** analisis eksploratif + uji yang memandu pemodelan. Ini menghasilkan **figures untuk skripsi**.
**Output:** `reports/figures/*.png`, `reports/results/eda_summary.json`.

**Steps (per deret & agregat):**
1. Plot deret waktu tiap store×brand (grid 4×5) → `figures/series_grid.png`.
2. Uji **ADF** per deret → tabel p-value (stasioner/tidak) → `eda_summary.json`.
3. **ACF/PACF** untuk beberapa deret representatif → `figures/acf_pacf_<series>.png`.
4. **seasonal_decompose** (period=52) untuk deret terpadat → `figures/decompose_<series>.png`.
5. Ringkasan musiman: rata-rata unit per bulan/minggu-tahun; soroti Ramadan/Lebaran & Harbolnas (Nov/Des).

**DoD:**
- Semua figure ter-generate tanpa error.
- `eda_summary.json` berisi p-value ADF per 20 deret + rekomendasi `d`/`D` awal.

### Tahap 5 — `src/features/build.py` (Bab III §3.1.5 rekayasa fitur)

**Objective:** ubah tiap deret jadi matriks supervised untuk RF & LSTM; siapkan eksogen untuk SARIMAX.
**Output:** `data/processed/features_<store>_<brand>.parquet` (atau satu file panel dengan index multi).

**Fitur (per baris minggu t, per deret):**
- Target: `units[t]`.
- **Lag**: `units[t-1..t-L]`, default L=8 (config).
- **Rolling**: `roll_mean_4`, `roll_std_4`, `roll_mean_8`.
- **Kalender**: `weekofyear`, `month`, `is_ramadan`, `is_lebaran_window`, `is_harbolnas` (11.11/12.12/Nov-Des), `year_trend` (indeks minggu berurutan untuk tren).
- **Eksogen**: `gt_index[t]` (dan opsional `gt_index[t-1]`).
- Scaling: **MinMax**, di-`fit` **hanya pada train** (simpan scaler per deret ke `models/scalers/`). Untuk RF sebenarnya scaling opsional; untuk LSTM wajib.

**Penting anti-leakage:** semua fitur lag/rolling dihitung dari masa lalu saja. `fit` scaler & imputasi statistik hanya dari partisi train (147 minggu pertama).

**DoD:**
- Tak ada baris dengan lag NaN yang bocor ke train/test (baris awal ber-NaN di-drop konsisten).
- Fungsi `make_supervised(series_df, cfg) -> (X, y, index)` ada + unit test bentuk output.

### Tahap 6 — Model (Bab III §3.1.6)

Semua model turun dari `src/models/base.py`:

```python
class Forecaster(ABC):
    def fit(self, train_df, exog_train=None): ...
    def predict(self, horizon, exog_future=None) -> np.ndarray: ...
    def name(self) -> str: ...
```

Latih **per deret store×brand** (20 model per algoritma) ATAU global dengan dummy deret — default **per deret** (lebih sederhana & sesuai naskah). Simpan artefak ke `models/<algo>/<store>_<brand>.*`.

**6a — `sarimax.py`:**
- Orde via `pmdarima.auto_arima` (seasonal=True, m=52) ATAU grid manual dipandu ACF/PACF dari Tahap 4.
- `gt_index` sebagai `exog`.
- Simpan orde terpilih per deret ke `reports/results/sarima_orders.csv`.
- **Gotcha:** m=52 pada statsmodels bisa lambat/berat. Jika tak konvergen, izinkan fallback ke SARIMA non-seasonal + fitur kalender, dan catat di log.

**6b — `random_forest.py`:**
- `RandomForestRegressor`; `GridSearchCV` dgn `cv=TimeSeriesSplit(n_splits=5)`.
- Grid awal: `n_estimators∈{200,500}`, `max_depth∈{None,10,20}`, `max_features∈{'sqrt',0.5}`, `min_samples_leaf∈{1,2}`.
- Simpan `feature_importances_` → `reports/figures/rf_importance_<series>.png`.

**6c — `lstm.py`:**
- Input sekuens sliding window (window=L dari config).
- Arsitektur awal: `LSTM(64) → Dropout(0.2) → Dense(1)`; loss=MSE, opt=Adam, `EarlyStopping(patience=10, restore_best_weights=True)`, `validation_split` temporal (bukan acak).
- Set seed (`tf`, `np`, `random`) untuk reproducibility.
- Simpan `.keras` + scaler.

**DoD (semua model):**
- Fungsi `train_all(algo, cfg)` menghasilkan 20 artefak per algoritma tanpa error (atau log fallback jelas untuk deret bermasalah).
- Prediksi horizon uji (37 minggu) tersedia untuk tiap deret → `reports/results/predictions_<algo>.parquet`.

### Tahap 7 — Evaluasi & Diebold-Mariano (Bab III §3.1.7)

**`metrics.py`:** implement MAPE (aman untuk aktual 0 → gunakan sMAPE atau MAPE dengan epsilon, dokumentasikan pilihan; **catatan:** deret ini punya minggu nol, jadi MAPE murni bisa meledak — pakai **sMAPE** sebagai pendamping dan laporkan keduanya), RMSE, MAE. Agregasi: per deret, lalu rata-rata (mean & weighted-by-volume) per algoritma.

**`diebold_mariano.py`:** implement uji DM (loss=squared error, horizon=1 atau multi; two-sided). Bandingkan pasangan: SARIMAX vs RF, SARIMAX vs LSTM, RF vs LSTM — per deret dan/atau atas galat gabungan. Output p-value + arah.

**Output:**
- `reports/results/metrics_summary.csv` — baris=algoritma, kolom=MAPE/sMAPE/RMSE/MAE (mean & weighted).
- `reports/results/dm_tests.csv` — pasangan, statistik DM, p-value, kesimpulan (α=0.05).
- `reports/figures/actual_vs_pred_<series>.png` untuk beberapa deret.

**DoD:**
- Tabel metrik lengkap 3 algoritma × 20 deret + ringkasan.
- Uji DM menghasilkan p-value valid; kesimpulan model terbaik ter-derive otomatis (algoritma dgn error terendah yang signifikan).

### Tahap 8 — Optimasi Inventori (Bab III §3.1.8)

**`inventory/optimize.py`:**
- Ambil **model terbaik** (dari Tahap 7) → galat peramalan pada test.
- `safety_stock = z * σ_forecast_error * sqrt(lead_time_weeks)` — `z` dari `service_level` (config, default 95% → z≈1.645), `lead_time_weeks` (config, default 1–2).
- `reorder_point = mean_demand_lead_time + safety_stock`.
- `order_up_to_level = reorder_point + review_period_demand` (order-up-to, sesuai [16]).
- Analisis dampak biaya: total holding cost, jumlah stockout pada simulasi test, bandingkan **antar-3-algoritma** (bukan hanya terbaik) untuk tabel dampak biaya.
- Sensitivitas: variasi `service_level` & `holding_cost` (grid kecil).

**Output:** `reports/results/inventory_params.csv` (per store×brand: SS, ROP, OUL), `reports/results/cost_impact.csv`.

**DoD:**
- Parameter inventori terhitung untuk 20 deret.
- Tabel dampak biaya membandingkan 3 algoritma dengan asumsi biaya terdokumentasi.

### Tahap 9 — DSS Dashboard (Bab III §3.1.9) — `app/dashboard.py`

**Streamlit**, tiga subsistem DSS:
- **Data**: muat hasil (`predictions_*`, `inventory_params.csv`, data historis).
- **Model**: dropdown pilih gerai + merek → tampilkan peramalan minggu depan.
- **UI/rekomendasi**: kartu status per merek → *"Stok saat ini X; ROP=Y → SARAN: pesan Z unit"* dalam bahasa awam; grafik aktual vs ramalan; indikator warna (aman/mendekati ROP/di bawah ROP).
- Sidebar: pilih gerai (Toko A–D). Input stok terkini manual (number_input) untuk hitung saran real-time.

**DoD:**
- `streamlit run app/dashboard.py` jalan tanpa error, memuat artefak dari `reports/results/` & `models/`.
- Untuk tiap gerai×merek menampilkan: ramalan, ROP, dan rekomendasi kuantitas pesan.

---

## 6. Orkestrasi & Reproducibility

`src/run_all.py`:
```
python -m src.run_all --config config.yaml [--no-trends] [--from-stage N] [--to-stage M]
```
- Jalankan tahap 1→8 berurutan; skip tahap yang artefaknya sudah ada kecuali `--force`.
- Set global seed di `utils/io.set_seed(42)`.
- Log tiap tahap (mulai/selesai/artefak) via `logging`.

`config.yaml` memuat: path, `lags`, `rolling_windows`, `seasonal_period=52`, `train_ratio=0.8`, `service_level`, `lead_time_weeks`, `holding_cost`, `KEYWORDS_BY_BRAND`, hyperparameter grid, `seed`.

---

## 7. Testing Strategy (`tests/`)

- `test_clean.py` — semua assertion Bagian 1.2.
- `test_aggregate.py` — 3.680 baris, no-NaN, zero-week mean ≤ 0.30, sum units konsisten.
- `test_features.py` — no leakage (scaler fit hanya train), bentuk X/y benar, tak ada lag-NaN di train/test.
- `test_splits.py` — 147/37, train.max_date < test.min_date.
- `test_metrics.py` — MAPE/RMSE/MAE benar pada contoh manual; sMAPE aman saat aktual 0.
- `test_dm.py` — DM pada dua deret identik → p-value ~1 (tak beda).
- `test_inventory.py` — SS naik saat service_level naik; ROP ≥ SS.

Target: `pytest -q` hijau sebelum tahap dianggap selesai.

---

## 8. Urutan Kerja untuk Claude Code (checklist eksekusi)

1. [ ] Scaffold struktur folder + `requirements.txt` + `config.yaml` + `README.md`.
2. [ ] Salin `pos_transactions_raw.csv` ke `data/raw/`.
3. [ ] Tahap 1 (clean) + test → hijau.
4. [ ] Tahap 2 (aggregate) + test → hijau, **verifikasi zero-week ≤ 0.30**.
5. [ ] Tahap 3 (google trends) + cache + fallback.
6. [ ] Tahap 4 (EDA) → figures + ADF summary.
7. [ ] Tahap 5 (features) + test anti-leakage.
8. [ ] Tahap 6a/6b/6c (model) → 20 artefak/algoritma + prediksi test.
9. [ ] Tahap 7 (metrics + DM) → tabel hasil + tentukan model terbaik.
10. [ ] Tahap 8 (inventory) → params + cost impact.
11. [ ] Tahap 9 (Streamlit DSS).
12. [ ] `src/run_all.py` end-to-end + `README` cara menjalankan.
13. [ ] `pytest -q` seluruhnya hijau.

---

## 9. Gotchas & Risiko (dari data riil — perhatikan!)

- **MAPE meledak pada minggu nol.** Deret gerai×merek masih punya ~26% minggu nol. MAPE murni → pembagian nol. **Wajib** laporkan sMAPE + MAE/RMSE; jangan andalkan MAPE tunggal untuk deret bernilai nol.
- **SARIMAX m=52 berat/lambat & rentan non-konvergen** pada 147 titik train. Siapkan fallback (non-seasonal + fitur kalender) dan catat per deret.
- **Data train pendek untuk LSTM** (147 minggu). Jaga model kecil (1 lapisan, unit sedikit), pakai EarlyStopping & Dropout; jangan over-parametrize. Ini sekaligus temuan menarik untuk pembahasan (deep learning belum tentu menang pada data terbatas — konsisten dgn ref [21] Sutisna dkk. di Bab II).
- **pytrends tak stabil** → cache wajib + fallback `--no-trends`.
- **Beberapa SKU lifecycle pendek** — tapi karena agregasi ke merek, efeknya teredam. Jangan kembali ke per-SKU tanpa menangani intermittency (mis. Croston) — di luar scope naskah saat ini.
- **`sales_no` bukan primary key** — jangan pakai untuk dedup; pakai `drop_duplicates()` baris penuh.
- **Reproducibility**: seed semua (`numpy`, `random`, `tensorflow`), dan `PYTHONHASHSEED`.

---

## 10. Deliverables akhir (yang harus ada saat selesai)

- Repo lengkap sesuai struktur Bagian 3.
- `reports/results/`: `metrics_summary.csv`, `dm_tests.csv`, `inventory_params.csv`, `cost_impact.csv`, `sarima_orders.csv`.
- `reports/figures/`: series grid, ACF/PACF, decompose, actual-vs-pred, RF importance.
- `models/`: artefak terlatih + scaler.
- `app/dashboard.py`: DSS berjalan.
- `README.md`: setup, cara run pipeline & dashboard, ringkasan hasil.
- `pytest` hijau.

Hasil numerik (metrik, orde, params) dari eksekusi ini akan mengisi angka-angka di **Bab IV (Hasil & Pembahasan)** skripsi.
