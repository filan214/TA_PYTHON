# Implementation Plan — Pipeline Bab III (Demand Forecasting & DSS Smartphone)

> **Untuk:** Claude Code
> **Konteks:** Implementasi metodologi Bab III skripsi Filan — "Sistem Pendukung Keputusan Pengadaan Stok Smartphone Berbasis Perbandingan LSTM, Random Forest, dan SARIMA di Indonesia".
> **Prinsip utama:** Setiap keputusan teknis di plan ini sudah disesuaikan dengan **karakteristik data riil** (`pos_transactions_raw.csv`, 6.320 baris, 4 gerai). Jangan mengganti keputusan desain kunci (Bagian 2) tanpa alasan berbasis data.

---

## STATUS PROGRES (update terakhir)

- ✅ **Tahap 1–4 SELESAI** (clean, aggregate, google_trends fetch+cache, EDA) — dijalankan sebelum revisi D5 di bawah. Artefak tahap ini **tetap valid**, tidak perlu diulang: cara data dibersihkan/diagregasi/di-fetch tidak berubah oleh revisi D5.
- 📊 **Temuan aktual Tahap 4 (EDA):** seasonal strength lemah untuk seluruh 20 deret (Fs<0,64, m=52), dengan musiman terkonsentrasi di minggu-minggu event (Lebaran/Harbolnas) — bukan siklus tahunan halus. Temuan ini memicu **revisi D6** (lihat di bawah).
- 🔄 **REVISI PADA RENCANA #1 (D5 + Tahap 5, 6, 7)**: Google Trends semula diperlakukan sebagai fitur eksogen wajib (langsung dipasang ke semua model). **Sekarang diubah menjadi ablation study**: tiap algoritma dilatih dalam dua varian — `baseline` (tanpa GT) dan `gt` (dengan GT) — supaya kontribusi GT diuji secara empiris (uji Diebold-Mariano), bukan diasumsikan. Lihat catatan lengkap di D5 (Bagian 2) dan Tahap 5–7 (Bagian 5) yang sudah direvisi.
- 🔄 **REVISI PADA RENCANA #2 (D6 + Tahap 5, 6a)**: berdasarkan temuan Tahap 4 di atas, **orde musiman SARIMA(X) kini diputuskan per-deret secara data-driven via AIC** — TIDAK dipaksa non-nol di semua 20 deret. Tahap 5 menambahkan fitur **Fourier(m=52)** di samping fitur kalender yang sudah ada, agar sinyal musiman tetap tertangkap oleh RF & LSTM juga. Lihat catatan lengkap di D6 (Bagian 2) dan Tahap 5, 6a (Bagian 5) yang sudah direvisi.
- 🔄 **REVISI PADA RENCANA #3 (D9 + Tahap 6, 7, 8)**: ditemukan saat implementasi Tahap 6a bahwa RF/LSTM alami *one-step-ahead* sedangkan SARIMAX default ke *fixed-origin* 37-langkah — asimetri ini merusak validitas uji DM (Tahap 7) dan formula *safety stock* (Tahap 8), bukan cuma soal keadilan. **Diputuskan: ketiga algoritma dievaluasi dengan one-step-ahead walk-forward yang identik** (SARIMAX via update state `append()`/`apply()`, orde tetap dari Tahap 6a, tanpa re-fit tiap langkah). Lihat catatan lengkap di D9 (Bagian 2) dan Tahap 6/7/8 (Bagian 5) yang sudah direvisi.
- ⏭️ **Posisi saat ini: Tahap 8 (Optimasi Inventori).** Tahap 5–7 selesai. Tahap 6 (SARIMAX/RF/LSTM, 2 varian × 20 deret, rezim walk-forward D9) dan Tahap 7 (metrik + Diebold-Mariano + ablation GT) hijau. **Hasil Tahap 7:** model terbaik = **RF(gt)**; peringkat signifikan **RF > LSTM > SARIMAX**; Google Trends terbukti membantu signifikan **hanya untuk RF** (p=0.013), tidak untuk SARIMAX (p=0.114) / LSTM (p=0.102). Tahap 8 memakai galat one-step-ahead model terbaik untuk safety stock (D8/D9).
- ⏳ **Tahap 8–9 BELUM dikerjakan** — lanjutkan dengan model terbaik dari Tahap 7 dan rezim evaluasi walk-forward seragam (D9) yang sudah dikunci.

---

## 0. Ringkasan & Definisi of Done keseluruhan

Bangun pipeline end-to-end yang, dari satu file CSV transaksi POS mentah + Google Trends, menghasilkan:

1. Data mingguan bersih level **gerai × merek** (20 deret, 184 minggu).
2. Tiga model peramalan terlatih (**SARIMA/SARIMAX, Random Forest, LSTM**), masing-masing dalam dua varian — **baseline** dan **+Google Trends** — dengan evaluasi komparatif antar-algoritma sekaligus ablation study kontribusi Google Trends (lihat D5, Bagian 2).
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
| D5 | **Google Trends `geo='ID'`** diuji sebagai **ablation study** (baseline vs +GT per algoritma), BUKAN diasumsikan otomatis membantu | Data toko riil berskala hyper-lokal (rata-rata 1–2 unit/minggu per deret); ada *scale mismatch* dgn sinyal pencarian nasional. Manfaat GT harus dibuktikan empiris pada data ini, bukan diasumsikan dari literatur berskala nasional/agregat (lih. catatan D5 di bawah & ref [21] Sutisna dkk.) |
| D6 | **Periode musiman s = 52** (mingguan tahunan) tetap jadi parameter kontrak, TAPI **orde musiman (P,D,Q) diputuskan per-deret secara data-driven** (AIC/validasi), bukan dipaksa non-nol di semua deret | EDA (Tahap 4) menunjukkan seasonal strength lemah (Fs<0,64 untuk seluruh 20 deret) dan musiman terkonsentrasi di minggu-minggu event (Lebaran/Harbolnas) — bukan pola musiman halus berulang. Memaksa (P,D,Q) non-nol pada semua deret berisiko tinggi non-konvergen (lih. §9) dan salah spesifikasi. |
| D7 | **Metrik = MAPE, RMSE, MAE**; target MAPE < 15%; uji **Diebold-Mariano** | Sesuai Bab II/III |
| D8 | **Safety stock berbasis σ galat peramalan** (bukan σ permintaan historis) | Pendekatan Prak dkk. [23] di Bab II |
| D9 | **Rezim evaluasi = one-step-ahead walk-forward untuk KETIGA algoritma** (bukan fixed-origin multi-step untuk SARIMAX vs one-step untuk RF/LSTM) | Tanpa penyeragaman ini: (a) formula *safety stock* di D8/Tahap 8 jadi tidak valid secara konseptual, karena mengasumsikan σ galat *one-step-ahead*, bukan galat multi-step yang terakumulasi; (b) uji Diebold-Mariano (D7) tidak sah karena membandingkan loss dari rezim peramalan yang berbeda, bukan forecast origin yang sepadan; (c) tidak merepresentasikan skenario riil DSS (toko cek stok mingguan, selalu punya data aktual minggu sebelumnya sebelum merekomendasikan pesanan minggu ini). |

**Catatan D9 — kenapa one-step walk-forward untuk ketiga algoritma (ditambahkan setelah Tahap 6a menemukan asimetri rezim):** RF dan LSTM secara alami melakukan peramalan *one-step-ahead* karena fitur lag/rolling-nya dihitung dari nilai permintaan aktual yang sudah diketahui hingga t-1. Jika SARIMAX dibiarkan melakukan *fixed-origin* 37-langkah (meramal seluruh periode uji sekaligus dari satu titik tanpa pernah "melihat" data aktual di antaranya), ketiga algoritma tidak lagi dievaluasi pada tugas yang sama — dan ini bukan sekadar isu keadilan naratif, tapi merusak validitas dua komponen lain yang sudah dikunci di kontrak:
- **D8 (safety stock)** memakai σ galat peramalan yang secara eksplisit diasumsikan *one-step-ahead* (praktik umum dalam teori inventori, distribusi galat 1-periode diskalakan `sqrt(lead_time)`). Galat dari peramalan 37-langkah terakumulasi secara berbeda dan akan membuat perhitungan *safety stock* salah secara konseptual, bukan hanya kurang akurat.
- **D7 (uji Diebold-Mariano)** mensyaratkan pasangan galat yang dihitung pada *forecast origin* yang sepadan antar-model. Membandingkan galat dari rezim yang berbeda (fixed-origin vs rolling) membuat hasil uji DM tidak sah secara statistik.

Karena itu, ketiga algoritma dievaluasi dalam rezim **one-step-ahead walk-forward** yang identik pada 37 minggu data uji: pada tiap langkah, model diberi seluruh data aktual hingga minggu t-1 (termasuk data uji yang sudah "terlewati"), meramal minggu t, lalu majunya satu minggu dan mengulang. Untuk SARIMAX, ini diimplementasikan lewat pembaruan state (`statsmodels` `append()`/`apply()`) menggunakan orde yang sudah dipilih di Tahap 6a — **bukan** re-fit `auto_arima` di tiap langkah (mahal dan mengubah identitas model 37 kali). Untuk RF/LSTM, tidak ada perubahan karena keduanya sudah alami one-step-ahead. Rezim ini juga paling konsisten dengan skenario riil DSS: pemilik toko selalu mengecek stok dan data penjualan minggu sebelumnya sebelum menentukan pesanan minggu ini — persis kondisi *walk-forward*, bukan meramal 37 minggu ke depan secara membabi buta.

**Catatan D6 — kenapa data-driven per-deret, bukan dipaksa (ditambahkan setelah Tahap 4 EDA aktual):** Hasil EDA riil menunjukkan seasonal strength (Fs) di bawah 0,64 untuk seluruh 20 deret gerai×merek, dengan komponen musiman yang jelas justru terkonsentrasi sempit di sekitar Lebaran dan Harbolnas (11.11/12.12), bukan berupa siklus tahunan yang halus dan berulang. Karakteristik ini secara statistik lebih cocok ditangkap lewat **fitur eksogen** (dummy kalender + Fourier) daripada lewat suku ARIMA musiman (P,D,Q)₅₂ yang didesain untuk autokorelasi musiman yang mulus — apalagi dengan hanya 147 titik latih (~2,8 siklus tahunan), estimasi suku musiman penuh rawan tidak stabil/non-konvergen (persis risiko yang sudah dicatat di Bagian 9). Karena itu:
- `s = 52` tetap menjadi parameter periodisitas yang tersedia untuk pencarian orde (tidak dihapus dari kontrak), tetapi (P,D,Q) musiman **tidak dipaksa non-nol** — dipilih AIC per deret di Tahap 6a, sehingga sebagian deret bisa saja berakhir non-seasonal jika itu yang terbaik.
- **Tahap 5 menambahkan fitur Fourier(m=52)** (pasangan sin/cos) di samping fitur kalender (`is_ramadan`, `is_lebaran_window`, `is_harbolnas`) yang sudah ada, agar sinyal musiman tetap tertangkap oleh **ketiga algoritma** (termasuk RF dan LSTM yang tidak punya struktur ARIMA musiman bawaan), bukan hanya oleh SARIMAX.
- Pendekatan ini konsisten dengan filosofi D5 (Google Trends): keputusan struktural diuji secara empiris per deret, bukan diasumsikan dari awal.

**Catatan D5 — kenapa ablation, bukan asumsi:** Google Trends `geo='ID'` mengukur minat pencarian **tingkat nasional**, sedangkan unit analisis penelitian ini adalah **penjualan toko fisik individual** dengan volume sangat kecil (~1–2 unit/minggu per deret gerai×merek). Rantai kausal dari "minat pencarian nasional naik" ke "satu toko lokal terjual lebih banyak minggu ini" panjang dan tidak otomatis berlaku — berbeda dari studi rujukan di Bab II (Choi & Varian [11], Sutisna dkk. [21]) yang seluruhnya menggunakan data **agregat nasional**, bukan satu unit usaha kecil. Karena itu, GT diperlakukan sebagai **hipotesis yang diuji secara empiris** (baseline vs +GT untuk tiap algoritma, lihat Tahap 5–7), bukan fitur yang dipasang begitu saja dengan asumsi pasti membantu. Ini juga menjaga konsistensi dengan judul skripsi saat ini, yang tidak lagi menempatkan Google Trends sebagai klaim utama penelitian.

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

**Objective:** ubah tiap deret jadi matriks supervised untuk RF & LSTM; siapkan dua **varian fitur** per deret — `baseline` (tanpa Google Trends) dan `gt` (dengan Google Trends) — agar kontribusi GT bisa diuji sebagai ablation study (lihat D5).
**Output:** `data/processed/features_<store>_<brand>.parquet` (satu file berisi seluruh kolom, termasuk `gt_index`; pemilihan varian dilakukan saat training via daftar kolom, bukan file terpisah).

**Fitur (per baris minggu t, per deret):**
- Target: `units[t]`.
- **Lag**: `units[t-1..t-L]`, default L=8 (config).
- **Rolling**: `roll_mean_4`, `roll_std_4`, `roll_mean_8`.
- **Kalender**: `weekofyear`, `month`, `is_ramadan`, `is_lebaran_window`, `is_harbolnas` (11.11/12.12/Nov-Des), `year_trend` (indeks minggu berurutan untuk tren).
- **Fourier musiman** *(baru — lih. D6)*: `fourier_sin_1`, `fourier_cos_1` (dan opsional harmonik ke-2: `fourier_sin_2`, `fourier_cos_2`) dengan periode m=52, dihitung dari indeks minggu berurutan. Fitur ini menyuntikkan sinyal periodisitas tahunan yang halus ke **RF dan LSTM** (yang tidak punya struktur ARIMA musiman bawaan), melengkapi dummy kalender yang menangkap lonjakan tajam di minggu-minggu event. Fitur ini ada di **kedua varian** (`baseline` maupun `gt`) — bukan bagian dari ablation GT.
- **Eksogen (khusus varian `gt`)**: `gt_index[t]` (dan opsional `gt_index[t-1]`). Kolom ini tetap dihitung & disimpan untuk semua deret, tapi **hanya dipakai** saat melatih varian `gt` — varian `baseline` mengecualikan kolom ini sepenuhnya (bukan di-nol-kan, karena akan mengubah struktur data untuk model non-pohon seperti LSTM).
- Scaling: **MinMax**, di-`fit` **hanya pada train** (simpan scaler per deret **per varian** ke `models/scalers/<variant>/`). Untuk RF sebenarnya scaling opsional; untuk LSTM wajib.

**Definisikan daftar kolom fitur secara eksplisit** di `config.yaml`:
```yaml
feature_variants:
  baseline: [lag_*, roll_*, weekofyear, month, is_ramadan, is_lebaran_window, is_harbolnas, year_trend, fourier_sin_1, fourier_cos_1]
  gt:       [lag_*, roll_*, weekofyear, month, is_ramadan, is_lebaran_window, is_harbolnas, year_trend, fourier_sin_1, fourier_cos_1, gt_index]
```

**Penting anti-leakage:** semua fitur lag/rolling dihitung dari masa lalu saja. `fit` scaler & imputasi statistik hanya dari partisi train (147 minggu pertama), dilakukan terpisah untuk tiap varian.

**DoD:**
- Tak ada baris dengan lag NaN yang bocor ke train/test (baris awal ber-NaN di-drop konsisten).
- Fungsi `make_supervised(series_df, cfg, variant='baseline'|'gt') -> (X, y, index)` ada + unit test bentuk output untuk **kedua varian** (assert kolom `gt_index` ada di varian `gt` dan tidak ada di `baseline`).

### Tahap 6 — Model (Bab III §3.1.6)

Semua model turun dari `src/models/base.py`:

```python
class Forecaster(ABC):
    def fit(self, train_df, exog_train=None): ...
    def predict(self, horizon, exog_future=None) -> np.ndarray: ...
    def name(self) -> str: ...
```

Latih **per deret store×brand, per varian fitur** (20 deret × 2 varian = 40 model per algoritma) — default **per deret** (lebih sederhana & sesuai naskah). Simpan artefak ke `models/<algo>/<variant>/<store>_<brand>.*`.

**Rezim evaluasi (D9):** ketiga algoritma dievaluasi dengan **one-step-ahead walk-forward** pada 37 minggu data uji — di tiap langkah, model diberi data aktual hingga t-1, meramal t, lalu maju satu minggu. Ini WAJIB seragam di ketiga algoritma (lihat Catatan D9 di atas); jangan biarkan SARIMAX melakukan fixed-origin multi-step sementara RF/LSTM one-step.

**6a — `sarimax.py`:**
- Orde via `pmdarima.auto_arima` (`seasonal=True`, `m=52`, `stepwise=True`) per deret, dipandu batas awal `d`/`D` dari `eda_summary.json` (Tahap 4) dan ACF/PACF. **Seasonal order (P,D,Q) TIDAK dipaksa non-nol** — biarkan `auto_arima` memilih via AIC, termasuk kemungkinan (0,0,0)₅₂ jika itu yang terbaik untuk deret tertentu (lih. D6). Ini keputusan data-driven per deret, bukan penyeragaman di awal.
- Varian `baseline` = **SARIMA** dengan Fourier+kalender sebagai `exog` (tanpa `gt_index`); varian `gt` = **SARIMAX** dengan Fourier+kalender+`gt_index` sebagai `exog`. Jadi kedua varian tetap punya exog — bedanya murni ada/tidaknya `gt_index`, agar perbandingan ablation benar-benar isolasi efek GT saja. Cari orde ARIMA terpisah untuk tiap varian (exog yang berbeda bisa mengubah orde optimal) — dokumentasikan kedua orde per deret.
- Simpan orde terpilih (termasuk kasus seasonal order = 0) per deret **per varian** ke `reports/results/sarima_orders.csv`, dengan kolom tambahan `has_seasonal_terms` (bool) untuk memudahkan pelaporan di Bab IV — mis. "X dari 20 deret memilih orde musiman non-nol".
- **Gotcha (termudahkan oleh D6):** karena seasonal order kini data-driven, risiko non-konvergen m=52 berkurang signifikan (deret dengan sinyal musiman lemah akan otomatis memilih orde rendah/nol). Tetap sediakan fallback ke non-seasonal jika `auto_arima` tetap gagal konvergen pada rentang orde manapun, dan catat di log.
- **Evaluasi (D9 — WAJIB):** setelah orde terpilih dari data latih, prediksi pada 37 minggu uji dilakukan sebagai **one-step-ahead walk-forward**, bukan fixed-origin 37-langkah. Gunakan `statsmodels` `append()`/`apply()` untuk memperbarui state model dengan data aktual tiap minggu tanpa re-fit orde (orde tetap sama dengan yang dipilih di data latih — hanya state/observasi yang diperbarui). Ini menjaga biaya komputasi rendah sekaligus menyamakan rezim evaluasi dengan RF/LSTM yang sudah alami one-step-ahead.

**6b — `random_forest.py`:**
- `RandomForestRegressor`; `GridSearchCV` dgn `cv=TimeSeriesSplit(n_splits=5)`.
- Grid awal: `n_estimators∈{200,500}`, `max_depth∈{None,10,20}`, `max_features∈{'sqrt',0.5}`, `min_samples_leaf∈{1,2}`.
- Latih dua kali per deret: fitur `baseline` dan fitur `gt` (lihat Tahap 5).
- Simpan `feature_importances_` → `reports/figures/rf_importance_<series>_<variant>.png`. Untuk varian `gt`, ini juga menjawab langsung seberapa penting `gt_index` relatif terhadap fitur lain.

**6c — `lstm.py`:**
- Input sekuens sliding window (window=L dari config).
- Arsitektur awal: `LSTM(64) → Dropout(0.2) → Dense(1)`; loss=MSE, opt=Adam, `EarlyStopping(patience=10, restore_best_weights=True)`, `validation_split` temporal (bukan acak).
- Varian `baseline`: sekuens tanpa kanal `gt_index`. Varian `gt`: kanal `gt_index` ditambahkan sebagai fitur tambahan per timestep. Arsitektur (jumlah unit/lapisan) **dibuat identik** antar-varian agar selisih performa murni berasal dari ada/tidaknya GT, bukan dari kapasitas model yang berbeda.
- Set seed (`tf`, `np`, `random`) untuk reproducibility.
- Simpan `.keras` + scaler per varian.

**DoD (semua model):**
- Fungsi `train_all(algo, variant, cfg)` menghasilkan 20 artefak per (algoritma, varian) tanpa error (atau log fallback jelas untuk deret bermasalah) → total 40 artefak per algoritma, 120 total.
- Prediksi horizon uji (37 minggu) tersedia untuk tiap deret × varian → `reports/results/predictions_<algo>_<variant>.parquet`, **dihasilkan lewat rezim one-step-ahead walk-forward yang identik untuk ketiga algoritma (D9)** — assert tidak ada algoritma yang menggunakan fixed-origin multi-step sementara yang lain one-step.

### Tahap 7 — Evaluasi, Diebold-Mariano, dan Ablation Study Google Trends (Bab III §3.1.7)

**`metrics.py`:** implement MAPE (aman untuk aktual 0 → gunakan sMAPE atau MAPE dengan epsilon, dokumentasikan pilihan; **catatan:** deret ini punya minggu nol, jadi MAPE murni bisa meledak — pakai **sMAPE** sebagai pendamping dan laporkan keduanya), RMSE, MAE. Agregasi: per deret, lalu rata-rata (mean & weighted-by-volume) per (algoritma, varian).

**`diebold_mariano.py`:** implement uji DM (loss=squared error, horizon=1 atau multi; two-sided). **Prasyarat validitas (D9):** galat yang dibandingkan harus berasal dari rezim evaluasi yang sama (one-step-ahead walk-forward) untuk kedua model dalam tiap pasangan — jangan menjalankan uji DM pada galat yang dihasilkan dari rezim berbeda. Dua kelompok perbandingan:
1. **Antar-algoritma** (pada varian terbaik masing-masing, biasanya `gt` jika terbukti membantu, atau `baseline` jika tidak): SARIMAX vs RF, SARIMAX vs LSTM, RF vs LSTM.
2. **Ablation per algoritma** (baseline vs gt, algoritma yang sama): SARIMA vs SARIMAX, RF-baseline vs RF-gt, LSTM-baseline vs LSTM-gt — ini yang **langsung menjawab pertanyaan "apakah Google Trends terbukti membantu di data toko riil ini"**, per algoritma, dengan p-value, bukan asumsi.

**Output:**
- `reports/results/metrics_summary.csv` — baris=(algoritma, varian), kolom=MAPE/sMAPE/RMSE/MAE (mean & weighted).
- `reports/results/dm_tests.csv` — pasangan, statistik DM, p-value, kesimpulan (α=0.05); mencakup kedua kelompok perbandingan di atas.
- `reports/results/gt_ablation_comparison.csv` — per (algoritma, deret): MAPE_baseline, MAPE_gt, Δ (%), apakah GT membantu (ya/tidak) berdasarkan uji DM ablation; plus baris ringkasan: jumlah deret yang membaik vs memburuk vs tak signifikan per algoritma.
- `reports/figures/actual_vs_pred_<series>_<variant>.png` untuk beberapa deret.

**DoD:**
- Tabel metrik lengkap 3 algoritma × 2 varian × 20 deret + ringkasan.
- Uji DM menghasilkan p-value valid untuk kedua kelompok perbandingan; kesimpulan model terbaik ter-derive otomatis (algoritma+varian dgn error terendah yang signifikan).
- `gt_ablation_comparison.csv` memberi jawaban eksplisit dan berbasis-bukti terhadap pertanyaan "apakah GT bermanfaat di skala data toko riil ini" — siap dikutip langsung di pembahasan Bab IV.

### Tahap 8 — Optimasi Inventori (Bab III §3.1.8)

**`inventory/optimize.py`:**
- Ambil **model terbaik** — yaitu kombinasi (algoritma, varian) dengan error terendah yang signifikan secara statistik dari Tahap 7, bisa jadi `baseline` atau `gt` tergantung hasil ablation — → galat peramalan pada test (**wajib galat one-step-ahead walk-forward sesuai D9** — formula di bawah tidak valid untuk galat multi-step).
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
- `test_features.py` — no leakage (scaler fit hanya train), bentuk X/y benar untuk **kedua varian** (`baseline` & `gt`), tak ada lag-NaN di train/test, assert `gt_index` absen di varian `baseline`.
- `test_splits.py` — 147/37, train.max_date < test.min_date.
- `test_metrics.py` — MAPE/RMSE/MAE benar pada contoh manual; sMAPE aman saat aktual 0.
- `test_dm.py` — DM pada dua deret identik → p-value ~1 (tak beda).
- `test_inventory.py` — SS naik saat service_level naik; ROP ≥ SS.

Target: `pytest -q` hijau sebelum tahap dianggap selesai.

---

## 8. Urutan Kerja untuk Claude Code (checklist eksekusi)

1. [x] Scaffold struktur folder + `requirements.txt` + `config.yaml` + `README.md`.
2. [x] Salin `pos_transactions_raw.csv` ke `data/raw/`.
3. [x] Tahap 1 (clean) + test → hijau.
4. [x] Tahap 2 (aggregate) + test → hijau, **verifikasi zero-week ≤ 0.30**.
5. [x] Tahap 3 (google trends) + cache + fallback.
6. [x] Tahap 4 (EDA) → figures + ADF summary.
7. [x] Tahap 5 (features, **2 varian: baseline & gt**) + test anti-leakage untuk kedua varian.
8. [x] Tahap 6a/6b/6c (model, **2 varian × 20 deret = 40 artefak/algoritma**) → prediksi test kedua varian. *(ketiganya selesai dgn rezim walk-forward D9)*
9. [x] Tahap 7 (metrics + DM antar-algoritma + **DM ablation baseline-vs-gt**) → `gt_ablation_comparison.csv` + model terbaik = **RF(gt)**. *(RF > LSTM > SARIMAX signifikan; GT signifikan hanya untuk RF)*
10. [ ] Tahap 8 (inventory) → params + cost impact.
11. [ ] Tahap 9 (Streamlit DSS).
12. [ ] `src/run_all.py` end-to-end + `README` cara menjalankan.
13. [ ] `pytest -q` seluruhnya hijau.

---

## 9. Gotchas & Risiko (dari data riil — perhatikan!)

- **MAPE meledak pada minggu nol.** Deret gerai×merek masih punya ~26% minggu nol. MAPE murni → pembagian nol. **Wajib** laporkan sMAPE + MAE/RMSE; jangan andalkan MAPE tunggal untuk deret bernilai nol.
- **SARIMAX m=52 berat/lambat & rentan non-konvergen** pada 147 titik train — risiko ini **sebagian besar dimitigasi oleh D6** (orde musiman data-driven per deret via AIC, tidak dipaksa non-nol). Tetap siapkan fallback non-seasonal untuk kasus `auto_arima` gagal konvergen di semua kandidat orde, dan catat per deret di `sarima_orders.csv`.
- **Data train pendek untuk LSTM** (147 minggu). Jaga model kecil (1 lapisan, unit sedikit), pakai EarlyStopping & Dropout; jangan over-parametrize. Ini sekaligus temuan menarik untuk pembahasan (deep learning belum tentu menang pada data terbatas — konsisten dgn ref [21] Sutisna dkk. di Bab II).
- **pytrends tak stabil** → cache wajib + fallback `--no-trends`.
- **Google Trends kini ablation study, bukan fitur wajib** (lih. D5). Ini menggandakan jumlah model terlatih (40 vs 20 per algoritma) — pastikan waktu komputasi masih wajar untuk skala data ini (147 titik train, model ringan); jika terlalu lambat, prioritaskan SARIMA/SARIMAX dulu (paling cepat) sebelum RF/LSTM. **Jangan** melaporkan GT sebagai "terbukti membantu" tanpa didukung `gt_ablation_comparison.csv` dan uji DM ablation yang signifikan.
- **Beberapa SKU lifecycle pendek** — tapi karena agregasi ke merek, efeknya teredam. Jangan kembali ke per-SKU tanpa menangani intermittency (mis. Croston) — di luar scope naskah saat ini.
- **Rezim evaluasi harus seragam (D9).** RF/LSTM alami one-step-ahead; SARIMAX **jangan** dibiarkan default ke fixed-origin multi-step forecast (`predict(steps=37)` sekali jalan) — ini akan merusak validitas uji DM (Tahap 7) dan formula *safety stock* (Tahap 8). Implementasikan walk-forward via update state (`append()`/`apply()`), orde tetap dari Tahap 6a, jangan re-fit tiap langkah.
- **`sales_no` bukan primary key** — jangan pakai untuk dedup; pakai `drop_duplicates()` baris penuh.
- **Reproducibility**: seed semua (`numpy`, `random`, `tensorflow`), dan `PYTHONHASHSEED`.

---

## 10. Deliverables akhir (yang harus ada saat selesai)

- Repo lengkap sesuai struktur Bagian 3.
- `reports/results/`: `metrics_summary.csv`, `dm_tests.csv`, `gt_ablation_comparison.csv`, `inventory_params.csv`, `cost_impact.csv`, `sarima_orders.csv`.
- `reports/figures/`: series grid, ACF/PACF, decompose, actual-vs-pred, RF importance.
- `models/`: artefak terlatih + scaler.
- `app/dashboard.py`: DSS berjalan.
- `README.md`: setup, cara run pipeline & dashboard, ringkasan hasil.
- `pytest` hijau.

Hasil numerik (metrik, orde, params) dari eksekusi ini akan mengisi angka-angka di **Bab IV (Hasil & Pembahasan)** skripsi.
