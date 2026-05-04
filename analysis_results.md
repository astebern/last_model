# 🔬 Analisis Pipeline V20 — Parameter & Optimasi

## Ringkasan Situasi

| Metrik | Nilai |
|---|---|
| Best Score (v20) | **2.8864** (MAE) |
| v21 (tweak smart rounding 0.08) | 2.8965 ❌ naik → revert |
| Tren 5 terakhir | 2.8893 → 2.8977 → 2.9042 → 2.8949 → **2.8864** |
| Plateau indicator | Perubahan < 0.01 sudah ~6 submisi terakhir |

---

## A. Parameter yang Ada di Kode TAPI Belum Pernah Dieksperimen

README hanya mendokumentasikan **6 parameter**. Tapi di `pipeline_v20.py` ada **18+ parameter tunable** yang belum pernah disentuh:

### 🔴 HIGH IMPACT — Belum Dicoba Sama Sekali

| # | Parameter | Lokasi | Nilai Saat Ini | Saran Eksperimen |
|---|---|---|---|---|
| 1 | `learning_rate` (LGB) | Baris 56 | `0.02` | Coba `0.01` (lebih lambat tapi lebih presisi, butuh n_estimators lebih tinggi) |
| 2 | `num_leaves` (LGB) | Baris 57 | `31` | Coba `15`, `20`, `63` — ini **sangat** mempengaruhi model complexity |
| 3 | `max_depth` (LGB+XGB) | Baris 57, 66 | `6` | Coba `4`, `5`, `8` |
| 4 | `min_child_samples` / `min_child_weight` | Baris 57, 66 | `50` | Coba `30`, `100` — regularisasi leaf-level |
| 5 | `subsample` | Baris 58, 67 | `0.8` | Coba `0.6`, `0.7`, `0.9` |
| 6 | `colsample_bytree` | Baris 58, 67 | `0.7` | Coba `0.5`, `0.6`, `0.8` |
| 7 | `reg_alpha` (L1) | Baris 59, 68 | `1.0` | Coba `0.1`, `0.5`, `3.0`, `5.0` |
| 8 | `reg_lambda` (L2) | Baris 59, 68 | `2.0` | Coba `0.5`, `1.0`, `5.0`, `10.0` |
| 9 | `n_estimators` | Baris 56, 65 | `3000` | Coba `5000` + early_stopping pada final train |
| 10 | `HOME_ADV` (Elo) | Baris 51 | `100` | Coba `50`, `75`, `150` — standar FIFA = ~100, tapi data kamu termasuk women's |
| 11 | `INITIAL_ELO` | Baris 50 | `1500` | Biasanya stabil, tapi bisa coba `1300` untuk menekan tim tanpa history |
| 12 | Rolling window (recent form) | Baris 212 | `10` (last 10 matches) | Coba `5`, `7`, `15`, `20` |

### 🟡 MEDIUM IMPACT — Structural Parameters

| # | Parameter | Lokasi | Nilai Saat Ini | Saran |
|---|---|---|---|---|
| 13 | `purge_days` (CV) | Baris 821 | `30` | Coba `7`, `14`, `60` |
| 14 | `n_folds` (CV) | Baris 821 | `5` | Coba `3` atau `7` |
| 15 | Tournament K-factors | Baris 79-86 | Hardcoded dict | Tweak K untuk "Friendly" (`20` → `15`?) dan "World Cup" (`60` → `50`?) |
| 16 | Tournament importance weights (sample_weight) | Baris 807-809 | `{5:2.0, 4:1.5, ...}` | Coba memperbesar gap: `{5:3.0, 4:2.0, 3:1.5, 2:1.0, 1:0.4}` |
| 17 | Era bins | Baris 407 | `[1870,1930,1960,1990,2010,2030]` | Bins ini sudah outdated; coba `[1950,1980,2000,2010,2020,2030]` |
| 18 | Final clip upper bound | Baris 890-891 | `15` | Coba `10` atau `8` — apakah ada tim yang mencetak >8 gol di test period? |

---

## B. Masalah Struktural & Peluang Optimasi Besar

### 🔥 1. Kamu Punya Optuna tapi TIDAK Pakai (`RUN_TUNING = False`)

```python
# Baris 46
RUN_TUNING = False  # ← INI!
```

Kamu sudah menulis fungsi `tune_hyperparameters()` (baris 609-656) tapi **tidak pernah mengaktifkannya** untuk final training. Optuna hanya di-tune pada LightGBM, dan hasilnya di-apply ke `LGB_PARAMS` saja — **XGB_PARAMS tidak pernah di-tune**.

> [!IMPORTANT]
> **Saran:** Ubah `RUN_TUNING = True` dan tingkatkan `TUNING_TRIALS = 50-100`. Lalu buat Optuna juga tuning XGB_PARAMS. Ini adalah **satu-satunya cara terbaik** menemukan parameter optimal secara sistematik dibanding coba-coba manual satu per satu.

### 🔥 2. Final Training TANPA Early Stopping = Overfitting Risk

```python
# Baris 842-846 — TIDAK ADA early stopping!
lgb_team = lgb.LGBMRegressor(**LGB_PARAMS)
lgb_team.fit(X_full, yt_full, sample_weight=w_full)  # ← langsung fit 3000 trees
```

Di CV kamu pakai `early_stopping(100)`, tapi di final training kamu fit penuh 3000 iterasi tanpa validation set. Ini artinya model kemungkinan **overfitting** di akhir.

> [!WARNING]
> **Saran:** Split sedikit data sebagai validation (misal 95:5), lalu gunakan early stopping juga di final training. Atau ambil `best_iteration_` dari CV dan set `n_estimators` ke rata-rata best iteration + buffer.

### 🔥 3. Fitur dari Dataset YANG TIDAK DIPAKAI SAMA SEKALI

Dataset `train.csv` punya kolom yang **ada di train tapi kamu abaikan**:

| Kolom Train | Status di Pipeline |
|---|---|
| `team_points_last5` | ❌ **Tidak dipakai** |
| `opp_points_last5` | ❌ **Tidak dipakai** |
| `points_last5_diff` | ❌ **Tidak dipakai** |
| `team_gd_last5` | ❌ **Tidak dipakai** |
| `opp_gd_last5` | ❌ **Tidak dipakai** |
| `gd_last5_diff` | ❌ **Tidak dipakai** |
| `h2h_points_last5` | ❌ **Tidak dipakai** |
| `h2h_gd_last5` | ❌ **Tidak dipakai** |
| `days_since_last_match_team` | ❌ **Tidak dipakai** (kamu buat sendiri dari scratch) |
| `days_since_last_match_opp` | ❌ **Tidak dipakai** |
| `team_points_last10` | ❌ **Tidak dipakai** |
| `opp_points_last10` | ❌ **Tidak dipakai** |
| `team_avg_goals_last5` | ❌ **Tidak dipakai** |
| `team_avg_conceded_last5` | ❌ **Tidak dipakai** |
| `opp_avg_goals_last5` | ❌ **Tidak dipakai** |
| `opp_avg_conceded_last5` | ❌ **Tidak dipakai** |
| `team_win_rate_last10` | ❌ **Tidak dipakai** |
| `opp_win_rate_last10` | ❌ **Tidak dipakai** |
| `elo_team` (raw dari train) | ❌ **Tidak dipakai** (kamu hitung sendiri) |
| `elo_opponent` (raw dari train) | ❌ **Tidak dipakai** |
| `rank_team` | ❌ **Tidak dipakai** |
| `rank_opponent` | ❌ **Tidak dipakai** |
| `rank_diff` | ❌ **Tidak dipakai** |

> [!CAUTION]
> **Ini masalah besar!** Kolom `rank_team`, `rank_opponent`, dan semua `*_last5`/`*_last10` tersedia di **train** tapi **tidak di test** (lihat header test.csv — hanya 20 kolom basic). Artinya kamu memang benar tidak bisa langsung memakainya di test. **TAPI** kamu bisa gunakan kolom-kolom ini **sebagai auxiliary target/validation signal** atau untuk men-derive fitur baru di train yang correlate-nya dengan goal scoring.

### 🔥 4. Model Architecture — Coba Pendekatan Berbeda

Kamu sudah stuck di **Poisson regression ensemble**. Beberapa alternatif:

| Pendekatan | Deskripsi | Potensi |
|---|---|---|
| **Ordinal Regression** | Goals itu ordinal (0,1,2,3,4). Ordinal objective bisa lebih baik dari Poisson | ⭐⭐⭐ |
| **CatBoost** | Tambahkan sebagai model ke-3 dalam ensemble | ⭐⭐⭐ |
| **Stacking** | Gunakan output LGB+XGB sebagai fitur → meta-learner (Ridge/Logistic) | ⭐⭐ |
| **Separate model per gender** | Gender M dan W punya distribusi gol yang sangat beda | ⭐⭐⭐ |
| **Separate model per tournament type** | Major vs Friendly punya pola berbeda | ⭐⭐ |

### 🔥 5. Feature Engineering Lanjutan yang Belum Ada

```
✅ = Sudah ada     ❌ = Belum ada
```

| Fitur | Status |
|---|---|
| Elo | ✅ |
| Team encoding (Bayesian) | ✅ |
| H2H stats | ✅ |
| Recent form (rolling 10) | ✅ |
| Tournament importance | ✅ |
| Geographic/socio-economic | ✅ |
| Time decay | ✅ |
| **Momentum/streak** (berapa kali menang beruntun) | ❌ |
| **Form trajectory** (apakah tim sedang naik/turun — slope of recent form) | ❌ |
| **Venue familiarity** (seberapa sering tim bermain di venue_country itu) | ❌ |
| **Opponent strength-adjusted metrics** (goals scored vs strong opponents only) | ❌ |
| **Time-since-tournament-start** (match day 1 vs final) | ❌ |
| **Rest days advantage** (days_since team minus days_since opp) | ❌ |
| **Goal scoring tendency per era/decade** | ❌ |
| **Confederation cross-match historical pattern** (UEFA vs CAF typically = ?) | ❌ |

---

## C. Quick Wins — Rekomendasi Prioritas

Berikut urutan yang saya sarankan, **satu per satu** sesuai metode kamu:

### Prioritas 1: Aktifkan Optuna (Impact: ⭐⭐⭐⭐⭐)
```python
RUN_TUNING = True
TUNING_TRIALS = 80
```
Hyperparameter tuning otomatis akan menangani parameter #1-#8 di tabel atas sekaligus. Ini **jauh lebih efektif** daripada ubah manual satu per satu.

### Prioritas 2: Early Stopping di Final Training (Impact: ⭐⭐⭐⭐)
Ubah final training agar pakai validation split + early stopping, bukan fit mentah 3000 trees.

### Prioritas 3: Tambahkan CatBoost ke Ensemble (Impact: ⭐⭐⭐⭐)
```python
import catboost as cb
# 3-model ensemble: W_LGB=0.5, W_XGB=0.2, W_CAT=0.3
```

### Prioritas 4: Separate Model per Gender (Impact: ⭐⭐⭐)
Training 2 model terpisah (Male & Female) karena distribusi gol dan kompetitivitas sangat berbeda.

### Prioritas 5: Feature Engineering Baru (Impact: ⭐⭐⭐)
Tambahkan **win streak**, **form slope**, dan **rest day advantage** — fitur-fitur ini murah tapi informatif.

### Prioritas 6: Rolling Window Size (Impact: ⭐⭐)
Ubah `rolling(10)` menjadi eksperimen: `rolling(5)` dan `rolling(15)`.

---

## D. Parameter yang Sudah Optimal / Tidak Perlu Diubah

Berdasarkan submission history, berikut yang sudah terbukti optimal:

| Parameter | Optimal | Evidence |
|---|---|---|
| Year cutoff | `>= 1950` | v2-v8 sudah eksplorasi range, 1950 terbaik |
| Clipping | `upper=4` | v14 vs v15 membuktikan 4 > 5 > 6 |
| Ensemble weight | `0.8/0.2` | v12 vs v13: 80:20 terbaik |
| Smart rounding threshold | `0.1` | v20 vs v21: 0.1 > 0.08 |
| Alpha (time decay) | `0.0005` | v19 vs v15: 0.0005 > 0.0001 |
| Bayesian prior weight | `10` | v15 vs v16-v18: 10 terbaik (7, 9, 12 lebih buruk) |

---

> [!TIP]
> **Bottom line:** Skor kamu plateau karena kamu sudah mengoptimalkan semua **"surface-level" hyperparameters** (clipping, weights, rounding). Yang belum tersentuh adalah **model internals** (LGB/XGB hyperparameters via Optuna), **model architecture** (CatBoost, gender split), dan **feature engineering baru** (streak, slope, venue familiarity). Di sinilah gain berikutnya akan datang.
