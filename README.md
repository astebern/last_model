## SUBMISSION HISTORY:
submission 2: 2.9467
    full scan (baseline)

submission 3: 2.9664
    dataset year >1998

submission 4: 2.9458
    dataset year >1970

submission 5: 2.9402 
    dataset year >1950 

submission 6: 2.9425
    dataset year >1945

submission 7: 2.9453
    dataset year >1950 + Recency Weighting, Delete Mismatch Friendlies, Geographic Smart Imputation     
    
submission 8: 2.9598
    dataset year >1962
    
submission 9: 3.6070
    dataset year >1950 + Recency Weighting, Delete Mismatch Friendlies, Geographic Smart Imputation, Penalty Conversion 

submission 10:2.9071
    skip

submission 11: 2.9071
    v5 + optuna

submission 12: 2.9039
    v11 + 70:30

submission 13: 2.9025
    v12 + 80:20

submission 14: 2.9020 
    v13 + clipping upper 5

submission 15: 2.8893 
    v14 + clipping upper 4

submission 16: 2.8977
    v15 + bayesian 7

submission 17: 2.9042
    v16 + bayesian 12

submission 18: 2.9025
    v17 + bayesian 9

submission 19: 2.8949
    v15 + alpha 0.0001

submission 20: 2.8864 (best)
    v15 + smart rounding 0.1

submission 21: 2.8965
    v20 + smart rounding 0.08
 
 submission 22: 2.9262
    optuna ON

## INSIGHT OPTIMAL: 
- Year : > 1950 
- Clipping : < 4
- Ensemble : 80:20
- alpha : 0.0005
- Smart Rounding: 0.1
- Bayesian : 10

## CUSTOMIZE PARAMETER
### 1. Rentang Tahun Data (Time Trimming)
Sepak bola tahun 1950 sangat berbeda dengan sepak bola modern. Mengambil data terlalu lama bisa menjadi noise, sementara mengambil terlalu sedikit bisa kekurangan sample size.
- *Lokasi di kode:* Baris 702 (train = train[train["date"] >= '1950-01-01'])
- *Nilai Saat Ini:* '1950-01-01'
- *Eksperimen yang patut dicoba:* 
  - '1998-01-01' (Era sepak bola modern/Format Piala Dunia baru)
  - '1990-01-01' (Transisi sepak bola)
  - '2000-01-01' atau '2010-01-01'

### 2. Batas Maksimal Gol (Outlier Clipping)
Model berbasis regresi (terutama Poisson) rentan rusak jika ada outlier gol yang sangat ekstrim (misal: 15-0 atau 31-0). Anda bisa membatasi maksimal skor wajar yang dipelajari model.
- *Lokasi di kode:* Baris 706 & 707 (train["team_goals"] = train["team_goals"].clip(upper=6))
- *Nilai Saat Ini:* 6
- *Eksperimen yang patut dicoba:* 
  - 4 atau 5 (Membuat model fokus belajar membedakan hasil skor ketat 1-0, 2-1)
  - 7 atau 8 (Membiarkan model belajar potensi pembantaian gol)

### 3. Bobot Ensemble Model (LightGBM vs XGBoost)
Kombinasi persentase kepercayaan dari dua model ini seringkali menjadi penentu akurasi desimal akhir.
- *Lokasi di kode:* Baris 72 (W_LGB, W_XGB = 0.6, 0.4)
- *Nilai Saat Ini:* 0.6 (60% LGBM) dan 0.4 (40% XGBoost)
- *Eksperimen yang patut dicoba:* 
  - 0.7 LGBM / 0.3 XGBoost
  - 0.5 LGBM / 0.5 XGBoost
  - 0.8 LGBM / 0.2 XGBoost

### 4. Tingkat Kecepatan Melupakan Sejarah (Exponential Time Decay)
Menentukan seberapa drastis model "meremehkan" data dari masa lalu menggunakan variabel parameter *alpha*.
- *Lokasi di kode:* Baris 812 (alpha = 0.0005)
- *Nilai Saat Ini:* 0.0005 (Dalam ~4 tahun bobot berkurang drastis)
- *Eksperimen yang patut dicoba:* 
  - 0.001 (Lupa sangat cepat, murni fokus ke tren jangka sangat pendek)
  - 0.0001 (Sangat lambat melupakan sejarah)
  - 0.0 (Mematikan efek time decay sepenuhnya)

### 5. Smart Rounding Threshold (Agresivitas Pembulatan)
Ini mengatur seberapa yakin model harus mengintervensi pembulatan standar untuk menjaga status "Kemenangan/Kekalahan".
- *Lokasi di kode:* Baris 668 di dalam fungsi smart_round (if raw_result != rounded_result and abs(diff) > 0.2:)
- *Nilai Saat Ini:* 0.2 (Jika desimal menang melebihi selisih 0.2 poin, paksa hasil agar tidak seri).
- *Eksperimen yang patut dicoba:* 
  - 0.1 (Sangat agresif: Pokoknya kalau prediksi model condong ke salah satu tim walau sedikit, paksa menang) 
  - 0.3 atau 0.4 (Lebih konservatif, biarkan seri kecuali model sangat yakin ada perbedaan ketat)

### 6. Bayesian Smoothing Factor (Kepercayaan pada Tim Jarang Main)
Mengatasi tim-tim "kuda hitam" atau negara terpencil yang jumlah pertandingannya sangat sedikit agar statistiknya tidak loncat sembarangan.
- *Lokasi di kode:* Baris 52 (PRIOR_WEIGHT = 10)
- *Nilai Saat Ini:* 10
- *Eksperimen yang patut dicoba:* 
  - 5 (Lebih percaya pada statistik asli tim tersebut meskipun datanya sedikit)
  - 20 (Lebih ragu-ragu: tarik paksa performa tim tersebut ke arah rata-rata dunia)

### Tips Eksekusi:
Saya sarankan Anda *jangan mengubah semuanya sekaligus. Ubah SATU parameter (misal: Tahun Potong Data ke 1998), buat file *submission, lalu submit. Evaluasi apakah skor naik atau turun. 
Jika naik, jadikan tahun 1998 sebagai batas permanen (base), lalu pindah bereksperimen dengan parameter kedua. Metodologi (Satu-Per-Satu) ini adalah cara paling efektif saat kompetisi.