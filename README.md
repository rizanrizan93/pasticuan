# IDX Super Scanner v7.4.0 — Statistical Gates & Quality Momentum

Scanner Streamlit untuk riset **Multibagger** dan **Core Swing** IDX, dengan
mesin arah IHSG tiga horizon. Sistem ini dibuat untuk meningkatkan kualitas
keputusan dan menjaga compounding; tidak menjanjikan profit, kekayaan, atau
akurasi arah pasar.

## Yang baru di v7.4.0

- Python 3.12 memakai `scipy==1.18.0` secara eksplisit; Spearman, BCa
  bootstrap, dan paired permutation test mempunyai NumPy fail-safe yang
  terlihat di audit;
- promosi AI tidak lagi memakai point estimate saja. Lower confidence bound
  Brier skill dan expectancy net harus positif, advantage versus baseline
  terkuat harus positif dan signifikan setelah koreksi tiga horizon, serta
  drawdown harus tetap di bawah guard;
- validasi statistik dihitung per tanggal walk-forward agar anggota satu
  cross-section tidak keliru dianggap sebagai observasi independen;
- Core Swing menambah 52-week-high proximity, momentum continuity, trend
  efficiency, jump concentration, drawdown, dan konsistensi 20D/60D/120D;
- Multibagger menambah lima pilar eksplisit: growth persistence,
  profitability termasuk gross profitability, cash conversion/accrual
  quality, balance-sheet safety, dan reinvestment runway;
- perbandingan kualitas sektoral hanya aktif bila tersedia minimal lima peer;
  data sektor yang tidak cukup tetap netral dan tidak difabrikasi;
- time-cycle/EOFF penuh ditunda untuk emiten yang gagal ambang kualitas
  strategis, sehingga CPU dipakai lebih dahulu untuk kandidat relevan;
- panel dan hasil evaluasi model mempunyai cache revision-aware. Perubahan
  data/config/version tetap otomatis menginvalidasi cache.

Deployment yang diuji menggunakan Python 3.12. Pilih Python 3.12 dari
**Advanced settings** Streamlit Community Cloud.

## Yang baru di v7.3.0

- cross-sectional selector memprediksi excess return net setelah biaya terhadap
  IHSG untuk horizon 5D, 20D, dan 60D;
- walk-forward membandingkan `RULE_ENGINE`, `INDEPENDENT_SELECTOR`,
  `AI_CHALLENGER`, dan `RELATIVE_STRENGTH` pada tanggal evaluasi yang tidak
  terlihat saat training;
- AI selector tetap shadow sampai mengalahkan semua baseline pada expectancy
  net, memiliki Brier skill positif, dan drawdown berada di bawah guard;
- Execution AI mempelajari `P(fill)`, `P(TP1-before-SL)`, expected R net, MFE,
  dan MAE. Bobotnya nol sampai Brier, expectancy, profit factor, dan drawdown
  OOS semuanya lolos;
- outcome AI, snapshot selector, outcome excess-return, dan evaluasi model
  disimpan persisten melalui database schema v6;
- Core Swing sekarang memilih teknikal + relative strength + Silent
  Accumulation lebih dahulu, lalu mencari setup. Saham `NO_SETUP` tetap berada
  di radar dengan trigger/invalidation yang ditunggu;
- Multibagger diranking berdasarkan kualitas strategis, confidence, dan Silent
  Accumulation lebih dahulu. Setup hanya menentukan waktu/deployment;
- setiap kandidat mempunyai alasan terstruktur: alasan dipilih, alasan belum
  entry, trigger yang ditunggu, invalidation, dan risiko utama;
- feature, Silent Accumulation, dan panel selector mempunyai cache
  revision-aware. Uji 10 saham turun dari sekitar 1,72 detik cold menjadi 0,36
  detik pada pengulangan input yang sama.

## Fondasi v7.2.1 yang tetap dipertahankan

- chronological OOS memakai cache per tanggal untuk hard gate dan Silent
  Accumulation; perhitungan rolling yang sama tidak diulang untuk empat setup;
- kandidat historis yang sudah pasti gagal hard gate dihentikan sebelum
  detector lengkap dijalankan, tanpa mengubah hasil keputusan;
- cohort OOS harian dibatasi secara deterministik dan berstrata likuiditas
  (default 60 ticker), bukan dipilih berdasarkan score/return;
- audit waktu per tahap tampil di aplikasi agar bottleneck CPU, network, cache,
  dan database dapat dibedakan;
- mode fundamental seluruh universe menjadi opt-in untuk riset
  mingguan/akhir pekan, bukan default scan harian;
- empat private legacy definition yang terbukti tidak mempunyai call-site
  dihapus; wrapper versi lama yang masih menjadi implementation layer aktif
  tetap dipertahankan;
- warning concat all-NA, DateOffset, dan DataFrame attrs untuk Streamlit
  dibersihkan;
- baris `NO_SETUP` tidak lagi dapat menampilkan quality grade `A/B`; partial
  detector score disimpan terpisah sebagai `setup_diagnostic_score`;
- perbaikan crash kolom duplikat Top 20 dari v7.2.0 tetap dipertahankan.

## Yang baru di v7.2.0

- prediksi arah IHSG 1/5/20 hari bursa: `UP`, `SIDEWAYS`, `DOWN`, atau
  `ABSTAIN`;
- fitur point-in-time: EMA20/50/200, momentum, volatility, drawdown, serta
  breadth universe yang diunggah;
- historical analogue dengan minimum jarak antar-event agar hari berdekatan
  tidak dihitung sebagai bukti independen;
- chronological walk-forward untuk setiap horizon;
- probabilitas produksi hanya bila Brier skill mengalahkan baseline, directional
  accuracy dan confidence lolos, serta candle EOD final;
- regime `BULL_CONFIRMED`, `BULL_FRAGILE`, `TRANSITION`, `BEAR_RALLY`, atau
  `BEAR_CONFIRMED`;
- IHSG hanya menjadi **risk-budget cap**. Ia tidak menaikkan Final Score, tidak
  mengubah saham tidak layak menjadi layak, dan tidak membuat order otomatis;
- pada `ACCOUNT_GUARDED`, cap diterapkan pada risk-per-trade, portfolio heat,
  dan budget Multibagger; pada `SIGNAL_FIRST` ia informasional;
- outcome point-in-time `^JKSE` disimpan dan diresolusikan setelah forward bars
  tersedia;
- snapshot IHSG baru di database melalui migration v5;
- crash detail Top 20 akibat requested column duplikat diperbaiki;
- cache fundamental melewati batas kedaluwarsa keras tidak lagi masuk scoring;
- semantic content hash mengabaikan timestamp transport/audit;
- default snapshot write cap dinaikkan dari 500 menjadi 2.000 baris.

## Hirarki keputusan

Top 20 memakai:

1. hard eligibility gate;
2. Final Score kategori—`core_priority_score`/`swing_selection_score` untuk
   Swing dan `multibagger_selection_score` untuk Multibagger;
3. Silent Accumulation;
4. base conviction;
5. action readiness;
6. ticker sebagai deterministic tie-break.

`REJECT`, `BLOCKED`, `NOT_QUALIFIED`, `NO_ALLOCATION`, data tidak
terskor/kedaluwarsa, dan non-syariah dikeluarkan. Best Buy Date/EOFF tetap
timing overlay. IHSG tidak ikut mengurutkan saham; kolom regime, consensus, dan
risk cap hanya menjadi konteks alokasi.

## Cara membaca IHSG Direction

- `raw_direction`: tebakan model sebelum production gate;
- `prediction_state`: arah yang boleh dipakai; menjadi `ABSTAIN` bila salah satu
  gate gagal;
- `P(UP)`, `P(SIDE)`, `P(DOWN)`: frekuensi berbobot historical analogue;
- `OOS_POSITIVE`: walk-forward mengalahkan baseline dan directional evidence
  cukup;
- `risk_budget_multiplier`: batas maksimum eksposur relatif, selalu `<= 1.0`;
- `NO_EDGE`: tidak ada horizon yang layak menjadi sinyal produksi.

Probabilitas bukan kepastian. Breadth bergantung pada universe CSV dan dapat
memiliki survivorship/selection bias. Input makro intraday, rupiah, obligasi,
arus asing, dan corporate action belum otomatis masuk model.

## Database prerequisite

Jalankan berurutan:

1. `database/schema_v2.sql` — hanya untuk proyek database baru;
2. `database/migration_v3_database_first.sql`;
3. `database/migration_v4_research_memory.sql`;
4. `database/migration_v5_ihsg_direction.sql`.
5. `database/migration_v6_selector_ai_outcomes.sql`.

Lalu jalankan `database/verify_v6_selector_ai_outcomes.sql`. Target health:
`HEALTHY_V6_SELECTOR_AI_OUTCOMES`.

## Deployment aman di Streamlit Community Cloud

1. Jalankan migration v5 lalu migration/verify v6 di Supabase.
2. Ganti seluruh isi root repository dengan isi ZIP v7.4.0 dalam satu commit
   atomik; jangan pernah membuat commit antara langkah hapus dan salin.
3. Jangan menghapus Streamlit Secrets.
4. Reboot app.
5. Uji **Arah IHSG** tanpa CSV.
6. Uji scan kecil 5–10 ticker.
7. Ulangi ticker yang sama untuk memeriksa cache/database hit.
8. Naikkan universe bertahap; jangan langsung menjalankan scan/backfill 300
   ticker saat instance sedang CPU-throttled.

## Validasi lokal

```bash
python -m py_compile app.py scanner.py scanner_focus.py scanner_database.py \
  dashboard_v660.py time_cycle.py eoff_reconstruction.py ai_engine.py \
  selector_engine.py research_maintenance.py ihsg_direction.py
python -m unittest discover -p "test*.py"
```

Lihat `BUILD_VALIDATION_V7_3_0.md`, `FIX_REPORT_V7_3_0.md`,
`TEN_STOCK_COMPARISON_V7_3_0_2026_07_28.md`, dan
`docs/SELECTOR_AND_EXECUTION_AI_V7_3.md`, serta
`docs/IHSG_DIRECTION_MODEL_V7_2.md` untuk detail.
