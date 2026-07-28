# IDX Super Scanner v7.1.2 — Mixed Datetime Normalization

Scanner Streamlit untuk Multibagger dan Core Swing IDX. v7.1.2 mempertahankan database-first, incremental OHLCV, P0 ranking safety, dan prioritas ranking **Final Score → Silent Accumulation**, lalu menambahkan P1 research-maintenance layer.

## Arsitektur v7.1.2

1. Bulk-read cache Supabase.
2. Periksa freshness, semantic model/parser version, `next_check_at`, dan unresolved material events.
3. Prioritaskan portfolio/execution candidates, event-triggered ticker, lalu cohort round-robin.
4. Panggil provider hanya untuk queue terbatas.
5. Simpan normalized facts, lineage, event fingerprint, backfill state, dan model registry.
6. Hitung ulang valuation/ranking dengan harga terbaru.
7. Simpan dan resolve outcome memory EOFF serta Silent Accumulation setelah forward bars cukup.

## Ranking produksi

Urutan Top 20 tetap:

1. hard eligibility gate;
2. Final Score tertinggi;
3. Silent Accumulation terkalibrasi tertinggi;
4. base conviction;
5. action readiness;
6. ticker sebagai deterministic tie-break.

Time-cycle/EOFF adalah timing overlay dan tidak menaikkan Multibagger Quality Score.

## P1 yang ditambahkan

- semantic model registry dan model-aware refresh;
- event-aware database refresh;
- bounded round-robin fundamental backfill;
- liquidity-bucket calibration untuk Silent Accumulation;
- durable EOFF/Silent Accumulation outcome memory;
- IDX trading-calendar hook dengan fallback yang diberi label unverified;
- unique-anchor gate agar proyeksi time-cycle dari anchor yang sama tidak dihitung sebagai bukti independen;
- dashboard hit/miss/backfill/outcome audit.

## Database prerequisite

Jalankan berurutan:

1. `database/schema_v2.sql` — hanya untuk proyek database baru;
2. `database/migration_v3_database_first.sql`;
3. `database/migration_v4_research_memory.sql`.

Untuk pengguna yang sudah menjalankan schema v2 dan migration v3, cukup jalankan migration v4.

## Deployment

1. Jalankan migration v4 dan verify v4 di Supabase SQL Editor.
2. Ganti seluruh isi root repository dengan paket v7.1.2.
3. Jangan menghapus Streamlit Secrets.
4. Reboot app.
5. Jalankan scan 5 ticker, lalu ulangi ticker yang sama.
6. Target health: `HEALTHY_V4_RESEARCH_MEMORY`.

Scan pertama setelah upgrade dapat melakukan refresh lebih banyak karena cache lama belum memiliki parser/model lineage v7.1. Setelah cache diperbarui, request provider kembali turun.
