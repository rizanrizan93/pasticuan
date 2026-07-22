# IDX Scanner v6.7.0 — Multibagger & Core Swing Focus

Scanner IDX berbasis Streamlit dengan dua fokus produksi:

1. **Multibagger** — mencari The Next Leader, turnaround/cyclical recovery, ekspansi kapasitas, rerating, dan pertumbuhan fundamental yang dapat dipertanggungjawabkan.
2. **Core Swing** — Pullback Continuation, Breakout Retest, Reversal Accumulation, dan Unicorn/ICT harian dengan entry, trigger, stop-loss, TP1, TP2, RR, expiry, serta validasi Stockbit.

Mesin fast-trade telah dihapus. Build ini tidak mempunyai menu, fungsi, export, shortlist, atau pengambilan intraday khusus untuk strategi cepat.

## Modul utama

- `app.py` — antarmuka Streamlit dan orchestration.
- `scanner.py` — OHLCV harian, indikator, setup Core Swing, fundamental, gate risiko, portfolio, dan provider resilience.
- `scanner_focus.py` — future fundamental, Multibagger scoring, capital allocation, Core Swing conviction ranking, dan Daily Focus.
- `time_cycle.py` — Time-Cycle dan Best Buy Date.
- `eoff_reconstruction.py` — rekonstruksi clean-room EOFF.
- `ai_engine.py` — AI lokal guarded untuk ranking.
- `dashboard_v660.py` — Top 20 Multibagger + Swing/Core.

## Alur scanner

1. Upload CSV ticker.
2. Ambil OHLCV harian dengan cache dan fallback provider.
3. Hitung empat setup Core Swing.
4. Ambil fundamental official-first: IDX/XBRL → cache → Yahoo/Twelve Data sebagai fallback atau cross-check terbatas.
5. Kumpulkan future fundamental: proyek, manajemen, ownership/governance evidence, dan catalyst.
6. Bangun Multibagger score dan capital allocation.
7. Terapkan Time-Cycle/EOFF hanya sebagai guarded timing overlay.
8. Tampilkan Top 20 unik, Core Swing Ranking, Multibagger, Daily Focus, portfolio review, dan audit data.

## Prinsip produksi

- Formula proprietary tidak disalin atau diklaim.
- EOFF tidak menggantikan fundamental dan struktur harga.
- Data yang tidak lengkap menurunkan confidence atau membuat sinyal fail-closed.
- `SIGNAL_READY` bukan instruksi beli.
- Semua order tetap diverifikasi dan dikirim manual melalui Stockbit.

## Deployment

Upload seluruh file root ke GitHub, lalu deploy `app.py` pada Streamlit Community Cloud. Simpan API key opsional di Streamlit Secrets. Jangan menyimpan credential di repository.
