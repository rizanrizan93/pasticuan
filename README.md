# IDX Super Scanner v8.0.2 — Two-Stage Slim Production

V8.0.2 mempertahankan kontrak scoring disjoint V8.0.0, lalu memperbaiki bottleneck scan 400 ticker dengan pipeline dua tahap dan cache-first.

## Arsitektur produksi

### Stage A — Full-universe local/cache scan

Seluruh ticker diproses menggunakan:

- OHLCV dan benchmark IHSG;
- indikator teknikal, SMC/ICT, setup, dan silent-accumulation proxy;
- fundamental, market status, dan news yang sudah tersedia di database/cache;
- preliminary routing score dengan **production weight 0%**.

Preliminary routing hanya menentukan ticker mana yang layak menerima enrichment mahal. Ia bukan final score dan tidak boleh memengaruhi alokasi modal.

### Stage B — Bounded external enrichment

Hanya shortlist default 60 ticker yang menerima:

- live fundamental snapshot secara terbatas;
- disclosure/news refresh;
- market-status refresh;
- execution snapshot;
- independent-price verification untuk kandidat teratas;
- final Multibagger dan Core Swing scoring.

Official IDX/XBRL berjalan round-robin default 40 ticker per scan. Cursor disimpan sehingga ticker yang belum diperbarui mendapat giliran pada scan berikutnya.

## Kontrak anti-double-counting

Final production score tetap dimiliki eksklusif oleh `production_scoring.py`.

### Multibagger

- Fundamental dan future fundamental: 55%
- Pure narrative dan issuer actions: 25%
- Market dan sector strength: 10%
- Silent accumulation: 10%

### Core Swing

- Technical execution: 45%
- Market, sector, dan liquidity: 15%
- Pure narrative dan issuer actions: 20%
- Silent accumulation: 15%
- Data quality dan OOS validation: 5%

AI, EOFF, Time-Cycle, Best Buy Date, hybrid conviction, dan legacy narrative/Emir additive overlays tidak memiliki bobot produksi.

## Batas komputasi harian

Daily scan menggunakan maksimal 750 bar per ticker. Batas ini tetap mencakup EMA200, struktur 52 minggu, dan selector lookback dengan buffer. Riwayat penuh hanya digunakan bila chronological OOS diaktifkan.

## Default operasional 400 ticker

- Stage-B shortlist: 60 ticker
- Yahoo/fundamental live subset: maksimal 24 ticker
- Official IDX/XBRL round-robin: 40 ticker
- Independent-price verification: mengikuti `max_automatic_price_candidates`, default maksimal 40
- OOS: nonaktif untuk daily scan

## Benchmark lokal

Synthetic benchmark 400 ticker × 750 bar, tanpa network/provider:

- Core local scan: 21,87–24,46 detik
- Lightweight preliminary routing: 0,49–0,79 detik
- Shortlist selection: sekitar 0,02 detik
- Final production scoring 60 ticker: 3,05–3,16 detik
- Total komputasi lokal: **25,43–28,42 detik**

Waktu end-to-end tetap bergantung pada provider, cache, rate limit, dan jumlah data yang perlu diperbarui. Benchmark ini mengukur komputasi lokal, bukan latency internet.

## Menjalankan aplikasi

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload CSV ticker pada kolom `ticker` atau `symbol`. Portfolio CSV bersifat opsional.

## File utama

- `app.py` — Streamlit orchestration dan UI
- `scanner.py` — data acquisition, core scan, gates, cache
- `scanner_focus.py` — final Multibagger/Core construction
- `production_scoring.py` — satu-satunya pemilik production score
- `two_stage_pipeline.py` — shortlist, round-robin refresh, coverage audit
- `scanner_database.py` — persistent database/readback
- `tests/test_v8_no_double_counting.py` — scoring contract
- `tests/test_v801_two_stage.py` — cache-first dan two-stage contract

## Status data

Scanner bersifat fail-closed. Data yang tidak tersedia tidak diberi nilai positif. Kekurangan evidence menurunkan coverage dan dapat menghasilkan `DATA_PENDING`, bukan skor default palsu.

## Hotfix integritas v8.0.2

V8.0.2 menambahkan empat proteksi produksi:

- conditional trigger wajib berada di bawah target;
- seluruh kandidat, termasuk trigger yang sudah confirmed, wajib lulus atomic entry-plan contract;
- tier sumber OHLCV dari `DownloadReport` wajib ditempelkan ke signal row sebelum execution gate;
- cache calculated-result v8.0.1 tidak digunakan ulang.

Laporan audit lengkap tersedia di `BUILD_VALIDATION_V8_0_2.md`.

> Catatan: hasil 400 ticker yang disertakan dalam folder `validation` merupakan deterministic engineering fixture, bukan scan live IDX atau rekomendasi transaksi.
