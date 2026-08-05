# IDX Super Scanner v9.1.0 — Database-First Macro Scanner

Jalur produksi:

```text
Persistent database
→ macro/regime
→ sector opportunity map
→ business quality + future fundamental
→ valuation + management/capital allocation
→ narrative–money flow
→ SMC/ICT execution
```

Output utama hanya:

1. **The Next Leader**
2. **Swing Ready**

EOFF, financial astrology, time-cycle, Best Buy Date, dan AI additive overlay tidak digunakan.

## Dua mode kerja

### Isi Database

Mode ini dijalankan lebih dahulu sampai status `READY_FOR_DAILY_SCAN`.

- Membaca seluruh evidence yang sudah tersimpan.
- Memilih ticker `MISSING` terlebih dahulu, kemudian `STALE`.
- Portfolio mendapat prioritas.
- Mengambil snapshot fundamental, histori fundamental, status pasar, disclosure/news, dan membangun narrative memory.
- Menyimpan hasil ke local cache dan Supabase bila dikonfigurasi.
- Tidak menjalankan full technical scan sehingga backfill lebih efisien.

Default batch adalah 80 ticker. Universe 400 ticker memerlukan sekitar lima proses dalam kondisi ideal; provider failure dapat menambah jumlah proses.

### Scan Harian

- Membaca fundamental dan narrative memory seluruh universe dari database.
- Hanya memperbarui ticker `MISSING/STALE` sesuai delta-refresh quota.
- OHLCV dan macro tetap diperbarui karena berubah setiap hari.
- Execution verification dibatasi maksimal 40 kandidat akhir.
- Daily Scan diblokir bila database inti belum memenuhi ambang, kecuali pengguna secara eksplisit mengaktifkan partial scan.

## Readiness gate

Default minimum:

- Fundamental snapshot: **90%** universe.
- Fundamental history dengan minimal dua periode: **80%** universe.

Market status dan news ditampilkan sebagai coverage dinamis, tetapi tidak menjadi hard gate database karena sifatnya lebih cepat berubah.

## Batas universe

- Maksimum **400 ticker total**, termasuk portfolio.
- Portfolio diprioritaskan.
- CSV di atas 400 ticker ditolak; tidak ada silent truncation.

## File produksi wajib

- `app.py`
- `database_first.py`
- `scanner.py`
- `scanner_database.py`
- `narrative_engine.py`
- `macro_engine.py`
- `simple_focus.py`
- `two_stage_pipeline.py`
- `free_data_providers.py`
- `ihsg_direction.py`
- `idx_trading_calendar.py`
- `incremental_store.py`
- `research_maintenance.py`
- `selector_engine.py`
- `requirements.txt`
- `runtime.txt`

## File/folder direkomendasikan

- `database/`
- `universes/`
- `STREAMLIT_SECRETS_EXAMPLE.toml`
- `tests/`, `pytest.ini`

## Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Secrets

```toml
ITICK_API_TOKEN = "..."
TWELVE_DATA_API_KEY = "..."
SUPABASE_URL = "..."
SUPABASE_KEY = "..."
```

Jangan commit secret asli ke repository publik.
