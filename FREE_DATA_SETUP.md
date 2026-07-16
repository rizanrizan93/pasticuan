# Free Data Setup — v4.3.1

Scanner ini tidak mewajibkan layanan berbayar. Arsitektur OHLCV harian berjalan dengan urutan berikut:

1. `CACHE_FRESH_VERIFIED` — memakai full history yang sudah pernah diverifikasi apabila bar EOD terakhir masih current.
2. `LIVE_YAHOO` — Yahoo/yfinance hanya dipanggil untuk ticker yang stale atau belum memiliki cache current.
3. `LIVE_IDX_EOD_PATCH` — bila Yahoo gagal tetapi cache historis tersedia, bar EOD terakhir ditambal dari IDX Stock Summary resmi.
4. `LIVE_ITICK_FREE_FALLBACK` — bila token iTick gratis tersedia, ticker yang masih gagal dicoba melalui historical Kline iTick.
5. `CACHE_FALLBACK` — cache lama tetap ditampilkan untuk riset, tetapi diblokir dari `BUY_LIMIT`.
6. `UNAVAILABLE` — seluruh provider gagal dan tidak ada cache layak.

## Mode tanpa API token

Tidak ada konfigurasi tambahan. Scanner memakai cache, Yahoo, IDX Stock Summary, Google News, Google Finance, dan halaman resmi IDX.

Mode ini sudah lebih tahan rate limit daripada v4.3.0 karena full history tidak diunduh ulang ketika cache EOD masih current.

## Menambahkan iTick free-tier

Token iTick bersifat opsional. Buat akun free-tier, lalu masukkan token ke Streamlit Community Cloud:

1. Buka aplikasi di Streamlit Community Cloud.
2. Buka **App settings**.
3. Pilih **Secrets**.
4. Tambahkan:

```toml
ITICK_API_TOKEN = "token_anda"
```

Jangan menaruh token asli di GitHub.

Scanner menerapkan guard internal maksimum 4 REST call per menit, di bawah batas free-tier 5 call per menit. Bila kuota internal habis, scanner tidak crash; ticker tetap masuk fallback berikutnya.

## Twelve Data

`TWELVE_DATA_API_KEY` tetap opsional dan hanya dipakai untuk validasi harga independen. Scanner tidak bergantung pada Twelve Data untuk OHLCV utama.

```toml
TWELVE_DATA_API_KEY = "key_opsional"
```

## Batasan realistis

- Pada cold start tanpa cache, full historical data masih terutama bergantung pada Yahoo karena IDX Stock Summary hanya menyediakan bar EOD per tanggal.
- iTick free-tier tidak dapat mengisi ratusan ticker sekaligus karena batas call per menit; adapter dipakai untuk ticker gagal secara bertahap.
- Cache Streamlit Community Cloud dapat hilang ketika aplikasi diredeploy, reboot, atau instance dibangun ulang.
- Intraday BPJS/BSJP tetap lebih sulit daripada daily OHLCV. iTick menjadi fallback opsional, tetapi tidak menjamin seluruh shortlist pulih dalam satu scan.
- Bar dari provider sekunder dapat memakai basis raw price. Scanner menyelaraskannya terhadap overlap cache bila stabil dan tetap memberi peringatan corporate action.

## Status yang aman untuk eksekusi

- `LIVE_YAHOO`, `LIVE_YAHOO_RETRY`, `LIVE_IDX_EOD_PATCH`, `LIVE_ITICK_FREE_FALLBACK`, dan `CACHE_FRESH_VERIFIED` dapat melanjutkan ke gate berikutnya.
- `CACHE_FALLBACK` dan `UNAVAILABLE` tidak dapat menghasilkan direct `BUY_LIMIT` dalam real-money mode.
- Harga independen, freshness, corporate action, data completeness, dan risk gate tetap harus lulus.
