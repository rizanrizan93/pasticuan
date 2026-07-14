# Audit Menyeluruh Super Scanner v4.2.6

## Executive conclusion

v4.2.6 bukan scanner yang rusak. Mesin risiko dan fail-closed logic sudah relatif disiplin, serta 96 unit test awal lulus. Namun paket yang diunggah belum layak disebut release deployment lengkap dan masih memiliki beberapa keterbatasan metodologis yang penting untuk real money.

Release integrasi v4.3.0 mempertahankan detektor inti, memperbaiki isu deployment/UI, memperluas verifikasi harga, memperjelas interpretasi OOS, serta menambah TradingView/Stockbit bridge.

## Scorecard sebelum perbaikan

| Area | Nilai | Catatan |
|---|---:|---|
| Setup logic dan risk gate | 8,2/10 | Structural entry, RR, completeness, context blocker, dan position risk cukup ketat |
| Data resilience | 7,2/10 | Cache/fallback baik, tetapi daily OHLCV dan fundamental masih sangat bergantung pada Yahoo |
| Statistical validation | 6,4/10 | Chronological holdout ada, tetapi statistik ditempel per setup, bukan probabilitas spesifik ticker |
| Specialty screens | 7,5/10 | BPJS/BSJP/ARA/Sniper dipisah dan fail-closed, tetapi intraday bergantung satu provider utama |
| Fundamental/multibagger | 6,8/10 | Faktor cukup lengkap, tetapi full-universe default off dan data laporan tetap provider-limited |
| Maintainability | 5,5/10 | scanner.py sangat besar dan memakai redefinisi/override bertahap |
| Test quality | 7,0/10 | 96 test lulus; coverage scanner sekitar 64%, UI belum tercakup memadai |
| Deployment package | 4,0/10 | ZIP hanya app.py, scanner.py, test_scanner.py; requirements/runtime/README tidak ada |
| Real-money readiness keseluruhan | 7,1/10 | Layak sebagai decision-support, bukan autonomous execution engine |

## Temuan kritis

### 1. Paket deployment tidak lengkap

ZIP v4.2.6 hanya memuat tiga source file. `requirements.txt`, `runtime.txt`, dokumentasi deploy, dan file integrasi tidak tersedia. Streamlit Cloud dapat gagal atau memakai dependency/Python version yang tidak terkontrol.

**Perbaikan v4.3.0:** manifest deployment lengkap ditambahkan.

### 2. Konsentrasi provider

- Daily OHLCV utama: Yahoo/yfinance.
- Intraday 5m utama: Yahoo/yfinance.
- Fundamental utama: Yahoo/yfinance.
- Harga independen sudah lebih baik: IDX Stock Summary → Google Finance → optional Twelve Data.

Cache meningkatkan availability tetapi cache bukan sumber independen. Ketika Yahoo rate-limited, kualitas fundamental/intraday tetap dapat turun.

**Perbaikan v4.3.0:** kandidat automatic price verification dinaikkan dari 24 menjadi 40. Arsitektur multi-provider fundamental penuh belum dapat dibuat tanpa sumber/API tambahan yang stabil.

### 3. OOS bukan probabilitas khusus saham

Backtest memakai chronological holdout dan fixed-rule detector. Hasil kemudian diagregasi berdasarkan setup dan ditempel ke seluruh saham dengan setup sama. Maka `P(TP1<SL)` adalah historical setup-level estimate, bukan probabilitas bahwa ticker tertentu akan mencapai TP1.

**Perbaikan v4.3.0:** label UI dan keterangan Validation diubah agar tidak memberi kesan ticker-specific forecast.

### 4. Backtest belum mencakup seluruh live execution stack

Historical simulation memvalidasi setup teknikal, fill, SL/TP, fee, dan slippage. Ia belum merekonstruksi secara point-in-time seluruh fundamental, news, IDX status, independent-price verification, dan portfolio heat yang digunakan live. Karena itu performa OOS tidak boleh dianggap backtest end-to-end dari dashboard production.

### 5. Multibagger universe dapat tidak lengkap

Default `Multibagger: fundamental seluruh universe` adalah off untuk menekan rate limit. Dalam mode ini fundamental diprioritaskan pada portfolio, execution candidates, dan top-ranked core signals. Emiten growth yang belum memiliki setup teknikal current dapat terlewat.

**Operasional:** gunakan CSV universe Multibagger khusus dan aktifkan full-universe saat riset malam/weekend, bukan saat live scan ratusan ticker.

### 6. Silent accumulation bukan broker-summary aktual

Scanner menggunakan OHLCV flow proxy seperti CMF, OBV slope, up/down value ratio, close location, dan volume behavior. Ini berguna, tetapi bukan beneficial-owner data dan bukan broker accumulation sebenarnya.

**Operasional:** validasi final di Stockbit Bandarmology/Big Accumulation atau broker summary bila tersedia.

### 7. Technical debt dari override bertahap

`scanner.py` sekitar 7.950 baris, 205 fungsi, dan memuat beberapa definisi ulang untuk class/fungsi yang sama. Python memakai definisi terakhir, sehingga aplikasi berjalan, tetapi audit dan perubahan selanjutnya lebih rawan regresi.

**Rekomendasi berikutnya:** refactor terkontrol menjadi modul data, indicators, setups, validation, specialty, risk, dan providers. Jangan dilakukan bersamaan dengan perubahan sinyal tanpa golden-master regression tests.

### 8. Bug dan inconsistency kecil

- Format percent Streamlit pada Risk/Spread/Broksum salah (`%.1%%`/`%.2%%`).
- Teks BPJS menyebut 09:15, sedangkan logic evaluasi efektif 09:20.
- Verifikasi harga otomatis hanya 24 kandidat sementara quote shortlist sampai 40.
- ZIP tidak menyediakan Pine/Stockbit bridge.

**Perbaikan v4.3.0:** seluruh poin di atas diperbaiki.

## Hal yang sudah baik

- Critical blockers fail closed.
- Minimum data completeness 80% dipisahkan dari execution confidence.
- Independent price family tidak menganggap cache Yahoo sebagai sumber kedua.
- Risk sizing account-level dan portfolio heat tersedia.
- Intraday BPJS membedakan opening bars, stale, dan provider unavailable.
- ARA Hunter tidak dipromosikan menjadi direct execution.
- Security review tidak menemukan eval/exec, unsafe pickle, shell execution, atau TLS verification bypass.
- 96 unit test awal lulus sebelum modifikasi.

## Perubahan release v4.3.0

1. Menambahkan `requirements.txt`, `runtime.txt`, `.gitignore`, README, audit report, Pine Script, dan Stockbit preset.
2. Menaikkan maximum automatic independent-price candidates dari 24 ke 40.
3. Mengoreksi window action BPJS menjadi 09:20–10:45 WIB.
4. Mengoreksi format percent UI.
5. Mengganti label OOS menjadi setup-level estimate.
6. Menambah tab TradingView/Stockbit.
7. Menambah normalized bridge CSV untuk entry zone, entry, SL, TP1, dan TP2.
8. Menambah test manifest/version/integration.

## Batas penggunaan

- Scanner adalah decision-support system, bukan jaminan profit.
- `EXECUTION_READY` tetap harus dikonfirmasi pada daily bar yang sudah final atau bar intraday yang sudah selesai.
- Directional score Pine adalah rule-based confluence, bukan calibrated probability.
- Jangan menjalankan ARA Hunter dengan ukuran posisi normal.
- Hasil fundamental yang tidak lengkap harus tetap fail closed; jangan mengisi angka sintetis.
