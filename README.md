# IDX Super Scanner v6.5.0 — Eye-of-Future Reconstruction AI

Scanner IDX berbasis Streamlit untuk core swing, Sniper/ICT, BPJS, BSJP,
ARA Hunter, dan riset/compounding Multibagger. Semua order tetap dieksekusi
manual melalui Stockbit.

## Clean-room Eye-of-Future Reconstruction (core swing, daily Sniper, dan Multibagger) v6.5.0

Rilis ini memperluas objective time-cycle menjadi clean-room Eye-of-Future reconstruction: geocentric ephemeris, Moon phase/declination, planetary aspect, ingress/retrograde/stationary, Sun annual cycle, minimum empat multi-anchor Fibonacci time projections, price geometry, harmonic/wave target, MA Envelope, gap, pattern, momentum, historical baseline/lift, dan Future Price Road Map.

Formula proprietary Eye of Future tidak diklaim atau disalin. Pengaruh hanya aktif setelah confluence dan evidence historis lolos, tetap berada dalam cap time-cycle. BPJS, BSJP, ARA, dan seluruh intraday tidak dipengaruhi.

Tab **Time-Cycle & Eye-of-Future Reconstruction** dapat menganalisis satu atau beberapa ticker yang diketik manual tanpa CSV.


## Prinsip keselamatan

- Rule engine tetap menentukan validitas struktur, harga, data, RR, likuiditas,
  dan konteks.
- `SIGNAL_READY` adalah radar, bukan instruksi beli.
- `READY_FOR_STOCKBIT_VERIFY` wajib diperiksa kembali terhadap harga live,
  bid/offer, spread, gap, batas ARA/ARB, dan lot di Stockbit.
- `EXECUTION_READY` hanya dapat muncul pada Account-Guarded dengan
  `autopilot_verified = True`.
- AI hanya mengubah urutan conviction. AI tidak dapat menghidupkan setup yang
  invalid atau mengubah status menjadi siap eksekusi.

## Validated Hybrid AI

AI lokal memakai logistic regularized untuk probabilitas tervalidasi,
similarity/KNN untuk diagnostik expected-R, prior Bayesian strategy-regime,
chronological calibration/evaluation, recency weighting, feature coverage, dan
drift guard. Prinsip v6.2.0 adalah:

> No resolved evidence = no probability and no AI influence.

Model fill dan model `TP1|fill` harus sama-sama mengalahkan baseline
probabilitas setidaknya 2% pada evaluasi kronologis terpisah. Jika salah satu
gagal, bobot ranking menjadi nol. Sampel strategi kurang, coverage fitur rendah,
duplikasi outcome, dan drift tinggi juga membatasi atau menonaktifkan pengaruh AI.

Multibagger peer intelligence sekarang membandingkan bank dengan bank dan,
bila tersedia minimal lima emiten, emiten umum dengan sektor ekonominya. Peer
rank dipadukan dengan absolute business quality. Pengaruh terhadap conviction
membutuhkan filing resmi terverifikasi dan current, minimal dua sumber, tanpa
konflik fundamental berat, serta peer sample yang memadai.

## Data otomatis

- OHLCV: cache current, Yahoo, IDX Stock Summary EOD, iTick opsional.
- Harga independen: IDX, Google Finance, iTick, Twelve Data opsional.
- Fundamental: Yahoo snapshot/history, IDX XBRL/iXBRL otomatis, Twelve Data
  opsional, CSV manual hanya sebagai fallback.
- Project/management: IDX/OJK/issuer IR dengan source quorum dan scenario model.
- Broker Summary dan Order Book: snapshot/export pengguna; proxy OHLCV tidak
  diklaim sebagai data broker aktual.

## Menjalankan

```bash
pip install -r requirements.txt
python -m unittest -v test_scanner.py
streamlit run app.py
```

Release validation: **223 tests**.

Baca `EOFF_RECONSTRUCTION_GUIDE_V6_5_0.md`, `FIX_REPORT_V6_5_0.md`,
`AI_LOCAL_GUIDE.md`, dan `AI_AUDIT_REPORT_V6_2_0.md` sebelum memakai hasil untuk real money.
