# IDX Super Scanner v7.16.1 — Database Schema v11 Readiness Hotfix

Scanner Streamlit untuk riset **Multibagger** dan **Core Swing** IDX, dengan
mesin arah IHSG tiga horizon. Sistem ini dibuat untuk meningkatkan kualitas
keputusan dan menjaga compounding; tidak menjanjikan profit, kekayaan, atau
akurasi arah pasar.


## Yang baru di v7.16.1

- Database bridge dinaikkan ke `scanner_schema_v11`.
- Database readiness sekarang memverifikasi kolom evidence comparability, production rank, distribution evidence, dan selector-universe controls dari migration v11.
- Status sehat menjadi `HEALTHY_V11_400_UNIVERSE_EVIDENCE`; database v10 yang belum dimigrasikan menghasilkan `MIGRATION_REQUIRED_V11`.
- SQLite incremental cache v7.15–v7.16 tetap self-initialising dan tidak memerlukan SQL manual.
- Upgrade Supabase bersifat idempotent dan tidak menghapus data lama.

## Yang baru di v7.16.0

### Persistent Core Engine cache

- prepared indicator dan local setup analysis disimpan pada 16 partisi ticker deterministik;
- perubahan satu ticker hanya membangun ulang satu partisi, bukan 400 ticker;
- summary Core penuh dipakai ulang pada scan identik, tetapi market breadth dan konteks tetap terikat pada fingerprint seluruh input;
- silent-accumulation profile dipulihkan ke `DataFrame.attrs`, sehingga Focus Engine menerima evidence yang sama pada cold maupun warm path;
- perubahan benchmark menginvalidasi seluruh prepared partition karena relative strength dan benchmark overlay memang berubah;
- history yang menjadi kosong setelah sanitasi OHLCV/volume sekarang dikeluarkan secara fail-closed, bukan memicu `index[-1]` crash;
- dashboard menggabungkan audit cache Core dan Focus dalam satu tabel;
- formula, bobot, entry, SL, TP, rank contract, evidence gate, dan allocation eligibility tidak berubah.

### Benchmark 400 ticker × 760 bar

- cold pipeline: 32,97 detik, termasuk penulisan sekitar 89,8 MB cache indikator;
- warm identik: Core 0,79 detik, Focus 1,61 detik, total 2,40 detik;
- fundamental-only refresh: total 6,04 detik; Core summary tetap hit;
- satu ticker OHLCV berubah: Core 2,37 detik; hanya 1/16 prepared partition dan 1/16 local-analysis partition dibangun ulang;
- 400 universe rows, 71 signals, 400 selector rows, dan 400 Multibagger rows tetap identik terhadap v7.15.0, selain timestamp runtime `narrative_as_of`;
- warm persisted output identik dengan cache-disabled output untuk signals, universe, selector, dan Multibagger setelah timestamp runtime dikecualikan.

Cold scan lebih lambat daripada v7.15 karena indikator 400 ticker harus dikompresi dan disimpan. Manfaat v7.16 muncul pada scan ulang dan perubahan parsial. Cache mempercepat komputasi lokal, bukan download provider.

## Yang baru di v7.15.0

### Incremental recomputation lintas proses

- Focus Engine memakai SQLite lokal dengan WAL untuk menyimpan stage yang mahal: valuation, silent profile, selector compact, narrative, Multibagger, order builder, dan full result;
- setiap entry diikat ke fingerprint deterministik seluruh input, konfigurasi keputusan, data `as_of`, tanggal run, dan versi pipeline;
- payload dikompresi, diverifikasi SHA-256 sebelum dibaca, mempunyai TTL dan LRU pruning, serta fail-soft bila database rusak atau tidak writable;
- selector cache hanya menyimpan ranking, model audit, dan training summary. Training panel 38.800 baris tidak dipersistenkan, sehingga payload 400 ticker hanya sekitar 123 KB;
- perubahan fundamental tidak memaksa selector teknikal dihitung ulang; perubahan OHLCV atau benchmark tetap menginvalidasi selector;
- tidak ada perubahan formula, bobot, evidence gate, rank contract, entry, SL, TP, atau allocation eligibility.

### Benchmark 400 ticker × 760 bar

- Core Engine tetap sekitar 13,9–14,0 detik;
- Focus cold build 14,47 detik;
- Focus warm identik 1,40 detik, terutama 1,34 detik untuk fingerprint input dan sekitar 54 ms untuk memuat full result;
- refresh fundamental-only 5,05 detik; selector 400 ticker dipakai ulang dari SQLite dalam sekitar 2,47 ms;
- top-20, jumlah baris, dan score sum warm identik sama dengan cold run;
- database setelah cold run sekitar 0,76 MB dan bertambah hanya untuk versi input yang benar-benar berbeda.

Cache mempercepat pengulangan komputasi, bukan download provider. Cold scan live tetap dipengaruhi jaringan, timeout, rate limit, retry, dan availability sumber.

## Yang baru di v7.14.0

### One-Pass Cross-Sectional Selector

- percentile seluruh tanggal dihitung secara vectorized dan tetap mempertahankan fail-neutral semantics;
- panel training dibangun per ticker secara blok, bukan loop bar-per-bar Python;
- technical feature frame dihitung satu kali per ticker dan dipakai ulang untuk current selector serta training panel;
- silent-accumulation profile dihitung satu kali di Core Engine lalu dipakai ulang pada tradeability, reversal setup, dan Focus Engine;
- tidak ada perubahan formula, bobot, evidence gate, rank contract, entry, SL, TP, atau allocation gate.

### Validasi performa dan equivalence

- benchmark deterministik 100 ticker × 760 bar: median core+focus turun 11,16 menjadi 7,43 detik (-33,4%); Focus Engine turun 49,6%;
- benchmark deterministik 400 ticker × 500 bar: median core+focus turun 30,35 menjadi 27,21 detik (-10,3%); Focus Engine turun 27,9%;
- stress 400 ticker × 760 bar: v7.14 selesai 27,67 detik pipeline / 29,02 detik wall-clock; v7.13 tidak selesai dalam ceiling 180 detik;
- 400 ticker menghasilkan 71 signals, 400 universe rows, dan 400 Multibagger rows; 656 kolom Multibagger identik terhadap v7.13 kecuali timestamp runtime `narrative_as_of`;
- 407 test terkumpul: 405 passed, 2 environment skips, 2 subtests passed, 0 failed.

Catatan: benchmark di atas adalah workload deterministik untuk mengukur CPU/pipeline, bukan replay harga live dan bukan estimasi durasi jaringan. Cold-start tetap dipengaruhi provider, timeout, rate limit, dan kecepatan koneksi.

## Yang baru di v7.13.0

### Auditable Valuation

- market cap, P/E, earnings yield, P/B, FCF yield, EV/EBITDA, dan PEG diturunkan hanya dari evidence bertanggal;
- 66 issuer dengan statement USD dikonversi memakai Bank Indonesia JISDOR official-first; missing/current FX gate bersifat fail-closed;
- PEG hanya valid untuk laba positif, growth minimal 10%, dan persistensi laba minimal 75%; base-effect growth dicap 100%;
- model valuation financial/non-financial mempunyai lineage eksplisit.

### Corporate Action & Market-Cap Reconciliation

- Yahoo split events menyimpan numerator, denominator, dan ratio di frame serta cache schema v6;
- statement shares hanya disesuaikan jika seluruh split ratio pasca-statement diketahui;
- dated market-cap reference menjadi cross-check untuk rights issue atau unexplained share change;
- split gap, stale statement, FX gap, share reconciliation, source tier, dan metric lineage tampil pada full audit serta decision summary.

### Validasi 400 Ticker v7.13.0

- 400/400 ticker pada output utama; 0 duplicate, 0 infinity, dan seluruh hard invariant lulus;
- coverage: market cap 400, P/E 296, earnings yield 342, P/B 397, FCF yield 280, EV/EBITDA 269, dan PEG 129;
- 332 statement IDR, 66 USD, dan dua tanpa statement currency;
- 277 ticker mempunyai `FULL_AUDITABLE_DERIVED_VALUATION`; tidak ada yang diklaim official verified pada replay ini;
- v7.13.0 vs v7.12.0: Spearman 0,9840 dan overlap Top-20 17/20;
- pembanding independen: Spearman 0,4427 dan overlap Top-20 7/20;
- direct rank dan allocation tetap nol tanpa authenticated broker/event evidence.

## Yang baru di v7.12.0

### Universe Completeness

- seluruh 400 ticker tetap muncul pada output Multibagger dan Core Swing;
- ticker dengan history di bawah 220 bar menjadi `DATA_NOT_SCORED_INSUFFICIENT_TECHNICAL_HISTORY`, bukan hilang diam-diam;
- jumlah bar, coverage history, dan alasan tidak comparable tersedia pada decision summary serta full audit.

### Broker Provenance Gate

- broker-summary upload manual tetap dapat menjadi observed-flow research, tetapi tidak dapat mengklaim direct evidence;
- direct flow membutuhkan authenticated provider/export provenance, data current, minimum lima hari observasi, dan quality gate;
- `broker_summary_direct_verified` dan `broker_summary_provenance_state` ikut menentukan direct rank dan allocation.

### Safe Cache & Fast Focus

- cache OHLCV cepat memakai fixed-schema NumPy `.npz` dengan `allow_pickle=False`; cache fundamental memakai JSON, sehingga scanner tidak mendeserialisasi pickle dari cache runtime;
- CSV OHLCV lama dimigrasikan otomatis ke safe companion; file `.pkl` legacy diabaikan;
- full time-cycle/EOFF Multibagger menjadi opt-in shadow research karena bobot produksi default 0%; Core Swing tetap memakai overlay bounded;
- EOFF berbobot nol tidak lagi memengaruhi ranking melalui confidence channel.

### Validasi 400 Ticker v7.12.0

- replay deterministik atas snapshot live sesi lengkap `2026-07-31` menghasilkan 400/400 ticker pada universe, Multibagger, Core Swing, dan decision summary;
- 392 ticker mempunyai history teknikal cukup dan 8 ticker berhistory pendek tetap terlihat sebagai `DATA_NOT_SCORED`; skor serta rank kedelapannya sengaja kosong;
- warm cache 400 ticker turun dari sekitar 41,9 detik menjadi 7,1 detik dan Focus Engine dari sekitar 103,8 detik menjadi 25,1 detik;
- ranking 392 ticker yang comparable tetap sangat stabil terhadap v7.11.1: Spearman 0,9999 dan overlap Top-20 20/20;
- pembanding fundamental/quality independen menghasilkan Spearman 0,4359 dan overlap Top-20 6/20. Ini pemeriksaan kewajaran lintas-model, bukan bukti return atau jaminan profit;
- tanpa sumber broker/event resmi terautentikasi, direct rank dan allocation tetap nol secara fail-closed.

## Yang baru di v7.11.1

### Bounded Direct-First Data Route

- OHLCV default memakai Yahoo Chart JSON direct dengan timeout dan retry terbatas; `yfinance` menjadi fallback opt-in melalui `IDX_SCANNER_ENABLE_YFINANCE_FALLBACK=1` karena timeout wrapper tidak selalu menjadi batas wall-clock;
- benchmark `^JKSE` memiliki route direct yang sama, sehingga market regime tidak lagi bergantung pada wrapper;
- HTTP 429/5xx diretry dengan exponential backoff; keberhasilan ticker lain tidak diulang;
- cache OHLCV menyimpan CSV portabel serta binary companion yang terverifikasi untuk warm start 400 ticker.

### Fundamental Parser & Coverage Integrity

- parser menerima `meta.type` Yahoo yang dapat berbentuk scalar atau list dan meminta `padTimeSeries=false`, sehingga observasi historis nyata tidak berubah menjadi padded null;
- field ekuitas memprioritaskan `TotalEquityGrossMinorityInterest`, dengan fallback `StockholdersEquity`, agar identitas aset–liabilitas–ekuitas konsisten;
- bila OCF tidak dilaporkan tetapi FCF dan capex tersedia, OCF direkonstruksi dari identitas akuntansi dan diberi lineage note; input yang tidak lengkap tetap `NaN`;
- history-only memakai coverage field history aktual. Ketiadaan quoteSummary tidak lagi membatasi coverage valid maksimum 55%; grade provenance tetap C untuk satu provider non-resmi;
- timeseries per ticker disimpan sebagai same-run cache agar snapshot dan history layer tidak menggandakan request.

### Direct Evidence, Proxy Research & Rank Contract

- evidence class menjadi `DIRECT_EVIDENCE_RANKING`, `FULL_PROXY_COVERAGE`, `PARTIAL_EVIDENCE_RESEARCH`, atau `DATA_PENDING`;
- direct production membutuhkan event resmi/review lengkap, direct issuer alignment, broker-summary coverage, fundamental lengkap, dan Emir production eligibility. OHLCV/structured-financial proxy tidak boleh menyamar sebagai direct evidence;
- urutan ranking: hard eligibility/narrative block → Final Score → Effective Silent Accumulation → Execution Readiness. `MULTIBAGGER_NOT_QUALIFIED` tidak dapat mendahului A/B hanya karena coverage tinggi;
- allocation sekarang fail-closed terhadap `multibagger_rank_eligible`;
- lane Growth/Turnaround ditentukan oleh gate yang benar-benar lolos, bukan label archetype deskriptif;
- dashboard default memakai ringkasan keputusan 41 kolom; frame 600+ kolom tetap tersedia sebagai audit teknis terpisah.

### Validasi Live 400 Ticker

- cold OHLCV: 400/400, 0 failure, 303 detik; warm: 400 cache hit, 0 provider call, 42 detik;
- seluruh bar terakhir `2026-07-31`, tanpa missing OHLCV; benchmark juga current;
- fundamental: 398/400 memiliki score, 333 complete untuk Multibagger, dua ticker tetap unresolved secara eksplisit;
- hasil final: 31 B-candidate plus satu turnaround research-only; 0 production/allocation karena direct event dan broker evidence tidak disuplai;
- perbandingan independen 392 ticker menghasilkan Spearman 0,4358. Ini korelasi moderat, bukan validasi profit atau jaminan performa.

### Database Migration

Tidak ada migrasi baru dari v7.11.0. Untuk instalasi lebih lama, jalankan `database/migration_v11_400_universe_evidence.sql`, lalu verifikasi dengan `database/verify_v11_400_universe_evidence.sql`.

## Yang baru di v7.11.0

## Yang baru di v7.10.0

### Data Completion Route

- OHLCV memiliki jalur baru `Yahoo Chart Direct` yang tidak bergantung pada paket `yfinance`; jalur ini digunakan untuk full history maupun incremental refresh dan menyimpan split/dividend metadata.
- Corporate action tidak lagi diperlakukan sebagai data rusak bila split event dan adjusted-price basis terverifikasi; lompatan tanpa event tetap menjadi blocker.
- Fundamental history memiliki fallback `Yahoo Fundamentals Timeseries Direct` untuk income statement, balance sheet, dan cash flow, lalu tetap dikonsolidasikan dengan IDX/XBRL official-first dan persistent cache.
- Universe hingga 120 ticker memakai mode `FULL_COMPLETION_SMALL_MEDIUM_UNIVERSE`: semua ticker mendapat giliran snapshot, history, dan structured narrative pada scan yang sama.
- Selector panel kosong atau schema tidak lengkap kembali sebagai `INSUFFICIENT_PANEL_SCHEMA`, bukan crash `KeyError: as_of`.

### Causal OOS Event Expansion

- Walk-forward menyimpan detector-confirmed plan dengan geometry valid sebagai `TRIGGER_CANDIDATE`, meskipun live production gate memblokir order.
- Setiap event menyimpan `production_gate_pass`, `production_gate_blockers`, dan `validation_event_tier`.
- AI boleh belajar fill/TP outcome dari causal trigger candidates, tetapi live execution tetap membutuhkan seluruh production gate.
- Statistik memisahkan `production_ready_events` dan `trigger_candidate_events`, sehingga evidence volume tidak disamarkan sebagai order-ready track record.

### IDX Session & Holiday Hardening

- kalender sesi memakai daftar penutupan Bursa resmi untuk 2026; weekend dan bar hari libur resmi dihapus sebelum indikator volume dihitung;
- fallback historis memakai konsensus lintas-universe yang konservatif, sehingga satu saham dengan volume nol tidak otomatis dianggap hari libur;
- freshness dihitung dalam **sesi Bursa**, bukan hanya hari kalender; data yang tertinggal satu sesi menjadi `daily_session_current=False`;
- cache stale tetap dapat dipakai sebagai bahan riset, tetapi market regime, completeness fallback, dan final execution gate bersifat fail-closed.

### Adaptive OOS Evidence

- histori OHLCV default menjadi 5 tahun;
- cohort chronological OOS berkembang deterministik `60 → 120 → 240 → cap` sampai minimum event genuine per setup tercapai atau batas ticker habis;
- default cap 180 ticker dan dapat dinaikkan sampai 300 dari UI;
- tidak ada synthetic win, relaksasi quality gate, atau aktivasi paksa. Bila edge OOS tidak terbukti, bobot AI tetap nol;
- all-NaN model features tidak lagi menghasilkan RuntimeWarning atau placeholder score.

### Progressive Narrative & Fundamental Completeness

- refresh narrative/news memprioritaskan portfolio dan kandidat, lalu merotasi cohort universe secara deterministik;
- structured financial evidence dapat mengisi narrative dan issuer alignment dari laporan yang memiliki lineage walaupun berita terbaru belum tersedia;
- status akuisisi evidence eksplisit: `SOURCE_EVENT_ACQUIRED`, `STRUCTURED_FINANCIAL_ACQUIRED`, `PARTIAL_*`, atau `NEWS_OR_DISCLOSURE_REFRESH_PENDING`;
- duplikasi provider/periode laporan dikoalesensikan per-field sehingga revenue, cash flow, neraca, atau saham beredar tidak hilang akibat row-level overwrite;
- Multibagger production gate meminta coverage laporan laba-rugi, neraca, arus kas, histori periode, freshness, source lineage, dan core-field completeness.

### Reclaim Trigger for Once-Daily Scan

- Breakout Retest menampilkan `retest_reference_price`, `reclaim_trigger_price`, dasar trigger, expiry, dan instruksi order;
- trigger memakai resistance 55D yang ditembus + satu tick IDX, bukan high candle terbaru;
- scanner dapat menampilkan conditional buy-stop untuk sesi berikutnya, tetapi order tetap diblokir bila RR, gap, atau invalidasi struktur tidak lolos.

### Absolute Swing Score

- universe di bawah 20 saham memakai absolute swing score 100%;
- relative/cross-sectional overlay dibatasi 0% / 5% / 10% / 15% sesuai jumlah peer;
- small-universe percentile tidak dapat lagi mengubah score kecil menjadi score eksekusi tinggi;
- missing Core base score menjadi `CORE_BASE_NOT_SCORED`, bukan fallback 50.

### Data Route Hardening

- provider retry hanya memproses ticker unresolved; hasil sukses dari batch sebelumnya dipertahankan;
- market status, news, dan fundamental memiliki route state serta `*_score_eligible`;
- unresolved fundamental/narrative/Emir/AI peer evidence menjadi `NaN/NOT_SCORED`, bukan neutral 50;
- cache fallback hanya boleh masuk scoring bila provenance, freshness, dan coverage masih lolos.

### AI Consistency Hardening

- post-AI Core Priority Score tidak lagi menghapus Emir adjustment;
- formula final Core: `70% Swing Selection + 30% Rule/Hybrid Conviction + Narrative + Emir`;
- missing rule score mematikan AI influence; cohort peer terlalu kecil menghasilkan `NaN`, bukan 50;
- AI tetap dibatasi maksimum 35% dan wajib lolos out-of-sample validation gate.

### Final Score Rebalance

- Growth Compounder: Narrative maksimum ±5 poin, Emir maksimum ±14 poin;
- Turnaround/Cyclical: Narrative maksimum ±7 poin, Emir maksimum ±16 poin;
- Core Swing: Narrative maksimum ±5 poin, Emir maksimum ±18 poin;
- Narrative standalone diturunkan untuk mengurangi double counting dengan narrative lifecycle dan issuer alignment di dalam Emir framework;
- score decomposition, Emir coverage, score state, dan formula final ditampilkan untuk audit.

### EOFF Reduction

- Core Swing outer time-cycle cap turun dari 10% menjadi 4%;
- Multibagger tetap 0%;
- EOFF hanya aktif saat `VALIDATED` dan tidak dapat melampaui hard gate fundamental, evidence, distribution, liquidity, atau execution integrity.

### Emir Public-Framework Layer

- Mengukur *stock-universe familiarity*, narrative lifecycle, smart-money behavior, dan issuer alignment.
- Membedakan `FLOW_LED_STORY_CONFIRMED`, `EARLY_NARRATIVE_FLOW_CONVERGENCE`, `STORY_AHEAD_OF_FLOW`, dan `LATE_CROWDED_OR_DISTRIBUTION`.
- Tidak menyamarkan OHLCV sebagai data broker: tanpa broker summary, mode wajib `OHLCV_PRICE_VOLUME_PROXY_ONLY`.
- Memblokir allocation/order builder pada distribusi, failed absorption, contradiction kritis, atau evidence lemah.
- Position cap maksimal 15%; maksimal 10% bila hanya memakai proxy OHLCV.

## Fondasi v7.6.3 yang dipertahankan

- placeholder neutral 50 tidak ditampilkan sebagai hasil analisis;
- source, coverage, score, dan state dipisahkan agar angka dapat diaudit;
- cache lama disanitasi dan Audit Nilai Sama tetap tersedia;
- missing evidence tidak boleh otomatis dianggap positif atau negatif.

## Fondasi v7.6.2 yang dipertahankan

- memisahkan **harga eksekusi**, **order trigger**, dan **confirmation level**.
  Trigger reclaim pada pullback/retest tidak lagi diperlakukan otomatis sebagai
  harga beli;
- seluruh level satu rekomendasi sekarang harus berasal dari satu kontrak
  atomik: entry/zone, trigger order, confirmation, SL, TP1, TP2, dan RR;
- RR dihitung ulang dari harga order yang benar. Limit/retest memakai `entry`,
  sedangkan buy-stop memakai `stockbit_order_price`;
- source execution gate tidak lagi menerbitkan `ENTRY_PLAN_READY` bila
  confirmation berada pada/di atas TP1, target geometry rusak, plan tidak
  lengkap, atau RR berada di bawah batas konfigurasi;
- plan invalid berubah menjadi `WAIT_FOR_VALID_PLAN`, lot dipaksa 0, dan level
  order tidak ditampilkan pada Top 20 maupun TradingView bridge. Angka mentah
  tetap tersedia hanya sebagai diagnostic fields;
- `entry_low == entry_high` ditampilkan sebagai **Entry RpX**, bukan zona palsu
  `RpX – RpX`;
- selector dan focus layer mempertahankan `entry_low`, `entry_high`,
  `entry_type`, stockbit order fields, serta minimum RR agar semantik tidak
  hilang saat ranking;
- ranking tetap **Final Score → Effective Silent Accumulation → Execution
  Readiness**. Perubahan ini tidak mengubah formula fundamental, narrative,
  EOFF, atau schema Supabase.

## Fondasi v7.6.1 yang dipertahankan


- **Narrative Event Database** menyimpan event point-in-time, waktu deteksi,
  sumber, kualitas sumber, novelty, materiality, financial bridge, decay, dan
  contradiction; event tidak dibuat bila sumber tidak ada;
- **Issuer Alignment Score** menggabungkan buyback/insider-controller action,
  eksekusi proyek, dilusi, share pledge, governance, dan ownership evidence.
  Konsentrasi kepemilikan tinggi tidak otomatis dianggap positif;
- **Narrative Conversion Rate** mengukur excess return 5D/20D/60D net terhadap
  IHSG sesudah `detected_at`, berikut MFE/MAE. Skor tetap shadow sampai minimal
  lima outcome resolved per ticker;
- **Retail Adoption Stage** memakai proxy attention dari event, volume, return,
  dan proximity ke high. Ini bukan klaim identitas investor ritel;
- **Narrative–Flow Convergence** menggabungkan story, Issuer Alignment, dan
  effective Silent Accumulation. Neutral 50 hanya boleh digunakan secara
  internal sebagai titik pusat matematika dengan bobot produksi nol; tampilan
  publik tetap kosong/`NOT_SCORED` ketika evidence belum ada;
- Growth Compounder, Turnaround/Cyclical, dan Core Swing menerima overlay
  narrative berbeda. Narrative hanya mengurutkan kandidat yang sudah memiliki
  evidence dasar; tidak dapat mengubah bisnis lemah menjadi Multibagger;
- event negatif resmi/material menjadi hard gate alokasi modal, tetapi saham
  tetap terlihat pada radar riset agar alasan blokir dapat diaudit;
- migration database v9 menambah `narrative_events`,
  `narrative_event_outcomes`, dan `narrative_snapshots`.

## Fondasi v7.5.4 yang dipertahankan

- nilai fundamental/teknikal yang tidak tersedia tidak lagi diperlakukan
  sebagai nol atau median palsu di ranking produksi;
- Multibagger menyimpan coverage growth, profitability, cashflow, safety,
  reinvestment runway, dan valuation. Gate produksi minimum 65%; kandidat A
  minimum 80%, disertai histori laporan yang cukup;
- selector menghitung coverage fitur per ticker, memakai missing indicators,
  menolak baris low-coverage dari training/ranking produksi, dan tetap
  mempertahankannya sebagai diagnostic radar;
- Silent Accumulation di-shrink menurut confidence. Bar dengan volume hilang
  dibuang dari indikator price-volume, bukan diubah menjadi volume nol;
- Core Swing memiliki `swing_component_coverage_pct`,
  `selector_production_gate`, dan `order_builder_eligible`;
- audit data contract membedakan `NO_CANDIDATE_VALID` dari
  `NOT_EVALUABLE_DATA_INSUFFICIENT`;
- kolom tabel yang seluruhnya null/kosong disembunyikan, tanpa mengisi angka
  buatan;
- backfill fundamental 40 ticker/scan diprioritaskan oleh technical quality,
  effective Silent Accumulation, relative strength, dan likuiditas, lalu tetap
  berputar secara round-robin;
- migration database v8 menyimpan state/coverage Multibagger dan selector.

## Fondasi v7.5.3 yang dipertahankan

- Top Multibagger tidak lagi berhenti pada pesan kosong. Bila belum ada saham
  yang qualified, dashboard langsung menampilkan **Provisional Research
  Queue** terpisah untuk Growth Compounder dan Turnaround/Cyclical;
- provisional bukan kandidat lolos gate: `candidate_state` selalu
  `PROVISIONAL_RESEARCH`, `capital_state` selalu
  `RESEARCH_ONLY_NO_ALLOCATION`, dan alokasi modal dipaksa 0;
- setiap provisional menampilkan blocker, evidence berikutnya yang harus
  dilengkapi, alasan dipilih, alasan belum entry, trigger yang ditunggu,
  invalidation, dan risiko utama;
- statement-history backfill memprioritaskan kombinasi snapshot
  quality/growth dan Silent Accumulation kandidat teratas, lalu tetap
  menyelesaikan cohort round-robin agar universe lain tidak kelaparan;
- audit database kini hanya menghitung pembacaan
  `SUPABASE_DATABASE_FIRST`. Baris scheduler/provider tidak lagi menggandakan
  denominator audit;
- semantic fundamental model tetap `7.5.2` karena formula fundamental tidak
  berubah; ini mencegah cache valid menjadi stale hanya akibat perubahan UI
  dan urutan backfill.

## Fondasi v7.5.2 yang dipertahankan

- memperbaiki collision kolom saat snapshot database yang pernah diperkaya
  digabung kembali dengan histori laporan. Sebelumnya kolom menjadi pasangan
  `_x/_y`, lalu grade/source kanonik tidak terbaca dan seluruh emiten dapat
  jatuh palsu ke grade D/source 0;
- mencegah recursive double-count: skor gabungan dari cache tidak lagi
  diperlakukan sebagai snapshot mentah pada scan berikutnya;
- memisahkan `fundamental_snapshot_source_count` dari
  `fundamental_history_source_count`; snapshot tidak lagi menyamar sebagai
  histori laporan;
- mode Multibagger seluruh universe kini membaca cache histori seluruh ticker,
  sementara refresh provider tetap dibatasi 40 ticker dan dirotasi per cohort.
  Universe di luar shortlist tidak lagi kelaparan backfill;
- menambahkan Coverage, Growth Near-Miss, Turnaround Near-Miss, Data Pending,
  dan Gate Blockers. Empty qualified table tidak lagi berarti “tidak ada
  potensi”;
- near-miss dan data-pending selalu `RESEARCH_ONLY_NO_ALLOCATION`. Gate modal
  grade A–C, kualitas, risiko, entry, dan likuiditas tidak dilonggarkan.

## Fondasi v7.5.1 yang dipertahankan

- Top Multibagger dan Top Swing/Core tidak lagi dipertandingkan pada satu
  ranking karena horizon, bukti, dan makna skornya berbeda;
- Multibagger mempunyai dua radar terpisah: **Growth Compounder** dan
  **Turnaround/Cyclical**;
- Growth Compounder memberi bobot langsung 15% pada reinvestment runway.
  Formula base: growth persistence 22%, profitability 19%, cash conversion
  16%, balance-sheet safety 14%, runway 15%, valuation 8%, liquidity 6%;
- Turnaround memakai infleksi fundamental point-in-time, akselerasi
  revenue/earnings, pemulihan gross margin dan cash conversion, safety, serta
  katalis. Minimal dua sinyal recovery diperlukan; konflik laporan,
  governance, related-party critical, dilusi/accrual/leverage ekstrem tetap
  hard blocker;
- status `research_eligible` dipisahkan dari
  `portfolio_allocation_eligible`. Kandidat riset tidak hilang hanya karena
  tidak masuk maksimal lima posisi portofolio;
- Silent Accumulation untuk seleksi Multibagger kini di-shrink ke netral
  menurut confidence dan di-cap ketika state menunjukkan distribution risk;
- outcome 12/24/36 bulan dan snapshot database menyimpan lane, skor recovery,
  alasan gate, dan status riset/alokasi.

## Fondasi v7.5.0 yang dipertahankan

- crash Streamlit akibat nama kolom Multibagger duplikat diperbaiki; semua
  tabel yang berpotensi berisi object campuran kini dinormalisasi sebelum
  serialisasi Arrow;
- selection rank dan execution rank dipisahkan. Final Score + Silent
  Accumulation tetap memilih saham, sedangkan validitas trigger/SL/TP,
  risk-reward, dan readiness menentukan prioritas eksekusi;
- trade plan bersifat atomik: trigger Best Buy tidak dapat lagi tercampur
  dengan SL/TP dari base setup. RR selalu dihitung kembali dari harga yang
  benar-benar ditampilkan;
- selector 5D/20D/60D menambah sector-relative strength yang hanya aktif bila
  sedikitnya lima peer tersedia, serta estimasi market-impact cost per bucket
  likuiditas;
- champion gate menambah CSCV/PBO. AI challenger tetap shadow bila
  probabilitas backtest overfitting di atas 50%, selain gate Brier skill,
  expectancy setelah biaya, significance, dan drawdown;
- Execution AI menambah discrete fill hazard, probabilitas fill 1D/3D,
  expected fill delay, dan outcome competing-risk `TP1_FIRST`, `SL_FIRST`,
  `TIME_EXIT`, atau censored;
- Multibagger menambah fundamental inflection (akselerasi revenue/earnings,
  perubahan gross margin, dan cash conversion) serta outcome relatif terhadap
  IHSG untuk 12/24/36 bulan;
- panel IHSG membedakan tegas probability leader riset dari sinyal resmi.
  Raw probability tetap dapat terlihat ketika keputusan produksi `ABSTAIN`.

## Fondasi statistik v7.4.0 yang dipertahankan

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

## Kebijakan Narrative Production v7.6.1

Atas keputusan pengguna, narrative positif tetap aktif sebagai input produksi sebelum
validasi OOS penuh. Pengaruhnya tetap dibatasi oleh kualitas sumber, materialitas,
novelty, decay, entity verification, lifecycle, dan confidence shrinkage. Scanner
tidak mengklaim narrative tersebut sudah tervalidasi secara statistik.

Kebijakan ranking:

1. hard eligibility dan narrative hard block;
2. Final Score;
3. Effective Silent Accumulation sebagai key terpisah;
4. execution readiness;
5. ticker sebagai tie-break deterministik.

Silent Accumulation tidak lagi dimasukkan langsung ke Final Score, selector score,
atau narrative rank adjustment. Event tanpa sumber/entitas terverifikasi tidak
mempunyai pengaruh produksi. Event negatif memakai outcome yang disesuaikan arah,
dan outcome dimulai pada sesi selesai pertama setelah `detected_at`.

## Database prerequisite

Jalankan berurutan:

1. `database/schema_v2.sql` — hanya untuk proyek database baru;
2. `database/migration_v3_database_first.sql`;
3. `database/migration_v4_research_memory.sql`;
4. `database/migration_v5_ihsg_direction.sql`.
5. `database/migration_v6_selector_ai_outcomes.sql`.
6. `database/migration_v7_multibagger_lanes.sql`.
7. `database/migration_v8_data_contract.sql`.
8. `database/migration_v9_narrative_intelligence.sql`;
9. `database/migration_v10_narrative_safety.sql`;
10. `database/migration_v11_400_universe_evidence.sql`.

Lalu jalankan `database/verify_v11_400_universe_evidence.sql`. Target health:
`HEALTHY_V11_400_UNIVERSE_EVIDENCE`.

## Deployment aman di Streamlit Community Cloud

1. Pastikan migration v5–v10 sudah terpasang, lalu jalankan migration v11 dan verify v11 di Supabase.
2. Ganti seluruh isi root repository dengan isi ZIP v7.10.0 dalam satu commit
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
  selector_engine.py research_maintenance.py ihsg_direction.py \
  narrative_engine.py
python -m unittest discover -p "test*.py"
```

Lihat `FIX_REPORT_V7_6_1.md`, `BUILD_VALIDATION_V7_6_1.md`,
`DASHBOARD_RANKING_GUIDE_V7_6_1.md`, dan
`docs/DATABASE_MIGRATION_V11_400_UNIVERSE_EVIDENCE.md` untuk perubahan terbaru.
Dokumen release lama tetap disertakan sebagai audit trail.

## Release research and validation

- `EMIR_METHOD_RESEARCH_AND_UPGRADE_V7_7_0.md`
- `FIX_REPORT_V7_7_0.md`
- `BUILD_VALIDATION_V7_7_0.md`
