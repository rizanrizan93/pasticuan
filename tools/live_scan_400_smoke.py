from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fast_scan_engine import run_fast_single_scan

TICKERS = """AADI.JK,AALI.JK,ABMM.JK,ACES.JK,ADCP.JK,ADHI.JK,ADMG.JK,ADMR.JK,ADRO.JK,AGAR.JK,AGII.JK,AGRO.JK,AIMS.JK,AISA.JK,AKPI.JK,AKRA.JK,AKSI.JK,ALDO.JK,ALII.JK,ALKA.JK,ALMI.JK,AMAN.JK,AMAR.JK,AMFG.JK,AMIN.JK,AMMN.JK,AMMS.JK,AMRT.JK,ANTM.JK,APEX.JK,APII.JK,APLI.JK,APLN.JK,ARCI.JK,AREA.JK,ARII.JK,ARKA.JK,ARKO.JK,ARNA.JK,ARTI.JK,ARTO.JK,ASGR.JK,ASHA.JK,ASII.JK,ASLC.JK,ASLI.JK,ASPI.JK,ASPR.JK,ASRI.JK,ASSA.JK,ATAP.JK,ATIC.JK,ATLA.JK,AUTO.JK,AVIA.JK,AWAN.JK,AXIO.JK,AYAM.JK,AYLS.JK,BABY.JK,BAJA.JK,BALI.JK,BANK.JK,BAPI.JK,BATA.JK,BATR.JK,BBCA.JK,BBHI.JK,BBNI.JK,BBRI.JK,BBRM.JK,BBSS.JK,BBTN.JK,BCIP.JK,BDKR.JK,BDMN.JK,BEBS.JK,BEKS.JK,BELI.JK,BELL.JK,BESS.JK,BEST.JK,BGTG.JK,BHIT.JK,BIKE.JK,BINO.JK,BIPI.JK,BIPP.JK,BIRD.JK,BISI.JK,BJBR.JK,BJTM.JK,BKDP.JK,BKSL.JK,BLES.JK,BLTA.JK,BLUE.JK,BMHS.JK,BMRI.JK,BMSR.JK,BMTR.JK,BNBR.JK,BNGA.JK,BNII.JK,BOAT.JK,BOBA.JK,BOGA.JK,BOLT.JK,BOSS.JK,BPTR.JK,BREN.JK,BRIS.JK,BRMS.JK,BRNA.JK,BRPT.JK,BRRC.JK,BSBK.JK,BSDE.JK,BSML.JK,BSSR.JK,BTON.JK,BTPN.JK,BTPS.JK,BUAH.JK,BUDI.JK,BUKA.JK,BUKK.JK,BULL.JK,BUMI.JK,BVIC.JK,BWPT.JK,BYAN.JK,CAKK.JK,CAMP.JK,CANI.JK,CARE.JK,CARS.JK,CASH.JK,CASS.JK,CBRE.JK,CCSI.JK,CEKA.JK,CFIN.JK,CGAS.JK,CHEK.JK,CHEM.JK,CHIP.JK,CINT.JK,CITA.JK,CITY.JK,CLEO.JK,CLPI.JK,CMNP.JK,CMNT.JK,CMPP.JK,CMRY.JK,CNKO.JK,COAL.JK,CPIN.JK,CPRO.JK,CRSN.JK,CSAP.JK,CSIS.JK,CSRA.JK,CTBN.JK,CTRA.JK,CTTH.JK,CUAN.JK,CYBR.JK,DADA.JK,DATA.JK,DAYA.JK,DCII.JK,DEAL.JK,DEWA.JK,DEWI.JK,DGIK.JK,DGNS.JK,DGWG.JK,DILD.JK,DIVA.JK,DKFT.JK,DMAS.JK,DMMX.JK,DMND.JK,DNAR.JK,DOID.JK,DOOH.JK,DPNS.JK,DRMA.JK,DSFI.JK,DSNG.JK,DSSA.JK,DUTI.JK,DVLA.JK,DWGL.JK,DYAN.JK,EDGE.JK,EKAD.JK,ELIT.JK,ELPI.JK,ELSA.JK,ELTY.JK,EMDE.JK,EMTK.JK,ENRG.JK,EPAC.JK,EPMT.JK,ERAA.JK,ERTX.JK,ESIP.JK,ESSA.JK,ETWA.JK,EURO.JK,EXCL.JK,FAST.JK,FASW.JK,FIMP.JK,FIRE.JK,FISH.JK,FLMC.JK,FMII.JK,FOLK.JK,FOOD.JK,FORE.JK,FPNI.JK,FWCT.JK,GDST.JK,GEMS.JK,GGRP.JK,GHON.JK,GIAA.JK,GJTL.JK,GLVA.JK,GMTD.JK,GOLD.JK,GOLF.JK,GOOD.JK,GOTO.JK,GPRA.JK,GPSO.JK,GRIA.JK,GRPM.JK,GTBO.JK,GTRA.JK,GTSI.JK,GULA.JK,GUNA.JK,GZCO.JK,HADE.JK,HAIS.JK,HALO.JK,HATM.JK,HBAT.JK,HDIT.JK,HEAL.JK,HELI.JK,HERO.JK,HEXA.JK,HILL.JK,HITS.JK,HOKI.JK,HOMI.JK,HOPE.JK,HRTA.JK,HRUM.JK,HUMI.JK,HYGN.JK,IATA.JK,IBFN.JK,IBST.JK,ICBP.JK,ICON.JK,IDPR.JK,IFII.JK,IFSH.JK,IGAR.JK,IKAI.JK,IKAN.JK,IKBI.JK,IKPM.JK,IMAS.JK,IMJS.JK,IMPC.JK,INAF.JK,INAI.JK,INCF.JK,INCI.JK,INCO.JK,INDF.JK,INDR.JK,INDX.JK,INDY.JK,INET.JK,INKP.JK,INPP.JK,INPS.JK,INRU.JK,INTA.JK,INTD.JK,INTP.JK,IOTF.JK,IPAC.JK,IPCC.JK,IPCM.JK,IPOL.JK,IRRA.JK,IRSX.JK,ISAT.JK,ISSP.JK,ITMA.JK,ITMG.JK,JARR.JK,JAST.JK,JATI.JK,JAWA.JK,JAYA.JK,JECC.JK,JGLE.JK,JKON.JK,JMAS.JK,JPFA.JK,JRPT.JK,JSMR.JK,JTPE.JK,KAEF.JK,KARW.JK,KAYU.JK,KBAG.JK,KBLI.JK,KBLM.JK,KDSI.JK,KEEN.JK,KEJU.JK,KETR.JK,KIAS.JK,KIJA.JK,KING.JK,KINO.JK,KIOS.JK,KJEN.JK,KKES.JK,KKGI.JK,KLAS.JK,KLBF.JK,KMDS.JK,KMTR.JK,KOBX.JK,KOCI.JK,KOIN.JK,KOKA.JK,KONI.JK,KOPI.JK,KRAS.JK,KREN.JK,KUAS.JK,LABA.JK,LABS.JK,LAJU.JK,LAND.JK,LEAD.JK,LION.JK,LMSH.JK,LOPI.JK,LPCK.JK,LPKR.JK,LPLI.JK,LPPF.JK,LRNA.JK,LSIP.JK,LTLS.JK,LUCK.JK,MAHA.JK,MAIN.JK,MANG.JK,MAPA.JK,MAPI.JK,MARK.JK,MAXI.JK,MBAP.JK,MBMA.JK,MBSS.JK,MBTO.JK,MCAS.JK,MCOL.JK,MDKA.JK,MDKI.JK,MDLA.JK,MDRN.JK,MEDC.JK,MEDS.JK,MEGA.JK,MERK.JK,META.JK,MFMI.JK,MHKI.JK,MIDI.JK,MIKA.JK,MINE.JK,MIRA.JK""".split(",")

assert len(TICKERS) == 400, len(TICKERS)

CONFIG = {
    "period": "3y",
    "evidence_refresh_cap": 20,
    "decision_evidence_cap": 12,
    "evidence_fundamental_cap": 20,
    "evidence_official_cap": 12,
    "evidence_snapshot_cap": 16,
    "evidence_market_cap": 6,
    "evidence_news_cap": 10,
    "execution_verification_cap": 8,
    "daily_market_refresh_limit": 6,
    "macro_external_enabled": True,
    "macro_timeout_seconds": 3,
    "lean_persistence": True,
    "lean_skip_narrative_history": True,
}


def _save(frame, path: str) -> int:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        pd.DataFrame().to_csv(path, index=False)
        return 0
    frame.to_csv(path, index=False)
    return len(frame)


result = run_fast_single_scan(TICKERS, config=CONFIG, runtime={})
focus = result.get("focus_screens", {}) or {}
leaders = focus.get("next_leaders", pd.DataFrame())
swings = focus.get("swing_ready", pd.DataFrame())
Path("live_scan_output").mkdir(exist_ok=True)
leader_n = _save(leaders, "live_scan_output/next_leaders.csv")
swing_n = _save(swings, "live_scan_output/swing_ready.csv")
coverage = result.get("scan_coverage_summary", pd.DataFrame())
_save(coverage, "live_scan_output/coverage.csv")

summary = {
    "scanner_version": result.get("scanner_version"),
    "scan_elapsed_seconds": result.get("scan_elapsed_seconds"),
    "database_transport_state": result.get("database_transport_state"),
    "feature_cache_hits": result.get("feature_cache_hits"),
    "feature_cache_refreshes": result.get("feature_cache_refreshes"),
    "leader_rows": leader_n,
    "swing_rows": swing_n,
}
Path("live_scan_output/summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
print(json.dumps(summary, indent=2, default=str))
if isinstance(leaders, pd.DataFrame) and not leaders.empty:
    cols = [c for c in ["ticker", "ranking_score", "final_score", "v9_next_leader_score", "production_rank_eligible", "real_money_authorization_state", "entry_low", "entry_high", "trigger", "stop_loss", "tp1", "tp2"] if c in leaders.columns]
    print("NEXT_LEADERS_TOP15")
    print(leaders.loc[:, cols].head(15).to_csv(index=False))
if isinstance(swings, pd.DataFrame) and not swings.empty:
    cols = [c for c in ["ticker", "ranking_score", "final_score", "v9_swing_score", "production_rank_eligible", "real_money_authorization_state", "entry_low", "entry_high", "trigger", "stop_loss", "tp1", "tp2"] if c in swings.columns]
    print("SWING_TOP15")
    print(swings.loc[:, cols].head(15).to_csv(index=False))
