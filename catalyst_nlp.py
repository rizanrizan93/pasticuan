"""Catalyst NLP helpers for Indonesian IDX/IHSG news filtering.

This module is tuned for Indonesian equities and macro catalysts.

Design goals:
- Reject rumor, clickbait, and retail noise.
- Prefer primary / authoritative sources.
- Classify government policy, macro, sector, and company-level catalysts.
- Keep the API compatible with the Streamlit app.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

# ---------------------------------------------------------------------
# Signals tuned for IDX / IHSG
# ---------------------------------------------------------------------

NEGATIVE_MARKERS = (
    "rumor",
    "rumour",
    "isu",
    "isue",
    "kabar burung",
    "bocoran",
    "leak",
    "spekulasi",
    "speculation",
    "katanya",
    "diduga",
    "disebut-sebut",
    "diperkirakan",
    "viral",
    "trending",
    "social media",
    "media sosial",
    "twitter",
    "x.com",
    "retweet",
    "heboh",
    "unconfirmed",
    "belum dikonfirmasi",
    "tak dikonfirmasi",
    "clickbait",
    "hype",
    "gorengan",
    "saham gorengan",
    "cuan cepat",
    "pump",
    "dump",
    "scalp",
    "retail frenzy",
    "like and subscribe",
)

INDONESIA_GOVERNMENT_BODIES = (
    "pemerintah",
    "presiden",
    "dpr",
    "dpr ri",
    "dpr-ri",
    "mpr",
    "dprd",
    "menkeu",
    "kemenkeu",
    "kementerian keuangan",
    "bi",
    "bank indonesia",
    "ojk",
    "bei",
    "idx",
    "bursa efek indonesia",
    "kemenperin",
    "kementerian perindustrian",
    "esdm",
    "kementerian esdm",
    "kemendag",
    "kementerian perdagangan",
    "kemenhub",
    "kementerian perhubungan",
    "kemenkominfo",
    "kominfo",
    "kementerian investasi",
    "bkpm",
    "bumn",
    "kementerian bumn",
    "sekretariat kabinet",
    "menko",
    "mahkamah agung",
    "mk",
    "bappebti",
    "kppu",
    "bea cukai",
)

POSITIVE_POLICY_MARKERS = (
    "peraturan",
    "regulation",
    "aturan",
    "kebijakan",
    "policy",
    "surat edaran",
    "keputusan",
    "perpres",
    "pmk",
    "pp",
    "uu",
    "revisi aturan",
    "insentif",
    "subsidi",
    "tarif",
    "bea masuk",
    "bea keluar",
    "pajak",
    "cukai",
    "royalti",
    "dhe",
    "tkdn",
    "hilirisasi",
    "kuota",
    "moratorium",
    "relaksasi",
    "pengampunan",
    "stimulus",
    "pelonggaran",
    "pengetatan",
    "deregulasi",
    "standardisasi",
    "izin",
    "perizinan",
)

POSITIVE_MACRO_MARKERS = (
    "bi rate",
    "suku bunga",
    "interest rate",
    "inflasi",
    "inflation",
    "nilai tukar",
    "rupiah",
    "cadangan devisa",
    "neraca perdagangan",
    "current account",
    "defisit fiskal",
    "apbn",
    "pdb",
    "gdp",
    "growth",
    "perlambatan ekonomi",
    "resesi",
    "likuiditas",
    "lcr",
    "gwm",
    "loan growth",
    "kredit",
    "permintaan domestik",
    "ekspor",
    "impor",
    "harga komoditas",
    "commodity",
    "batubara",
    "coal",
    "nickel",
    "nikel",
    "cpo",
    "palm oil",
    "emas",
    "gold",
    "minyak",
    "oil",
    "gas",
)

POSITIVE_COMPANY_MARKERS = (
    "guidance",
    "earnings",
    "financial results",
    "results",
    "laporan keuangan",
    "laba",
    "laba bersih",
    "pendapatan",
    "revenue",
    "profit",
    "net income",
    "margin",
    "dividen",
    "dividend",
    "contract",
    "kontrak",
    "tender",
    "order",
    "pesanan",
    "acquisition",
    "akuisisi",
    "merger",
    "divestment",
    "divestasi",
    "spin-off",
    "rights issue",
    "buyback",
    "buy back",
    "capex",
    "capital expenditure",
    "debt restructuring",
    "restrukturisasi utang",
    "profit warning",
    "guidance raise",
    "guidance cut",
    "capacity expansion",
    "ekspansi kapasitas",
    "pabrik",
    "plant",
    "smelter",
    "production start",
    "commercial operation",
    "operasi komersial",
    "approval",
    "persetujuan",
    "license",
    "licence",
    "izin",
    "ipo",
    "listing",
    "offer",
    "offtake",
    "mou",
    "memorandum of understanding",
    "project",
    "proyek",
)

POSITIVE_SECTOR_MARKERS = (
    "bank",
    "banking",
    "insurance",
    "asuransi",
    "property",
    "properti",
    "telecom",
    "telekomunikasi",
    "consumer",
    "consumer goods",
    "retail",
    "mining",
    "pertambangan",
    "coal",
    "batubara",
    "nickel",
    "nikel",
    "gold",
    "emas",
    "oil",
    "gas",
    "energy",
    "energi",
    "cpo",
    "palm",
    "smelter",
    "industrial",
    "industri",
    "healthcare",
    "farmasi",
    "technology",
    "teknologi",
    "shipping",
    "logistics",
    "transport",
    "infrastructure",
    "infrastruktur",
    "data center",
    "renewable",
    "geothermal",
)

PRIMARY_SOURCE_HINTS = (
    "official",
    "exchange",
    "regulator",
    "statement",
    "filing",
    "company",
    "annual report",
    "quarterly report",
    "reuters",
    "bloomberg",
    "ap",
    "wsj",
    "ft",
    "the jakarta post",
    "bank indonesia",
    "ojk",
    "idx",
    "bei",
    "kementerian",
    "kemendag",
    "kemenkeu",
    "esdm",
    "bkpm",
    "bumn",
    "antara",
)

DOMESTIC_PRIMARY_SOURCE_HINTS = (
    "antara",
    "bisnis indonesia",
    "bisnis.com",
    "kontan",
    "cnbc indonesia",
    "cnn indonesia",
    "tempo",
    "kompas",
    "detik",
)

SYSTEM_PROMPT = """You are CatalystNLP, a strict institutional-grade news filter for Indonesian equities.

Task:
Decide whether a news item is a STRUCTURAL catalyst worth passing to the trading system.

Rules:
- Reject rumor, speculation, clickbait, and retail sentiment noise.
- Pass only verifiable, material, structural news.
- Prefer primary sources: company filings, exchange notices, regulator statements, central bank statements, ministry statements, audited results, or reputable wire reports quoting primary documents.
- Classify macro regime changes, policy actions, sector rotation catalysts, and company-level fundamentals that materially affect earnings, cash flow, valuation, or capital allocation.
- If the story is mainly about price movement, public excitement, or social chatter, reject it.
- For Indonesian equities, prioritize Bank Indonesia, OJK, BEI/IDX, Kementerian Keuangan, ESDM, BKPM, BUMN, DPR, and other policy-related institutions.
- When uncertain, return WATCH rather than PASS.

Return only JSON with keys:
decision, category, confidence, impact_horizon, reasons, tags, summary, red_flags, source_quality, materiality
"""


@dataclass(frozen=True)
class CatalystDecision:
    decision: str
    category: str
    confidence: int
    impact_horizon: str
    reasons: list[str]
    tags: list[str]
    summary: str
    red_flags: list[str]
    source_quality: str = "unknown"
    materiality: str = "medium"

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


def build_catalyst_system_prompt() -> str:
    return SYSTEM_PROMPT


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def _compile_term_pattern(term: str) -> re.Pattern[str]:
    term = str(term or "").strip().lower()
    if not term:
        return re.compile(r"a^")
    escaped = re.escape(term)
    if re.fullmatch(r"[a-z0-9]{1,4}", term):
        return re.compile(rf"\b{escaped}\b", re.IGNORECASE)
    return re.compile(rf"\b{escaped}\b", re.IGNORECASE)


def _contains_term(text: str, term: str) -> bool:
    if not text or not term:
        return False
    term_l = str(term).strip().lower()
    if not term_l:
        return False

    # Short acronyms need word boundaries to avoid false positives (e.g. "bi" in "biasa").
    if re.fullmatch(r"[a-z0-9]{1,4}", term_l):
        return re.search(rf"\b{re.escape(term_l)}\b", text) is not None

    # Generic multi-word or longer terms.
    return term_l in text


def _find_terms(text: str, phrases: Iterable[str]) -> list[str]:
    hits: list[str] = []
    for phrase in phrases:
        if _contains_term(text, phrase):
            hits.append(phrase)
    return hits


def _count_hits(text: str, phrases: Iterable[str]) -> int:
    return len(_find_terms(text, phrases))


def _is_indonesia_policy_source(source: str, text: str) -> bool:
    src = _normalize_text(source)
    return any(_contains_term(src, h) for h in INDONESIA_GOVERNMENT_BODIES) or any(
        _contains_term(text, h) for h in INDONESIA_GOVERNMENT_BODIES
    )


def _source_quality(source: str, text: str) -> str:
    src = _normalize_text(source)

    if any(_contains_term(src, h) for h in PRIMARY_SOURCE_HINTS) or any(
        _contains_term(text, h) for h in PRIMARY_SOURCE_HINTS
    ):
        return "primary"

    if any(_contains_term(src, h) for h in DOMESTIC_PRIMARY_SOURCE_HINTS) or any(
        _contains_term(text, h) for h in DOMESTIC_PRIMARY_SOURCE_HINTS
    ):
        return "primary"

    if src:
        return "secondary"

    return "unknown"


def _policy_tags(text: str) -> list[str]:
    tags: list[str] = []
    if any(k in text for k in ("bi rate", "suku bunga", "inflasi", "rupiah", "cadangan devisa")):
        tags.append("macro_policy")
    if any(k in text for k in ("ojk", "bank indonesia", "bi", "bea cukai", "idx", "bei")):
        tags.append("regulator")
    if any(k in text for k in ("kemenkeu", "menkeu", "apbn", "pajak", "cukai", "subsidi")):
        tags.append("fiscal_policy")
    if any(k in text for k in ("esdm", "royalti", "nikel", "batubara", "coal", "hilirisasi", "tkdn")):
        tags.append("industrial_policy")
    if any(k in text for k in ("dpr", "uu", "revisi", "peraturan", "perpres", "keputusan")):
        tags.append("policy_change")
    return tags


def _sector_tags(text: str) -> list[str]:
    tags: list[str] = []
    mapping = {
        "banking": ("bank", "banking", "kredit", "lcr", "gwm", "bi rate", "suku bunga"),
        "property": ("properti", "property", "mortgage", "kpr"),
        "consumer": ("consumer", "consumer goods", "ritel", "retail", "subsidi"),
        "telecom": ("telekomunikasi", "telecom", "digi", "data"),
        "mining": ("mining", "pertambangan", "batubara", "coal", "nikel", "nickel", "emas", "gold"),
        "energy": ("oil", "gas", "energy", "energi", "cpo", "palm", "minyak"),
        "healthcare": ("healthcare", "farmasi", "pharma", "obat"),
        "technology": ("technology", "teknologi", "digital", "software", "platform"),
        "industrial": ("industrial", "industri", "smelter", "factory", "pabrik", "manufacturing"),
        "infrastructure": ("infrastructure", "infrastruktur", "logistics", "shipping", "transport"),
    }
    for tag, kws in mapping.items():
        if any(k in text for k in kws):
            tags.append(tag)
    return tags


def _category_profile(text: str) -> dict[str, dict[str, Any]]:
    """Collect evidence by category without letting one bucket overwrite another."""
    return {
        "policy": {
            "hits": _find_terms(text, POSITIVE_POLICY_MARKERS),
            "priority": 4,
            "horizon": "weeks",
            "base_materiality": "high",
            "reason": "policy_or_regulatory_signal",
            "tags": _policy_tags(text),
        },
        "macro": {
            "hits": _find_terms(text, POSITIVE_MACRO_MARKERS),
            "priority": 3,
            "horizon": "weeks",
            "base_materiality": "high",
            "reason": "macro_regime_relevant",
            "tags": ["macro"],
        },
        "company_structural": {
            "hits": _find_terms(text, POSITIVE_COMPANY_MARKERS),
            "priority": 2,
            "horizon": "weeks",
            "base_materiality": "high",
            "reason": "company_structural_catalyst",
            "tags": ["fundamental"],
        },
        "sector": {
            "hits": _find_terms(text, POSITIVE_SECTOR_MARKERS),
            "priority": 1,
            "horizon": "days",
            "base_materiality": "medium",
            "reason": "sector_rotation_relevant",
            "tags": ["sector"],
        },
        "noise": {
            "hits": _find_terms(text, NEGATIVE_MARKERS),
            "priority": 0,
            "horizon": "days",
            "base_materiality": "low",
            "reason": "rumor_or_noise_language",
            "tags": ["noise"],
        },
    }


def _detect_category(text: str) -> tuple[str, str, list[str], list[str], str]:
    """Return dominant category, horizon, reasons, tags, and materiality.

    Priority order for IDX:
    policy > macro > company_structural > sector > noise
    """
    reasons: list[str] = []
    tags: list[str] = []
    materiality = "medium"
    impact_horizon = "days"
    category = "unknown"

    profile = _category_profile(text)
    signal_counts = {cat: len(info["hits"]) for cat, info in profile.items()}

    # Choose the highest-priority category that has at least one real hit.
    for cat in ("policy", "macro", "company_structural", "sector", "noise"):
        if signal_counts.get(cat, 0) > 0:
            category = cat
            impact_horizon = str(profile[cat]["horizon"])
            materiality = str(profile[cat]["base_materiality"])
            reasons.append(str(profile[cat]["reason"]))
            tags.extend(profile[cat]["tags"])
            break

    # Add contextual evidence from lower-priority buckets without changing category.
    if category != "unknown":
        if category != "policy" and signal_counts["policy"] > 0:
            reasons.append("secondary_policy_context")
            tags.extend(profile["policy"]["tags"])
        if category not in ("policy", "macro") and signal_counts["macro"] > 0:
            reasons.append("secondary_macro_context")
            tags.extend(profile["macro"]["tags"])
        if category not in ("policy", "macro", "company_structural") and signal_counts["company_structural"] > 0:
            reasons.append("secondary_company_context")
            tags.extend(profile["company_structural"]["tags"])
        if category == "noise":
            materiality = "low"

    # If no structural bucket hit but institutional references exist, elevate to policy.
    if category == "unknown" and any(
        _contains_term(text, h) for h in ("bank indonesia", "ojk", "idx", "bei", "kemenkeu", "esdm", "bkpm", "dpr")
    ):
        category = "policy"
        impact_horizon = "weeks"
        materiality = "high"
        reasons.append("institutional_reference")
        tags.extend(_policy_tags(text))

    # Deduplicate while preserving order.
    tags = list(dict.fromkeys(tags))
    reasons = list(dict.fromkeys(reasons))

    return category, impact_horizon, reasons, tags, materiality


def _decision_thresholds(score: int, category: str, red_flags: list[str]) -> str:
    if category == "noise" and score < 60:
        return "REJECT"
    if "rumor_or_noise_language" in red_flags and score < 70:
        return "REJECT"
    if score >= 82:
        return "PASS"
    if score >= 60:
        return "WATCH"
    return "REJECT"


def score_news_item(title: str, summary: str = "", source: str = "") -> CatalystDecision:
    title_text = _normalize_text(title)
    summary_text = _normalize_text(summary)
    source_text = _normalize_text(source)
    text = " ".join([title_text, summary_text, source_text]).strip()

    reasons: list[str] = []
    red_flags: list[str] = []
    tags: list[str] = []

    # Start neutral, then move up/down.
    score = 45
    category = "unknown"
    impact_horizon = "days"
    source_quality = _source_quality(source, text)
    materiality = "medium"

    if not text:
        return CatalystDecision(
            decision="REJECT",
            category="unknown",
            confidence=0,
            impact_horizon="days",
            reasons=["empty_news_item"],
            tags=["news"],
            summary="No title",
            red_flags=["empty_input"],
            source_quality="unknown",
            materiality="low",
        )

    # Noise / rumor penalty.
    negative_hits = _find_terms(text, NEGATIVE_MARKERS)
    if negative_hits:
        score -= 30
        red_flags.append("rumor_or_noise_language")
        category = "noise"
        source_quality = "rumor"
        materiality = "low"
        reasons.append("negative_language_detected")
        tags.append("noise")

    # Very short / vague headlines are less trustworthy.
    if len(title_text) < 24 and len(summary_text) < 40:
        score -= 6
        red_flags.append("thin_context")

    cat, horizon, cat_reasons, cat_tags, cat_materiality = _detect_category(text)
    if cat != "unknown":
        category = cat
        impact_horizon = horizon
        reasons.extend(cat_reasons)
        tags.extend(cat_tags)
        materiality = cat_materiality

        if cat == "policy":
            score += 28
        elif cat == "macro":
            score += 22
        elif cat == "company_structural":
            score += 20
        elif cat == "sector":
            score += 14
        elif cat == "noise":
            score -= 8

    # Primary source and local authoritative source boost.
    if source_quality == "primary":
        score += 12
        tags.append("primary_source")
    elif source_quality == "secondary":
        score += 4

    # Strong government / regulator references are especially valuable for IDX.
    if _is_indonesia_policy_source(source, text):
        score += 8
        tags.append("id_policy")

    # Positive wording that usually indicates structural change.
    structural_hits = _count_hits(
        text,
        (
            "audited",
            "results",
            "earnings",
            "guidance",
            "buyback",
            "rights issue",
            "contract",
            "tender",
            "approval",
            "license",
            "licence",
            "capex",
            "capacity expansion",
            "operasi komersial",
            "ekspansi kapasitas",
            "investasi",
            "proyek",
        ),
    )
    if structural_hits:
        score += 6
        tags.append("corporate_event")

    # Retail hype / analyst chatter / price-target noise.
    if _count_hits(
        text,
        (
            "social media",
            "viral",
            "trending",
            "analyst says",
            "price target",
            "target price",
            "unconfirmed",
            "rumored",
            "rumour",
        ),
    ):
        score -= 20
        red_flags.append("retail_hype")
        tags.append("noise")

    # Price action only news should not be treated as structural.
    if _count_hits(text, ("shares jump", "stock rallies", "stock falls", "price surges", "price tumbles", "sentimen", "market reacts")):
        score -= 10
        red_flags.append("price_action_only")

    # If the item mentions a concrete policy instrument, increase conviction.
    if _count_hits(text, ("tarif", "subsidi", "pajak", "cukai", "royalti", "dhe", "tkdn", "insentif", "kuota", "relaksasi", "pengetatan")):
        score += 6
        tags.append("policy_instrument")

    # Higher-quality local official context.
    if _count_hits(text, ("bank indonesia", "ojk", "idx", "bei", "kemenkeu", "esdm", "bkpm", "bumn", "dpr")):
        score += 4
        tags.append("id_institution")

    # Ensure tags and reasons are populated.
    if not reasons:
        reasons.append("insufficient_structural_signal")
    if not tags:
        tags.append("news")

    # Cap score.
    score = int(max(0, min(100, score)))

    decision = _decision_thresholds(score, category, red_flags)

    return CatalystDecision(
        decision=decision,
        category=category,
        confidence=score,
        impact_horizon=impact_horizon,
        reasons=reasons[:5],
        tags=list(dict.fromkeys(tags))[:8],
        summary=title.strip()[:180] if title else "No title",
        red_flags=red_flags[:5],
        source_quality=source_quality,
        materiality=materiality,
    )


def filter_news_items(items: Iterable[dict]) -> list[CatalystDecision]:
    out: list[CatalystDecision] = []
    for item in items:
        if not isinstance(item, dict):
            continue

        # Support common news item shapes from Yahoo / search / manual input.
        title = str(
            item.get("title")
            or item.get("headline")
            or item.get("name")
            or item.get("article_title")
            or ""
        )
        summary = str(
            item.get("summary")
            or item.get("description")
            or item.get("content")
            or item.get("snippet")
            or item.get("body")
            or ""
        )
        source = str(
            item.get("source")
            or item.get("publisher")
            or item.get("provider")
            or item.get("site")
            or ""
        )
        out.append(score_news_item(title=title, summary=summary, source=source))
    return out


def _strip_code_fences(payload: str) -> str:
    text = str(payload or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_catalyst_response(payload: str) -> CatalystDecision:
    try:
        data = json.loads(_strip_code_fences(payload))
    except Exception:
        return CatalystDecision(
            decision="WATCH",
            category="unknown",
            confidence=50,
            impact_horizon="days",
            reasons=["invalid_json_response"],
            tags=["news"],
            summary="Parser fallback",
            red_flags=["parse_error"],
            source_quality="unknown",
            materiality="medium",
        )

    reasons = data.get("reasons", [])
    tags = data.get("tags", [])
    red_flags = data.get("red_flags", [])

    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    if not isinstance(tags, list):
        tags = [str(tags)]
    if not isinstance(red_flags, list):
        red_flags = [str(red_flags)]

    return CatalystDecision(
        decision=str(data.get("decision", "WATCH")),
        category=str(data.get("category", "unknown")),
        confidence=int(data.get("confidence", 50)),
        impact_horizon=str(data.get("impact_horizon", "days")),
        reasons=[str(x) for x in reasons],
        tags=[str(x) for x in tags],
        summary=str(data.get("summary", "")),
        red_flags=[str(x) for x in red_flags],
        source_quality=str(data.get("source_quality", "unknown")),
        materiality=str(data.get("materiality", "medium")),
    )
