"""Focused Multibagger and Core Swing intelligence for IDX Super Scanner.

This module contains forward-fundamental research, Multibagger scoring, capital
allocation, and a focused Core Swing order/ranking layer. Intraday fast-trade
engines are intentionally excluded.
"""
from __future__ import annotations

import hashlib
import io
import logging
import warnings
import ipaddress
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse

try:
    import requests
except ModuleNotFoundError:  # Deployment installs requirements; core tests stay importable.
    requests = None
try:
    from bs4 import BeautifulSoup
except ModuleNotFoundError:  # Automatic forward review fails soft when optional parsing is absent.
    BeautifulSoup = None

from scanner import Any, BinaryIO, Iterable, Mapping, ScanConfig, ThreadPoolExecutor, read_csv_input, safe_number, safe_text, silent_accumulation_metrics, silent_accumulation_profile, truthy, as_completed, math, normalize_idx_ticker, np, pd


from ai_engine import LocalAIConfig, enrich_profit_ranking_with_ai, enrich_multibagger_with_peer_ai
from selector_engine import (
    SelectorConfig,
    attach_setups_to_selector,
    build_cross_sectional_selector,
    current_silent_profiles,
)
from narrative_engine import (
    attach_narrative_profiles,
    build_narrative_intelligence,
)
from time_cycle import TimeCycleConfig, analyze_time_cycle, setup_time_alignment

warnings.filterwarnings(
    "ignore",
    message=r"The 'generic' unit for NumPy timedelta is deprecated.*",
    category=DeprecationWarning,
    module=r"yfinance(?:\..*)?",
)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

_FORWARD_PROJECT_TERMS = (
    'project', 'proyek', 'expansion', 'ekspansi', 'capacity', 'kapasitas',
    'plant', 'pabrik', 'smelter', 'mine', 'tambang', 'construction',
    'konstruksi', 'commissioning', 'commercial operation', 'cod',
    'acquisition', 'akuisisi', 'contract', 'kontrak', 'capex', 'investment',
    'investasi', 'development', 'pengembangan', 'pipeline', 'refinery',
    'warehouse', 'gudang', 'data center', 'power plant', 'pembangkit',
)
_FORWARD_IR_TERMS = (
    'investor', 'annual report', 'laporan tahunan', 'public expose', 'pubex',
    'presentation', 'presentasi', 'disclosure', 'keterbukaan', 'project',
    'proyek', 'expansion', 'ekspansi', 'management', 'direksi', 'board',
)
_FORWARD_STAGE_SCORES = {
    'PLANNING': 22.0, 'FEASIBILITY': 30.0, 'PERMITTING': 40.0,
    'FINANCING': 52.0, 'CONSTRUCTION': 68.0, 'COMMISSIONING': 86.0,
    'OPERATING': 100.0,
}
_FORWARD_STAGE_PATTERNS = (
    ('OPERATING', ('commercial operation', 'telah beroperasi', 'operational', 'beroperasi penuh')),
    ('COMMISSIONING', ('commissioning', 'uji coba', 'trial operation', 'ramp-up', 'ramp up')),
    ('CONSTRUCTION', ('construction', 'konstruksi', 'groundbreaking', 'pembangunan', 'progress')),
    ('FINANCING', ('financial close', 'funding secured', 'pendanaan', 'financing', 'pinjaman')),
    ('PERMITTING', ('permit', 'perizinan', 'izin lingkungan', 'amdal')),
    ('FEASIBILITY', ('feasibility', 'studi kelayakan', 'bankable feasibility')),
    ('PLANNING', ('plan', 'rencana', 'planned', 'akan membangun', 'proposal')),
)
_FORWARD_GOVERNANCE_NEGATIVE = (
    'restatement', 'penyajian kembali', 'qualified opinion', 'opini wajar dengan pengecualian',
    'disclaimer opinion', 'tidak menyatakan pendapat', 'fraud', 'korupsi', 'bribery',
    'suap', 'sanction', 'sanksi', 'lawsuit', 'gugatan', 'default', 'gagal bayar',
    'related party concern', 'benturan kepentingan', 'investigation', 'penyelidikan',
)
_FORWARD_NAV_NOISE = (
    'home', 'menu', 'contact us', 'hubungi kami', 'privacy policy', 'kebijakan privasi',
    'career', 'karir', 'download', 'site map', 'sitemap', 'copyright', 'all rights reserved',
    'internal audit', 'komite audit', 'audit committee', 'corporate secretary',
    'whistleblowing', 'investor relation menu', 'board charter', 'piagam komite',
)
_FORWARD_PROJECT_STRONG_ACTIONS = (
    'membangun', 'pembangunan', 'konstruksi', 'construction', 'commissioning',
    'beroperasi', 'commercial operation', 'groundbreaking', 'ekspansi kapasitas',
    'menambah kapasitas', 'akuisisi', 'acquisition', 'kontrak baru', 'new contract',
    'investasi', 'investment', 'capex', 'offtake', 'financial close',
)
_FORWARD_PROJECT_OBJECTS = (
    'pabrik', 'plant', 'smelter', 'tambang', 'mine', 'refinery', 'kilang',
    'data center', 'warehouse', 'gudang', 'power plant', 'pembangkit', 'jalan tol',
    'pelabuhan', 'port', 'kapasitas produksi', 'production capacity', 'proyek', 'project',
)


def _forward_cache_root() -> Path:
    base = os.getenv('IDX_SCANNER_CACHE_DIR', '').strip()
    root = Path(base).expanduser() if base else Path.home() / '.cache' / 'idx_super_scanner'
    path = root / 'forward_intelligence'
    path.mkdir(parents=True, exist_ok=True)
    return path


def _forward_cache_path(ticker: str) -> Path:
    safe = re.sub(r'[^A-Z0-9_.-]+', '_', normalize_idx_ticker(ticker).upper())
    return _forward_cache_root() / f'{safe}.json'


def _read_forward_cache(ticker: str, max_age_days: int) -> pd.DataFrame:
    path = _forward_cache_path(ticker)
    if not path.exists():
        return pd.DataFrame()
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
        written = pd.Timestamp(payload.get('written_at'))
        if written.tzinfo is None:
            written = written.tz_localize('Asia/Jakarta')
        age_days = (pd.Timestamp.now(tz='Asia/Jakarta') - written).total_seconds() / 86400.0
        if age_days > max(0, int(max_age_days)):
            return pd.DataFrame()
        rows = payload.get('rows') or []
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def _write_forward_cache(ticker: str, frame: pd.DataFrame) -> None:
    try:
        payload = {
            'written_at': pd.Timestamp.now(tz='Asia/Jakarta').isoformat(),
            'schema_version': 1,
            'rows': frame.replace({np.nan: None}).to_dict('records') if frame is not None else [],
        }
        _forward_cache_path(ticker).write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    except Exception:
        pass


def _safe_public_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        if parsed.scheme not in {'http', 'https'} or not parsed.hostname:
            return False
        host = parsed.hostname.lower().strip('.')
        if host in {'localhost', '0.0.0.0'} or host.endswith('.local'):
            return False
        try:
            ip = ipaddress.ip_address(host)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
        except ValueError:
            pass
        return True
    except Exception:
        return False


def _normalized_host(url: str) -> str:
    try:
        host = (urlparse(url).hostname or '').lower()
        return host[4:] if host.startswith('www.') else host
    except Exception:
        return ''


def _source_family(url: str, company_host: str = '') -> tuple[str, bool]:
    host = _normalized_host(url)
    company_host = company_host.lower().removeprefix('www.')
    if host == 'idx.co.id' or host.endswith('.idx.co.id'):
        return ('IDX_OFFICIAL', True)
    if host == 'ojk.go.id' or host.endswith('.ojk.go.id'):
        return ('OJK_OFFICIAL', True)
    if company_host and (host == company_host or host.endswith('.' + company_host)):
        return ('COMPANY_IR', True)
    return ('OTHER_PUBLIC', False)


def _decode_search_result_url(value: str) -> str:
    if not value:
        return ''
    value = urljoin('https://html.duckduckgo.com', value)
    parsed = urlparse(value)
    query = parse_qs(parsed.query)
    candidate = query.get('uddg', [''])[0]
    return unquote(candidate) if candidate else value


def _fetch_document(url: str, timeout: float, max_bytes: int = 8_000_000, max_pdf_pages: int = 45) -> tuple[str, str, str]:
    if not _safe_public_url(url):
        return ('', '', 'UNSAFE_URL')
    if requests is None or BeautifulSoup is None:
        return ('', url, 'OPTIONAL_FORWARD_DEPENDENCY_UNAVAILABLE')
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; IDXSuperScanner/6.0; research-only)',
        'Accept': 'text/html,application/pdf;q=0.9,*/*;q=0.7',
    }
    try:
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True, stream=True)
        response.raise_for_status()
        final_url = response.url
        content = bytearray()
        truncated = False
        for chunk in response.iter_content(chunk_size=65536):
            if not chunk:
                continue
            content.extend(chunk)
            if len(content) > max_bytes:
                truncated = True
                break
        content_type = (response.headers.get('content-type') or '').lower()
        raw = bytes(content[:max_bytes])
        if 'pdf' in content_type or final_url.lower().split('?')[0].endswith('.pdf'):
            if truncated:
                return ('', final_url, 'DOCUMENT_TOO_LARGE')
            pdf_probe = raw.lstrip()
            if not pdf_probe.startswith(b'%PDF-'):
                return ('', final_url, 'INVALID_PDF_PAYLOAD')
            if b'%%EOF' not in raw[-65536:]:
                return ('', final_url, 'INCOMPLETE_PDF_PAYLOAD')
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(raw), strict=False)
                pages = []
                for page in reader.pages[:max_pdf_pages]:
                    try:
                        pages.append(page.extract_text() or '')
                    except Exception:
                        continue
                return ('\n'.join(pages), final_url, 'PDF')
            except Exception as exc:
                return ('', final_url, f'PDF_PARSE_ERROR:{type(exc).__name__}')
        soup = BeautifulSoup(raw, 'html.parser')
        for tag in soup(['script', 'style', 'noscript', 'svg']):
            tag.decompose()
        return (' '.join(soup.get_text(' ', strip=True).split()), final_url, 'HTML')
    except requests.Timeout:
        return ('', url, 'PROVIDER_TIMEOUT')
    except requests.RequestException as exc:
        return ('', url, f'PROVIDER_CONNECTION_ERROR:{type(exc).__name__}')
    except Exception as exc:
        return ('', url, f'PROGRAMMING_ERROR:{type(exc).__name__}')


def _discover_ir_links(website: str, timeout: float, max_links: int = 8) -> list[str]:
    if not website or not _safe_public_url(website):
        return []
    if requests is None or BeautifulSoup is None:
        return []
    try:
        response = requests.get(
            website,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; IDXSuperScanner/6.0)'},
            timeout=timeout,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        scored: list[tuple[int, str]] = []
        base_host = _normalized_host(response.url)
        for tag in soup.find_all('a', href=True):
            href = urljoin(response.url, tag.get('href', ''))
            if not _safe_public_url(href) or _normalized_host(href) != base_host:
                continue
            label = f"{tag.get_text(' ', strip=True)} {href}".lower()
            score = sum(1 for term in _FORWARD_IR_TERMS if term in label)
            if score:
                scored.append((score, href))
        result = []
        landing_pages = []
        for _, href in sorted(scored, key=lambda item: item[0], reverse=True):
            if href not in result:
                result.append(href)
                if not href.lower().split('?')[0].endswith('.pdf'):
                    landing_pages.append(href)
            if len(result) >= max_links:
                break
        # Crawl a bounded second level so an Investor Relations landing page can
        # reveal annual reports, public-expose decks, and project presentations.
        for landing in landing_pages[:3]:
            try:
                nested = requests.get(landing, headers={'User-Agent': 'Mozilla/5.0 (compatible; IDXSuperScanner/6.0)'}, timeout=timeout)
                nested.raise_for_status()
                nested_soup = BeautifulSoup(nested.content, 'html.parser')
                nested_scored = []
                for tag in nested_soup.find_all('a', href=True):
                    href = urljoin(nested.url, tag.get('href', ''))
                    if not _safe_public_url(href) or _normalized_host(href) != base_host:
                        continue
                    label = f"{tag.get_text(' ', strip=True)} {href}".lower()
                    score = sum(1 for term in _FORWARD_IR_TERMS if term in label)
                    if href.lower().split('?')[0].endswith('.pdf'):
                        score += 2
                    if score:
                        nested_scored.append((score, href))
                for _, href in sorted(nested_scored, key=lambda item: item[0], reverse=True):
                    if href not in result:
                        result.append(href)
                    if len(result) >= max_links:
                        return result
            except Exception:
                continue
        return result
    except Exception:
        return []


def _search_forward_links(ticker: str, company_name: str, company_website: str, timeout: float, max_results: int = 10) -> list[str]:
    if requests is None or BeautifulSoup is None:
        return []
    code = normalize_idx_ticker(ticker).replace('.JK', '')
    company_host = _normalized_host(company_website)
    queries = [
        f'"{code}" "{company_name}" site:idx.co.id/StaticData/NewsAndAnnouncement project OR proyek OR "public expose" OR "annual report"',
        f'"{company_name}" project expansion management annual report investor relations',
    ]
    links: list[str] = []
    for query in queries:
        try:
            url = 'https://html.duckduckgo.com/html/?q=' + quote_plus(query)
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            for tag in soup.select('a.result__a, a[data-testid="result-title-a"]'):
                target = _decode_search_result_url(tag.get('href', ''))
                host = _normalized_host(target)
                allowed = host == 'idx.co.id' or host.endswith('.idx.co.id') or (company_host and (host == company_host or host.endswith('.' + company_host)))
                if allowed and _safe_public_url(target) and target not in links:
                    links.append(target)
                if len(links) >= max_results:
                    return links
        except Exception:
            continue
    return links


def _parse_localized_number(text: str) -> float:
    value = re.sub(r'[^0-9,.-]', '', str(text or ''))
    if not value:
        return np.nan
    if ',' in value and '.' in value:
        if value.rfind(',') > value.rfind('.'):
            value = value.replace('.', '').replace(',', '.')
        else:
            value = value.replace(',', '')
    elif ',' in value:
        tail = value.rsplit(',', 1)[-1]
        value = value.replace(',', '.') if len(tail) <= 2 else value.replace(',', '')
    try:
        return float(value)
    except Exception:
        return np.nan


def _parse_idr_amount(text: str) -> float:
    pattern = re.compile(
        r'(?:(Rp\.?|IDR)\s*)?([0-9]+(?:[.,][0-9]+)?)\s*'
        r'(triliun|trillion|tn|miliar|billion|bn|juta|million|mn)?', re.I,
    )
    best = np.nan
    for prefix, number, scale in pattern.findall(str(text or '')):
        if not prefix and not scale:
            continue
        value = _parse_localized_number(number)
        if not np.isfinite(value):
            continue
        factor = {
            'triliun': 1e12, 'trillion': 1e12, 'tn': 1e12,
            'miliar': 1e9, 'billion': 1e9, 'bn': 1e9,
            'juta': 1e6, 'million': 1e6, 'mn': 1e6,
        }.get(scale.lower(), 1.0) if scale else 1.0
        candidate = value * factor
        if not np.isfinite(best) or candidate > best:
            best = candidate
    return best


def _extract_percent(sentence: str, labels: tuple[str, ...]) -> float:
    lower = sentence.lower()
    for label in labels:
        position = lower.find(label)
        if position < 0:
            continue
        window = sentence[max(0, position - 60): position + len(label) + 90]
        match = re.search(r'([0-9]+(?:[.,][0-9]+)?)\s*%', window)
        if match:
            value = _parse_localized_number(match.group(1))
            return value / 100.0 if np.isfinite(value) else np.nan
    return np.nan


def _project_stage(sentence: str) -> str:
    lower = sentence.lower()
    future_markers = ('target', 'ditargetkan', 'expected', 'akan', 'planned', 'rencana')
    construction_markers = ('construction', 'konstruksi', 'pembangunan', 'progress', 'groundbreaking')
    if any(term in lower for term in construction_markers) and any(term in lower for term in future_markers):
        return 'CONSTRUCTION'
    for stage, patterns in _FORWARD_STAGE_PATTERNS:
        if any(pattern in lower for pattern in patterns):
            return stage
    return 'PLANNING'


def _extract_year(sentence: str) -> int | None:
    years = [int(value) for value in re.findall(r'\b(20[2-4][0-9])\b', sentence)]
    return min(years) if years else None


def _extract_management_name(text: str) -> tuple[str, str]:
    patterns = (
        r'(?:Direktur Utama|President Director|Chief Executive Officer|CEO)\s*(?:adalah|is|:|-)?\s*([A-Z][A-Za-zÀ-ÖØ-öø-ÿ.\' -]{3,80})',
        r'([A-Z][A-Za-zÀ-ÖØ-öø-ÿ.\' -]{3,80})\s*(?:menjabat sebagai|serves as)\s*(Direktur Utama|President Director|Chief Executive Officer|CEO)',
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            if len(match.groups()) == 1:
                return (' '.join(match.group(1).split())[:80], 'President Director/CEO')
            return (' '.join(match.group(1).split())[:80], match.group(2))
    return ('', '')


def _clean_project_excerpt(sentence: str) -> str:
    text = ' '.join(str(sentence or '').split())
    # Remove common breadcrumb/menu fragments without altering the evidence body.
    text = re.sub(r'(?i)^(home|beranda|investor relations?|hubungan investor)\s*[>›|:/-]+\s*', '', text)
    text = re.sub(r'\s{2,}', ' ', text).strip(' -|>›')
    return text[:500]


def _is_project_evidence(sentence: str) -> bool:
    lower = sentence.lower()
    if len(sentence) < 45 or len(sentence) > 1600:
        return False
    noise_hits = sum(term in lower for term in _FORWARD_NAV_NOISE)
    strong_action = any(term in lower for term in _FORWARD_PROJECT_STRONG_ACTIONS)
    project_object = any(term in lower for term in _FORWARD_PROJECT_OBJECTS)
    numeric_evidence = bool(re.search(r'\b(?:rp|idr|usd)\s*[0-9]|[0-9]+(?:[.,][0-9]+)?\s*%|\b20[2-4][0-9]\b', lower))
    stage_evidence = any(pattern in lower for _, patterns in _FORWARD_STAGE_PATTERNS for pattern in patterns)
    # Governance/menu prose is not a project even if it happens to contain the
    # generic word "development" or "management".
    if noise_hits >= 2 and not (strong_action and numeric_evidence):
        return False
    return bool(project_object and strong_action and (numeric_evidence or stage_evidence))


def _sanitize_management_name(name: str) -> str:
    value = re.sub(r'(?i)\b(mr|mrs|ms|dr|ir|prof|h|hj)\.?\s+', '', str(name or '')).strip(' .,:;-')
    value = re.sub(r'\s+', ' ', value)
    if len(value) < 4 or len(value.split()) > 8 or not re.search(r'[A-Za-z]', value):
        return ''
    if any(term in value.lower() for term in ('committee', 'komite', 'audit', 'director board', 'website')):
        return ''
    return value[:80]


def _extract_forward_rows(ticker: str, text: str, source_url: str, source_family: str, verified: bool, fund: Mapping[str, Any]) -> list[dict[str, Any]]:
    clean = ' '.join(str(text or '').split())
    if not clean:
        return []
    sentences = re.split(r'(?<=[.!?])\s+|\n+', clean)
    rows: list[dict[str, Any]] = []
    consumed: set[int] = set()
    for position, sentence in enumerate(sentences):
        if position in consumed:
            continue
        sentence = _clean_project_excerpt(sentence)
        lower = sentence.lower()
        if not _is_project_evidence(sentence):
            continue
        if position + 1 < len(sentences):
            follow = sentences[position + 1]
            follow_lower = follow.lower()
            continuation_terms = ('progress', 'completion', 'selesai', 'commercial operation', 'commissioning', 'cod', 'funding', 'pendanaan', 'offtake', 'kapasitas')
            if any(term in follow_lower for term in continuation_terms):
                sentence = f'{sentence} {follow}'
                lower = sentence.lower()
                consumed.add(position + 1)
        stage = _project_stage(sentence)
        completion = _extract_percent(sentence, ('progress', 'completion', 'penyelesaian', 'selesai'))
        funding = _extract_percent(sentence, ('funding', 'pendanaan', 'financing'))
        offtake = _extract_percent(sentence, ('offtake', 'contracted', 'kontrak penjualan'))
        ownership = _extract_percent(sentence, ('ownership', 'kepemilikan', 'economic interest'))
        overrun = _extract_percent(sentence, ('cost overrun', 'pembengkakan biaya'))
        amount = _parse_idr_amount(sentence)
        capex = amount if any(term in lower for term in ('capex', 'investment', 'investasi', 'nilai proyek')) else np.nan
        expected_revenue = amount if any(term in lower for term in ('expected revenue', 'revenue contribution', 'tambahan pendapatan', 'kontribusi pendapatan')) else np.nan
        expected_ebitda = amount if 'ebitda' in lower else np.nan
        cod_year = _extract_year(sentence) if any(term in lower for term in ('cod', 'commercial operation', 'beroperasi', 'commissioning')) else None
        strategic = any(term in lower for term in ('national strategic project', 'proyek strategis nasional', 'psn', 'strategic project'))
        risk = 'HIGH' if any(term in lower for term in ('delay', 'tertunda', 'cost overrun', 'dispute', 'sengketa')) else 'LOW' if stage in {'COMMISSIONING', 'OPERATING'} else 'MEDIUM'
        row = {
            'ticker': normalize_idx_ticker(ticker),
            'as_of': pd.Timestamp.now(tz='Asia/Jakarta').date().isoformat(),
            'source_url': source_url,
            'source_family': source_family,
            'source_verified': bool(verified),
            'project_name': _clean_project_excerpt(sentence)[:180],
            'project_stage': stage,
            'project_completion_pct': completion,
            'project_capex_idr': capex,
            'funding_secured_pct': funding,
            'offtake_secured_pct': offtake,
            'expected_revenue_idr': expected_revenue,
            'expected_ebitda_idr': expected_ebitda,
            'expected_cod': str(cod_year or ''),
            'ownership_pct': ownership if np.isfinite(ownership) else np.nan,
            'project_delay_months': np.nan,
            'cost_overrun_pct': overrun,
            'strategic_project': strategic,
            'project_risk': risk,
            'evidence_excerpt': sentence[:500],
        }
        rows.append(row)
        if len(rows) >= 8:
            break
    ceo_name, ceo_title = _extract_management_name(clean)
    ceo_name = _sanitize_management_name(ceo_name or safe_text(fund.get('ceo_name')))
    ceo_title = ceo_title or safe_text(fund.get('ceo_title'))
    appointment_year = None
    if ceo_name:
        pos = clean.lower().find(ceo_name.lower())
        if pos >= 0:
            appointment_year = _extract_year(clean[max(0, pos - 180):pos + 280])
    current_year = pd.Timestamp.now(tz='Asia/Jakarta').year
    tenure = current_year - appointment_year if appointment_year and appointment_year <= current_year else np.nan
    governance_hits = [term for term in _FORWARD_GOVERNANCE_NEGATIVE if term in clean.lower()]
    governance_risks = [safe_number(fund.get(key), np.nan) for key in ('governance_overall_risk', 'governance_board_risk', 'governance_audit_risk')]
    governance_risks = [value for value in governance_risks if np.isfinite(value)]
    governance_score = 100.0 - 10.0 * float(np.mean(governance_risks)) if governance_risks else np.nan
    capital_allocation = 50.0
    roic = safe_number(fund.get('history_roic_proxy'), np.nan)
    cash_conversion = safe_number(fund.get('history_cash_conversion'), np.nan)
    dilution = safe_number(fund.get('history_share_dilution_yoy'), np.nan)
    if np.isfinite(roic):
        capital_allocation += 20.0 if roic >= 0.12 else 10.0 if roic >= 0.07 else -10.0
    if np.isfinite(cash_conversion):
        capital_allocation += 15.0 if 0.8 <= cash_conversion <= 1.8 else -10.0 if cash_conversion < 0.5 else 5.0
    if np.isfinite(dilution):
        capital_allocation -= 25.0 if dilution > 0.12 else 10.0 if dilution > 0.05 else 0.0
    management_row = {
        'ticker': normalize_idx_ticker(ticker),
        'as_of': pd.Timestamp.now(tz='Asia/Jakarta').date().isoformat(),
        'source_url': source_url,
        'source_family': source_family,
        'source_verified': bool(verified),
        'project_name': '', 'project_stage': '',
        'ceo_name': ceo_name,
        'management_team': ceo_title,
        'ceo_tenure_years': tenure,
        'board_avg_tenure_years': np.nan,
        'management_revenue_cagr': safe_number(fund.get('history_revenue_cagr_3y'), safe_number(fund.get('revenue_growth'), np.nan)),
        'management_roic_change_pct': safe_number(fund.get('history_roic_change_3y'), np.nan),
        'capital_allocation_score': max(0.0, min(100.0, capital_allocation)),
        'governance_score': max(0.0, min(100.0, governance_score)) if np.isfinite(governance_score) else np.nan,
        'board_turnover_3y': np.nan,
        'insider_ownership_pct': np.nan,
        'audit_clean': not bool(governance_hits),
        'related_party_risk': 'HIGH' if any(term in clean.lower() for term in ('benturan kepentingan', 'related party concern')) else 'UNKNOWN',
        'legal_governance_flags': ' • '.join(governance_hits[:5]),
        'management_source_url': source_url,
        'management_verified': bool(verified and ceo_name),
        'evidence_excerpt': (f'CEO: {ceo_name}; source {source_family}' if ceo_name else f'Management source {source_family}'),
    }
    rows.append(management_row)
    return rows


def _automatic_forward_one(ticker: str, fund: Mapping[str, Any], cfg: ScanConfig, force_refresh: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache_days = max(1, int(getattr(cfg, 'automatic_forward_quality_cache_days', 14)))
    if not force_refresh:
        cached = _read_forward_cache(ticker, cache_days)
        if not cached.empty:
            return cached, {'ticker': ticker, 'state': 'CACHE_HIT', 'documents': 0, 'rows': len(cached), 'source_families': safe_text(cached.get('source_family', pd.Series(dtype=str)).dropna().unique().tolist())}
    company_name = safe_text(fund.get('company_name'))
    website = safe_text(fund.get('company_website'))
    timeout = float(getattr(cfg, 'automatic_forward_quality_timeout_seconds', 8.0))
    max_docs = max(1, int(getattr(cfg, 'automatic_forward_quality_max_documents', 5)))
    company_host = _normalized_host(website)
    links = _discover_ir_links(website, timeout, max_links=max_docs)
    for link in _search_forward_links(ticker, company_name, website, timeout, max_results=max_docs * 2):
        if link not in links:
            links.append(link)
        if len(links) >= max_docs:
            break
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    families: set[str] = set()
    documents = 0
    for link in links[:max_docs]:
        text, final_url, state = _fetch_document(link, timeout)
        if not text:
            errors.append(state)
            continue
        family, verified = _source_family(final_url, company_host)
        families.add(family)
        documents += 1
        rows.extend(_extract_forward_rows(ticker, text, final_url, family, verified, fund))
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame['automatic_discovery'] = True
        frame['source_quorum_count'] = len(families)
        frame['source_quorum_verified'] = len({value for value in families if value in {'IDX_OFFICIAL', 'OJK_OFFICIAL', 'COMPANY_IR'}}) >= 2
        _write_forward_cache(ticker, frame)
        state = 'AUTO_VERIFIED' if bool(frame['source_quorum_verified'].any()) else 'AUTO_SINGLE_SOURCE'
    else:
        state = 'NO_DOCUMENT_EVIDENCE'
    return frame, {
        'ticker': ticker, 'state': state, 'documents': documents, 'rows': len(frame),
        'source_families': ' • '.join(sorted(families)), 'errors': ' • '.join(dict.fromkeys(errors[:5])),
    }


def collect_automatic_forward_quality(
    fundamentals: pd.DataFrame | None,
    tickers: Iterable[str] | None = None,
    config: ScanConfig | None = None,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Automatically collect project and management evidence for top candidates.

    The collector prioritizes official IDX/OJK documents and the issuer's own
    investor-relations domain. It deliberately limits live requests, caches the
    evidence, and labels single-source or modelled conclusions explicitly.
    """
    cfg = config or ScanConfig()
    if not truthy(getattr(cfg, 'automatic_forward_quality_enabled', True)) or fundamentals is None or fundamentals.empty:
        return pd.DataFrame(), pd.DataFrame()
    frame = fundamentals.copy()
    if 'ticker' not in frame:
        return pd.DataFrame(), pd.DataFrame()
    requested = {normalize_idx_ticker(value) for value in (tickers or frame['ticker'].tolist())}
    frame['ticker'] = frame['ticker'].map(normalize_idx_ticker)
    frame = frame[frame['ticker'].isin(requested)].copy()
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    def _numeric_series(*names: str, default: float = 0.0) -> pd.Series:
        """Return an index-aligned numeric Series even when every source column is absent.

        ``DataFrame.get(name, 0)`` returns a scalar when ``name`` is missing.
        Calling ``fillna`` on that scalar caused the production crash seen in
        v6.7.0 after the focus cleanup removed some optional snapshot fields.
        """
        for name in names:
            if name in frame.columns:
                return pd.to_numeric(frame[name], errors='coerce').reindex(frame.index).fillna(default)
        return pd.Series(float(default), index=frame.index, dtype='float64')

    score = _numeric_series('fundamental_score_10', 'fundamental_score', default=0.0)
    if not score.empty and score.max() > 10.0:
        score = score / 10.0
    coverage = _numeric_series('fundamental_coverage', default=0.0)
    growth = _numeric_series('revenue_growth', default=0.0).clip(-0.5, 1.0)
    frame['_forward_priority'] = 10.0 * score + 0.12 * coverage + 10.0 * growth
    top_n = max(1, int(getattr(cfg, 'automatic_forward_quality_top_n', 12)))
    selected = frame.sort_values('_forward_priority', ascending=False).drop_duplicates('ticker').head(top_n)
    workers = max(1, min(int(getattr(cfg, 'automatic_forward_quality_workers', 4)), len(selected)))
    evidence: list[pd.DataFrame] = []
    reports: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_automatic_forward_one, row['ticker'], row.to_dict(), cfg, force_refresh): row['ticker']
            for _, row in selected.iterrows()
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                result, report = future.result()
                if not result.empty:
                    evidence.append(result)
                reports.append(report)
            except Exception as exc:
                reports.append({'ticker': ticker, 'state': 'PROGRAMMING_ERROR', 'documents': 0, 'rows': 0, 'errors': f'{type(exc).__name__}: {str(exc)[:160]}'})
    evidence_records = [
        record
        for frame in evidence
        if frame is not None and not frame.empty
        for record in frame.dropna(axis=1, how="all").to_dict(orient="records")
    ]
    return (
        pd.DataFrame.from_records(evidence_records)
        if evidence_records else pd.DataFrame(),
        pd.DataFrame(reports),
    )


def merge_project_management_reviews(automatic: pd.DataFrame | None, manual: pd.DataFrame | None) -> pd.DataFrame:
    frames = []
    for frame, source in ((automatic, 'AUTOMATIC'), (manual, 'MANUAL_OVERRIDE')):
        if frame is None or frame.empty:
            continue
        local = frame.copy()
        local['review_origin'] = source
        frames.append(local)
    if not frames:
        return pd.DataFrame()
    records = [
        record
        for frame in frames
        for record in frame.dropna(axis=1, how="all").to_dict(orient="records")
    ]
    out = pd.DataFrame.from_records(records)
    if 'ticker' in out:
        out['ticker'] = out['ticker'].map(normalize_idx_ticker)
    return out



PROJECT_STAGE_PRIORS: dict[str, float] = {
    'ANNOUNCEMENT': 0.05,
    'PLANNING': 0.05,
    'FEASIBILITY': 0.10,
    'PERMITTING': 0.15,
    'FUNDING_SECURED': 0.25,
    'FINANCING': 0.25,
    'CONSTRUCTION_LT50': 0.30,
    'CONSTRUCTION': 0.45,
    'CONSTRUCTION_GT50': 0.45,
    'MECHANICAL_COMPLETION': 0.60,
    'COMMISSIONING': 0.70,
    'FIRST_SALE': 0.80,
    'OPERATING': 0.95,
    'STABLE_UTILISATION': 0.95,
    'EXPANSION': 0.75,
    'UNKNOWN': 0.20,
}


def _normalised_project_stage(pm: Mapping[str, Any]) -> tuple[str, float, str]:
    """Return conservative stage prior and provenance.

    Explicit stage text wins. Otherwise the stage is inferred from completion,
    funding, first-sale/COD and utilisation fields. These are research priors,
    not calibrated success probabilities; the production impact model uses
    them as a ceiling to avoid giving announced projects full credit.
    """
    raw = ' '.join(
        safe_text(pm.get(key)).upper()
        for key in ('project_stage', 'project_status', 'execution_stage', 'project_phase')
        if safe_text(pm.get(key))
    )
    aliases = (
        (('STABLE UTIL', 'FULL CAPACITY', 'STEADY STATE'), 'STABLE_UTILISATION'),
        (('FIRST SALE', 'FIRST SHIPMENT', 'COMMERCIAL SALE'), 'FIRST_SALE'),
        (('OPERATING', 'COMMERCIAL OPERATION', 'PRODUCTION'), 'OPERATING'),
        (('COMMISSION', 'TRIAL RUN', 'TESTING'), 'COMMISSIONING'),
        (('MECHANICAL COMPLETION', 'MECHANICALLY COMPLETE'), 'MECHANICAL_COMPLETION'),
        (('CONSTRUCTION', 'EPC'), 'CONSTRUCTION'),
        (('FUNDING SECURED', 'FINANCIAL CLOSE'), 'FUNDING_SECURED'),
        (('FINANCING',), 'FINANCING'),
        (('PERMIT',), 'PERMITTING'),
        (('FEASIBILITY', 'FS '), 'FEASIBILITY'),
        (('ANNOUNC', 'PLANNING', 'PLAN'), 'ANNOUNCEMENT'),
        (('EXPANSION',), 'EXPANSION'),
    )
    for tokens, stage in aliases:
        if any(token in raw for token in tokens):
            return stage, PROJECT_STAGE_PRIORS[stage], 'EXPLICIT_STAGE'
    completion = _pct_fraction(pm.get('project_completion_pct'))
    funding = _pct_fraction(pm.get('project_funding_secured_pct'))
    utilisation = _pct_fraction(pm.get('project_utilisation_pct'))
    first_sale = safe_text(pm.get('project_first_sale_date')) or safe_text(pm.get('actual_cod'))
    if np.isfinite(utilisation) and utilisation >= 0.70:
        return 'STABLE_UTILISATION', PROJECT_STAGE_PRIORS['STABLE_UTILISATION'], 'INFERRED_UTILISATION'
    if first_sale:
        return 'FIRST_SALE', PROJECT_STAGE_PRIORS['FIRST_SALE'], 'INFERRED_FIRST_SALE'
    if np.isfinite(completion) and completion >= 0.95:
        return 'MECHANICAL_COMPLETION', PROJECT_STAGE_PRIORS['MECHANICAL_COMPLETION'], 'INFERRED_COMPLETION'
    if np.isfinite(completion) and completion >= 0.50:
        return 'CONSTRUCTION_GT50', PROJECT_STAGE_PRIORS['CONSTRUCTION_GT50'], 'INFERRED_COMPLETION'
    if np.isfinite(completion) and completion > 0:
        return 'CONSTRUCTION_LT50', PROJECT_STAGE_PRIORS['CONSTRUCTION_LT50'], 'INFERRED_COMPLETION'
    if np.isfinite(funding) and funding >= 0.90:
        return 'FUNDING_SECURED', PROJECT_STAGE_PRIORS['FUNDING_SECURED'], 'INFERRED_FUNDING'
    if safe_number(pm.get('project_capex_idr'), 0.0) > 0 or safe_number(pm.get('project_count'), 0.0) > 0:
        return 'ANNOUNCEMENT', PROJECT_STAGE_PRIORS['ANNOUNCEMENT'], 'INFERRED_PROJECT_EXISTS'
    return 'UNKNOWN', PROJECT_STAGE_PRIORS['UNKNOWN'], 'MISSING_STAGE'


def _economic_earnings_profile(fund: Mapping[str, Any]) -> dict[str, Any]:
    """Research-only economic earnings diagnostics.

    The score is deliberately not included in the production Multibagger
    quality weight until a persistent database supplies consistent NPATMI,
    consolidated profit, inventory and receivable histories.
    """
    ebitda = safe_number(fund.get('history_ebitda_ttm'), safe_number(fund.get('ebitda'), np.nan))
    ocf = safe_number(fund.get('history_operating_cash_flow_ttm'), safe_number(fund.get('operating_cash_flow'), np.nan))
    fcf = safe_number(fund.get('history_fcf_ttm'), safe_number(fund.get('free_cash_flow'), np.nan))
    npatmi = safe_number(
        fund.get('history_npatmi_ttm'),
        safe_number(fund.get('net_income_to_common'), safe_number(fund.get('net_income'), np.nan)),
    )
    consolidated = safe_number(
        fund.get('history_consolidated_profit_ttm'),
        safe_number(fund.get('consolidated_net_profit'), safe_number(fund.get('net_income_including_noncontrolling_interests'), np.nan)),
    )
    inventory_growth = safe_number(fund.get('history_inventory_growth_yoy'), np.nan)
    receivable_growth = safe_number(fund.get('history_receivables_growth_yoy'), np.nan)
    revenue_growth = safe_number(fund.get('revenue_growth'), np.nan)
    ocf_ebitda = ocf / ebitda if np.isfinite(ocf) and np.isfinite(ebitda) and ebitda > 0 else np.nan
    minority_leakage = max(0.0, min(1.0, 1.0 - npatmi / consolidated)) if np.isfinite(npatmi) and np.isfinite(consolidated) and consolidated > 0 and npatmi >= 0 else np.nan
    components: list[tuple[float, float]] = []
    if np.isfinite(ocf_ebitda):
        components.append((100.0 if ocf_ebitda >= 0.80 else 75.0 if ocf_ebitda >= 0.55 else 45.0 if ocf_ebitda >= 0.30 else 15.0, 0.35))
    if np.isfinite(fcf):
        components.append((90.0 if fcf > 0 else 25.0, 0.20))
    if np.isfinite(minority_leakage):
        components.append((100.0 if minority_leakage <= 0.10 else 75.0 if minority_leakage <= 0.25 else 45.0 if minority_leakage <= 0.45 else 15.0, 0.20))
    if np.isfinite(inventory_growth):
        inventory_gap = inventory_growth - (revenue_growth if np.isfinite(revenue_growth) else 0.0)
        components.append((90.0 if inventory_gap <= 0.05 else 65.0 if inventory_gap <= 0.20 else 30.0, 0.125))
    if np.isfinite(receivable_growth):
        receivable_gap = receivable_growth - (revenue_growth if np.isfinite(revenue_growth) else 0.0)
        components.append((90.0 if receivable_gap <= 0.05 else 65.0 if receivable_gap <= 0.20 else 30.0, 0.125))
    weight = sum(item_weight for _, item_weight in components)
    score = sum(item_score * item_weight for item_score, item_weight in components) / weight if weight > 0 else np.nan
    confidence = min(100.0, 100.0 * weight)
    if confidence >= 70 and np.isfinite(score) and score >= 72:
        state = 'CASH_BACKED_EARNINGS'
    elif confidence >= 45 and np.isfinite(score) and score >= 55:
        state = 'PARTIALLY_CONFIRMED'
    elif confidence >= 45 and np.isfinite(score):
        state = 'EARNINGS_QUALITY_RISK'
    else:
        state = 'INSUFFICIENT_ECONOMIC_DATA'
    return {
        'economic_earnings_score': round(score, 1) if np.isfinite(score) else np.nan,
        'economic_earnings_confidence': round(confidence, 1),
        'economic_earnings_state': state,
        'economic_earnings_production_weight_pct': 0.0,
        'ocf_ebitda_conversion': round(ocf_ebitda, 3) if np.isfinite(ocf_ebitda) else np.nan,
        'minority_leakage_pct': round(100.0 * minority_leakage, 1) if np.isfinite(minority_leakage) else np.nan,
        'inventory_growth_yoy': round(inventory_growth, 4) if np.isfinite(inventory_growth) else np.nan,
        'receivables_growth_yoy': round(receivable_growth, 4) if np.isfinite(receivable_growth) else np.nan,
    }


def _future_fundamental_impact(pm: Mapping[str, Any], fund: Mapping[str, Any]) -> dict[str, Any]:
    current_revenue = safe_number(fund.get('history_revenue_ttm'), safe_number(fund.get('total_revenue'), np.nan))
    current_ebitda = safe_number(fund.get('history_ebitda_ttm'), safe_number(fund.get('ebitda'), np.nan))
    current_net_margin = safe_number(fund.get('net_margin'), np.nan)
    current_fcf = safe_number(fund.get('history_fcf_ttm'), safe_number(fund.get('free_cash_flow'), np.nan))
    current_debt = safe_number(fund.get('total_debt'), np.nan)
    capex = max(0.0, safe_number(pm.get('project_capex_idr'), 0.0))
    expected_revenue = max(0.0, safe_number(pm.get('project_expected_revenue_idr'), 0.0))
    expected_ebitda = max(0.0, safe_number(pm.get('project_expected_ebitda_idr'), 0.0))
    project_score = max(0.0, min(100.0, safe_number(pm.get('project_pipeline_score_observed'), 0.0)))
    coverage = max(0.0, min(100.0, safe_number(pm.get('project_data_coverage'), 0.0)))
    completion = max(0.0, min(1.0, safe_number(pm.get('project_completion_pct'), 0.0)))
    funding = max(0.0, min(1.0, safe_number(pm.get('project_funding_secured_pct'), 0.0)))
    ownership_raw = safe_number(pm.get('project_ownership_pct'), np.nan)
    ownership_known = bool(np.isfinite(ownership_raw))
    ownership = max(0.0, min(1.0, ownership_raw)) if ownership_known else 0.0
    project_stage, stage_prior, stage_source = _normalised_project_stage(pm)
    evidence_probability = max(0.05, min(0.95, 0.35 + 0.004 * project_score + 0.15 * completion + 0.10 * funding))
    # Research-maintenance policy v6.9.0: stage is a conservative ceiling.
    # An announced project cannot receive commissioning-level success credit.
    probability = min(evidence_probability, stage_prior)
    if not ownership_known:
        probability = min(probability, 0.20)
    model = 'NO_NUMERIC_PROJECT_IMPACT'
    gross_revenue = expected_revenue * ownership
    gross_ebitda = expected_ebitda * ownership
    if gross_revenue > 0:
        model = 'DISCLOSED_PROJECT_REVENUE'
    elif capex > 0 and np.isfinite(current_revenue) and current_revenue > 0:
        historical_capex = abs(safe_number(fund.get('history_capex_ttm'), np.nan))
        if np.isfinite(historical_capex) and historical_capex > 0:
            productivity = max(0.25, min(3.0, current_revenue / historical_capex))
        else:
            productivity = 0.75
        gross_revenue = capex * productivity * ownership
        model = 'MODELLED_COMPANY_CAPEX_PRODUCTIVITY'
    margin = current_ebitda / current_revenue if np.isfinite(current_ebitda) and np.isfinite(current_revenue) and current_revenue > 0 else max(0.03, min(0.35, safe_number(fund.get('operating_margin'), 0.12)))
    if gross_ebitda <= 0 and gross_revenue > 0:
        gross_ebitda = gross_revenue * margin
    base_revenue = gross_revenue * probability
    bear_revenue = gross_revenue * max(0.10, probability - 0.25)
    bull_revenue = gross_revenue * min(1.0, probability + 0.20)
    base_ebitda = gross_ebitda * probability
    tax_profit_factor = max(0.35, min(0.80, safe_number(fund.get('net_margin'), 0.08) / max(margin, 1e-6))) if np.isfinite(current_net_margin) else 0.55
    base_profit = base_ebitda * tax_profit_factor
    remaining_capex = capex * max(0.0, 1.0 - completion) * ownership
    debt_need = remaining_capex * max(0.0, 1.0 - funding)
    revenue_uplift = 100.0 * base_revenue / current_revenue if np.isfinite(current_revenue) and current_revenue > 0 else np.nan
    ebitda_uplift = 100.0 * base_ebitda / current_ebitda if np.isfinite(current_ebitda) and current_ebitda > 0 else np.nan
    profit_base = current_revenue * current_net_margin if np.isfinite(current_revenue) and np.isfinite(current_net_margin) else np.nan
    profit_uplift = 100.0 * base_profit / profit_base if np.isfinite(profit_base) and profit_base > 0 else np.nan
    fcf_pressure = max(0.0, remaining_capex - max(0.0, current_fcf if np.isfinite(current_fcf) else 0.0))
    debt_change_pct = 100.0 * debt_need / current_debt if np.isfinite(current_debt) and current_debt > 0 else np.nan
    upside_score = 50.0
    if np.isfinite(revenue_uplift):
        upside_score += max(-15.0, min(25.0, 0.8 * revenue_uplift))
    if np.isfinite(ebitda_uplift):
        upside_score += max(-10.0, min(20.0, 0.5 * ebitda_uplift))
    if fcf_pressure > 0 and np.isfinite(current_revenue) and current_revenue > 0:
        upside_score -= min(20.0, 50.0 * fcf_pressure / current_revenue)
    if np.isfinite(debt_change_pct):
        upside_score -= min(20.0, 0.15 * max(0.0, debt_change_pct - 20.0))
    quorum = truthy(pm.get('project_source_quorum_verified'))
    mature_stage = project_stage in {'COMMISSIONING', 'FIRST_SALE', 'OPERATING', 'STABLE_UTILISATION'}
    confidence = 'LOW' if not ownership_known else 'HIGH' if quorum and coverage >= 75 and expected_revenue > 0 and expected_ebitda > 0 and mature_stage else 'MEDIUM' if coverage >= 50 and (expected_revenue > 0 or capex > 0) else 'LOW'
    if not ownership_known:
        model = 'NO_VERIFIED_ECONOMIC_OWNERSHIP'
    return {
        'future_fundamental_impact_score': round(max(0.0, min(100.0, upside_score)), 1),
        'future_impact_confidence': confidence,
        'future_impact_model': model,
        'future_impact_horizon': '12–36 months; project-stage dependent',
        'future_revenue_uplift_bear_pct': round(100.0 * bear_revenue / current_revenue, 1) if np.isfinite(current_revenue) and current_revenue > 0 else np.nan,
        'future_revenue_uplift_base_pct': round(revenue_uplift, 1) if np.isfinite(revenue_uplift) else np.nan,
        'future_revenue_uplift_bull_pct': round(100.0 * bull_revenue / current_revenue, 1) if np.isfinite(current_revenue) and current_revenue > 0 else np.nan,
        'future_ebitda_uplift_base_pct': round(ebitda_uplift, 1) if np.isfinite(ebitda_uplift) else np.nan,
        'future_net_profit_uplift_base_pct': round(profit_uplift, 1) if np.isfinite(profit_uplift) else np.nan,
        'future_fcf_pressure_idr': round(fcf_pressure, 0),
        'future_net_debt_change_idr': round(debt_need, 0),
        'future_net_debt_change_pct': round(debt_change_pct, 1) if np.isfinite(debt_change_pct) else np.nan,
        'project_success_probability_pct': round(100.0 * probability, 1),
        'project_stage': project_stage,
        'project_stage_probability_pct': round(100.0 * stage_prior, 1),
        'project_stage_probability_source': stage_source,
        'project_evidence_probability_pct': round(100.0 * evidence_probability, 1),
    }





def _fundamental_records(fundamentals: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if fundamentals is None or fundamentals.empty or 'ticker' not in fundamentals:
        return {}
    return {str(row['ticker']): row.to_dict() for _, row in fundamentals.drop_duplicates('ticker', keep='last').iterrows()}















def _bounded_score(value: Any, maximum: float) -> float:
    """Normalize a non-negative component to a transparent 0-100 scale."""
    numeric = max(0.0, safe_number(value, 0.0))
    return float(max(0.0, min(100.0, 100.0 * numeric / max(1e-9, float(maximum)))))


def _range_quality_score(value: Any, weak: float, strong: float) -> float:
    """Map a fundamental ratio to 0-100 without rewarding missing evidence."""
    numeric = safe_number(value, np.nan)
    if not np.isfinite(numeric):
        return np.nan
    if abs(strong - weak) <= 1e-12:
        return 50.0
    return float(np.clip(100.0 * (numeric - weak) / (strong - weak), 0.0, 100.0))



def _pct_fraction(value: Any) -> float:
    numeric = safe_number(value, np.nan)
    if not np.isfinite(numeric):
        return np.nan
    return numeric / 100.0 if abs(numeric) > 1.5 else numeric


def parse_project_management_csv(source: bytes | BinaryIO | pd.DataFrame) -> pd.DataFrame:
    """Parse optional forward-project and management due-diligence rows.

    The scanner never infers a director's track record from a name alone. Scores
    require structured evidence such as project stage/funding, historical ROIC
    improvement, capital-allocation discipline and governance flags. Multiple
    project rows per ticker are aggregated later by the Multibagger engine.
    """
    frame = read_csv_input(source)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if 'ticker' not in frame:
        raise ValueError('Project/management CSV wajib memiliki kolom ticker')
    out = pd.DataFrame(index=frame.index)
    out['ticker'] = frame['ticker'].map(normalize_idx_ticker)
    aliases = {
        'as_of': ('as_of', 'date', 'review_date'),
        'source_url': ('source_url', 'project_source_url'),
        'source_verified': ('source_verified', 'verified'),
        'project_name': ('project_name', 'project'),
        'project_stage': ('project_stage', 'stage'),
        'project_completion_pct': ('project_completion_pct', 'completion_pct'),
        'project_capex_idr': ('project_capex_idr', 'capex_idr'),
        'funding_secured_pct': ('funding_secured_pct', 'funding_pct'),
        'offtake_secured_pct': ('offtake_secured_pct', 'offtake_pct', 'contracted_revenue_pct'),
        'expected_revenue_idr': ('expected_revenue_idr', 'incremental_revenue_idr'),
        'expected_ebitda_idr': ('expected_ebitda_idr', 'incremental_ebitda_idr'),
        'expected_cod': ('expected_cod', 'commercial_operation_date', 'target_completion_date'),
        'ownership_pct': ('ownership_pct', 'economic_interest_pct'),
        'project_delay_months': ('project_delay_months', 'delay_months'),
        'cost_overrun_pct': ('cost_overrun_pct', 'overrun_pct'),
        'strategic_project': ('strategic_project', 'national_strategic_project'),
        'project_risk': ('project_risk', 'execution_risk'),
        'ceo_name': ('ceo_name', 'president_director', 'direktur_utama'),
        'management_team': ('management_team', 'board_summary'),
        'ceo_tenure_years': ('ceo_tenure_years', 'president_director_tenure_years'),
        'board_avg_tenure_years': ('board_avg_tenure_years', 'board_tenure_years'),
        'management_revenue_cagr': ('management_revenue_cagr', 'revenue_cagr_under_management'),
        'management_roic_change_pct': ('management_roic_change_pct', 'roic_change_under_management'),
        'capital_allocation_score': ('capital_allocation_score',),
        'governance_score': ('governance_score',),
        'board_turnover_3y': ('board_turnover_3y', 'director_turnover_3y'),
        'insider_ownership_pct': ('insider_ownership_pct',),
        'audit_clean': ('audit_clean', 'clean_audit_opinion'),
        'related_party_risk': ('related_party_risk',),
        'legal_governance_flags': ('legal_governance_flags', 'governance_flags'),
        'management_source_url': ('management_source_url', 'board_source_url'),
        'management_verified': ('management_verified', 'board_verified'),
    }
    for canonical, candidates in aliases.items():
        source_column = next((name for name in candidates if name in frame.columns), None)
        out[canonical] = frame[source_column] if source_column else np.nan
    for column in (
        'project_completion_pct', 'funding_secured_pct', 'offtake_secured_pct',
        'ownership_pct', 'cost_overrun_pct', 'management_revenue_cagr',
        'management_roic_change_pct', 'insider_ownership_pct',
    ):
        out[column] = pd.to_numeric(out[column], errors='coerce').map(_pct_fraction)
    for column in (
        'project_capex_idr', 'expected_revenue_idr', 'expected_ebitda_idr',
        'project_delay_months', 'ceo_tenure_years', 'board_avg_tenure_years',
        'capital_allocation_score', 'governance_score', 'board_turnover_3y',
    ):
        out[column] = pd.to_numeric(out[column], errors='coerce')
    for column in ('source_verified', 'strategic_project', 'audit_clean', 'management_verified'):
        out[column] = out[column].map(truthy)
    out['as_of'] = pd.to_datetime(out['as_of'], errors='coerce')
    out['expected_cod'] = pd.to_datetime(out['expected_cod'], errors='coerce')
    for column in ('project_stage', 'project_risk', 'related_party_risk'):
        out[column] = out[column].fillna('').astype(str).str.upper().str.strip()
    for column in ('project_name', 'ceo_name', 'management_team', 'legal_governance_flags', 'source_url', 'management_source_url'):
        out[column] = out[column].fillna('').astype(str).str.strip()
    out = out[out['ticker'].astype(str).str.len().gt(0)].reset_index(drop=True)
    return out


def _project_management_records(frame: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if frame is None or frame.empty or 'ticker' not in frame:
        return {}
    stage_scores = {
        'PLANNING': 18.0, 'FEASIBILITY': 24.0, 'PERMITTING': 34.0,
        'FINANCING': 45.0, 'CONSTRUCTION': 66.0, 'COMMISSIONING': 86.0,
        'OPERATING': 100.0, 'EXPANSION': 78.0,
    }
    records: dict[str, dict[str, Any]] = {}
    for ticker, group in frame.groupby('ticker', sort=False):
        project_rows = group[(group['project_name'].str.len().gt(0)) | (group['project_stage'].str.len().gt(0))].copy()
        project_scores: list[tuple[float, float]] = []
        expected_revenue = 0.0
        expected_ebitda = 0.0
        capex_total = 0.0
        weighted_completion = 0.0
        weighted_funding = 0.0
        weighted_offtake = 0.0
        weighted_ownership = 0.0
        metric_weight_total = 0.0
        source_families: set[str] = set()
        project_source_urls: list[str] = []
        project_names: list[str] = []
        project_flags: list[str] = []
        project_stages: list[tuple[str, float]] = []
        verified_projects = 0
        for _, item in project_rows.iterrows():
            stage = safe_text(item.get('project_stage')).upper()
            stage_score = stage_scores.get(stage, 20.0 if stage else 0.0)
            completion = 100.0 * max(0.0, min(1.0, safe_number(item.get('project_completion_pct'), 0.0)))
            funding = 100.0 * max(0.0, min(1.0, safe_number(item.get('funding_secured_pct'), 0.0)))
            offtake = 100.0 * max(0.0, min(1.0, safe_number(item.get('offtake_secured_pct'), 0.0)))
            name = safe_text(item.get('project_name'))
            ownership_raw = safe_number(item.get('ownership_pct'), np.nan)
            ownership_known = bool(np.isfinite(ownership_raw))
            ownership = 100.0 * max(0.0, min(1.0, ownership_raw)) if ownership_known else 0.0
            strategic = 100.0 if truthy(item.get('strategic_project')) else 45.0
            delay = max(0.0, safe_number(item.get('project_delay_months'), 0.0))
            overrun = 100.0 * max(0.0, safe_number(item.get('cost_overrun_pct'), 0.0))
            risk_text = safe_text(item.get('project_risk')).upper()
            risk_penalty = {'LOW': 0.0, 'MEDIUM': 8.0, 'HIGH': 22.0, 'CRITICAL': 40.0}.get(risk_text, 5.0 if risk_text else 0.0)
            score = (
                0.24 * stage_score + 0.20 * completion + 0.20 * funding
                + 0.14 * offtake + 0.10 * ownership + 0.12 * strategic
                - min(24.0, 1.5 * delay) - min(25.0, 0.55 * overrun) - risk_penalty
            )
            score = max(0.0, min(100.0, score))
            capex = max(0.0, safe_number(item.get('project_capex_idr'), 0.0))
            weight = capex if capex > 0 else 1.0
            project_scores.append((score, weight))
            if stage:
                project_stages.append((stage, weight))
            capex_total += capex
            expected_revenue += max(0.0, safe_number(item.get('expected_revenue_idr'), 0.0))
            expected_ebitda += max(0.0, safe_number(item.get('expected_ebitda_idr'), 0.0))
            metric_weight = capex if capex > 0 else 1.0
            weighted_completion += max(0.0, min(1.0, safe_number(item.get('project_completion_pct'), 0.0))) * metric_weight
            weighted_funding += max(0.0, min(1.0, safe_number(item.get('funding_secured_pct'), 0.0))) * metric_weight
            weighted_offtake += max(0.0, min(1.0, safe_number(item.get('offtake_secured_pct'), 0.0))) * metric_weight
            if ownership_known:
                weighted_ownership += max(0.0, min(1.0, ownership_raw)) * metric_weight
            else:
                project_flags.append(f'{name or "Project"}: ownership belum terverifikasi')
            metric_weight_total += metric_weight
            source_url = safe_text(item.get('source_url'))
            family = safe_text(item.get('source_family'))
            if not family and source_url:
                family, _ = _source_family(source_url)
            if family:
                source_families.add(family)
            if source_url:
                project_source_urls.append(source_url)
            if name:
                project_names.append(name)
            if truthy(item.get('source_verified')):
                verified_projects += 1
            if delay >= 6:
                project_flags.append(f'{name or "Project"}: delay {delay:.0f} bulan')
            if overrun >= 20:
                project_flags.append(f'{name or "Project"}: cost overrun {overrun:.0f}%')
        if project_scores:
            total_weight = sum(weight for _, weight in project_scores)
            project_score = sum(score * weight for score, weight in project_scores) / max(total_weight, 1e-9)
            project_coverage = min(100.0, 35.0 + 20.0 * len(project_scores) + 15.0 * verified_projects)
            project_source = 'VERIFIED_PROJECT_PIPELINE' if verified_projects else 'USER_PROJECT_REVIEW'
        else:
            project_score = np.nan
            project_coverage = 0.0
            project_source = 'MISSING'

        latest = group.sort_values('as_of').iloc[-1]
        ceo_name = safe_text(latest.get('ceo_name'))
        capital_allocation = safe_number(latest.get('capital_allocation_score'), np.nan)
        governance = safe_number(latest.get('governance_score'), np.nan)
        revenue_cagr = safe_number(latest.get('management_revenue_cagr'), np.nan)
        roic_change = safe_number(latest.get('management_roic_change_pct'), np.nan)
        ceo_tenure = safe_number(latest.get('ceo_tenure_years'), np.nan)
        board_tenure = safe_number(latest.get('board_avg_tenure_years'), np.nan)
        board_turnover = safe_number(latest.get('board_turnover_3y'), np.nan)
        insider = safe_number(latest.get('insider_ownership_pct'), np.nan)
        audit_clean = truthy(latest.get('audit_clean'))
        related_risk = safe_text(latest.get('related_party_risk')).upper()
        governance_flags = safe_text(latest.get('legal_governance_flags'))
        management_components: list[tuple[float, float]] = []
        if np.isfinite(capital_allocation):
            management_components.append((max(0.0, min(100.0, capital_allocation)), 0.25))
        if np.isfinite(governance):
            management_components.append((max(0.0, min(100.0, governance)), 0.22))
        if np.isfinite(revenue_cagr):
            management_components.append((max(0.0, min(100.0, 50.0 + 180.0 * revenue_cagr)), 0.14))
        if np.isfinite(roic_change):
            management_components.append((max(0.0, min(100.0, 50.0 + 250.0 * roic_change)), 0.16))
        if np.isfinite(ceo_tenure):
            tenure_score = 90.0 if 3 <= ceo_tenure <= 10 else 70.0 if 1 <= ceo_tenure < 3 or 10 < ceo_tenure <= 15 else 45.0
            management_components.append((tenure_score, 0.08))
        if np.isfinite(board_tenure):
            management_components.append((85.0 if 2 <= board_tenure <= 10 else 55.0, 0.05))
        if np.isfinite(board_turnover):
            management_components.append((100.0 if board_turnover <= 1 else 65.0 if board_turnover <= 3 else 25.0, 0.05))
        if np.isfinite(insider):
            management_components.append((90.0 if 0.05 <= insider <= 0.60 else 65.0 if insider > 0 else 40.0, 0.03))
        if audit_clean:
            management_components.append((100.0, 0.02))
        if management_components:
            weights = sum(weight for _, weight in management_components)
            management_score = sum(score * weight for score, weight in management_components) / max(weights, 1e-9)
            management_coverage = min(100.0, 100.0 * weights / 1.0 + (10.0 if truthy(latest.get('management_verified')) else 0.0))
            management_source = 'VERIFIED_MANAGEMENT_REVIEW' if truthy(latest.get('management_verified')) else 'USER_MANAGEMENT_REVIEW'
        else:
            management_score = np.nan
            management_coverage = 0.0
            management_source = 'MISSING'
        management_penalty = 0.0
        if related_risk in {'HIGH', 'CRITICAL'}:
            management_penalty += 20.0 if related_risk == 'HIGH' else 35.0
        if governance_flags:
            management_penalty += 25.0
        if np.isfinite(management_score):
            management_score = max(0.0, management_score - management_penalty)
        records[str(ticker)] = {
            'project_pipeline_score_observed': round(project_score, 1) if np.isfinite(project_score) else np.nan,
            'project_data_coverage': round(project_coverage, 1),
            'project_data_source': project_source,
            'project_count': int(len(project_scores)),
            'project_names': ' • '.join(dict.fromkeys(project_names[:5])),
            'project_stage': max(project_stages, key=lambda item: item[1])[0] if project_stages else '',
            'project_capex_idr': capex_total,
            'project_expected_revenue_idr': expected_revenue,
            'project_expected_ebitda_idr': expected_ebitda,
            'project_completion_pct': weighted_completion / metric_weight_total if metric_weight_total > 0 else np.nan,
            'project_funding_secured_pct': weighted_funding / metric_weight_total if metric_weight_total > 0 else np.nan,
            'project_offtake_secured_pct': weighted_offtake / metric_weight_total if metric_weight_total > 0 else np.nan,
            'project_ownership_pct': weighted_ownership / metric_weight_total if metric_weight_total > 0 and weighted_ownership > 0 else np.nan,
            'project_source_family_count': len(source_families),
            'project_source_families': ' • '.join(sorted(source_families)),
            'project_source_urls': ' • '.join(dict.fromkeys(project_source_urls[:8])),
            'project_source_quorum_verified': len({value for value in source_families if value in {'IDX_OFFICIAL', 'OJK_OFFICIAL', 'COMPANY_IR'}}) >= 2,
            'project_execution_flags': ' • '.join(dict.fromkeys(project_flags)),
            'ceo_name_reviewed': ceo_name,
            'management_quality_score_observed': round(management_score, 1) if np.isfinite(management_score) else np.nan,
            'management_data_coverage': round(management_coverage, 1),
            'management_data_source': management_source,
            'management_governance_flags': governance_flags,
            'management_related_party_risk': related_risk,
            'management_verified': truthy(latest.get('management_verified')),
            'management_source_urls': ' • '.join(dict.fromkeys([value for value in (safe_text(latest.get('management_source_url')), safe_text(latest.get('source_url'))) if value])),
        }
    return records


def _automatic_forward_quality_proxy(fund: Mapping[str, Any]) -> dict[str, Any]:
    """Low-confidence proxy used only when no structured project review exists."""
    revenue = safe_number(fund.get('history_revenue_ttm'), safe_number(fund.get('total_revenue'), np.nan))
    capex = abs(safe_number(fund.get('history_capex_ttm'), np.nan))
    ocf = safe_number(fund.get('history_ocf_ttm'), safe_number(fund.get('operating_cash_flow'), np.nan))
    fcf = safe_number(fund.get('history_fcf_ttm'), safe_number(fund.get('free_cash_flow'), np.nan))
    growth = safe_number(fund.get('history_revenue_cagr_3y'), safe_number(fund.get('revenue_growth'), np.nan))
    capex_intensity = capex / revenue if np.isfinite(capex) and np.isfinite(revenue) and revenue > 0 else np.nan
    project_proxy = 45.0
    if np.isfinite(capex_intensity):
        project_proxy += 18.0 if 0.05 <= capex_intensity <= 0.30 else 8.0 if capex_intensity > 0.30 else 3.0
    if np.isfinite(ocf) and ocf > 0:
        project_proxy += 10.0
    if np.isfinite(fcf) and fcf < 0 and np.isfinite(ocf) and ocf > 0:
        project_proxy -= 6.0  # investment phase; not automatically bad
    if np.isfinite(growth):
        project_proxy += max(-10.0, min(15.0, 60.0 * growth))
    project_proxy = max(20.0, min(68.0, project_proxy))

    roic = safe_number(fund.get('history_roic_proxy'), np.nan)
    cash_conversion = safe_number(fund.get('history_cash_conversion'), np.nan)
    margin_stability = safe_number(fund.get('history_margin_stability'), np.nan)
    dilution = safe_number(fund.get('history_share_dilution_yoy'), np.nan)
    revenue_cagr = safe_number(fund.get('history_revenue_cagr_3y'), np.nan)
    management_proxy = 45.0
    management_proxy += 15.0 if np.isfinite(roic) and roic >= 0.12 else 8.0 if np.isfinite(roic) and roic >= 0.07 else 0.0
    management_proxy += 10.0 if np.isfinite(cash_conversion) and 0.8 <= cash_conversion <= 1.8 else 4.0 if np.isfinite(cash_conversion) and cash_conversion > 0.5 else -8.0 if np.isfinite(cash_conversion) else 0.0
    management_proxy += 10.0 if np.isfinite(margin_stability) and margin_stability >= 0.75 else 4.0 if np.isfinite(margin_stability) and margin_stability >= 0.55 else 0.0
    management_proxy += max(-8.0, min(10.0, 50.0 * revenue_cagr)) if np.isfinite(revenue_cagr) else 0.0
    management_proxy -= 18.0 if np.isfinite(dilution) and dilution > 0.12 else 8.0 if np.isfinite(dilution) and dilution > 0.05 else 0.0
    governance_values = [
        safe_number(fund.get('governance_overall_risk'), np.nan),
        safe_number(fund.get('governance_board_risk'), np.nan),
        safe_number(fund.get('governance_audit_risk'), np.nan),
    ]
    governance_values = [value for value in governance_values if np.isfinite(value)]
    if governance_values:
        average_risk = float(np.mean(governance_values))
        management_proxy += 8.0 if average_risk <= 3.0 else 3.0 if average_risk <= 5.0 else -8.0 if average_risk >= 8.0 else 0.0
    management_proxy = max(20.0, min(68.0, management_proxy))
    management_coverage = 35.0 + (5.0 if safe_text(fund.get('ceo_name')) else 0.0) + (10.0 if governance_values else 0.0)
    return {
        'project_pipeline_score_proxy': round(project_proxy, 1),
        'project_proxy_coverage': 30.0 if np.isfinite(capex_intensity) else 15.0,
        'project_proxy_basis': f'CAPEX intensity {capex_intensity:.1%}' if np.isfinite(capex_intensity) else 'CAPEX data unavailable',
        'management_quality_score_proxy': round(management_proxy, 1),
        'management_proxy_coverage': management_coverage,
        'management_proxy_basis': 'ROIC/cash conversion/margin stability/dilution proxy',
    }

def _multibagger_data_integrity_score(row: Mapping[str, Any]) -> float:
    grade = safe_text(row.get('fundamental_data_grade')).upper() or 'D'
    reliability = safe_text(row.get('fundamental_reliability')).upper() or 'UNKNOWN'
    grade_score = {'A': 100.0, 'B': 82.0, 'C': 60.0, 'D': 25.0}.get(grade, 20.0)
    reliability_score = {'HIGH': 100.0, 'MEDIUM': 72.0, 'LOW': 42.0, 'UNKNOWN': 20.0}.get(reliability, 20.0)
    consensus = max(0.0, min(100.0, safe_number(row.get('fundamental_consensus_score'), 0.0)))
    history = max(0.0, min(100.0, safe_number(row.get('fundamental_history_coverage'), 0.0)))
    official = 100.0 if truthy(row.get('fundamental_official_verified')) else 55.0 if truthy(row.get('fundamental_official_reference')) else 25.0
    source_count = max(0.0, safe_number(row.get('fundamental_source_count'), 0.0))
    source_score = 100.0 if source_count >= 3 else 80.0 if source_count >= 2 else 35.0 if source_count >= 1 else 0.0
    return round(
        0.25 * grade_score
        + 0.20 * reliability_score
        + 0.20 * consensus
        + 0.15 * history
        + 0.10 * official
        + 0.10 * source_score,
        1,
    )


def _multibagger_solvency_strength(row: Mapping[str, Any]) -> float:
    if safe_text(row.get('fundamental_model')).upper() == 'FINANCIAL':
        car = safe_number(row.get('car'), np.nan)
        npl = safe_number(row.get('npl_gross'), np.nan)
        ldr = safe_number(row.get('ldr'), np.nan)
        car_score = 100.0 if np.isfinite(car) and car >= 0.20 else 82.0 if np.isfinite(car) and car >= 0.15 else 65.0 if np.isfinite(car) and car >= 0.12 else 0.0
        npl_score = 100.0 if np.isfinite(npl) and npl <= 0.015 else 78.0 if np.isfinite(npl) and npl <= 0.03 else 40.0 if np.isfinite(npl) and npl <= 0.05 else 0.0
        ldr_score = 100.0 if np.isfinite(ldr) and 0.75 <= ldr <= 0.90 else 75.0 if np.isfinite(ldr) and 0.65 <= ldr <= 1.00 else 25.0 if np.isfinite(ldr) else 0.0
        return round(0.40 * car_score + 0.35 * npl_score + 0.25 * ldr_score, 1)
    balance = _bounded_score(row.get('balance_sheet_score'), 12.0)
    coverage = max(0.0, min(100.0, safe_number(row.get('solvency_coverage'), 0.0)))
    return round(0.78 * balance + 0.22 * coverage, 1)


def _multibagger_technical_timing_score(state: Any) -> float:
    value = safe_text(state).upper()
    if value in {'EXECUTION_READY', 'READY_FOR_STOCKBIT_VERIFY'}:
        return 100.0
    if value in {'ENTRY_PLAN_READY', 'READY_FOR_PRICE_VERIFY'}:
        return 82.0
    if value == 'SIGNAL_READY':
        return 68.0
    if value in {'WATCHLIST_ENTRY', 'PENDING_DATA', 'PENDING_CLOSE'}:
        return 42.0
    return 30.0


def _multibagger_capital_tier(row: Mapping[str, Any], conviction: float) -> tuple[str, float]:
    status = safe_text(row.get('multibagger_status')).upper()
    grade_a = status == 'MULTIBAGGER_A_CANDIDATE'
    grade_b = status == 'MULTIBAGGER_B_CANDIDATE'
    if grade_a and conviction >= 88.0:
        return ('CORE_COMPOUNDING', 1.35)
    if grade_a and conviction >= 80.0:
        return ('HIGH_CONVICTION', 1.10)
    if grade_b and conviction >= 78.0:
        return ('SATELLITE_GROWTH', 0.82)
    if grade_b and conviction >= 72.0:
        return ('STARTER_ONLY', 0.58)
    return ('WATCH_ONLY', 0.0)


def _multibagger_tier_cap_pct(tier: str, cfg: ScanConfig) -> float:
    values = {
        'CORE_COMPOUNDING': 100.0 * max(0.0, min(1.0, cfg.multibagger_core_cap_pct)),
        'HIGH_CONVICTION': 100.0 * max(0.0, min(1.0, cfg.multibagger_high_cap_pct)),
        'SATELLITE_GROWTH': 100.0 * max(0.0, min(1.0, cfg.multibagger_satellite_cap_pct)),
        'STARTER_ONLY': 100.0 * max(0.0, min(1.0, cfg.multibagger_starter_cap_pct)),
    }
    return float(values.get(tier, 0.0))


def _capped_weight_distribution(raw_weights: Mapping[object, float], caps: Mapping[object, float]) -> dict[object, float]:
    """Normalize weights with per-name concentration caps; unused weight stays cash."""
    remaining = 100.0
    active = {idx for idx, value in raw_weights.items() if value > 0 and caps.get(idx, 0.0) > 0}
    allocated = {idx: 0.0 for idx in raw_weights}
    while active and remaining > 1e-8:
        raw_total = sum(max(0.0, raw_weights[idx]) for idx in active)
        if raw_total <= 0:
            break
        provisional = {idx: remaining * max(0.0, raw_weights[idx]) / raw_total for idx in active}
        capped_now = [idx for idx in active if provisional[idx] >= caps[idx] - allocated[idx] - 1e-8]
        if not capped_now:
            for idx in active:
                allocated[idx] += provisional[idx]
            remaining = 0.0
            break
        for idx in capped_now:
            room = max(0.0, caps[idx] - allocated[idx])
            allocated[idx] += room
            remaining -= room
            active.remove(idx)
    return {idx: round(max(0.0, value), 2) for idx, value in allocated.items()}


def allocate_multibagger_capital(candidates: pd.DataFrame, config: ScanConfig | None=None) -> pd.DataFrame:
    """Rank where the Multibagger sleeve should place the most capital.

    Strategic target weights are conviction-based and capped per tier. Actual
    deployment is allowed only for ACCUMULATE_NOW/STARTER_NOW; weights assigned
    to candidates waiting for an entry zone remain explicit cash reserve.
    """
    cfg = config or ScanConfig()
    if candidates is None or candidates.empty:
        return candidates.copy() if isinstance(candidates, pd.DataFrame) else pd.DataFrame()
    out = candidates.copy()
    out = enrich_multibagger_with_peer_ai(
        out, enabled=bool(getattr(cfg, 'ai_enabled', True)),
        max_weight=min(0.25, max(0.0, safe_number(getattr(cfg, 'ai_max_weight', 0.35), 0.35) * 0.65)),
    )
    defaults = {
        'capital_conviction_score': 0.0,
        'capital_tier': 'WATCH_ONLY',
        'capital_priority_score': 0.0,
        'capital_priority_rank': np.nan,
        'allocation_eligible': False,
        'allocation_cap_pct': 0.0,
        'strategic_target_weight_pct': 0.0,
        'deploy_now_weight_pct': 0.0,
        'strategic_target_amount_idr': 0.0,
        'recommended_allocation_idr': 0.0,
        'recommended_lots': 0,
        'estimated_order_value_idr': 0.0,
        'allocation_reference_price': np.nan,
        'allocation_action': 'NO_ALLOCATION',
        'allocation_reason': '',
        'multibagger_budget_idr': max(0.0, cfg.multibagger_capital_budget_idr),
        'multibagger_cash_reserve_idr': max(0.0, cfg.multibagger_capital_budget_idr),
    }
    for column, default in defaults.items():
        if column not in out:
            out[column] = default

    pillar_columns = {
        'Growth': ('growth_persistence_pillar', 100.0),
        'Profitability': ('profitability_pillar', 100.0),
        'Cash-flow quality': ('cash_conversion_pillar', 100.0),
        'Balance-sheet safety': ('balance_sheet_safety_pillar', 100.0),
        'Reinvestment runway': ('reinvestment_runway_pillar', 100.0),
        'Valuation': ('valuation_score', 8.0),
        'Momentum': ('momentum_score', 12.0),
        'Smart-money proxy': ('accumulation_score', 10.0),
    }
    raw_weights: dict[object, float] = {}
    caps: dict[object, float] = {}
    eligible_rows: list[tuple[float, object]] = []
    minimum = max(0.0, min(100.0, cfg.multibagger_min_capital_conviction))

    for idx, row in out.iterrows():
        scoring_state = safe_text(row.get('multibagger_scoring_state')).upper()
        score_value = safe_number(row.get('multibagger_score'), np.nan)
        if scoring_state.startswith('DATA_NOT_SCORED') or not np.isfinite(score_value):
            out.at[idx, 'capital_conviction_score'] = np.nan
            out.at[idx, 'capital_tier'] = 'WATCH_ONLY'
            out.at[idx, 'allocation_action'] = 'WAIT_FOR_FUNDAMENTAL_DATA'
            out.at[idx, 'allocation_reason'] = safe_text(row.get('multibagger_score_reason')) or 'Multibagger score unavailable.'
            continue
        normalized = {label: _bounded_score(row.get(column), maximum) for label, (column, maximum) in pillar_columns.items()}
        # Backward-compatible database snapshots may predate the explicit v7.4
        # pillars. Reconstruct them from the frozen legacy components instead
        # of turning missing columns into zero quality.
        if not np.isfinite(safe_number(row.get('growth_persistence_pillar'), np.nan)):
            normalized['Growth'] = _bounded_score(row.get('growth_score'), 22.0)
        if not np.isfinite(safe_number(row.get('profitability_pillar'), np.nan)):
            normalized['Profitability'] = _bounded_score(row.get('profitability_score'), 18.0)
        if not np.isfinite(safe_number(row.get('cash_conversion_pillar'), np.nan)):
            normalized['Cash-flow quality'] = _bounded_score(row.get('earnings_quality_score'), 18.0)
        if not np.isfinite(safe_number(row.get('balance_sheet_safety_pillar'), np.nan)):
            normalized['Balance-sheet safety'] = _multibagger_solvency_strength(row)
        if not np.isfinite(safe_number(row.get('reinvestment_runway_pillar'), np.nan)):
            normalized['Reinvestment runway'] = max(
                0.0,
                min(
                    100.0,
                    0.50 * safe_number(row.get('project_pipeline_score'), 50.0)
                    + 0.50 * safe_number(row.get('future_fundamental_impact_score'), 50.0),
                ),
            )
        data_integrity = _multibagger_data_integrity_score(row)
        solvency = _multibagger_solvency_strength(row)
        timing = _multibagger_technical_timing_score(row.get('technical_entry_state'))
        project_score = max(0.0, min(100.0, safe_number(row.get('project_pipeline_score'), 50.0)))
        management_score = max(0.0, min(100.0, safe_number(row.get('management_quality_score'), 50.0)))
        future_impact_score = max(0.0, min(100.0, safe_number(row.get('future_fundamental_impact_score'), 50.0)))
        project_coverage = max(0.0, min(100.0, safe_number(row.get('project_data_coverage_effective'), 0.0)))
        management_coverage = max(0.0, min(100.0, safe_number(row.get('management_data_coverage_effective'), 0.0)))
        impact_confidence = safe_text(row.get('future_impact_confidence')).upper()
        cycle_score = max(0.0, min(100.0, safe_number(row.get('multibagger_time_cycle_score'), 50.0)))
        explicit_quality = safe_number(row.get('multibagger_quality_score'), np.nan)
        derived_quality = (
            0.30 * normalized['Growth'] + 0.24 * normalized['Profitability']
            + 0.18 * normalized['Cash-flow quality']
            + 0.12 * normalized['Balance-sheet safety']
            + 0.08 * normalized['Reinvestment runway']
            + 0.08 * normalized['Valuation']
        )
        quality_score = max(0.0, min(100.0, explicit_quality if np.isfinite(explicit_quality) else derived_quality))
        execution_readiness = max(0.0, min(100.0, safe_number(row.get('execution_readiness_score'), 0.55 * normalized['Momentum'] + 0.45 * normalized['Smart-money proxy'])))
        evidence_strength = 0.65 * data_integrity + 0.35 * solvency
        rule_base = 0.67 * quality_score + 0.18 * evidence_strength + 0.15 * execution_readiness
        # Verified forward information may influence capital conviction slightly,
        # but it cannot replace the underlying quality score.
        project_weight = 0.05 * project_coverage / 100.0
        management_weight = 0.05 * management_coverage / 100.0
        future_impact_weight = 0.04 * {'HIGH': 1.0, 'MEDIUM': 0.65, 'LOW': 0.30}.get(impact_confidence, 0.0)
        overlay_weight = min(0.14, project_weight + management_weight + future_impact_weight)
        rule_conviction = round(
            (1.0 - overlay_weight) * rule_base
            + project_weight * project_score
            + management_weight * management_score
            + future_impact_weight * future_impact_score,
            1,
        )
        cycle_weight = 0.0
        ai_peer_score = max(0.0, min(100.0, safe_number(row.get('ai_multibagger_peer_score'), 50.0)))
        ai_weight = max(0.0, min(0.25, safe_number(row.get('ai_multibagger_effective_weight_pct'), 0.0) / 100.0))
        if safe_text(getattr(cfg, 'ai_mode', 'HYBRID_GUARDED')).upper() != 'HYBRID_GUARDED':
            ai_weight = 0.0
        conviction = round((1.0 - ai_weight) * rule_conviction + ai_weight * ai_peer_score, 1)
        tier, tier_multiplier = _multibagger_capital_tier(row, conviction)
        red_text = safe_text(row.get('red_flags')).upper()
        severe_tokens = ('OCF NEGATIF', 'MARGIN BERSIH NEGATIF', 'DER TINGGI', 'DILUSI TINGGI')
        governance_critical = bool(safe_text(row.get('management_governance_flags')).strip()) or safe_text(row.get('management_related_party_risk')).upper() == 'CRITICAL'
        project_critical = 'CRITICAL' in safe_text(row.get('project_execution_flags')).upper()
        severe = (
            truthy(row.get('severe_fundamental_flags'))
            or bool(safe_text(row.get('fundamental_conflicts')).strip())
            or any(token in red_text for token in severe_tokens)
            or governance_critical or project_critical
            or truthy(row.get('narrative_hard_block'))
        )
        status = safe_text(row.get('multibagger_status')).upper()
        eligible = bool(
            tier != 'WATCH_ONLY'
            and conviction >= minimum
            and status in {'MULTIBAGGER_A_CANDIDATE', 'MULTIBAGGER_B_CANDIDATE'}
            and not severe
        )
        cap_pct = _multibagger_tier_cap_pct(tier, cfg) if eligible else 0.0
        priority = conviction + (6.0 if status == 'MULTIBAGGER_A_CANDIDATE' else 0.0) + (3.0 if safe_text(row.get('compounding_state')).upper() == 'ACCUMULATE_NOW' else 1.0 if safe_text(row.get('compounding_state')).upper() == 'STARTER_NOW' else 0.0) + (2.0 if truthy(row.get('fundamental_official_verified')) else 0.0)
        out.at[idx, 'rule_capital_conviction_score'] = rule_conviction
        out.at[idx, 'capital_conviction_score'] = conviction
        out.at[idx, 'capital_tier'] = tier
        out.at[idx, 'capital_priority_score'] = round(priority, 1)
        out.at[idx, 'allocation_eligible'] = eligible
        out.at[idx, 'allocation_cap_pct'] = round(cap_pct, 1)
        out.at[idx, 'data_integrity_score'] = data_integrity
        out.at[idx, 'solvency_strength_score'] = solvency
        out.at[idx, 'technical_timing_score'] = timing
        out.at[idx, 'multibagger_quality_score'] = quality_score
        out.at[idx, 'execution_readiness_score'] = execution_readiness
        out.at[idx, 'project_capital_weight_pct'] = round(100.0 * project_weight, 2)
        out.at[idx, 'management_capital_weight_pct'] = round(100.0 * management_weight, 2)
        out.at[idx, 'future_impact_capital_weight_pct'] = round(100.0 * future_impact_weight, 2)
        out.at[idx, 'time_cycle_capital_weight_pct'] = round(100.0 * cycle_weight, 2)
        out.at[idx, 'multibagger_time_cycle_score'] = round(cycle_score, 1)
        strongest = sorted({**normalized, 'Asset quality': quality_score, 'Execution readiness': execution_readiness, 'Data integrity': data_integrity, 'Solvency': solvency, 'Project pipeline': project_score, 'Management': management_score, 'Future impact': future_impact_score, 'Time-cycle': cycle_score}.items(), key=lambda item: item[1], reverse=True)[:3]
        strongest_text = ', '.join(f'{label} {value:.0f}' for label, value in strongest)
        ai_note = f"; peer-AI {ai_peer_score:.1f} weight {ai_weight*100:.1f}%" if ai_weight > 0 else ''
        cycle_note = f"; time-cycle {cycle_score:.1f} weight {cycle_weight*100:.1f}%" if cycle_weight > 0 else ''
        narrative_note = (
            f"; narrative gate: {safe_text(row.get('narrative_primary_risk'))}"
            if truthy(row.get('narrative_hard_block')) else ''
        )
        out.at[idx, 'allocation_reason'] = f'{tier}; conviction {conviction:.1f}/100 (rule {rule_conviction:.1f}){ai_note}{cycle_note}{narrative_note}; strongest pillars: {strongest_text}'
        if eligible:
            eligible_rows.append((priority, idx))

    max_names = max(1, int(cfg.multibagger_max_holdings))
    selected = {idx for _, idx in sorted(eligible_rows, reverse=True)[:max_names]}
    for idx in out.index:
        if idx not in selected:
            out.at[idx, 'allocation_eligible'] = False
            out.at[idx, 'allocation_cap_pct'] = 0.0
            continue
        conviction = safe_number(out.at[idx, 'capital_conviction_score'], 0.0)
        tier = safe_text(out.at[idx, 'capital_tier'])
        tier_multiplier = {'CORE_COMPOUNDING': 1.35, 'HIGH_CONVICTION': 1.10, 'SATELLITE_GROWTH': 0.82, 'STARTER_ONLY': 0.58}.get(tier, 0.0)
        raw_weights[idx] = max(1.0, conviction - 55.0) ** 2 * tier_multiplier
        caps[idx] = safe_number(out.at[idx, 'allocation_cap_pct'], 0.0)

    weights = _capped_weight_distribution(raw_weights, caps)
    ranked_indices = [idx for _, idx in sorted(((safe_number(out.at[idx, 'capital_priority_score']), idx) for idx in selected), reverse=True)]
    for rank, idx in enumerate(ranked_indices, start=1):
        target_weight = weights.get(idx, 0.0)
        state = safe_text(out.at[idx, 'compounding_state']).upper()
        deploy = target_weight if state in {'ACCUMULATE_NOW', 'STARTER_NOW'} else 0.0
        action = 'ALLOCATE_LARGEST' if rank == 1 and deploy > 0 else 'ALLOCATE_NOW' if deploy > 0 else 'WAIT_ENTRY_ZONE'
        out.at[idx, 'capital_priority_rank'] = rank
        out.at[idx, 'strategic_target_weight_pct'] = target_weight
        out.at[idx, 'deploy_now_weight_pct'] = round(deploy, 2)
        out.at[idx, 'allocation_action'] = action

    budget = max(0.0, safe_number(cfg.multibagger_capital_budget_idr, 0.0))
    deployed_value = 0.0
    for idx, row in out.iterrows():
        deploy_weight = safe_number(row.get('deploy_now_weight_pct'), 0.0)
        target_amount = budget * safe_number(row.get('strategic_target_weight_pct'), 0.0) / 100.0
        proposed = budget * deploy_weight / 100.0
        entry = safe_number(row.get('entry'), np.nan)
        last_price = safe_number(row.get('last_price'), np.nan)
        reference = entry if np.isfinite(entry) and entry > 0 else last_price
        lots = int(math.floor(proposed / (reference * 100.0))) if proposed > 0 and np.isfinite(reference) and reference > 0 else 0
        order_value = lots * reference * 100.0 if lots > 0 else 0.0
        out.at[idx, 'multibagger_budget_idr'] = budget
        out.at[idx, 'allocation_reference_price'] = reference
        out.at[idx, 'strategic_target_amount_idr'] = round(target_amount, 0)
        out.at[idx, 'recommended_allocation_idr'] = round(proposed, 0)
        out.at[idx, 'recommended_lots'] = lots
        out.at[idx, 'estimated_order_value_idr'] = round(order_value, 0)
        deployed_value += order_value
        if proposed > 0 and lots <= 0:
            out.at[idx, 'allocation_action'] = 'BUDGET_BELOW_ONE_LOT'
    reserve = max(0.0, budget - deployed_value)
    out['multibagger_cash_reserve_idr'] = round(reserve, 0)
    out['capital_rank_selected'] = out.index.to_series().isin(selected)
    return out


def _multibagger_liquidity_quality(adtv: Any) -> float:
    value = max(0.0, safe_number(adtv, 0.0))
    if value >= 3_000_000_000:
        return 100.0
    if value >= 1_500_000_000:
        return 85.0
    if value >= 750_000_000:
        return 65.0
    if value >= 300_000_000:
        return 45.0
    if value > 0:
        return 20.0
    return 0.0


def _multibagger_candidate_type(
    *, revenue_growth: float, earnings_growth: float, profitability_score: float,
    earnings_quality_score: float, roic_proxy: float, positive_earnings_ratio: float,
    share_dilution: float, project_score: float, future_impact_score: float,
    free_cash_flow: float, project_capex: float,
) -> str:
    """Classify the economic path; descriptive only, never a production gate."""
    dilution_ok = (not np.isfinite(share_dilution)) or share_dilution <= 0.05
    if (
        profitability_score >= 14.0 and earnings_quality_score >= 13.0
        and np.isfinite(roic_proxy) and roic_proxy >= 0.12
        and positive_earnings_ratio >= 0.875 and dilution_ok
        and revenue_growth >= 0.08
    ):
        return 'TRUE_COMPOUNDER'
    if (
        project_score >= 65.0 and future_impact_score >= 58.0
        and (project_capex > 0 or (np.isfinite(free_cash_flow) and free_cash_flow <= 0))
    ):
        return 'TRANSFORMATIONAL_GROWTH'
    if (
        earnings_growth >= 0.25 and revenue_growth >= 0
        and np.isfinite(positive_earnings_ratio) and positive_earnings_ratio < 0.625
    ):
        return 'TURNAROUND'
    if earnings_growth >= 0.25 and revenue_growth >= 0 and positive_earnings_ratio < 0.875:
        return 'CYCLICAL_OR_EARNINGS_RECOVERY'
    if revenue_growth >= 0.18 and earnings_growth >= 0.18:
        return 'EMERGING_GROWTH_LEADER'
    if project_score >= 60.0 or future_impact_score >= 58.0:
        return 'EVENT_DRIVEN_RERATING'
    return 'QUALITY_GROWTH_WATCH'


def _growth_compounder_base_score(
    *,
    growth_persistence: Any,
    profitability: Any,
    cash_conversion: Any,
    balance_sheet_safety: Any,
    reinvestment_runway: Any,
    valuation: Any,
    liquidity: Any,
) -> float:
    """Score durable compounders without hiding reinvestment runway.

    v7.5.0 calculated reinvestment runway but only used it as a gate.  The
    v7.5.2 score gives runway an explicit 15% weight because a good business
    without a place to reinvest is not automatically a multibagger.
    """
    values = {
        'growth': max(0.0, min(100.0, safe_number(growth_persistence, 0.0))),
        'profitability': max(0.0, min(100.0, safe_number(profitability, 0.0))),
        'cash': max(0.0, min(100.0, safe_number(cash_conversion, 0.0))),
        'safety': max(0.0, min(100.0, safe_number(balance_sheet_safety, 0.0))),
        'runway': max(0.0, min(100.0, safe_number(reinvestment_runway, 0.0))),
        'valuation': max(0.0, min(100.0, safe_number(valuation, 0.0))),
        'liquidity': max(0.0, min(100.0, safe_number(liquidity, 0.0))),
    }
    return round(
        0.22 * values['growth']
        + 0.19 * values['profitability']
        + 0.16 * values['cash']
        + 0.14 * values['safety']
        + 0.15 * values['runway']
        + 0.08 * values['valuation']
        + 0.06 * values['liquidity'],
        2,
    )


def _coverage_adjusted_component(
    components: Iterable[tuple[Any, float, float]],
    *,
    neutral: float = 50.0,
) -> tuple[float, float, float]:
    """Return effective score, observed coverage, and observed-only raw score.

    Missing evidence is not a zero and it is not a full-strength neutral 50.
    Available evidence is first averaged using its intended weight, then the
    result is shrunk toward neutral in proportion to observed coverage.
    ``observed_fraction`` may be between 0 and 1 for bounded proxy evidence.
    """
    parsed: list[tuple[float, float, float]] = []
    total_weight = 0.0
    observed_weight = 0.0
    for value, weight, observed_fraction in components:
        clean_weight = max(0.0, safe_number(weight, 0.0))
        total_weight += clean_weight
        fraction = max(0.0, min(1.0, safe_number(observed_fraction, 0.0)))
        number = safe_number(value, np.nan)
        if clean_weight <= 0.0 or fraction <= 0.0 or not np.isfinite(number):
            continue
        weighted_observation = clean_weight * fraction
        parsed.append((
            max(0.0, min(100.0, number)),
            weighted_observation,
            fraction,
        ))
        observed_weight += weighted_observation
    if total_weight <= 0.0 or observed_weight <= 0.0:
        return (float(neutral), 0.0, float(neutral))
    raw = sum(value * weight for value, weight, _ in parsed) / observed_weight
    coverage = max(0.0, min(1.0, observed_weight / total_weight))
    effective = neutral + coverage * (raw - neutral)
    return (
        round(max(0.0, min(100.0, effective)), 2),
        round(100.0 * coverage, 2),
        round(max(0.0, min(100.0, raw)), 2),
    )


def _threshold_quality(
    value: Any,
    tiers: Iterable[tuple[float, float]],
    *,
    default: float = 0.0,
) -> float:
    """Map a finite metric to a 0–100 quality score using descending tiers."""
    number = safe_number(value, np.nan)
    if not np.isfinite(number):
        return np.nan
    for threshold, score in tiers:
        if number >= threshold:
            return float(score)
    return float(default)


def _turnaround_research_profile(
    *,
    growth_compounder_score: Any,
    candidate_type: Any,
    fundamental_inflection: Any,
    revenue_growth_acceleration: Any,
    earnings_growth_acceleration: Any,
    gross_margin_change_yoy: Any,
    cash_conversion_change_yoy: Any,
    safety_pillar: Any,
    cashflow_pillar: Any,
    project_score: Any,
    future_impact_score: Any,
    management_score: Any,
    valuation_quality: Any,
    liquidity_quality: Any,
    overall_confidence: Any,
    fundamental_coverage: Any,
    inflection_coverage: Any,
    data_grade: Any,
    statement_current: bool,
    critical_research_flags: bool,
) -> dict[str, Any]:
    """Build a point-in-time recovery lane without weakening capital gates.

    Operational weakness (for example a still-negative margin) may be part of
    a turnaround thesis. Governance, reporting conflict, aggressive dilution,
    accrual, or leverage deterioration remain hard research blockers.
    """
    inflection = max(0.0, min(100.0, safe_number(fundamental_inflection, 0.0)))
    earnings_accel = _range_quality_score(
        earnings_growth_acceleration, -0.10, 0.30,
    )
    revenue_accel = _range_quality_score(
        revenue_growth_acceleration, -0.08, 0.20,
    )
    margin_recovery = _range_quality_score(
        gross_margin_change_yoy, -0.05, 0.05,
    )
    cash_recovery = _range_quality_score(
        cash_conversion_change_yoy, -0.20, 0.30,
    )
    catalyst = (
        0.45 * max(0.0, min(100.0, safe_number(project_score, 50.0)))
        + 0.35 * max(0.0, min(100.0, safe_number(future_impact_score, 50.0)))
        + 0.20 * max(0.0, min(100.0, safe_number(management_score, 50.0)))
    )
    components = {
        'inflection': inflection,
        'earnings_acceleration': safe_number(earnings_accel, 0.0),
        'revenue_acceleration': safe_number(revenue_accel, 0.0),
        'margin_recovery': safe_number(margin_recovery, 0.0),
        'cash_recovery': safe_number(cash_recovery, 0.0),
        'safety': max(0.0, min(100.0, safe_number(safety_pillar, 0.0))),
        'cashflow': max(0.0, min(100.0, safe_number(cashflow_pillar, 0.0))),
        'catalyst': max(0.0, min(100.0, catalyst)),
        'valuation': max(0.0, min(100.0, safe_number(valuation_quality, 0.0))),
        'liquidity': max(0.0, min(100.0, safe_number(liquidity_quality, 0.0))),
    }
    score = round(
        0.20 * components['inflection']
        + 0.12 * components['earnings_acceleration']
        + 0.10 * components['revenue_acceleration']
        + 0.10 * components['margin_recovery']
        + 0.10 * components['cash_recovery']
        + 0.15 * components['safety']
        + 0.08 * components['cashflow']
        + 0.10 * components['catalyst']
        + 0.03 * components['valuation']
        + 0.02 * components['liquidity'],
        1,
    )
    recovery_signals = sum((
        inflection >= 60.0,
        safe_number(earnings_growth_acceleration, -1.0) >= 0.05,
        safe_number(revenue_growth_acceleration, -1.0) >= 0.03,
        safe_number(gross_margin_change_yoy, -1.0) >= 0.01,
        safe_number(cash_conversion_change_yoy, -1.0) >= 0.05,
    ))
    confidence = max(0.0, min(100.0, safe_number(overall_confidence, 0.0)))
    coverage = max(0.0, min(100.0, safe_number(fundamental_coverage, 0.0)))
    inflection_cov = max(0.0, min(100.0, safe_number(inflection_coverage, 0.0)))
    safety = components['safety']
    grade_ok = safe_text(data_grade).upper() in {'A', 'B', 'C'}
    confirmed = bool(
        score >= 72.0 and confidence >= 50.0 and coverage >= 60.0
        and inflection_cov >= 50.0 and recovery_signals >= 2
        and safety >= 45.0 and grade_ok and statement_current
        and not critical_research_flags
    )
    early = bool(
        score >= 62.0 and confidence >= 40.0 and coverage >= 50.0
        and inflection_cov >= 50.0 and recovery_signals >= 2
        and safety >= 35.0 and grade_ok and statement_current
        and not critical_research_flags
    )
    state = (
        'TURNAROUND_CONFIRMED' if confirmed
        else 'TURNAROUND_EARLY' if early
        else 'NOT_QUALIFIED'
    )
    gate_reasons: list[str] = []
    if critical_research_flags:
        gate_reasons.append('CRITICAL_GOVERNANCE_OR_ACCOUNTING_RISK')
    if not statement_current:
        gate_reasons.append('STATEMENT_NOT_CURRENT')
    if not grade_ok:
        gate_reasons.append('DATA_GRADE_BELOW_C')
    if coverage < 50.0:
        gate_reasons.append('FUNDAMENTAL_COVERAGE_BELOW_50')
    if inflection_cov < 50.0:
        gate_reasons.append('INFLECTION_COVERAGE_BELOW_50')
    if recovery_signals < 2:
        gate_reasons.append('RECOVERY_SIGNALS_BELOW_2')
    if safety < 35.0:
        gate_reasons.append('BALANCE_SHEET_SAFETY_BELOW_35')
    if score < 62.0:
        gate_reasons.append('TURNAROUND_SCORE_BELOW_62')

    type_name = safe_text(candidate_type).upper()
    turnaround_types = {
        'TURNAROUND', 'CYCLICAL_OR_EARNINGS_RECOVERY',
    }
    growth_score = max(
        0.0, min(100.0, safe_number(growth_compounder_score, 0.0)),
    )
    turnaround_lane = bool(
        type_name in turnaround_types
        or (
            state != 'NOT_QUALIFIED'
            and type_name != 'TRUE_COMPOUNDER'
            and score >= growth_score + 3.0
        )
        or (
            type_name == 'EVENT_DRIVEN_RERATING'
            and score >= growth_score
            and state != 'NOT_QUALIFIED'
        )
    )
    return {
        'multibagger_lane': (
            'TURNAROUND_CYCLICAL' if turnaround_lane
            else 'GROWTH_COMPOUNDER'
        ),
        'turnaround_recovery_score': score,
        'turnaround_research_state': state,
        'turnaround_recovery_signals': int(recovery_signals),
        'turnaround_gate_reasons': ' | '.join(gate_reasons),
        'turnaround_research_eligible': state in {
            'TURNAROUND_CONFIRMED', 'TURNAROUND_EARLY',
        },
    }


def _effective_silent_accumulation_score(
    score: Any, confidence: Any, state: Any,
) -> float:
    """Shrink uncertain flow evidence toward neutral and cap distribution."""
    raw = max(0.0, min(100.0, safe_number(score, 50.0)))
    weight = max(0.0, min(1.0, safe_number(confidence, 0.0) / 100.0))
    adjusted = 50.0 + weight * (raw - 50.0)
    if safe_text(state).upper() == 'DISTRIBUTION_RISK':
        adjusted = min(adjusted, 25.0)
    return round(max(0.0, min(100.0, adjusted)), 1)


def _multibagger_execution_readiness(
    *, momentum_score: float, silent_accumulation_score: float, time_score: float,
    technical_state: Any, broker_signal: Any='UNAVAILABLE',
) -> tuple[float, float]:
    """Separate timing from quality; broker data is only a bounded proxy."""
    momentum = max(0.0, min(100.0, 100.0 * safe_number(momentum_score, 0.0) / 12.0))
    accumulation = max(0.0, min(100.0, safe_number(silent_accumulation_score, 0.0)))
    timing = max(0.0, min(100.0, safe_number(time_score, 50.0)))
    technical = _multibagger_technical_timing_score(technical_state)
    broker = safe_text(broker_signal).upper()
    adjustment = 4.0 if broker == 'ACCUMULATION_PROXY' else -10.0 if broker == 'DISTRIBUTION_PROXY' else 0.0
    score = 0.30 * momentum + 0.35 * accumulation + 0.20 * timing + 0.15 * technical + adjustment
    return (round(max(0.0, min(100.0, score)), 1), adjustment)


def _confidence_grade(score: float) -> str:
    value = max(0.0, min(100.0, safe_number(score, 0.0)))
    if value >= 80.0:
        return 'HIGH'
    if value >= 62.0:
        return 'MEDIUM'
    if value >= 45.0:
        return 'LOW'
    return 'VERY_LOW'


def _multibagger_confidence_profile(
    *, coverage: float, data_grade: str, reliability: str, official_verified: bool,
    history_coverage: float, consensus_score: float, project_coverage: float,
    management_coverage: float, future_impact_confidence: str,
    accumulation_confidence: float, execution_readiness: float,
    core_execution_confidence: float, time_cycle_confidence: float,
    best_buy_confidence: float, eoff_validation_state: str,
    eoff_events: float, eoff_lift: float,
) -> dict[str, Any]:
    """Confidence is evidence quality, not expected return.

    It can reduce rank when evidence is incomplete, but it cannot raise the
    underlying Multibagger quality score above the asset-quality calculation.
    """
    grade_score = {'A': 100.0, 'B': 82.0, 'C': 60.0, 'D': 30.0}.get(safe_text(data_grade).upper(), 25.0)
    reliability_score = {'HIGH': 100.0, 'MEDIUM': 70.0, 'LOW': 42.0}.get(safe_text(reliability).upper(), 30.0)
    official_score = 100.0 if official_verified else 45.0
    data_confidence = max(0.0, min(100.0,
        0.42 * max(0.0, min(100.0, coverage))
        + 0.20 * grade_score
        + 0.20 * reliability_score
        + 0.18 * official_score
    ))
    consensus = max(0.0, min(100.0, consensus_score if np.isfinite(consensus_score) else 40.0))
    fundamental_confidence = max(0.0, min(100.0,
        0.55 * data_confidence
        + 0.25 * max(0.0, min(100.0, history_coverage))
        + 0.20 * consensus
    ))
    impact_score = {'HIGH': 100.0, 'MEDIUM': 65.0, 'LOW': 30.0}.get(safe_text(future_impact_confidence).upper(), 15.0)
    future_confidence = max(0.0, min(100.0,
        0.42 * max(0.0, min(100.0, project_coverage))
        + 0.33 * max(0.0, min(100.0, management_coverage))
        + 0.25 * impact_score
    ))
    technical_confidence = max(0.0, min(100.0,
        0.45 * max(0.0, min(100.0, accumulation_confidence))
        + 0.35 * max(0.0, min(100.0, execution_readiness))
        + 0.20 * max(0.0, min(100.0, core_execution_confidence))
    ))
    validation_state = safe_text(eoff_validation_state).upper()
    event_score = min(100.0, max(0.0, safe_number(eoff_events, 0.0)) / 100.0 * 100.0)
    lift_value = safe_number(eoff_lift, 0.0)
    lift_score = max(0.0, min(100.0, 50.0 + 5.0 * lift_value))
    validation_score = 100.0 if validation_state in {'VALIDATED', 'ROBUST', 'PUBLIC_VALIDATED'} else 70.0 if 'LIMITED' in validation_state else 35.0
    eoff_confidence = max(0.0, min(100.0,
        0.40 * max(0.0, min(100.0, time_cycle_confidence))
        + 0.25 * max(0.0, min(100.0, best_buy_confidence))
        + 0.15 * validation_score
        + 0.10 * event_score
        + 0.10 * lift_score
    ))
    overall = max(0.0, min(100.0,
        0.40 * fundamental_confidence
        + 0.20 * future_confidence
        + 0.25 * technical_confidence
        + 0.15 * eoff_confidence
    ))
    return {
        'data_confidence_score': round(data_confidence, 1),
        'fundamental_confidence_score': round(fundamental_confidence, 1),
        'future_fundamental_confidence_score': round(future_confidence, 1),
        'technical_confidence_score': round(technical_confidence, 1),
        'eoff_confidence_score': round(eoff_confidence, 1),
        'overall_research_confidence': round(overall, 1),
        'overall_research_confidence_grade': _confidence_grade(overall),
    }


def _multibagger_reason_profile(
    *, revenue_growth: float, earnings_growth: float, roe: float, roic_proxy: float,
    cash_conversion: float, free_cash_flow: float, debt_equity: float,
    net_debt_ebitda: float, share_dilution: float, project_score: float,
    gross_profitability: float, accruals_to_assets: float,
    quality_pillars_strong: int,
    future_impact_score: float, accumulation: Mapping[str, Any],
    technical_state: str, data_confidence: float, governance_flags: str,
    project_flags: str, fundamental_conflicts: str,
) -> dict[str, str]:
    positives: list[str] = []
    negatives: list[str] = []
    codes: list[str] = []
    if revenue_growth >= 0.18:
        positives.append(f'Revenue tumbuh {100*revenue_growth:.1f}%')
        codes.append('REVENUE_ACCELERATION')
    if earnings_growth >= 0.20:
        positives.append(f'Laba tumbuh {100*earnings_growth:.1f}%')
        codes.append('EARNINGS_ACCELERATION')
    if roe >= 0.15:
        positives.append(f'ROE {100*roe:.1f}%')
        codes.append('HIGH_ROE')
    if np.isfinite(roic_proxy) and roic_proxy >= 0.12:
        positives.append(f'ROIC proxy {100*roic_proxy:.1f}%')
        codes.append('HIGH_ROIC')
    if np.isfinite(gross_profitability) and gross_profitability >= 0.20:
        positives.append(f'Gross profitability {100*gross_profitability:.1f}% aset')
        codes.append('GROSS_PROFITABILITY_PREMIUM')
    if quality_pillars_strong >= 4:
        positives.append(f'{quality_pillars_strong}/5 pilar kualitas kuat')
        codes.append('MULTI_PILLAR_QUALITY')
    if np.isfinite(cash_conversion) and cash_conversion >= 0.75:
        positives.append(f'Cash conversion {cash_conversion:.2f}x')
        codes.append('CASH_BACKED_EARNINGS')
    if np.isfinite(free_cash_flow) and free_cash_flow > 0:
        positives.append('FCF positif')
        codes.append('POSITIVE_FCF')
    if project_score >= 65:
        positives.append(f'Project pipeline {project_score:.0f}')
        codes.append('PROJECT_PIPELINE')
    if future_impact_score >= 60:
        positives.append(f'Future impact {future_impact_score:.0f}')
        codes.append('FUTURE_FUNDAMENTAL')
    accumulation_state = safe_text(accumulation.get('silent_accumulation_state')).upper()
    if accumulation_state == 'SILENT_ACCUMULATION_CONFIRMED':
        positives.append('Silent accumulation terkonfirmasi')
        codes.append('SILENT_ACCUMULATION')
    persistence = safe_number(accumulation.get('accumulation_persistence_score'), np.nan)
    if np.isfinite(persistence) and persistence >= 60:
        positives.append(f'Akumulasi persisten {persistence:.0f}')
        codes.append('ACCUMULATION_PERSISTENCE')
    if np.isfinite(debt_equity) and debt_equity > 1.5:
        negatives.append(f'DER tinggi {debt_equity:.2f}x')
        codes.append('LEVERAGE_RISK')
    if np.isfinite(net_debt_ebitda) and net_debt_ebitda > 3.0:
        negatives.append(f'Net debt/EBITDA {net_debt_ebitda:.2f}x')
        codes.append('REFINANCING_RISK')
    if np.isfinite(share_dilution) and share_dilution > 0.05:
        negatives.append(f'Dilusi {100*share_dilution:.1f}%')
        codes.append('DILUTION_RISK')
    if np.isfinite(accruals_to_assets) and accruals_to_assets > 0.08:
        negatives.append(f'Akrual tinggi {100*accruals_to_assets:.1f}% aset')
        codes.append('ACCRUAL_QUALITY_RISK')
    if np.isfinite(free_cash_flow) and free_cash_flow < 0:
        negatives.append('FCF negatif')
        codes.append('FCF_PRESSURE')
    if accumulation_state in {'DISTRIBUTION_RISK', 'WEAK_OR_DISTRIBUTION'}:
        negatives.append(f'Flow {accumulation_state}')
        codes.append('DISTRIBUTION_RISK')
    if safe_text(technical_state).upper() in {'NO_ACTIVE_ENTRY_SETUP', 'WATCH_ONLY'}:
        negatives.append('Belum ada entry setup aktif')
        codes.append('TIMING_NOT_READY')
    if data_confidence < 60:
        negatives.append(f'Data confidence {data_confidence:.0f}')
        codes.append('LOW_DATA_CONFIDENCE')
    for label, value, code in (
        ('Governance', governance_flags, 'GOVERNANCE_FLAG'),
        ('Project', project_flags, 'PROJECT_EXECUTION_FLAG'),
        ('Fundamental conflict', fundamental_conflicts, 'FUNDAMENTAL_CONFLICT'),
    ):
        if safe_text(value):
            negatives.append(f'{label}: {safe_text(value)[:90]}')
            codes.append(code)
    return {
        'top_positive_drivers': ' • '.join(positives[:5]) or 'Belum ada driver kuat yang terverifikasi',
        'top_negative_drivers': ' • '.join(negatives[:5]) or 'Tidak ada red flag material pada data tersedia',
        'scoring_reason_codes': ' | '.join(dict.fromkeys(codes)),
    }


def scan_multibagger_candidates(
    prepared: Mapping[str, pd.DataFrame],
    fundamentals: pd.DataFrame | None,
    core_signals: pd.DataFrame | None=None,
    project_management: pd.DataFrame | None=None,
    config: ScanConfig | None=None,
    selector_ranking: pd.DataFrame | None=None,
    silent_profiles: Mapping[str, Mapping[str, Any]] | None=None,
    narrative_profiles: pd.DataFrame | None=None,
) -> pd.DataFrame:
    """Rank long-horizon growth/quality candidates; not a return guarantee."""
    cfg = config or ScanConfig()
    f_map = _fundamental_records(fundamentals)
    project_management_map = _project_management_records(project_management)
    accumulation_profiles = {
        safe_text(ticker).upper(): dict(profile)
        for ticker, profile in (silent_profiles or {}).items()
    }
    signal_map: dict[str, dict[str, Any]] = {}
    if core_signals is not None and (not core_signals.empty) and ('ticker' in core_signals):
        ranked = core_signals.copy()
        ranked['_q'] = pd.to_numeric(ranked.get('composite_score', ranked.get('quality_score')), errors='coerce').fillna(0)
        for ticker, group in ranked.sort_values('_q', ascending=False).groupby('ticker', sort=False):
            signal_map[str(ticker)] = group.iloc[0].to_dict()
    rows: list[dict[str, Any]] = []
    for ticker, frame in prepared.items():
        if frame is None or frame.empty or len(frame) < 220:
            continue
        fund = f_map.get(ticker, {})
        pm = project_management_map.get(ticker, {})
        forward_proxy = _automatic_forward_quality_proxy(fund)
        future_impact = _future_fundamental_impact(pm, fund)
        economic_earnings = _economic_earnings_profile(fund)
        coverage = safe_number(fund.get('fundamental_coverage'), 0.0)
        row = frame.iloc[-1]
        preliminary_accumulation = accumulation_profiles.get(
            safe_text(ticker).upper(),
        ) or silent_accumulation_profile(frame)
        if coverage <= 0:
            rows.append({
                'ticker': ticker,
                'multibagger_status': 'DATA_NOT_SCORED',
                'multibagger_scoring_state': 'DATA_NOT_SCORED_NO_FUNDAMENTAL',
                'multibagger_lane': 'UNCLASSIFIED',
                'research_eligible': False,
                'growth_research_eligible': False,
                'turnaround_research_eligible': False,
                'turnaround_research_state': 'NOT_QUALIFIED',
                'research_eligibility_reason': 'FUNDAMENTAL_SNAPSHOT_PENDING',
                'multibagger_score': np.nan,
                'multibagger_score_reason': 'Fundamental coverage 0%; scanner tidak mengubah missing data menjadi skor nol.',
                'multibagger_metric_coverage_pct': 0.0,
                'multibagger_metric_data_gate': False,
                'growth_pillar_coverage_pct': 0.0,
                'profitability_pillar_coverage_pct': 0.0,
                'cashflow_pillar_coverage_pct': 0.0,
                'safety_pillar_coverage_pct': 0.0,
                'runway_pillar_coverage_pct': 0.0,
                'valuation_pillar_coverage_pct': 0.0,
                'fundamental_score': np.nan,
                'fundamental_score_10': np.nan,
                'fundamental_coverage': coverage,
                'fundamental_data_grade': safe_text(fund.get('fundamental_data_grade')).upper() or 'D',
                'fundamental_reliability': safe_text(fund.get('fundamental_reliability')).upper() or 'UNKNOWN',
                'fundamental_source_count': int(safe_number(fund.get('fundamental_source_count'), 0.0)),
                'fundamental_history_source_count': int(safe_number(fund.get('fundamental_history_source_count'), 0.0)),
                'fundamental_snapshot_source_count': int(safe_number(fund.get('fundamental_snapshot_source_count'), 0.0)),
                'fundamental_history_quarters': int(safe_number(fund.get('fundamental_history_quarters'), 0.0)),
                'fundamental_history_years': int(safe_number(fund.get('fundamental_history_years'), 0.0)),
                'last_price': safe_number(row.get('Close'), np.nan),
                'adtv20_idr': safe_number(row.get('ADTV20'), 0.0),
                'silent_accumulation_score': preliminary_accumulation.get('silent_accumulation_score'),
                'silent_accumulation_state': preliminary_accumulation.get('silent_accumulation_state'),
                'silent_accumulation_confidence': preliminary_accumulation.get('silent_accumulation_confidence'),
                'silent_accumulation_reason': preliminary_accumulation.get('silent_accumulation_reason'),
                'accumulation_persistence_score': preliminary_accumulation.get('accumulation_persistence_score'),
                'accumulation_regime': preliminary_accumulation.get('accumulation_regime'),
                'data_confidence_score': 0.0,
                'fundamental_confidence_score': 0.0,
                'future_fundamental_confidence_score': 0.0,
                'technical_confidence_score': preliminary_accumulation.get('silent_accumulation_confidence'),
                'eoff_confidence_score': 0.0,
                'overall_research_confidence': 0.0,
                'overall_research_confidence_grade': 'VERY_LOW',
                'confidence_adjusted_multibagger_score': np.nan,
                'top_positive_drivers': 'Price-volume dapat dihitung, tetapi kualitas aset belum dapat dinilai',
                'top_negative_drivers': 'Fundamental coverage 0%',
                'scoring_reason_codes': 'DATA_NOT_SCORED_NO_FUNDAMENTAL',
                'up_down_value_ratio20': preliminary_accumulation.get('up_down_value_ratio20'),
                'up_down_value_ratio60': preliminary_accumulation.get('up_down_value_ratio60'),
                'distribution_days20': preliminary_accumulation.get('distribution_days20'),
                'capital_conviction_score': np.nan,
                'capital_tier': 'WATCH_ONLY',
                'allocation_action': 'WAIT_FOR_FUNDAMENTAL_DATA',
                'allocation_reason': 'Tidak dihitung sampai data laporan keuangan tersedia.',
                'compounding_state': 'WAIT_FOR_EVIDENCE',
                'review_action': 'REFRESH_FUNDAMENTAL_DATABASE',
                'note': 'N/A means not scored; it does not mean the issuer quality is zero.',
            })
            continue
        close = safe_number(row.get('Close'), 0.0)
        adtv = safe_number(row.get('ADTV20'), 0.0)
        revenue_growth = safe_number(fund.get('revenue_growth'), np.nan)
        earnings_growth = safe_number(fund.get('earnings_growth'), np.nan)
        roe = safe_number(fund.get('roe'), np.nan)
        roa = safe_number(fund.get('roa'), np.nan)
        net_margin = safe_number(fund.get('net_margin'), np.nan)
        operating_margin = safe_number(fund.get('operating_margin'), np.nan)
        debt_equity = safe_number(fund.get('debt_equity'), np.nan)
        current_ratio = safe_number(fund.get('current_ratio'), np.nan)
        cash_to_debt = safe_number(fund.get('cash_to_debt'), np.nan)
        ocf = safe_number(fund.get('operating_cash_flow'), np.nan)
        fcf = safe_number(fund.get('free_cash_flow'), np.nan)
        peg = safe_number(fund.get('peg_ratio'), np.nan)
        fcf_yield = safe_number(fund.get('fcf_yield'), np.nan)
        market_cap = safe_number(fund.get('market_cap'), np.nan)
        fundamental_reliability = safe_text(fund.get('fundamental_reliability')).upper() or 'UNKNOWN'
        fundamental_data_grade = safe_text(fund.get('fundamental_data_grade')).upper() or 'D'
        fundamental_score_10 = safe_number(fund.get('fundamental_score_10'), safe_number(fund.get('fundamental_score'), np.nan) / 10.0)
        source_count = int(safe_number(fund.get('fundamental_source_count'), 0.0))
        source_families = safe_text(fund.get('fundamental_source_families'))
        history_quarters = int(safe_number(fund.get('fundamental_history_quarters'), 0.0))
        history_years = int(safe_number(fund.get('fundamental_history_years'), 0.0))
        history_coverage = safe_number(fund.get('fundamental_history_coverage'), 0.0)
        consensus_score = safe_number(fund.get('fundamental_consensus_score'), np.nan)
        fundamental_conflicts = safe_text(fund.get('fundamental_conflicts'))
        official_reference = truthy(fund.get('fundamental_official_reference', False))
        official_verified = truthy(fund.get('fundamental_official_verified', False))
        cash_conversion = safe_number(fund.get('history_cash_conversion'), np.nan)
        positive_ocf_ratio = safe_number(fund.get('history_positive_ocf_ratio'), np.nan)
        positive_earnings_ratio = safe_number(fund.get('history_positive_earnings_ratio'), np.nan)
        margin_stability = safe_number(fund.get('history_margin_stability'), np.nan)
        share_dilution = safe_number(fund.get('history_share_dilution_yoy'), np.nan)
        roic_proxy = safe_number(fund.get('history_roic_proxy'), np.nan)
        gross_profitability = safe_number(fund.get('history_gross_profitability'), np.nan)
        gross_margin = safe_number(fund.get('history_gross_margin'), np.nan)
        gross_profit_growth = safe_number(fund.get('history_gross_profit_growth'), np.nan)
        fundamental_inflection = safe_number(fund.get('fundamental_inflection_score'), np.nan)
        fundamental_inflection_coverage = safe_number(fund.get('fundamental_inflection_coverage_pct'), 0.0)
        revenue_growth_acceleration = safe_number(fund.get('history_revenue_growth_acceleration'), np.nan)
        earnings_growth_acceleration = safe_number(fund.get('history_earnings_growth_acceleration'), np.nan)
        gross_margin_change_yoy = safe_number(fund.get('history_gross_margin_change_yoy'), np.nan)
        cash_conversion_change_yoy = safe_number(fund.get('history_cash_conversion_change_yoy'), np.nan)
        accruals_to_assets = safe_number(fund.get('history_accruals_to_assets'), np.nan)
        leverage_change_yoy = safe_number(fund.get('history_leverage_change_yoy'), np.nan)
        net_debt_ebitda = safe_number(fund.get('history_net_debt_ebitda'), np.nan)
        interest_coverage = safe_number(fund.get('history_interest_coverage'), np.nan)
        statement_age_days = safe_number(fund.get('statement_age_days'), np.nan)
        fundamental_model = safe_text(fund.get('fundamental_model')) or 'GENERAL'
        is_financial = fundamental_model == 'FINANCIAL'
        red_flags = safe_text(fund.get('fundamental_red_flags'))
        roc60 = safe_number(row.get('ROC60'), -1.0)
        roc120 = safe_number(row.get('ROC120'), -1.0)
        rs60 = safe_number(row.get('REL_STRENGTH60'), -1.0)
        dist_high = safe_number(row.get('DIST_52W_HIGH'), -1.0)
        cmf_v = safe_number(row.get('CMF20'), -1.0)
        obv_up = safe_number(row.get('OBV_SLOPE10'), -1.0) > 0
        accumulation_profile = preliminary_accumulation
        accumulation = safe_number(accumulation_profile.get('silent_accumulation_score'), 0.0)
        up_down = safe_number(accumulation_profile.get('up_down_value_ratio20'), np.nan)
        revenue_quality = _threshold_quality(
            revenue_growth,
            ((0.20, 100.0), (0.10, 63.64), (0.0, 27.27)),
        )
        earnings_growth_quality = _threshold_quality(
            earnings_growth,
            ((0.25, 100.0), (0.12, 63.64), (0.0, 27.27)),
        )
        growth_quality, growth_metric_coverage, growth_quality_raw = (
            _coverage_adjusted_component((
                (revenue_quality, 11.0, float(np.isfinite(revenue_growth))),
                (
                    earnings_growth_quality,
                    11.0,
                    float(np.isfinite(earnings_growth)),
                ),
            ))
        )
        growth_score = (
            22.0 * growth_quality / 100.0
            if growth_metric_coverage > 0.0 else 0.0
        )

        roe_quality = _threshold_quality(
            roe, ((0.20, 100.0), (0.12, 64.29), (0.08, 28.57)),
        )
        roa_quality = _threshold_quality(
            roa, ((0.08, 100.0), (0.04, 50.0)),
        )
        operating_margin_quality = _threshold_quality(
            operating_margin, ((0.15, 100.0), (0.08, 50.0)),
        )
        net_margin_quality = _threshold_quality(
            net_margin, ((0.12, 100.0), (0.06, 50.0)),
        )
        (
            profitability_quality,
            profitability_metric_coverage,
            profitability_quality_raw,
        ) = _coverage_adjusted_component((
            (roe_quality, 7.0, float(np.isfinite(roe))),
            (roa_quality, 3.0, float(np.isfinite(roa))),
            (
                operating_margin_quality,
                4.0,
                float(np.isfinite(operating_margin)),
            ),
            (
                net_margin_quality,
                4.0,
                float(np.isfinite(net_margin)),
            ),
        ))
        profitability_score = (
            18.0 * profitability_quality / 100.0
            if profitability_metric_coverage > 0.0 else 0.0
        )

        cash_conversion_quality = (
            100.0 if np.isfinite(cash_conversion)
            and 0.8 <= cash_conversion <= 1.8
            else 60.0 if np.isfinite(cash_conversion)
            and cash_conversion >= 0.6
            else 0.0 if np.isfinite(cash_conversion)
            else np.nan
        )
        fcf_quality = (
            100.0 if np.isfinite(fcf) and fcf > 0.0
            else 0.0 if np.isfinite(fcf) else np.nan
        )
        positive_ocf_quality = _threshold_quality(
            positive_ocf_ratio, ((0.875, 100.0), (0.625, 50.0)),
        )
        positive_earnings_quality = _threshold_quality(
            positive_earnings_ratio, ((0.875, 100.0), (0.625, 50.0)),
        )
        dilution_quality = (
            100.0 if np.isfinite(share_dilution) and share_dilution <= 0.02
            else 50.0 if np.isfinite(share_dilution)
            and share_dilution <= 0.05
            else 0.0 if np.isfinite(share_dilution)
            else np.nan
        )
        (
            cashflow_quality,
            cashflow_metric_coverage,
            cashflow_quality_raw,
        ) = _coverage_adjusted_component((
            (
                cash_conversion_quality,
                5.0,
                float(np.isfinite(cash_conversion)),
            ),
            (fcf_quality, 4.0, float(np.isfinite(fcf))),
            (
                positive_ocf_quality,
                4.0,
                float(np.isfinite(positive_ocf_ratio)),
            ),
            (
                positive_earnings_quality,
                3.0,
                float(np.isfinite(positive_earnings_ratio)),
            ),
            (
                dilution_quality,
                2.0,
                float(np.isfinite(share_dilution)),
            ),
        ))
        earnings_quality_score = (
            18.0 * cashflow_quality / 100.0
            if cashflow_metric_coverage > 0.0 else 0.0
        )

        debt_equity_quality = (
            100.0 if np.isfinite(debt_equity) and debt_equity <= 0.8
            else 50.0 if np.isfinite(debt_equity) and debt_equity <= 1.5
            else 0.0 if np.isfinite(debt_equity) else np.nan
        )
        current_ratio_quality = (
            100.0 if np.isfinite(current_ratio) and current_ratio >= 1.5
            else 50.0 if np.isfinite(current_ratio) and current_ratio >= 1.0
            else 0.0 if np.isfinite(current_ratio) else np.nan
        )
        cash_to_debt_quality = (
            100.0 if np.isfinite(cash_to_debt) and cash_to_debt >= 0.5
            else 50.0 if np.isfinite(cash_to_debt)
            and cash_to_debt >= 0.2
            else 0.0 if np.isfinite(cash_to_debt) else np.nan
        )
        net_debt_quality = (
            100.0 if np.isfinite(net_debt_ebitda)
            and net_debt_ebitda <= 1.5
            else 50.0 if np.isfinite(net_debt_ebitda)
            and net_debt_ebitda <= 3.0
            else 0.0 if np.isfinite(net_debt_ebitda)
            else np.nan
        )
        interest_coverage_quality = (
            100.0 if np.isfinite(interest_coverage)
            and interest_coverage >= 6.0
            else 50.0 if np.isfinite(interest_coverage)
            and interest_coverage >= 3.0
            else 0.0 if np.isfinite(interest_coverage)
            else np.nan
        )
        if not is_financial:
            (
                balance_quality,
                balance_metric_coverage,
                balance_quality_raw,
            ) = _coverage_adjusted_component((
                (
                    debt_equity_quality,
                    4.0,
                    float(np.isfinite(debt_equity)),
                ),
                (
                    current_ratio_quality,
                    2.0,
                    float(np.isfinite(current_ratio)),
                ),
                (
                    cash_to_debt_quality,
                    2.0,
                    float(np.isfinite(cash_to_debt)),
                ),
                (
                    net_debt_quality,
                    2.0,
                    float(np.isfinite(net_debt_ebitda)),
                ),
                (
                    interest_coverage_quality,
                    2.0,
                    float(np.isfinite(interest_coverage)),
                ),
            ))
        else:
            financial_snapshot_quality = (
                10.0 * fundamental_score_10
                if np.isfinite(fundamental_score_10) else np.nan
            )
            (
                balance_quality,
                balance_metric_coverage,
                balance_quality_raw,
            ) = _coverage_adjusted_component((
                (
                    financial_snapshot_quality,
                    1.0,
                    min(1.0, max(0.0, coverage / 100.0)),
                ),
            ))
        balance_score = (
            12.0 * balance_quality / 100.0
            if balance_metric_coverage > 0.0 else 0.0
        )
        solvency_fields = (debt_equity, current_ratio, cash_to_debt)
        solvency_coverage = 100.0 * sum((np.isfinite(value) for value in solvency_fields)) / len(solvency_fields)
        peg_quality = (
            100.0 if np.isfinite(peg) and 0.0 < peg <= 1.5
            else 50.0 if np.isfinite(peg) and 1.5 < peg <= 2.5
            else 0.0 if np.isfinite(peg) else np.nan
        )
        fcf_yield_quality = (
            100.0 if np.isfinite(fcf_yield) and fcf_yield >= 0.04
            else 50.0 if np.isfinite(fcf_yield) and fcf_yield > 0.0
            else 0.0 if np.isfinite(fcf_yield) else np.nan
        )
        (
            valuation_quality,
            valuation_metric_coverage,
            valuation_quality_raw,
        ) = _coverage_adjusted_component((
            (peg_quality, 4.0, float(np.isfinite(peg))),
            (
                fcf_yield_quality,
                4.0,
                float(np.isfinite(fcf_yield)),
            ),
        ))
        valuation_score = (
            8.0 * valuation_quality / 100.0
            if valuation_metric_coverage > 0.0 else 0.0
        )
        momentum_score = 0.0
        momentum_score += 4.0 if roc60 >= 0.15 else 2.5 if roc60 >= 0.07 else 0.0
        momentum_score += 3.0 if roc120 >= 0.25 else 2.0 if roc120 >= 0.12 else 0.0
        momentum_score += 2.0 if rs60 > 0 else 0.0
        momentum_score += 2.0 if dist_high >= -0.15 else 1.0 if dist_high >= -0.3 else 0.0
        momentum_score += 1.0 if close > safe_number(row.get('EMA200'), float('inf')) else 0.0
        accumulation_score = 0.0
        accumulation_state = safe_text(accumulation_profile.get('silent_accumulation_state')).upper()
        distribution_days = int(safe_number(accumulation_profile.get('distribution_days20'), 0.0))
        accumulation_score += 6.0 if accumulation >= 80 else 4.0 if accumulation >= 70 else 2.0 if accumulation >= 60 else 0.0
        accumulation_score += 2.0 if accumulation_state == 'SILENT_ACCUMULATION_CONFIRMED' else 1.0 if accumulation_state == 'EARLY_ACCUMULATION' else 0.0
        accumulation_score += 2.0 if safe_number(accumulation_profile.get('up_down_value_ratio60'), np.nan) >= 1.10 and distribution_days <= 2 else 0.0
        if distribution_days >= 4 or accumulation_state == 'DISTRIBUTION_RISK':
            accumulation_score = min(accumulation_score, 2.0)

        observed_project = safe_number(pm.get('project_pipeline_score_observed'), np.nan)
        observed_management = safe_number(pm.get('management_quality_score_observed'), np.nan)
        project_score = observed_project if np.isfinite(observed_project) else safe_number(forward_proxy.get('project_pipeline_score_proxy'), 45.0)
        management_score = observed_management if np.isfinite(observed_management) else safe_number(forward_proxy.get('management_quality_score_proxy'), 45.0)
        project_coverage = safe_number(pm.get('project_data_coverage'), 0.0) if np.isfinite(observed_project) else safe_number(forward_proxy.get('project_proxy_coverage'), 0.0)
        management_coverage = safe_number(pm.get('management_data_coverage'), 0.0) if np.isfinite(observed_management) else safe_number(forward_proxy.get('management_proxy_coverage'), 0.0)
        project_source = safe_text(pm.get('project_data_source')) if np.isfinite(observed_project) else 'AUTOMATIC_CAPEX_PROXY'
        management_source = safe_text(pm.get('management_data_source')) if np.isfinite(observed_management) else 'AUTOMATIC_OPERATING_TRACK_RECORD_PROXY'
        ceo_name = safe_text(pm.get('ceo_name_reviewed')) or safe_text(fund.get('ceo_name'))
        sig = signal_map.get(ticker, {})
        technical_entry_state = safe_text(sig.get('status')) or 'NO_ACTIVE_ENTRY_SETUP'
        broker_signal = safe_text(sig.get('broksum_signal')).upper() or 'UNAVAILABLE'

        # Research architecture v6.9.0: asset quality and entry timing are separate.
        # Momentum, accumulation and EOFF may change readiness, but may not turn a weak
        # business into a high-quality multibagger candidate.
        gross_profitability_quality = _range_quality_score(
            gross_profitability, 0.03, 0.30,
        )
        gross_margin_quality = _range_quality_score(gross_margin, 0.08, 0.40)
        (
            profitability_pillar,
            profitability_pillar_coverage,
            profitability_pillar_raw,
        ) = _coverage_adjusted_component((
            (
                profitability_quality_raw,
                0.65,
                profitability_metric_coverage / 100.0,
            ),
            (
                gross_profitability_quality,
                0.25,
                float(np.isfinite(gross_profitability)),
            ),
            (
                gross_margin_quality,
                0.10,
                float(np.isfinite(gross_margin)),
            ),
        ))
        gross_growth_quality = _range_quality_score(gross_profit_growth, -0.05, 0.18)
        (
            growth_persistence_pillar,
            growth_persistence_pillar_coverage,
            growth_persistence_pillar_raw,
        ) = _coverage_adjusted_component((
            (
                growth_quality_raw,
                0.45,
                growth_metric_coverage / 100.0,
            ),
            (
                100.0 * positive_earnings_ratio,
                0.15,
                float(np.isfinite(positive_earnings_ratio)),
            ),
            (
                100.0 * margin_stability,
                0.10,
                float(np.isfinite(margin_stability)),
            ),
            (
                gross_growth_quality,
                0.15,
                float(np.isfinite(gross_profit_growth)),
            ),
            (
                fundamental_inflection,
                0.15,
                min(
                    1.0,
                    max(0.0, fundamental_inflection_coverage / 100.0),
                ),
            ),
        ))
        accrual_quality = _range_quality_score(accruals_to_assets, 0.10, -0.02)
        (
            cashflow_pillar,
            cashflow_pillar_coverage,
            cashflow_pillar_raw,
        ) = _coverage_adjusted_component((
            (
                cashflow_quality_raw,
                0.80,
                cashflow_metric_coverage / 100.0,
            ),
            (
                accrual_quality,
                0.20,
                float(np.isfinite(accruals_to_assets)),
            ),
        ))
        leverage_trend_quality = _range_quality_score(
            leverage_change_yoy, 0.08, -0.04,
        )
        (
            safety_pillar,
            safety_pillar_coverage,
            safety_pillar_raw,
        ) = _coverage_adjusted_component((
            (
                balance_quality_raw,
                0.80,
                balance_metric_coverage / 100.0,
            ),
            (
                leverage_trend_quality,
                0.20,
                float(
                    np.isfinite(leverage_change_yoy) and not is_financial
                ),
            ),
        ))
        liquidity_quality = _multibagger_liquidity_quality(adtv)
        project_weight = 0.09 * max(0.0, min(1.0, project_coverage / 100.0))
        management_weight = 0.09 * max(0.0, min(1.0, management_coverage / 100.0))
        impact_confidence = safe_text(future_impact.get('future_impact_confidence')).upper()
        impact_coverage = {'HIGH': 1.0, 'MEDIUM': 0.65, 'LOW': 0.30}.get(impact_confidence, 0.0)
        impact_weight = 0.08 * impact_coverage
        base_weight = max(0.0, 1.0 - project_weight - management_weight - impact_weight)
        future_impact_score = safe_number(future_impact.get('future_fundamental_impact_score'), 50.0)
        roic_quality = _range_quality_score(roic_proxy, 0.04, 0.18)
        (
            reinvestment_runway_pillar,
            reinvestment_runway_pillar_coverage,
            reinvestment_runway_pillar_raw,
        ) = _coverage_adjusted_component((
            (
                roic_quality,
                0.30,
                float(np.isfinite(roic_proxy)),
            ),
            (
                project_score,
                0.25,
                min(1.0, max(0.0, project_coverage / 100.0)),
            ),
            (
                future_impact_score,
                0.25,
                impact_coverage,
            ),
            (
                growth_persistence_pillar_raw,
                0.20,
                growth_persistence_pillar_coverage / 100.0,
            ),
        ))
        base_quality = _growth_compounder_base_score(
            growth_persistence=growth_persistence_pillar,
            profitability=profitability_pillar,
            cash_conversion=cashflow_pillar,
            balance_sheet_safety=safety_pillar,
            reinvestment_runway=reinvestment_runway_pillar,
            valuation=valuation_quality,
            liquidity=liquidity_quality,
        )
        quality_pillar_coverage = float(np.mean([
            growth_persistence_pillar_coverage,
            profitability_pillar_coverage,
            cashflow_pillar_coverage,
            safety_pillar_coverage,
            reinvestment_runway_pillar_coverage,
        ]))
        liquidity_metric_coverage = 100.0 if adtv > 0.0 else 0.0
        multibagger_metric_coverage = (
            0.22 * growth_persistence_pillar_coverage
            + 0.19 * profitability_pillar_coverage
            + 0.16 * cashflow_pillar_coverage
            + 0.14 * safety_pillar_coverage
            + 0.15 * reinvestment_runway_pillar_coverage
            + 0.08 * valuation_metric_coverage
            + 0.06 * liquidity_metric_coverage
        )
        pillar_values = {
            'growth_persistence': growth_persistence_pillar,
            'profitability': profitability_pillar,
            'cash_conversion': cashflow_pillar,
            'balance_sheet_safety': safety_pillar,
            'reinvestment_runway': reinvestment_runway_pillar,
        }
        quality_pillars_strong = sum(value >= 60.0 for value in pillar_values.values())
        quality_pillars_critical = sum(value < 35.0 for value in pillar_values.values())
        total = (
            base_weight * base_quality
            + project_weight * project_score
            + management_weight * management_score
            + impact_weight * future_impact_score
        )
        run_full_timing = bool(
            getattr(cfg, 'time_cycle_enabled', True)
            and (
                total >= 55.0
                or (
                    fundamental_inflection_coverage >= 50.0
                    and fundamental_inflection >= 60.0
                )
                or technical_entry_state in {
                    'EXECUTION_READY', 'READY_FOR_STOCKBIT_VERIFY',
                    'ENTRY_PLAN_READY', 'READY_FOR_PRICE_VERIFY',
                }
            )
        )
        time_cycle = analyze_time_cycle(
            frame,
            TimeCycleConfig(
                min_bars=int(getattr(cfg, 'time_cycle_min_history_bars', 180)),
                lunar_enabled=bool(getattr(cfg, 'time_cycle_lunar_enabled', True)),
                eoff_enabled=bool(getattr(cfg, 'eoff_enabled', True)),
                eoff_ephemeris_enabled=bool(getattr(cfg, 'eoff_ephemeris_enabled', True)),
                eoff_min_fib_cluster=int(getattr(cfg, 'eoff_min_fib_cluster', 4)),
                eoff_min_unique_anchors=int(getattr(cfg, 'eoff_min_unique_anchors', 3)),
                eoff_max_dominant_anchor_share=float(getattr(cfg, 'eoff_max_dominant_anchor_share', 0.55)),
                eoff_aspect_orb_deg=float(getattr(cfg, 'eoff_aspect_orb_deg', 3.0)),
                eoff_require_astro_fib_confluence=bool(getattr(cfg, 'eoff_require_astro_fib_confluence', True)),
                idx_trading_holidays=tuple(getattr(cfg, 'idx_trading_holidays', ()) or ()),
                idx_official_open_dates=tuple(getattr(cfg, 'idx_official_open_dates', ()) or ()),
                idx_official_closed_dates=tuple(getattr(cfg, 'idx_official_closed_dates', ()) or ()),
                require_official_idx_calendar=bool(getattr(cfg, 'require_official_idx_calendar', False)),
            ),
        ) if run_full_timing else {
            'time_cycle_state': (
                'SKIPPED_LOW_MULTIBAGGER_QUALITY'
                if bool(getattr(cfg, 'time_cycle_enabled', True)) else 'DISABLED'
            ),
            'time_cycle_confidence': 0.0,
            'bullish_timing_score': 50.0,
            'continuation_timing_score': 50.0,
            'time_cycle_explanation': (
                'Full time-cycle/EOFF deferred because strategic quality is below the research threshold.'
            ),
        }
        multibagger_time_score = max(
            safe_number(time_cycle.get('bullish_timing_score'), 50.0),
            safe_number(time_cycle.get('continuation_timing_score'), 50.0),
        )
        execution_readiness, broker_flow_adjustment = _multibagger_execution_readiness(
            momentum_score=momentum_score,
            silent_accumulation_score=accumulation,
            time_score=multibagger_time_score,
            technical_state=technical_entry_state,
            broker_signal=broker_signal,
        )
        candidate_type = _multibagger_candidate_type(
            revenue_growth=revenue_growth, earnings_growth=earnings_growth,
            profitability_score=profitability_score, earnings_quality_score=earnings_quality_score,
            roic_proxy=roic_proxy, positive_earnings_ratio=positive_earnings_ratio,
            share_dilution=share_dilution, project_score=project_score,
            future_impact_score=future_impact_score, free_cash_flow=fcf,
            project_capex=safe_number(pm.get('project_capex_idr'), 0.0),
        )
        operational_recovery_flags = any((
            flag in red_flags
            for flag in ('Margin bersih negatif', 'OCF negatif', 'DER tinggi')
        ))
        governance_flags = safe_text(pm.get('management_governance_flags'))
        related_party_risk = safe_text(pm.get('management_related_party_risk')).upper()
        project_execution_flags = safe_text(pm.get('project_execution_flags'))
        critical_research_flags = bool(
            fundamental_conflicts
            or (np.isfinite(share_dilution) and share_dilution > 0.12)
            or (np.isfinite(accruals_to_assets) and accruals_to_assets > 0.12)
            or (np.isfinite(leverage_change_yoy) and leverage_change_yoy > 0.15)
            or governance_flags or related_party_risk == 'CRITICAL'
            or 'CRITICAL' in project_execution_flags.upper()
        )
        severe_flags = bool(operational_recovery_flags or critical_research_flags)
        if severe_flags:
            total = min(total, 69.0)
        if np.isfinite(market_cap) and market_cap < 300000000000:
            total -= 4.0
        total = max(0.0, min(100.0, total))
        below_minimum_score = bool(total < 60)
        statement_current = bool(np.isfinite(statement_age_days) and 0 <= statement_age_days <= cfg.max_statement_age_days)
        minimum_metric_coverage = max(
            0.0,
            min(
                100.0,
                safe_number(
                    getattr(
                        cfg,
                        'multibagger_min_metric_coverage_pct',
                        65.0,
                    ),
                    65.0,
                ),
            ),
        )
        a_minimum_metric_coverage = max(
            minimum_metric_coverage,
            min(
                100.0,
                safe_number(
                    getattr(
                        cfg,
                        'multibagger_a_min_metric_coverage_pct',
                        80.0,
                    ),
                    80.0,
                ),
            ),
        )
        history_current_enough = bool(
            history_coverage >= 35.0
            and (history_quarters >= 4 or history_years >= 2)
        )
        metric_data_gate = bool(
            multibagger_metric_coverage >= minimum_metric_coverage
            and statement_current
            and history_current_enough
        )
        multibagger_scoring_state = (
            'SCORED_COMPLETE'
            if (
                multibagger_metric_coverage >= a_minimum_metric_coverage
                and statement_current
                and history_coverage >= 55.0
            )
            else 'SCORED_SUFFICIENT'
            if metric_data_gate
            else 'SCORED_PARTIAL_RESEARCH_ONLY'
            if multibagger_metric_coverage >= 45.0
            else 'DATA_INSUFFICIENT_NOT_PRODUCTION'
        )
        car_raw = safe_number(fund.get('history_car'), np.nan)
        npl_raw = safe_number(fund.get('history_npl_gross'), np.nan)
        ldr_raw = safe_number(fund.get('history_ldr'), np.nan)
        car = car_raw / 100.0 if car_raw > 1.5 else car_raw
        npl_gross = npl_raw / 100.0 if npl_raw > 1.0 else npl_raw
        ldr = ldr_raw / 100.0 if ldr_raw > 2.0 else ldr_raw
        bank_prudential_gate = bool(
            is_financial and np.isfinite(car) and np.isfinite(npl_gross) and np.isfinite(ldr)
            and car >= 0.12 and npl_gross <= 0.03 and 0.65 <= ldr <= 1.00
        )
        general_solvency_gate = bool((not is_financial) and solvency_coverage >= 66.0)
        fundamental_data_gate = bool(
            coverage >= 70.0 and fundamental_data_grade in {'A', 'B'}
            and fundamental_reliability == 'HIGH' and statement_current
            and history_coverage >= 55.0
        )
        grade_a_gate = bool(
            fundamental_data_grade == 'A' and source_count >= 2
            and official_verified
            and (history_quarters >= 8 or history_years >= 3)
            and np.isfinite(consensus_score) and consensus_score >= 75.0
            and not fundamental_conflicts
        )
        quality_pillar_gate = bool(
            quality_pillar_coverage >= 60.0
            and quality_pillars_strong >= 3
            and quality_pillars_critical <= 1
        )
        quality_pillar_a_gate = bool(
            quality_pillar_coverage >= 80.0
            and quality_pillars_strong >= 4
            and quality_pillars_critical == 0
        )
        forward_quality_coverage = min(100.0, 0.5 * project_coverage + 0.5 * management_coverage)
        forward_quality_score = (project_score + management_score) / 2.0
        forward_quality_gate = bool(
            forward_quality_coverage < 50.0
            or (project_score >= 50.0 and management_score >= 50.0 and not governance_flags and related_party_risk != 'CRITICAL')
        )
        confidence_profile = _multibagger_confidence_profile(
            coverage=coverage, data_grade=fundamental_data_grade, reliability=fundamental_reliability,
            official_verified=official_verified, history_coverage=history_coverage,
            consensus_score=consensus_score, project_coverage=project_coverage,
            management_coverage=management_coverage, future_impact_confidence=impact_confidence,
            accumulation_confidence=safe_number(accumulation_profile.get('silent_accumulation_confidence'), 0.0),
            execution_readiness=execution_readiness,
            core_execution_confidence=safe_number(sig.get('execution_confidence_score'), safe_number(sig.get('data_completeness_score'), 50.0)),
            time_cycle_confidence=safe_number(time_cycle.get('time_cycle_confidence'), 0.0),
            best_buy_confidence=safe_number(time_cycle.get('best_buy_confidence'), 0.0),
            eoff_validation_state=safe_text(time_cycle.get('eoff_public_validation_state')),
            eoff_events=safe_number(time_cycle.get('eoff_public_directional_events'), 0.0),
            eoff_lift=safe_number(time_cycle.get('eoff_public_lift'), 0.0),
        )
        overall_confidence = safe_number(confidence_profile.get('overall_research_confidence'), 0.0)
        confidence_adjusted_score = max(0.0, min(total, total * (0.78 + 0.22 * overall_confidence / 100.0)))
        archetype_profile = _turnaround_research_profile(
            growth_compounder_score=total,
            candidate_type=candidate_type,
            fundamental_inflection=fundamental_inflection,
            revenue_growth_acceleration=revenue_growth_acceleration,
            earnings_growth_acceleration=earnings_growth_acceleration,
            gross_margin_change_yoy=gross_margin_change_yoy,
            cash_conversion_change_yoy=cash_conversion_change_yoy,
            safety_pillar=safety_pillar,
            cashflow_pillar=cashflow_pillar,
            project_score=project_score,
            future_impact_score=future_impact_score,
            management_score=management_score,
            valuation_quality=valuation_quality,
            liquidity_quality=liquidity_quality,
            overall_confidence=overall_confidence,
            fundamental_coverage=coverage,
            inflection_coverage=fundamental_inflection_coverage,
            data_grade=fundamental_data_grade,
            statement_current=statement_current,
            critical_research_flags=critical_research_flags,
        )
        if not metric_data_gate:
            archetype_profile['turnaround_research_eligible'] = False
            if archetype_profile.get('turnaround_research_state') != 'NOT_QUALIFIED':
                archetype_profile['turnaround_research_state'] = (
                    'DATA_PENDING_METRICS'
                )
            existing_turnaround_reasons = safe_text(
                archetype_profile.get('turnaround_gate_reasons')
            )
            archetype_profile['turnaround_gate_reasons'] = ' | '.join(
                part for part in (
                    existing_turnaround_reasons,
                    (
                        'MULTIBAGGER_METRIC_COVERAGE_BELOW_'
                        f'{minimum_metric_coverage:.0f}'
                    ),
                    'CURRENT_STATEMENT_HISTORY_REQUIRED',
                ) if part
            )
        reason_profile = _multibagger_reason_profile(
            revenue_growth=revenue_growth, earnings_growth=earnings_growth, roe=roe,
            roic_proxy=roic_proxy, cash_conversion=cash_conversion, free_cash_flow=fcf,
            debt_equity=debt_equity, net_debt_ebitda=net_debt_ebitda,
            share_dilution=share_dilution, project_score=project_score,
            gross_profitability=gross_profitability,
            accruals_to_assets=accruals_to_assets,
            quality_pillars_strong=quality_pillars_strong,
            future_impact_score=future_impact_score, accumulation=accumulation_profile,
            technical_state=technical_entry_state, data_confidence=safe_number(confidence_profile.get('data_confidence_score'), 0.0),
            governance_flags=governance_flags, project_flags=project_execution_flags,
            fundamental_conflicts=fundamental_conflicts,
        )
        if below_minimum_score:
            status = 'MULTIBAGGER_NOT_QUALIFIED'
        elif total >= 82 and overall_confidence >= 70.0 and fundamental_score_10 >= 8.0 and fundamental_data_gate and grade_a_gate and quality_pillar_a_gate and multibagger_metric_coverage >= a_minimum_metric_coverage and forward_quality_gate and (adtv >= 1500000000) and (not severe_flags) and (general_solvency_gate or bank_prudential_gate):
            status = 'MULTIBAGGER_A_CANDIDATE'
        elif total >= 72 and overall_confidence >= 55.0 and fundamental_score_10 >= 7.0 and fundamental_data_grade in {'A', 'B', 'C'} and coverage >= 60 and metric_data_gate and quality_pillar_gate and (not severe_flags):
            status = 'MULTIBAGGER_B_CANDIDATE'
        else:
            status = 'MULTIBAGGER_WATCHLIST'
        growth_research_eligible = status in {
            'MULTIBAGGER_A_CANDIDATE', 'MULTIBAGGER_B_CANDIDATE',
        }
        turnaround_research_eligible = bool(
            archetype_profile.get('turnaround_research_eligible', False)
        )
        if turnaround_research_eligible and not growth_research_eligible:
            # A recovery-only qualification must never leak into the Growth
            # radar merely because the descriptive classifier was ambiguous.
            archetype_profile['multibagger_lane'] = 'TURNAROUND_CYCLICAL'
        research_eligible = bool(
            growth_research_eligible or turnaround_research_eligible
        )
        if growth_research_eligible:
            research_eligibility_reason = (
                f'{status}; durable growth/quality gates passed'
            )
        elif turnaround_research_eligible:
            research_eligibility_reason = (
                f"{archetype_profile.get('turnaround_research_state')}; "
                'recovery evidence passed the research-only gate'
            )
        elif not metric_data_gate:
            research_eligibility_reason = (
                f'{multibagger_scoring_state}; metric coverage '
                f'{multibagger_metric_coverage:.1f}% (minimum '
                f'{minimum_metric_coverage:.0f}%); current statement/history '
                'belum lengkap'
            )
        else:
            research_eligibility_reason = (
                archetype_profile.get('turnaround_gate_reasons')
                or 'Neither growth-compounder nor turnaround research gate passed'
            )
        actionable_entry_states = {
            'EXECUTION_READY', 'READY_FOR_STOCKBIT_VERIFY',
            'ENTRY_PLAN_READY', 'READY_FOR_PRICE_VERIFY',
        }
        if status == 'MULTIBAGGER_A_CANDIDATE' and technical_entry_state in actionable_entry_states and execution_readiness >= 72:
            compounding_state = 'ACCUMULATE_NOW'
            review_action = 'ADD_GRADUALLY_WITH_CORE_ENTRY_PLAN'
        elif status == 'MULTIBAGGER_A_CANDIDATE':
            compounding_state = 'WAIT_ACCUMULATION_ZONE'
            review_action = 'KEEP_REALIZED_PROFIT_AS_COMPOUNDING_CASH'
        elif status == 'MULTIBAGGER_B_CANDIDATE' and technical_entry_state in actionable_entry_states and execution_readiness >= 75:
            compounding_state = 'STARTER_NOW'
            review_action = 'OPEN_SMALL_STARTER_ONLY; ADD AFTER QUARTERLY CONFIRMATION'
        elif status == 'MULTIBAGGER_B_CANDIDATE':
            compounding_state = 'RESEARCH_AND_WAIT'
            review_action = 'VERIFY_QUARTERLY_TREND_BEFORE_ADDING'
        else:
            compounding_state = 'RESEARCH_ONLY'
            review_action = 'NO_COMPOUNDING_ALLOCATION'
        max_allocation = min(cfg.max_position_pct, 0.20 if status == 'MULTIBAGGER_A_CANDIDATE' else 0.12 if status == 'MULTIBAGGER_B_CANDIDATE' else 0.0)
        recommendation_status = (
            'BUY_ZONE' if status in {'MULTIBAGGER_A_CANDIDATE', 'MULTIBAGGER_B_CANDIDATE'} and overall_confidence >= 65 and execution_readiness >= 75 and technical_entry_state in actionable_entry_states
            else 'WATCH' if status in {'MULTIBAGGER_A_CANDIDATE', 'MULTIBAGGER_B_CANDIDATE'}
            else 'WATCH' if turnaround_research_eligible
            else 'WAIT'
        )
        rows.append({'ticker': ticker, 'multibagger_status': status, 'multibagger_score': round(total, 1), 'multibagger_quality_score': round(total, 1), 'execution_readiness_score': execution_readiness, 'research_recommendation_status': recommendation_status, 'multibagger_candidate_type': candidate_type, 'multibagger_scoring_state': multibagger_scoring_state, 'multibagger_score_reason': 'Coverage-aware quality/future-fundamental score; execution timing is separate' if total >= 60 else 'Coverage-aware quality score below minimum 60', 'growth_score': round(growth_score, 1), 'profitability_score': round(profitability_score, 1), 'earnings_quality_score': round(earnings_quality_score, 1), 'economic_earnings_score': economic_earnings.get('economic_earnings_score'), 'economic_earnings_confidence': economic_earnings.get('economic_earnings_confidence'), 'economic_earnings_state': economic_earnings.get('economic_earnings_state'), 'economic_earnings_production_weight_pct': economic_earnings.get('economic_earnings_production_weight_pct'), 'ocf_ebitda_conversion': economic_earnings.get('ocf_ebitda_conversion'), 'minority_leakage_pct': economic_earnings.get('minority_leakage_pct'), 'inventory_growth_yoy': economic_earnings.get('inventory_growth_yoy'), 'receivables_growth_yoy': economic_earnings.get('receivables_growth_yoy'), 'balance_sheet_score': round(balance_score, 1), 'valuation_score': round(valuation_score, 1), 'liquidity_quality_score': round(liquidity_quality, 1), 'momentum_score': round(momentum_score, 1), 'accumulation_score': round(accumulation_score, 1), 'base_multibagger_score': round(base_quality, 1), 'broker_flow_signal': broker_signal, 'broker_flow_adjustment': broker_flow_adjustment, 'project_pipeline_score': round(project_score, 1), 'project_data_coverage_effective': round(project_coverage, 1), 'project_data_source': project_source, 'project_count': int(safe_number(pm.get('project_count'), 0.0)), 'project_names': safe_text(pm.get('project_names')), 'project_capex_idr': safe_number(pm.get('project_capex_idr'), 0.0), 'project_expected_revenue_idr': safe_number(pm.get('project_expected_revenue_idr'), 0.0), 'project_expected_ebitda_idr': safe_number(pm.get('project_expected_ebitda_idr'), 0.0), 'project_execution_flags': project_execution_flags, 'project_proxy_basis': safe_text(forward_proxy.get('project_proxy_basis')), 'management_quality_score': round(management_score, 1), 'management_data_coverage_effective': round(management_coverage, 1), 'management_data_source': management_source, 'ceo_name': ceo_name, 'ceo_title': safe_text(fund.get('ceo_title')), 'management_governance_flags': governance_flags, 'management_related_party_risk': related_party_risk, 'management_proxy_basis': safe_text(forward_proxy.get('management_proxy_basis')), 'forward_quality_score': round(forward_quality_score, 1), 'forward_quality_coverage': round(forward_quality_coverage, 1), 'forward_quality_gate': forward_quality_gate, 'future_fundamental_impact_score': future_impact.get('future_fundamental_impact_score'), 'future_impact_confidence': future_impact.get('future_impact_confidence'), 'future_impact_model': future_impact.get('future_impact_model'), 'future_impact_horizon': future_impact.get('future_impact_horizon'), 'future_revenue_uplift_bear_pct': future_impact.get('future_revenue_uplift_bear_pct'), 'future_revenue_uplift_base_pct': future_impact.get('future_revenue_uplift_base_pct'), 'future_revenue_uplift_bull_pct': future_impact.get('future_revenue_uplift_bull_pct'), 'future_ebitda_uplift_base_pct': future_impact.get('future_ebitda_uplift_base_pct'), 'future_net_profit_uplift_base_pct': future_impact.get('future_net_profit_uplift_base_pct'), 'future_fcf_pressure_idr': future_impact.get('future_fcf_pressure_idr'), 'future_net_debt_change_idr': future_impact.get('future_net_debt_change_idr'), 'future_net_debt_change_pct': future_impact.get('future_net_debt_change_pct'), 'project_success_probability_pct': future_impact.get('project_success_probability_pct'), 'project_stage': future_impact.get('project_stage'), 'project_stage_probability_pct': future_impact.get('project_stage_probability_pct'), 'project_stage_probability_source': future_impact.get('project_stage_probability_source'), 'project_evidence_probability_pct': future_impact.get('project_evidence_probability_pct'), 'project_source_families': safe_text(pm.get('project_source_families')), 'project_source_urls': safe_text(pm.get('project_source_urls')), 'project_source_quorum_verified': truthy(pm.get('project_source_quorum_verified')), 'management_source_urls': safe_text(pm.get('management_source_urls')), 'multibagger_time_cycle_score': round(multibagger_time_score, 1), 'time_cycle_score': time_cycle.get('time_cycle_score'), 'time_cycle_confidence': time_cycle.get('time_cycle_confidence'), 'time_cycle_state': time_cycle.get('time_cycle_state'), 'time_cycle_direction_bias': time_cycle.get('time_cycle_direction_bias'), 'time_cycle_phase': time_cycle.get('time_cycle_phase'), 'dominant_cycle_bars': time_cycle.get('dominant_cycle_bars'), 'cycle_historical_hit_rate': time_cycle.get('cycle_historical_hit_rate'), 'cycle_validation_samples': time_cycle.get('cycle_validation_samples'), 'next_reversal_window_start': time_cycle.get('next_reversal_window_start'), 'next_reversal_window_end': time_cycle.get('next_reversal_window_end'), 'bars_to_reversal_window': time_cycle.get('bars_to_reversal_window'), 'lunar_phase': time_cycle.get('lunar_phase'), 'lunar_days_to_major_marker': time_cycle.get('lunar_days_to_major_marker'), 'time_cycle_explanation': time_cycle.get('time_cycle_explanation'), 'quick_buy_state': time_cycle.get('quick_buy_state'), 'quick_buy_action': time_cycle.get('quick_buy_action'), 'best_buy_date': time_cycle.get('best_buy_date'), 'best_buy_raw_date': time_cycle.get('best_buy_raw_date'), 'best_buy_calendar_state': time_cycle.get('best_buy_calendar_state'), 'best_buy_calendar_verified': time_cycle.get('best_buy_calendar_verified'), 'best_buy_date_adjustment_days': time_cycle.get('best_buy_date_adjustment_days'), 'best_buy_date_basis': time_cycle.get('best_buy_date_basis'), 'best_buy_window_start': time_cycle.get('best_buy_window_start'), 'best_buy_window_end': time_cycle.get('best_buy_window_end'), 'best_buy_score': time_cycle.get('best_buy_score'), 'best_buy_confidence': time_cycle.get('best_buy_confidence'), 'best_buy_entry_low': time_cycle.get('best_buy_entry_low'), 'best_buy_entry_high': time_cycle.get('best_buy_entry_high'), 'best_buy_trigger': time_cycle.get('best_buy_trigger'), 'best_buy_stop_loss': time_cycle.get('best_buy_stop_loss'), 'best_buy_tp1': time_cycle.get('best_buy_tp1'), 'best_buy_tp2': time_cycle.get('best_buy_tp2'), 'best_buy_rr1': time_cycle.get('best_buy_rr1'), 'best_buy_rr2': time_cycle.get('best_buy_rr2'), 'best_buy_order_plan': time_cycle.get('best_buy_order_plan'), 'best_buy_reason': time_cycle.get('best_buy_reason'), 'best_buy_no_trade_condition': time_cycle.get('best_buy_no_trade_condition'), 'best_buy_summary': time_cycle.get('best_buy_summary'), 'eoff_version': time_cycle.get('eoff_version'), 'eoff_state': time_cycle.get('eoff_state'), 'eoff_reconstruction_score': time_cycle.get('eoff_reconstruction_score'), 'eoff_strength_label': time_cycle.get('eoff_strength_label'), 'eoff_signal_active': time_cycle.get('eoff_signal_active'), 'eoff_direction_bias': time_cycle.get('eoff_direction_bias'), 'eoff_time_power_score': time_cycle.get('eoff_time_power_score'), 'eoff_price_power_score': time_cycle.get('eoff_price_power_score'), 'eoff_pattern_score': time_cycle.get('eoff_pattern_score'), 'eoff_momentum_score': time_cycle.get('eoff_momentum_score'), 'eoff_astro_score': time_cycle.get('eoff_astro_score'),'eoff_core_astro_score': time_cycle.get('eoff_core_astro_score'), 'eoff_adaptive_astro_score': time_cycle.get('eoff_adaptive_astro_score'), 'eoff_adaptive_total_weight_pct': time_cycle.get('eoff_adaptive_total_weight_pct'), 'eoff_adaptive_active_factors': time_cycle.get('eoff_adaptive_active_factors'), 'eoff_adaptive_validation_state': time_cycle.get('eoff_adaptive_validation_state'), 'eoff_validation_path': time_cycle.get('eoff_validation_path'), 'eoff_fib_cluster_count': time_cycle.get('eoff_fib_cluster_count'), 'eoff_fib_unique_anchor_count': time_cycle.get('eoff_fib_unique_anchor_count'), 'eoff_fib_unique_anchor_ratio': time_cycle.get('eoff_fib_unique_anchor_ratio'), 'eoff_fib_dominant_anchor_share': time_cycle.get('eoff_fib_dominant_anchor_share'), 'eoff_unique_anchor_gate': time_cycle.get('eoff_unique_anchor_gate'), 'eoff_unique_anchor_signature': time_cycle.get('eoff_unique_anchor_signature'), 'eoff_historical_hit_rate': time_cycle.get('eoff_historical_hit_rate'), 'eoff_historical_baseline_rate': time_cycle.get('eoff_historical_baseline_rate'), 'eoff_historical_lift': time_cycle.get('eoff_historical_lift'), 'eoff_confluence_historical_hit_rate': time_cycle.get('eoff_confluence_historical_hit_rate'), 'eoff_confluence_historical_events': time_cycle.get('eoff_confluence_historical_events'), 'eoff_confluence_historical_lift': time_cycle.get('eoff_confluence_historical_lift'), 'eoff_historical_events': time_cycle.get('eoff_historical_events'), 'eoff_reversal_date': time_cycle.get('eoff_reversal_date'), 'eoff_ephemeris_state': time_cycle.get('eoff_ephemeris_state'), 'eoff_ephemeris_date': time_cycle.get('eoff_ephemeris_date'), 'eoff_astro_events': time_cycle.get('eoff_astro_events'), 'eoff_active_aspects': time_cycle.get('eoff_active_aspects'), 'eoff_retrograde_planets': time_cycle.get('eoff_retrograde_planets'), 'eoff_retrograde_transition_events': time_cycle.get('eoff_retrograde_transition_events'), 'eoff_stationary_planets': time_cycle.get('eoff_stationary_planets'), 'eoff_ingress_events': time_cycle.get('eoff_ingress_events'), 'eoff_moon_declination_deg': time_cycle.get('eoff_moon_declination_deg'), 'eoff_moon_phase': time_cycle.get('eoff_moon_phase'), 'eoff_sun_sign': time_cycle.get('eoff_sun_sign'), 'eoff_sun_annual_cycle_bias': time_cycle.get('eoff_sun_annual_cycle_bias'), 'eoff_roadmap_json': time_cycle.get('eoff_roadmap_json'), 'eoff_internal_weight_pct': time_cycle.get('eoff_internal_weight_pct'), 'eoff_explanation': time_cycle.get('eoff_explanation'), 'fundamental_coverage': coverage, 'fundamental_score': fund.get('fundamental_score'), 'fundamental_score_10': fundamental_score_10, 'fundamental_reliability': fundamental_reliability, 'fundamental_data_grade': fundamental_data_grade, 'fundamental_source_count': source_count, 'fundamental_source_families': source_families, 'fundamental_history_quarters': history_quarters, 'fundamental_history_years': history_years, 'fundamental_history_coverage': history_coverage, 'fundamental_consensus_score': consensus_score, 'fundamental_conflicts': fundamental_conflicts, 'fundamental_official_reference': official_reference, 'fundamental_official_verified': official_verified, 'statement_age_days': statement_age_days, 'statement_current': statement_current, 'statement_age_state': 'CURRENT' if statement_current else 'UNKNOWN' if not np.isfinite(statement_age_days) else 'STALE', 'peg_valid_for_valuation': bool(np.isfinite(peg) and peg > 0), 'fundamental_data_gate': fundamental_data_gate, 'grade_a_gate': grade_a_gate, 'severe_fundamental_flags': severe_flags, 'revenue_growth': revenue_growth, 'earnings_growth': earnings_growth, 'roe': roe, 'roa': roa, 'net_margin': net_margin, 'debt_equity': debt_equity, 'current_ratio': current_ratio, 'cash_to_debt': cash_to_debt, 'operating_cash_flow': ocf, 'free_cash_flow': fcf, 'cash_conversion_ttm': cash_conversion, 'positive_ocf_ratio': positive_ocf_ratio, 'positive_earnings_ratio': positive_earnings_ratio, 'margin_stability': margin_stability, 'share_dilution_yoy': share_dilution, 'roic_proxy': roic_proxy, 'net_debt_ebitda': net_debt_ebitda, 'interest_coverage': interest_coverage, 'solvency_coverage': round(solvency_coverage, 1), 'fundamental_model': fundamental_model, 'car': car, 'npl_gross': npl_gross, 'ldr': ldr, 'bank_prudential_gate': bank_prudential_gate, 'peg_ratio': peg, 'fcf_yield': fcf_yield, 'market_cap': market_cap, 'last_price': close, 'roc60': roc60, 'roc120': roc120, 'relative_strength60': rs60, 'distance_52w_high': dist_high, 'silent_accumulation_version': accumulation_profile.get('silent_accumulation_version'), 'silent_accumulation_score': accumulation, 'silent_accumulation_raw_score': accumulation_profile.get('silent_accumulation_raw_score'), 'silent_accumulation_liquidity_adjustment': accumulation_profile.get('silent_accumulation_liquidity_adjustment'), 'silent_accumulation_liquidity_min_confirmation': accumulation_profile.get('silent_accumulation_liquidity_min_confirmation'), 'silent_accumulation_calibration_policy': accumulation_profile.get('silent_accumulation_calibration_policy'), 'liquidity_bucket': accumulation_profile.get('liquidity_bucket'), 'silent_accumulation_base_score_v2': accumulation_profile.get('silent_accumulation_base_score_v2'), 'silent_accumulation_v3_adjustment': accumulation_profile.get('silent_accumulation_v3_adjustment'), 'silent_accumulation_state': accumulation_profile.get('silent_accumulation_state'), 'silent_accumulation_confidence': accumulation_profile.get('silent_accumulation_confidence'), 'silent_accumulation_reason': accumulation_profile.get('silent_accumulation_reason'), 'up_down_value_ratio20': up_down, 'up_down_value_ratio60': accumulation_profile.get('up_down_value_ratio60'), 'weighted_close_location20': accumulation_profile.get('weighted_close_location20'), 'cmf20_accumulation': accumulation_profile.get('cmf20_accumulation'), 'cmf60_accumulation': accumulation_profile.get('cmf60_accumulation'), 'adl_slope20': accumulation_profile.get('adl_slope20'), 'obv_slope20': accumulation_profile.get('obv_slope20'), 'pullback_volume_ratio20': accumulation_profile.get('pullback_volume_ratio20'), 'accumulation_days20': accumulation_profile.get('accumulation_days20'), 'distribution_days20': accumulation_profile.get('distribution_days20'), 'churning_support_days20': accumulation_profile.get('churning_support_days20'), 'absorption_confirmed_days20': accumulation_profile.get('absorption_confirmed_days20'), 'failed_absorption_days20': accumulation_profile.get('failed_absorption_days20'), 'effort_result_absorption20': accumulation_profile.get('effort_result_absorption20'), 'effort_result_distribution20': accumulation_profile.get('effort_result_distribution20'), 'lower_wick_support_days20': accumulation_profile.get('lower_wick_support_days20'), 'persistent_bid_score': accumulation_profile.get('persistent_bid_score'), 'supply_pressure_ratio20': accumulation_profile.get('supply_pressure_ratio20'), 'silent_accumulation_data_coverage': accumulation_profile.get('silent_accumulation_data_coverage'), 'adtv20_idr': adtv, 'active_setup': sig.get('setup', ''), 'technical_entry_state': technical_entry_state, 'entry_low': sig.get('entry_low', np.nan), 'entry_high': sig.get('entry_high', np.nan), 'entry': sig.get('entry', np.nan), 'entry_type': sig.get('entry_type', ''), 'action': sig.get('action', ''), 'trigger': sig.get('trigger', np.nan), 'stockbit_trigger_price': sig.get('stockbit_trigger_price', np.nan), 'stockbit_limit_price': sig.get('stockbit_limit_price', np.nan), 'stockbit_order_price': sig.get('stockbit_order_price', np.nan), 'order_instruction': sig.get('order_instruction', ''), 'execution_timing': sig.get('execution_timing', ''), 'stop_loss': sig.get('stop_loss', np.nan), 'tp1': sig.get('tp1', np.nan), 'tp2': sig.get('tp2', np.nan), 'rr1': sig.get('rr1', np.nan), 'rr2': sig.get('rr2', np.nan), 'entry_plan_min_rr1': safe_number(sig.get('entry_plan_min_rr1'), cfg.min_rr1), 'entry_plan_min_rr2': safe_number(sig.get('entry_plan_min_rr2'), cfg.min_rr2), 'compounding_state': compounding_state, 'review_action': review_action, 'profit_allocation_pct': 100.0 * cfg.multibagger_profit_allocation_pct, 'max_position_pct_equity': 100.0 * max_allocation, 'horizon': '12–36 months; quarterly review', 'red_flags': ' • '.join(part for part in (red_flags, fundamental_conflicts, governance_flags, project_execution_flags) if part), 'note': 'Bank grade A requires CAR/NPL/LDR history plus verified IDX/XBRL and multi-source consensus' if is_financial else 'Candidate ranking, not a forecast or guaranteed multiple'})
        rows[-1].update({
            **confidence_profile,
            **reason_profile,
            **archetype_profile,
            'fundamental_history_source_count': int(
                safe_number(
                    fund.get('fundamental_history_source_count'),
                    source_count,
                )
            ),
            'fundamental_snapshot_source_count': int(
                safe_number(
                    fund.get('fundamental_snapshot_source_count'),
                    1 if coverage > 0 else 0,
                )
            ),
            'growth_compounder_score': round(total, 1),
            'growth_compounder_base_score': round(base_quality, 1),
            'growth_research_eligible': bool(growth_research_eligible),
            'research_eligible': bool(research_eligible),
            'research_eligibility_reason': research_eligibility_reason,
            'critical_research_flags': bool(critical_research_flags),
            'operational_recovery_flags': bool(operational_recovery_flags),
            'confidence_adjusted_multibagger_score': round(confidence_adjusted_score, 1),
            'confidence_penalty_pct': round(100.0 * (1.0 - confidence_adjusted_score / total), 1) if total > 0 else 0.0,
            'growth_persistence_pillar': round(growth_persistence_pillar, 1),
            'profitability_pillar': round(profitability_pillar, 1),
            'cash_conversion_pillar': round(cashflow_pillar, 1),
            'balance_sheet_safety_pillar': round(safety_pillar, 1),
            'reinvestment_runway_pillar': round(reinvestment_runway_pillar, 1),
            'growth_pillar_coverage_pct': round(
                growth_persistence_pillar_coverage, 1,
            ),
            'profitability_pillar_coverage_pct': round(
                profitability_pillar_coverage, 1,
            ),
            'cashflow_pillar_coverage_pct': round(
                cashflow_pillar_coverage, 1,
            ),
            'safety_pillar_coverage_pct': round(
                safety_pillar_coverage, 1,
            ),
            'runway_pillar_coverage_pct': round(
                reinvestment_runway_pillar_coverage, 1,
            ),
            'valuation_pillar_coverage_pct': round(
                valuation_metric_coverage, 1,
            ),
            'multibagger_metric_coverage_pct': round(
                multibagger_metric_coverage, 1,
            ),
            'multibagger_metric_data_gate': bool(metric_data_gate),
            'quality_pillar_coverage_pct': round(quality_pillar_coverage, 1),
            'quality_pillars_strong': int(quality_pillars_strong),
            'quality_pillars_critical': int(quality_pillars_critical),
            'quality_pillar_gate': quality_pillar_gate,
            'quality_pillar_a_gate': quality_pillar_a_gate,
            'gross_profitability': gross_profitability,
            'gross_margin': gross_margin,
            'gross_profit_growth': gross_profit_growth,
            'fundamental_inflection_score': fundamental_inflection,
            'fundamental_inflection_coverage_pct': fundamental_inflection_coverage,
            'revenue_growth_acceleration': revenue_growth_acceleration,
            'earnings_growth_acceleration': earnings_growth_acceleration,
            'gross_margin_change_yoy': gross_margin_change_yoy,
            'cash_conversion_change_yoy': cash_conversion_change_yoy,
            'accruals_to_assets': accruals_to_assets,
            'leverage_change_yoy': leverage_change_yoy,
            'time_cycle_evaluation_mode': (
                'FULL_CANDIDATE' if run_full_timing else 'DEFERRED_LOW_QUALITY'
            ),
            'silent_accumulation_v4_adjustment': accumulation_profile.get('silent_accumulation_v4_adjustment'),
            'silent_accumulation_adaptive_adjustment': accumulation_profile.get('silent_accumulation_adaptive_adjustment'),
            'accumulation_persistence_score': accumulation_profile.get('accumulation_persistence_score'),
            'accumulation_positive_windows_pct': accumulation_profile.get('accumulation_positive_windows_pct'),
            'accumulation_longest_run': accumulation_profile.get('accumulation_longest_run'),
            'accumulation_regime': accumulation_profile.get('accumulation_regime'),
            'accumulation_weight_profile': accumulation_profile.get('accumulation_weight_profile'),
        })
        rows[-1].update({
            'best_buy_target_basis': time_cycle.get('best_buy_target_basis'),
            'eoff_public_validation_state': time_cycle.get('eoff_public_validation_state'),
            'eoff_public_validation_method': time_cycle.get('eoff_public_validation_method'),
            'eoff_public_directional_events': time_cycle.get('eoff_public_directional_events'),
            'eoff_public_reversal_hit_rate': time_cycle.get('eoff_public_reversal_hit_rate'),
            'eoff_public_baseline_rate': time_cycle.get('eoff_public_baseline_rate'),
            'eoff_public_lift': time_cycle.get('eoff_public_lift'),
            'eoff_public_forward_hit_rate': time_cycle.get('eoff_public_forward_hit_rate'),
            'eoff_public_median_directional_return_pct': time_cycle.get('eoff_public_median_directional_return_pct'), 'eoff_declination_validation_state': time_cycle.get('eoff_declination_validation_state'), 'eoff_declination_oos_events': time_cycle.get('eoff_declination_oos_events'), 'eoff_declination_oos_lift': time_cycle.get('eoff_declination_oos_lift'), 'eoff_declination_oos_forward_hit_rate': time_cycle.get('eoff_declination_oos_forward_hit_rate'), 'eoff_declination_oos_median_return_pct': time_cycle.get('eoff_declination_oos_median_return_pct'), 'eoff_declination_weight_pct': time_cycle.get('eoff_declination_weight_pct'), 'eoff_declination_current_active': time_cycle.get('eoff_declination_current_active'), 'eoff_declination_current_score': time_cycle.get('eoff_declination_current_score'), 'eoff_ingress_validation_state': time_cycle.get('eoff_ingress_validation_state'), 'eoff_ingress_oos_events': time_cycle.get('eoff_ingress_oos_events'), 'eoff_ingress_oos_lift': time_cycle.get('eoff_ingress_oos_lift'), 'eoff_ingress_oos_forward_hit_rate': time_cycle.get('eoff_ingress_oos_forward_hit_rate'), 'eoff_ingress_oos_median_return_pct': time_cycle.get('eoff_ingress_oos_median_return_pct'), 'eoff_ingress_weight_pct': time_cycle.get('eoff_ingress_weight_pct'), 'eoff_ingress_current_active': time_cycle.get('eoff_ingress_current_active'), 'eoff_ingress_current_score': time_cycle.get('eoff_ingress_current_score'), 'eoff_retrograde_validation_state': time_cycle.get('eoff_retrograde_validation_state'), 'eoff_retrograde_oos_events': time_cycle.get('eoff_retrograde_oos_events'), 'eoff_retrograde_oos_lift': time_cycle.get('eoff_retrograde_oos_lift'), 'eoff_retrograde_oos_forward_hit_rate': time_cycle.get('eoff_retrograde_oos_forward_hit_rate'), 'eoff_retrograde_oos_median_return_pct': time_cycle.get('eoff_retrograde_oos_median_return_pct'), 'eoff_retrograde_weight_pct': time_cycle.get('eoff_retrograde_weight_pct'), 'eoff_retrograde_current_active': time_cycle.get('eoff_retrograde_current_active'), 'eoff_retrograde_current_score': time_cycle.get('eoff_retrograde_current_score'), 'eoff_sun_validation_state': time_cycle.get('eoff_sun_validation_state'), 'eoff_sun_oos_events': time_cycle.get('eoff_sun_oos_events'), 'eoff_sun_oos_lift': time_cycle.get('eoff_sun_oos_lift'), 'eoff_sun_oos_forward_hit_rate': time_cycle.get('eoff_sun_oos_forward_hit_rate'), 'eoff_sun_oos_median_return_pct': time_cycle.get('eoff_sun_oos_median_return_pct'), 'eoff_sun_weight_pct': time_cycle.get('eoff_sun_weight_pct'), 'eoff_sun_current_active': time_cycle.get('eoff_sun_current_active'), 'eoff_sun_current_score': time_cycle.get('eoff_sun_current_score'),
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        # Preserve the economic peer label so Multibagger AI can compare a
        # general issuer with its sector when at least five peers are present.
        result['sector'] = result['ticker'].map(
            lambda ticker: safe_text(f_map.get(safe_text(ticker).upper(), {}).get('sector'))
        )
        result = _attach_selector_overlay(result, selector_ranking)
        result = attach_narrative_profiles(result, narrative_profiles)
        for column, default in (
            ('selector_relative_strength_score', 50.0),
            ('selector_trend_score', 50.0),
            ('multibagger_timing_selector_score', 50.0),
        ):
            if column not in result:
                result[column] = default
        growth_quality = pd.to_numeric(
            result.get('confidence_adjusted_multibagger_score', result.get('multibagger_score')),
            errors='coerce',
        )
        turnaround_raw = pd.to_numeric(
            result.get('turnaround_recovery_score'), errors='coerce',
        )
        research_confidence = pd.to_numeric(
            result.get('overall_research_confidence'), errors='coerce',
        ).fillna(0.0).clip(0.0, 100.0)
        result['confidence_adjusted_turnaround_score'] = (
            turnaround_raw * (0.75 + 0.25 * research_confidence / 100.0)
        ).clip(0.0, 100.0).round(1)
        turnaround_quality = pd.to_numeric(
            result['confidence_adjusted_turnaround_score'], errors='coerce',
        )
        result['sector_peer_count'] = (
            result.groupby('sector')['ticker'].transform('count')
            .where(result['sector'].fillna('').astype(str).str.len().gt(0), 0)
            .fillna(0).astype(int)
        )
        result['sector_relative_quality_score'] = 50.0
        result['turnaround_sector_relative_score'] = 50.0
        growth_sector = (
            result['sector'].fillna('').astype(str).str.len().gt(0)
            & result['sector_peer_count'].ge(5)
            & growth_quality.notna()
        )
        turnaround_sector = (
            result['sector'].fillna('').astype(str).str.len().gt(0)
            & result['sector_peer_count'].ge(5)
            & turnaround_quality.notna()
        )
        if growth_sector.any():
            result.loc[growth_sector, 'sector_relative_quality_score'] = (
                growth_quality.loc[growth_sector]
                .groupby(result.loc[growth_sector, 'sector'])
                .rank(method='average', pct=True)
                .mul(100.0)
            )
        if turnaround_sector.any():
            result.loc[
                turnaround_sector, 'turnaround_sector_relative_score'
            ] = (
                turnaround_quality.loc[turnaround_sector]
                .groupby(result.loc[turnaround_sector, 'sector'])
                .rank(method='average', pct=True)
                .mul(100.0)
            )
        result['sector_relative_state'] = np.where(
            result['sector_peer_count'].ge(5),
            'SECTOR_RELATIVE_ACTIVE',
            'INSUFFICIENT_SECTOR_PEERS',
        )
        growth_peer_quality = pd.to_numeric(
            result['sector_relative_quality_score'], errors='coerce',
        ).fillna(50.0)
        turnaround_peer_quality = pd.to_numeric(
            result['turnaround_sector_relative_score'], errors='coerce',
        ).fillna(50.0)
        growth_for_selection = np.where(
            growth_sector,
            0.90 * growth_quality.fillna(0.0) + 0.10 * growth_peer_quality,
            growth_quality,
        )
        growth_for_selection = pd.Series(
            growth_for_selection, index=result.index, dtype=float,
        )
        turnaround_for_selection = np.where(
            turnaround_sector,
            0.90 * turnaround_quality.fillna(0.0)
            + 0.10 * turnaround_peer_quality,
            turnaround_quality,
        )
        turnaround_for_selection = pd.Series(
            turnaround_for_selection, index=result.index, dtype=float,
        )
        raw_silent = pd.to_numeric(
            result.get('silent_accumulation_score'), errors='coerce',
        ).fillna(50.0)
        silent_confidence = pd.to_numeric(
            result.get('silent_accumulation_confidence'), errors='coerce',
        ).fillna(0.0)
        silent_state = result.get(
            'silent_accumulation_state', pd.Series('', index=result.index),
        )
        result['effective_silent_accumulation_score'] = [
            _effective_silent_accumulation_score(score, confidence, state)
            for score, confidence, state in zip(
                raw_silent, silent_confidence, silent_state,
            )
        ]
        effective_silent = pd.to_numeric(
            result['effective_silent_accumulation_score'], errors='coerce',
        ).fillna(50.0)
        timing_selector = pd.to_numeric(
            result.get('multibagger_timing_selector_score'), errors='coerce',
        ).fillna(50.0)
        # Growth and recovery theses have different evidence clocks and are
        # ranked in separate lanes.  Setup detection remains downstream.
        result['growth_compounder_selection_score'] = np.where(
            growth_quality.notna(),
            growth_for_selection.fillna(0.0),
            np.nan,
        )
        result['turnaround_selection_score'] = np.where(
            turnaround_quality.notna(),
            turnaround_for_selection.fillna(0.0),
            np.nan,
        )
        growth_narrative_adjustment = pd.to_numeric(
            result.get(
                'narrative_growth_rank_adjustment',
                pd.Series(0.0, index=result.index),
            ),
            errors='coerce',
        ).fillna(0.0)
        turnaround_narrative_adjustment = pd.to_numeric(
            result.get(
                'narrative_turnaround_rank_adjustment',
                pd.Series(0.0, index=result.index),
            ),
            errors='coerce',
        ).fillna(0.0)
        result['growth_compounder_selection_score_pre_narrative'] = (
            result['growth_compounder_selection_score']
        )
        result['turnaround_selection_score_pre_narrative'] = (
            result['turnaround_selection_score']
        )
        result['growth_compounder_selection_score'] = (
            pd.to_numeric(
                result['growth_compounder_selection_score'],
                errors='coerce',
            )
            + growth_narrative_adjustment
        ).clip(0.0, 100.0)
        result['turnaround_selection_score'] = (
            pd.to_numeric(
                result['turnaround_selection_score'],
                errors='coerce',
            )
            + turnaround_narrative_adjustment
        ).clip(0.0, 100.0)
        lane = result.get(
            'multibagger_lane',
            pd.Series('GROWTH_COMPOUNDER', index=result.index),
        ).fillna('GROWTH_COMPOUNDER').astype(str)
        result['multibagger_selection_score'] = np.where(
            lane.eq('TURNAROUND_CYCLICAL'),
            result['turnaround_selection_score'],
            result['growth_compounder_selection_score'],
        )
        for column in (
            'growth_compounder_selection_score',
            'turnaround_selection_score',
            'multibagger_selection_score',
        ):
            result[column] = pd.to_numeric(
                result[column], errors='coerce',
            ).round(1)
        selector_reason = result.get(
            'selector_selected_reason', pd.Series('', index=result.index),
        ).fillna('').astype(str)
        fundamental_reason = result.get(
            'top_positive_drivers', pd.Series('', index=result.index),
        ).fillna('').astype(str)
        narrative_reason = result.get(
            'narrative_primary_reason', pd.Series('', index=result.index),
        ).fillna('').astype(str)
        lane_reason = [
            (
                f"Turnaround {safe_number(turn_score, 0.0):.1f}; "
                f"{safe_text(turn_state) or 'recovery gate pending'}; "
                f"effective silent accumulation {safe_number(silent_score, 50.0):.1f}"
                if safe_text(lane_value).upper() == 'TURNAROUND_CYCLICAL'
                else (
                    f"Growth compounder {safe_number(growth_score_value, 0.0):.1f}; "
                    f"reinvestment runway {safe_number(runway, 0.0):.1f}; "
                    f"effective silent accumulation {safe_number(silent_score, 50.0):.1f}"
                )
            )
            for (
                lane_value, turn_score, turn_state, growth_score_value,
                runway, silent_score,
            ) in zip(
                result.get(
                    'multibagger_lane',
                    pd.Series('GROWTH_COMPOUNDER', index=result.index),
                ),
                result.get(
                    'turnaround_recovery_score',
                    pd.Series(np.nan, index=result.index),
                ),
                result.get(
                    'turnaround_research_state',
                    pd.Series('', index=result.index),
                ),
                result.get(
                    'growth_compounder_score',
                    pd.Series(np.nan, index=result.index),
                ),
                result.get(
                    'reinvestment_runway_pillar',
                    pd.Series(np.nan, index=result.index),
                ),
                result['effective_silent_accumulation_score'],
            )
        ]
        result['selected_reason'] = [
            ' • '.join(
                part for part in (
                    lane_summary, fundamental, narrative, technical,
                )
                if safe_text(part)
            )
            or 'Belum ada alasan pemilihan yang cukup kuat'
            for lane_summary, fundamental, narrative, technical in zip(
                lane_reason, fundamental_reason, narrative_reason,
                selector_reason,
            )
        ]
        entry_state = result.get(
            'technical_entry_state', pd.Series('NO_ACTIVE_ENTRY_SETUP', index=result.index),
        ).fillna('NO_ACTIVE_ENTRY_SETUP').astype(str)
        result['not_entry_reason'] = [
            (
                'Belum ada setup/entry zone aktif; kandidat tetap dipantau karena lolos seleksi strategis'
                if safe_text(state).upper() in {'NO_ACTIVE_ENTRY_SETUP', 'WATCH_ONLY', ''}
                else f'Entry belum dieksekusi: status teknikal {safe_text(state)}'
            )
            for state in entry_state
        ]
        entry_price = pd.to_numeric(
            result.get('entry', pd.Series(np.nan, index=result.index)), errors='coerce',
        )
        result['trigger_waiting'] = [
            f'Tunggu konfirmasi entry/trigger di sekitar {value:,.0f}'
            if np.isfinite(value) else 'Tunggu setup, entry zone, dan konfirmasi volume'
            for value in entry_price
        ]
        stop = pd.to_numeric(
            result.get('stop_loss', pd.Series(np.nan, index=result.index)), errors='coerce',
        )
        result['invalidation_reason'] = [
            (
                (
                    f'Tesis timing invalid bila struktur menembus SL {value:,.0f}; '
                    'tesis recovery invalid bila infleksi laba/margin/cash conversion berbalik'
                    if np.isfinite(value)
                    else 'Tesis recovery invalid bila infleksi laba/margin/cash conversion berbalik atau safety memburuk'
                )
                if safe_text(lane_value).upper() == 'TURNAROUND_CYCLICAL'
                else (
                    f'Tesis timing invalid bila struktur menembus SL {value:,.0f}; '
                    'tesis compounder ditinjau ulang bila ROIC/runway/cash conversion memburuk'
                    if np.isfinite(value)
                    else 'Tesis compounder ditinjau ulang bila ROIC, reinvestment runway, cash conversion, atau safety memburuk'
                )
            )
            for value, lane_value in zip(
                stop,
                result.get(
                    'multibagger_lane',
                    pd.Series('GROWTH_COMPOUNDER', index=result.index),
                ),
            )
        ]
        selector_risks = result.get(
            'selector_selection_risks', pd.Series('', index=result.index),
        ).fillna('').astype(str)
        fundamental_risks = result.get(
            'top_negative_drivers', pd.Series('', index=result.index),
        ).fillna('').astype(str)
        narrative_risks = result.get(
            'narrative_primary_risk', pd.Series('', index=result.index),
        ).fillna('').astype(str)
        narrative_blocks = result.get(
            'narrative_hard_block', pd.Series(False, index=result.index),
        ).fillna(False).map(truthy)
        result['primary_risk'] = [
            (
                safe_text(narrative)
                if blocked
                else safe_text(fundamental) or safe_text(technical)
                or safe_text(narrative)
                or 'Risiko utama belum terstruktur'
            ).split(' • ')[0]
            for fundamental, technical, narrative, blocked in zip(
                fundamental_risks, selector_risks,
                narrative_risks, narrative_blocks,
            )
        ]
        result['selection_before_setup'] = True
        rank = {'MULTIBAGGER_A_CANDIDATE': 0, 'MULTIBAGGER_B_CANDIDATE': 1, 'MULTIBAGGER_WATCHLIST': 2, 'MULTIBAGGER_NOT_QUALIFIED': 7, 'DATA_NOT_SCORED': 9}
        result['_research_rank'] = np.where(
            result.get(
                'research_eligible',
                pd.Series(False, index=result.index),
            ).fillna(False).astype(bool),
            0,
            1,
        )
        result['_rank'] = result['multibagger_status'].map(rank).fillna(9)
        result = result.sort_values(
            ['_research_rank', '_rank', 'multibagger_selection_score',
             'effective_silent_accumulation_score',
             'selector_relative_strength_score', 'selector_trend_score', 'adtv20_idr'],
            ascending=[True, True, False, False, False, False, False],
            kind='stable',
            na_position='last',
        ).drop(columns=['_research_rank', '_rank']).reset_index(drop=True)
        result['multibagger_selection_rank'] = np.arange(1, len(result) + 1)
        result['growth_compounder_rank'] = np.nan
        result['turnaround_rank'] = np.nan
        growth_indices = result.index[
            result.get(
                'research_eligible',
                pd.Series(False, index=result.index),
            ).fillna(False).astype(bool)
            & result.get(
                'multibagger_lane',
                pd.Series('', index=result.index),
            ).eq('GROWTH_COMPOUNDER')
        ].tolist()
        turnaround_indices = result.index[
            result.get(
                'research_eligible',
                pd.Series(False, index=result.index),
            ).fillna(False).astype(bool)
            & result.get(
                'multibagger_lane',
                pd.Series('', index=result.index),
            ).eq('TURNAROUND_CYCLICAL')
        ].tolist()
        result.loc[growth_indices, 'growth_compounder_rank'] = np.arange(
            1, len(growth_indices) + 1,
        )
        result.loc[turnaround_indices, 'turnaround_rank'] = np.arange(
            1, len(turnaround_indices) + 1,
        )
        result = allocate_multibagger_capital(result, cfg)
        result['portfolio_allocation_eligible'] = result.get(
            'allocation_eligible',
            pd.Series(False, index=result.index),
        ).fillna(False).astype(bool)
        result['_research_rank'] = np.where(
            result.get(
                'research_eligible',
                pd.Series(False, index=result.index),
            ).fillna(False).astype(bool),
            0,
            1,
        )
        result['_rank'] = result['multibagger_status'].map(rank).fillna(9)
        # Capital priority is a deployment annotation, never the selector.
        result = result.sort_values(
            ['_research_rank', '_rank', 'multibagger_selection_score',
             'effective_silent_accumulation_score',
             'selector_relative_strength_score', 'selector_trend_score'],
            ascending=[True, True, False, False, False, False],
            kind='stable',
            na_position='last',
        ).drop(columns=['_research_rank', '_rank']).reset_index(drop=True)
        result['multibagger_selection_rank'] = np.arange(1, len(result) + 1)
    return result




















def build_multibagger_diagnostic_views(
    multibagger: pd.DataFrame | None,
    *,
    expected_ticker_count: int | None = None,
    limit: int = 20,
) -> dict[str, pd.DataFrame]:
    """Expose coverage, near-misses, and pending-data queues without relaxing gates.

    Qualified Growth/Turnaround tables are intentionally strict. This helper
    prevents an empty qualified table from being misread as "the universe has
    no potential": candidates with incomplete evidence remain visible as
    research queues, while every returned near-miss stays allocation-ineligible.
    """
    empty = {
        'coverage': pd.DataFrame(),
        'gate_failures': pd.DataFrame(),
        'growth_near_miss': pd.DataFrame(),
        'turnaround_near_miss': pd.DataFrame(),
        'growth_research_queue': pd.DataFrame(),
        'turnaround_research_queue': pd.DataFrame(),
        'data_pending': pd.DataFrame(),
    }
    if not isinstance(multibagger, pd.DataFrame) or multibagger.empty:
        return empty

    frame = multibagger.copy()
    total = int(frame.get(
        'ticker', pd.Series(index=frame.index, dtype=str),
    ).astype(str).replace('', np.nan).nunique())
    requested = max(total, int(expected_ticker_count or 0))
    denominator = max(requested, 1)

    def numeric(name: str, default: float = np.nan) -> pd.Series:
        source = frame.get(name, pd.Series(default, index=frame.index))
        return pd.to_numeric(source, errors='coerce')

    def boolean(name: str) -> pd.Series:
        source = frame.get(name, pd.Series(False, index=frame.index))
        return source.map(
            lambda value: (
                False
                if value is None
                or (
                    isinstance(value, (float, np.floating))
                    and not np.isfinite(value)
                )
                else truthy(value)
            )
        ).astype(bool)

    coverage = numeric('fundamental_coverage', 0.0).fillna(0.0)
    fundamental_score = numeric('fundamental_score')
    history_sources = numeric('fundamental_history_source_count')
    if history_sources.isna().all():
        history_sources = numeric('fundamental_source_count', 0.0)
    history_sources = history_sources.fillna(0.0)
    history_periods = (
        numeric('fundamental_history_quarters', 0.0).fillna(0.0)
        + numeric('fundamental_history_years', 0.0).fillna(0.0)
    )
    grade = frame.get(
        'fundamental_data_grade', pd.Series('D', index=frame.index),
    ).fillna('D').astype(str).str.upper()
    research_eligible = boolean('research_eligible')
    allocation_eligible = boolean('portfolio_allocation_eligible')
    snapshot_available = coverage.gt(0.0) & fundamental_score.notna()
    history_available = history_sources.gt(0.0) | history_periods.gt(0.0)
    grade_abc = grade.isin({'A', 'B', 'C'})
    metric_coverage = numeric('multibagger_metric_coverage_pct')
    if metric_coverage.isna().all():
        metric_coverage = coverage.copy()
    metric_coverage = metric_coverage.fillna(0.0).clip(0.0, 100.0)
    metric_threshold_ready = metric_coverage.ge(65.0)
    statement_ready = (
        boolean('statement_current')
        if 'statement_current' in frame
        else pd.Series(True, index=frame.index)
    )
    if 'multibagger_metric_data_gate' in frame:
        metric_data_gate = boolean('multibagger_metric_data_gate')
    else:
        metric_data_gate = (
            snapshot_available
            & history_available
            & metric_threshold_ready
            & statement_ready
        )
    snapshot_pending = ~snapshot_available
    history_pending = ~history_available
    metric_pending = ~metric_threshold_ready
    statement_pending = snapshot_available & ~statement_ready
    other_data_gate_pending = (
        ~metric_data_gate
        & ~snapshot_pending
        & ~history_pending
        & ~metric_pending
        & ~statement_pending
    )
    evidence_pending = (
        snapshot_pending
        | history_pending
        | metric_pending
        | statement_pending
        | other_data_gate_pending
    )
    data_pending = evidence_pending
    scored_blocked = snapshot_available & ~research_eligible & ~evidence_pending

    coverage_rows = [
        (
            'Universe diminta', requested,
            'Jumlah ticker pada CSV; selisih dengan OHLCV harus diaudit.',
        ),
        (
            'OHLCV siap dinilai', total,
            'Ticker yang memiliki bar cukup untuk membentuk baris Multibagger.',
        ),
        (
            'Snapshot fundamental tersedia', int(snapshot_available.sum()),
            'Snapshot cukup untuk antrean riset, belum sama dengan histori laporan terverifikasi.',
        ),
        (
            'Histori laporan tersedia', int(history_available.sum()),
            'Minimal satu sumber/periode laporan historis berhasil dinormalisasi.',
        ),
        (
            'Coverage metrik Multibagger memadai', int(metric_threshold_ready.sum()),
            'Metrik fundamental minimum teramati; nilai kosong tidak diperlakukan sebagai nol.',
        ),
        (
            'Data grade A–C', int(grade_abc.sum()),
            'Kualitas data minimum yang dapat melewati gate riset.',
        ),
        (
            'Lolos gate riset', int(research_eligible.sum()),
            'Growth Compounder atau Turnaround/Cyclical yang benar-benar qualified.',
        ),
        (
            'Layak alokasi modal', int(allocation_eligible.sum()),
            'Qualified sekaligus memenuhi seluruh gate alokasi.',
        ),
        (
            'Scored tetapi masih terblokir', int(scored_blocked.sum()),
            'Near-miss; wajib memperbaiki evidence/quality sebelum dipromosikan.',
        ),
        (
            'Menunggu evidence fundamental', int(data_pending.sum()),
            'Snapshot, histori, atau coverage metrik belum cukup; belum boleh disimpulkan buruk.',
        ),
    ]
    coverage_view = pd.DataFrame(
        coverage_rows, columns=['stage', 'count', 'meaning'],
    )
    coverage_view['pct_of_requested'] = (
        100.0 * coverage_view['count'] / denominator
    ).round(1)
    coverage_view = coverage_view[
        ['stage', 'count', 'pct_of_requested', 'meaning']
    ]

    reason_series = frame.get(
        'research_eligibility_reason', pd.Series('', index=frame.index),
    ).fillna('').astype(str)
    reason_records: list[dict[str, Any]] = []
    for index in frame.index[~research_eligible]:
        if snapshot_pending.loc[index]:
            reasons = ['FUNDAMENTAL_SNAPSHOT_PENDING']
        elif metric_pending.loc[index]:
            reasons = [
                'MULTIBAGGER_METRIC_COVERAGE_PENDING_'
                f'{metric_coverage.loc[index]:.0f}PCT'
            ]
        elif history_pending.loc[index]:
            reasons = ['STATEMENT_HISTORY_PENDING']
        elif statement_pending.loc[index]:
            reasons = ['STATEMENT_STALE_OR_DATE_UNKNOWN']
        elif other_data_gate_pending.loc[index]:
            reasons = ['MULTIBAGGER_DATA_CONTRACT_GATE_PENDING']
        else:
            text = reason_series.loc[index].strip()
            reasons = [
                token.strip()
                for token in re.split(r'\s*(?:\||•)\s*', text)
                if token.strip()
            ] or ['QUALITY_OR_RECOVERY_GATE_PENDING']
        ticker = safe_text(frame.at[index, 'ticker']) if 'ticker' in frame else ''
        for reason in dict.fromkeys(reasons):
            reason_records.append({'ticker': ticker, 'gate_reason': reason})
    gate_failures = pd.DataFrame(reason_records)
    if not gate_failures.empty:
        gate_failures = (
            gate_failures.groupby('gate_reason', as_index=False)
            .agg(ticker_count=('ticker', 'nunique'))
            .sort_values(['ticker_count', 'gate_reason'], ascending=[False, True])
            .reset_index(drop=True)
        )
        gate_failures['pct_of_prepared'] = (
            100.0 * gate_failures['ticker_count'] / max(total, 1)
        ).round(1)

    effective_silent = numeric(
        'effective_silent_accumulation_score',
    ).fillna(numeric('silent_accumulation_score', 50.0)).fillna(50.0)
    frame['effective_silent_accumulation_score'] = effective_silent
    trend = numeric('selector_trend_score', 50.0).fillna(50.0)
    relative_strength = numeric(
        'selector_relative_strength_score', 50.0,
    ).fillna(50.0)
    narrative_priority = numeric(
        'narrative_flow_effective_score', 50.0,
    ).fillna(50.0)
    adtv = numeric('adtv20_idr', 0.0).fillna(0.0).clip(lower=0.0)
    liquidity = pd.Series(0.0, index=frame.index)
    positive_liquidity = adtv.gt(0.0)
    liquidity.loc[positive_liquidity] = (
        20.0
        + 50.0
        * np.log10(
            adtv.loc[positive_liquidity].clip(lower=250_000_000.0)
            / 250_000_000.0
        )
        / np.log10(40.0)
    ).clip(0.0, 100.0)
    frame['data_refresh_priority_score'] = (
        0.30 * effective_silent
        + 0.20 * relative_strength
        + 0.20 * trend
        + 0.15 * liquidity
        + 0.15 * narrative_priority
    ).round(1)
    growth_selection = numeric(
        'growth_compounder_selection_score',
    )
    turnaround_selection = numeric(
        'turnaround_selection_score',
    )
    fundamental_quality_for_queue = (
        50.0
        + metric_coverage.div(100.0)
        * (fundamental_score.fillna(50.0).clip(0.0, 100.0) - 50.0)
    )
    # This score only orders the due-diligence queue. Snapshot quality and
    # coverage receive explicit weight so an exciting but weakly evidenced
    # cyclical does not outrank a cleaner compounder merely because both still
    # await statement-history verification.
    frame['growth_provisional_priority_score'] = (
        0.55 * growth_selection
        + 0.20 * fundamental_quality_for_queue
        + 0.10 * effective_silent
        + 0.10 * coverage
        + 0.05 * liquidity
    ).round(1)
    frame['turnaround_provisional_priority_score'] = (
        0.55 * turnaround_selection
        + 0.15 * fundamental_quality_for_queue
        + 0.15 * effective_silent
        + 0.10 * coverage
        + 0.05 * liquidity
    ).round(1)
    frame['near_miss_state'] = np.select(
        [
            snapshot_pending,
            metric_pending,
            history_pending,
            statement_pending | other_data_gate_pending,
            boolean('critical_research_flags'),
            ~grade_abc,
        ],
        [
            'DATA_PENDING_SNAPSHOT',
            'METRIC_COVERAGE_PENDING',
            'STATEMENT_HISTORY_PENDING',
            'STATEMENT_OR_DATA_GATE_PENDING',
            'CRITICAL_RISK_BLOCKED',
            'DATA_PROVENANCE_PENDING',
        ],
        default='QUALITY_GATE_PENDING',
    )
    frame['near_miss_reason'] = reason_series.where(
        reason_series.str.strip().ne(''),
        frame.get(
            'multibagger_score_reason',
            pd.Series('Quality/recovery gate belum lolos', index=frame.index),
        ).fillna('Quality/recovery gate belum lolos').astype(str),
    )
    frame['candidate_state'] = np.where(
        data_pending,
        'DATA_PENDING_RESEARCH',
        np.where(
            research_eligible,
            'QUALIFIED_RESEARCH',
            'PROVISIONAL_RESEARCH',
        ),
    )
    frame['next_required_evidence'] = np.select(
        [
            snapshot_pending,
            metric_pending,
            history_pending,
            statement_pending | other_data_gate_pending,
            boolean('critical_research_flags'),
            ~grade_abc,
        ],
        [
            'REFRESH_FUNDAMENTAL_SNAPSHOT',
            'BACKFILL_REQUIRED_MULTIBAGGER_METRICS',
            'BACKFILL_AND_VALIDATE_STATEMENT_HISTORY',
            'REFRESH_AND_VALIDATE_CURRENT_STATEMENT',
            'RESOLVE_OR_REJECT_CRITICAL_RISK',
            'IMPROVE_DATA_PROVENANCE_TO_GRADE_C_OR_BETTER',
        ],
        default='PASS_QUALITY_OR_RECOVERY_GATE',
    )
    frame['capital_state'] = 'RESEARCH_ONLY_NO_ALLOCATION'
    frame['portfolio_allocation_eligible'] = False

    common_columns = [
        'ticker', 'candidate_state', 'near_miss_state',
        'next_required_evidence', 'fundamental_score',
        'fundamental_score_10', 'fundamental_coverage',
        'fundamental_data_grade', 'fundamental_history_source_count',
        'fundamental_source_count', 'fundamental_history_quarters',
        'fundamental_history_years', 'effective_silent_accumulation_score',
        'narrative_effective_score',
        'narrative_evidence_coverage_pct',
        'issuer_alignment_effective_score', 'issuer_alignment_state',
        'retail_adoption_stage', 'narrative_flow_effective_score',
        'narrative_flow_convergence_state',
        'narrative_conversion_rate_20d_pct',
        'narrative_conversion_resolved_20d',
        'narrative_crowding_risk_score', 'narrative_hard_block',
        'multibagger_scoring_state', 'multibagger_metric_coverage_pct',
        'multibagger_metric_data_gate', 'growth_pillar_coverage_pct',
        'profitability_pillar_coverage_pct',
        'cashflow_pillar_coverage_pct', 'safety_pillar_coverage_pct',
        'runway_pillar_coverage_pct', 'valuation_pillar_coverage_pct',
        'adtv20_idr',
        'selector_trend_score', 'selector_relative_strength_score',
        'overall_research_confidence', 'technical_entry_state',
        'selected_reason', 'not_entry_reason', 'trigger_waiting',
        'invalidation_reason', 'primary_risk',
        'research_eligibility_reason',
        'near_miss_reason', 'portfolio_allocation_eligible', 'capital_state',
    ]

    scored_near_miss = frame.loc[scored_blocked].copy()
    scored_near_miss['_provisional_state_rank'] = (
        scored_near_miss['near_miss_state'].map({
            'QUALITY_GATE_PENDING': 0,
            'CRITICAL_RISK_BLOCKED': 9,
        }).fillna(5)
    )
    for column, default in (
        ('growth_compounder_selection_score', np.nan),
        ('turnaround_selection_score', np.nan),
        ('effective_silent_accumulation_score', 50.0),
        ('overall_research_confidence', 0.0),
        ('ticker', ''),
    ):
        if column not in scored_near_miss:
            scored_near_miss[column] = default
    growth_mask = (
        pd.to_numeric(
            scored_near_miss.get(
                'growth_compounder_selection_score',
                pd.Series(np.nan, index=scored_near_miss.index),
            ),
            errors='coerce',
        ).notna()
    )
    growth_near_miss = scored_near_miss.loc[growth_mask].sort_values(
        [
            '_provisional_state_rank',
            'growth_provisional_priority_score',
            'effective_silent_accumulation_score',
            'overall_research_confidence',
            'ticker',
        ],
        ascending=[True, False, False, False, True],
        kind='stable',
        na_position='last',
    )
    turnaround_mask = (
        pd.to_numeric(
            scored_near_miss.get(
                'turnaround_selection_score',
                pd.Series(np.nan, index=scored_near_miss.index),
            ),
            errors='coerce',
        ).notna()
    )
    turnaround_near_miss = scored_near_miss.loc[
        turnaround_mask
    ].sort_values(
        [
            '_provisional_state_rank',
            'turnaround_provisional_priority_score',
            'effective_silent_accumulation_score',
            'overall_research_confidence',
            'ticker',
        ],
        ascending=[True, False, False, False, True],
        kind='stable',
        na_position='last',
    )
    pending_view = frame.loc[data_pending].sort_values(
        ['data_refresh_priority_score', 'effective_silent_accumulation_score', 'ticker'],
        ascending=[False, False, True],
        kind='stable',
        na_position='last',
    )

    growth_columns = [
        'growth_provisional_priority_score',
        'growth_compounder_selection_score', 'growth_compounder_score',
        'reinvestment_runway_pillar', *common_columns,
    ]
    turnaround_columns = [
        'turnaround_provisional_priority_score',
        'turnaround_selection_score', 'turnaround_recovery_score',
        'turnaround_recovery_signals', 'turnaround_gate_reasons',
        *common_columns,
    ]
    pending_columns = [
        'ticker', 'candidate_state', 'data_refresh_priority_score',
        'silent_accumulation_score', 'effective_silent_accumulation_score',
        'silent_accumulation_state', 'selector_trend_score',
        'selector_relative_strength_score', 'adtv20_idr',
        'near_miss_state', 'next_required_evidence',
        'multibagger_scoring_state', 'multibagger_metric_coverage_pct',
        'multibagger_metric_data_gate',
        'near_miss_reason', 'capital_state',
    ]

    def select_columns(local: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        unique = list(dict.fromkeys(column for column in columns if column in local))
        selected = (
            local.loc[:, unique]
            .head(max(1, int(limit)))
            .reset_index(drop=True)
        )
        if not selected.empty:
            selected.insert(
                0, 'research_queue_rank',
                np.arange(1, len(selected) + 1),
            )
        return selected

    growth_queue = select_columns(growth_near_miss, growth_columns)
    turnaround_queue = select_columns(
        turnaround_near_miss, turnaround_columns,
    )
    pending_queue = select_columns(pending_view, pending_columns)

    return {
        'coverage': coverage_view,
        'gate_failures': gate_failures,
        'growth_near_miss': growth_queue,
        'turnaround_near_miss': turnaround_queue,
        'growth_research_queue': growth_queue.copy(),
        'turnaround_research_queue': turnaround_queue.copy(),
        'data_pending': pending_queue,
    }


def build_scanner_data_contract_audit(
    expected_tickers: Iterable[str],
    *,
    histories: Mapping[str, pd.DataFrame] | None = None,
    prepared: Mapping[str, pd.DataFrame] | None = None,
    fundamentals: pd.DataFrame | None = None,
    fundamental_history: pd.DataFrame | None = None,
    selector: pd.DataFrame | None = None,
    multibagger: pd.DataFrame | None = None,
    core_signals: pd.DataFrame | None = None,
    order_builder: pd.DataFrame | None = None,
    order_builder_coverage: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return an explicit observed/derived/decision coverage contract.

    A zero qualified-candidate count is only labelled a valid decision when
    its upstream evidence is sufficient. Otherwise it is reported as not
    evaluable, preventing an empty table from masquerading as a negative call.
    """
    expected_names = {
        safe_text(value).upper()
        for value in expected_tickers
        if safe_text(value)
    }
    expected_count = len(expected_names)
    denominator = max(expected_count, 1)

    def mapping_ready_count(
        source: Mapping[str, pd.DataFrame] | None,
    ) -> int:
        if not isinstance(source, Mapping):
            return 0
        return len({
            safe_text(ticker).upper()
            for ticker, frame in source.items()
            if safe_text(ticker)
            and (
                not expected_names
                or safe_text(ticker).upper() in expected_names
            )
            and isinstance(frame, pd.DataFrame)
            and not frame.empty
        })

    def dataframe_ticker_count(
        source: pd.DataFrame | None,
        mask: pd.Series | None = None,
    ) -> int:
        if (
            not isinstance(source, pd.DataFrame)
            or source.empty
            or 'ticker' not in source
        ):
            return 0
        local = source
        if expected_names:
            local_ticker = (
                local['ticker'].fillna('').astype(str).str.upper().str.strip()
            )
            local = local.loc[local_ticker.isin(expected_names)]
        if mask is not None:
            local = local.loc[mask.reindex(local.index, fill_value=False)]
        return int(
            local['ticker'].fillna('').astype(str)
            .str.upper().str.strip().replace('', np.nan).nunique()
        )

    history_count = mapping_ready_count(histories)
    prepared_count = mapping_ready_count(prepared)
    volume_ready_names: set[str] = set()
    if isinstance(prepared, Mapping):
        for ticker, frame in prepared.items():
            if not isinstance(frame, pd.DataFrame) or len(frame) < 25:
                continue
            normalized_ticker = safe_text(ticker).upper()
            if expected_names and normalized_ticker not in expected_names:
                continue
            volume = pd.to_numeric(
                frame.get(
                    'Volume',
                    pd.Series(np.nan, index=frame.index),
                ),
                errors='coerce',
            )
            observed = float(
                100.0 * volume.tail(min(60, len(volume))).notna().mean()
            )
            declared = pd.to_numeric(
                frame.get(
                    'VOLUME_DATA_COVERAGE20',
                    pd.Series(np.nan, index=frame.index),
                ),
                errors='coerce',
            ).dropna()
            if not declared.empty:
                observed = min(observed, float(declared.iloc[-1]))
            if observed >= 90.0:
                volume_ready_names.add(normalized_ticker)
    volume_ready_count = len(volume_ready_names)

    snapshot_count = 0
    if isinstance(fundamentals, pd.DataFrame) and not fundamentals.empty:
        snapshot_coverage = pd.to_numeric(
            fundamentals.get(
                'fundamental_coverage',
                pd.Series(0.0, index=fundamentals.index),
            ),
            errors='coerce',
        ).fillna(0.0)
        snapshot_score = pd.to_numeric(
            fundamentals.get(
                'fundamental_score',
                pd.Series(np.nan, index=fundamentals.index),
            ),
            errors='coerce',
        )
        snapshot_count = dataframe_ticker_count(
            fundamentals,
            snapshot_coverage.gt(0.0) & snapshot_score.notna(),
        )
    statement_history_count = dataframe_ticker_count(fundamental_history)

    selector_count = 0
    if isinstance(selector, pd.DataFrame) and not selector.empty:
        selector_mask = (
            selector.get(
                'selector_rank_eligible',
                pd.Series(False, index=selector.index),
            ).map(truthy)
        )
        selector_count = dataframe_ticker_count(selector, selector_mask)

    metric_gate_count = 0
    qualified_count = 0
    if isinstance(multibagger, pd.DataFrame) and not multibagger.empty:
        metric_gate = multibagger.get(
            'multibagger_metric_data_gate',
            pd.Series(False, index=multibagger.index),
        ).map(truthy)
        metric_gate_count = dataframe_ticker_count(multibagger, metric_gate)
        research_gate = multibagger.get(
            'research_eligible',
            pd.Series(False, index=multibagger.index),
        ).map(truthy)
        qualified_count = dataframe_ticker_count(
            multibagger,
            research_gate & metric_gate,
        )

    setup_count = dataframe_ticker_count(core_signals)
    order_count = 0
    if isinstance(order_builder, pd.DataFrame) and not order_builder.empty:
        order_gate = order_builder.get(
            'order_builder_eligible',
            pd.Series(True, index=order_builder.index),
        ).map(truthy)
        order_count = dataframe_ticker_count(order_builder, order_gate)
    swing_evaluable_count = 0
    if (
        isinstance(order_builder_coverage, pd.DataFrame)
        and not order_builder_coverage.empty
    ):
        coverage_gate = order_builder_coverage.get(
            'order_builder_eligible',
            pd.Series(False, index=order_builder_coverage.index),
        ).map(truthy)
        swing_evaluable_count = dataframe_ticker_count(
            order_builder_coverage,
            coverage_gate,
        )
    elif order_count > 0:
        swing_evaluable_count = order_count

    stages = [
        (
            'UNIVERSE_EXPECTED', 'OBSERVED', expected_count,
            'Ticker unik dari universe input.',
        ),
        (
            'OHLCV_DOWNLOADED', 'OBSERVED', history_count,
            'OHLCV provider/cache tersedia dan tidak kosong.',
        ),
        (
            'INDICATORS_READY', 'DERIVED', prepared_count,
            'OHLCV lolos normalisasi dan minimum bar indikator.',
        ),
        (
            'PRICE_VOLUME_EVIDENCE_READY', 'DERIVED', volume_ready_count,
            'Coverage volume terbaru ≥90%; missing volume bukan nol.',
        ),
        (
            'FUNDAMENTAL_SNAPSHOT_READY', 'OBSERVED', snapshot_count,
            'Snapshot memiliki score dan coverage teramati.',
        ),
        (
            'STATEMENT_HISTORY_READY', 'OBSERVED', statement_history_count,
            'Histori laporan terverifikasi berhasil dinormalisasi.',
        ),
        (
            'SELECTOR_FEATURES_SUFFICIENT', 'DERIVED', selector_count,
            'Coverage fitur cross-sectional memenuhi gate produksi.',
        ),
        (
            'MULTIBAGGER_METRICS_SUFFICIENT', 'DERIVED', metric_gate_count,
            'Coverage quality/growth/cashflow/safety/reinvestment/valuation memadai.',
        ),
        (
            'MULTIBAGGER_QUALIFIED', 'DECISION', qualified_count,
            'Kandidat lolos seluruh gate riset; bukan janji return.',
        ),
        (
            'CORE_SETUP_DETECTED', 'DECISION', setup_count,
            'Setup teknikal valid terdeteksi.',
        ),
        (
            'SWING_COMPONENTS_SUFFICIENT', 'DERIVED',
            swing_evaluable_count,
            'Coverage komponen swing dan selector memenuhi gate produksi.',
        ),
        (
            'SWING_ORDER_ELIGIBLE', 'DECISION', order_count,
            'Setup, komponen, selector, dan execution evidence memadai.',
        ),
    ]
    rows: list[dict[str, Any]] = []
    swing_decision_upstream = (
        swing_evaluable_count
        if swing_evaluable_count > 0
        else prepared_count
        if setup_count == 0 and prepared_count > 0
        else 0
    )
    upstream = {
        'MULTIBAGGER_QUALIFIED': metric_gate_count,
        'CORE_SETUP_DETECTED': prepared_count,
        'SWING_ORDER_ELIGIBLE': swing_decision_upstream,
    }
    for stage, data_class, count, meaning in stages:
        missing = max(0, expected_count - int(count))
        if data_class == 'DECISION' and count == 0:
            state = (
                'NO_CANDIDATE_VALID'
                if upstream.get(stage, 0) > 0
                else 'NOT_EVALUABLE_DATA_INSUFFICIENT'
            )
        elif count >= expected_count and expected_count > 0:
            state = 'READY'
        elif count > 0:
            state = 'PARTIAL'
        else:
            state = 'NOT_AVAILABLE'
        rows.append({
            'stage': stage,
            'data_class': data_class,
            'count': int(count),
            'pct_of_universe': round(100.0 * int(count) / denominator, 1),
            'missing_count': missing,
            'state': state,
            'meaning': meaning,
        })
    return pd.DataFrame(rows)


def _focus_liquidity_score(adtv: Any) -> float:
    value = max(0.0, safe_number(adtv, 0.0))
    if value <= 0:
        return 0.0
    return float(max(0.0, min(100.0, 20.0 + 50.0 * np.log10(max(value, 250_000_000.0) / 250_000_000.0) / np.log10(40.0))))


def _focus_target_score(rr1: Any, rr2: Any, target_valid: Any=True) -> float:
    if not truthy(target_valid):
        return 0.0
    first = safe_number(rr1, np.nan)
    second = safe_number(rr2, np.nan)
    if not np.isfinite(first) or not np.isfinite(second):
        return 20.0
    return float(max(0.0, min(100.0, 18.0 + 24.0 * min(2.0, max(0.0, first)) + 17.0 * min(3.0, max(0.0, second)))))


def _focus_grade(score: float) -> str:
    if score >= 88.0:
        return 'A+'
    if score >= 80.0:
        return 'A'
    if score >= 72.0:
        return 'B+'
    if score >= 64.0:
        return 'B'
    return 'C'


_SELECTOR_OVERLAY_RENAME = {
    'trend_score': 'selector_trend_score',
    'momentum_score': 'selector_momentum_score',
    'relative_strength_score': 'selector_relative_strength_score',
    'flow_score': 'selector_flow_score',
    'selected_reason': 'selector_selected_reason',
    'selection_risks': 'selector_selection_risks',
    'selection_reason_codes': 'selector_selection_reason_codes',
}


def _attach_selector_overlay(
    frame: pd.DataFrame,
    selector_ranking: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach selection-only fields without overwriting strategy score fields."""
    if frame is None or frame.empty or selector_ranking is None or selector_ranking.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    if 'ticker' not in frame or 'ticker' not in selector_ranking:
        return frame.copy()
    selector_columns = [
        'ticker', 'as_of', 'selection_rank', 'production_selection_rank',
        'selector_rank_eligible', 'selector_data_state',
        'technical_feature_coverage_pct', 'selector_missing_feature_count',
        'selector_missing_features', 'swing_selection_score',
        'technical_selection_score', 'multibagger_timing_selector_score',
        'trend_score', 'momentum_score', 'relative_strength_score', 'flow_score',
        'effective_silent_accumulation_score',
        'silent_accumulation_confidence',
        'sector', 'sector_peer_count', 'sector_relative_strength_score',
        'sector_relative_state', 'liquidity_bucket',
        'estimated_market_impact_cost_pct', 'estimated_total_cost_pct',
        'selector_model_state', 'selector_version',
        'selected_reason', 'selection_risks', 'selection_reason_codes',
    ]
    for horizon in (5, 20, 60):
        selector_columns.extend([
            f'selector_expected_excess_return_{horizon}d_pct',
            f'selector_outperform_probability_{horizon}d_pct',
            f'selector_score_{horizon}d',
            f'selector_ai_weight_{horizon}d_pct',
            f'selector_model_state_{horizon}d',
            f'selector_champion_{horizon}d',
        ])
    overlay = selector_ranking.loc[
        :, [column for column in selector_columns if column in selector_ranking]
    ].copy()
    overlay['ticker'] = overlay['ticker'].astype(str).str.upper().str.strip()
    overlay = overlay.drop_duplicates('ticker', keep='first').rename(
        columns=_SELECTOR_OVERLAY_RENAME,
    )
    out = frame.copy()
    out['_selector_ticker'] = out['ticker'].astype(str).str.upper().str.strip()
    overlay = overlay.rename(columns={'ticker': '_selector_ticker'})
    out = out.merge(overlay, on='_selector_ticker', how='left', suffixes=('', '_selector'))
    return out.drop(columns='_selector_ticker')


def build_focus_order_builder(
    core_signals: pd.DataFrame | None,
    config: ScanConfig | None=None,
    validation_events: pd.DataFrame | None=None,
    ai_memory: pd.DataFrame | None=None,
    selector_ranking: pd.DataFrame | None=None,
    narrative_profiles: pd.DataFrame | None=None,
) -> pd.DataFrame:
    """Rank only daily Core Swing setups.

    No intraday, fast-trade, or auto-rejection model is evaluated here. The
    output remains compatible with the Top-20 dashboard and local AI engine.
    """
    cfg = config or ScanConfig()
    rows: list[dict[str, Any]] = []
    specs = {
        'PULLBACK_CONTINUATION': {
            'weights': (0.28, 0.20, 0.12, 0.10, 0.12, 0.10, 0.08),
            'horizon': '2–20 trading days',
            'action_scores': {'READY_TRIGGER': 100.0, 'READY_LIMIT': 90.0, 'WAIT_PULLBACK_CONFIRMATION': 78.0, 'WAIT_STRICT_FLOW_CONFIRMATION': 70.0},
        },
        'BREAKOUT_RETEST': {
            'weights': (0.30, 0.22, 0.12, 0.08, 0.12, 0.08, 0.08),
            'horizon': '2–25 trading days',
            'action_scores': {'READY_TRIGGER': 100.0, 'READY_LIMIT': 92.0, 'WAIT_CURRENT_RETEST_CONFIRMATION': 80.0, 'WAIT_RETEST': 72.0},
        },
        'REVERSAL_ACCUMULATION': {
            'weights': (0.30, 0.20, 0.18, 0.08, 0.10, 0.08, 0.06),
            'horizon': '5–40 trading days',
            'action_scores': {'READY_TRIGGER': 100.0, 'READY_LIMIT': 90.0, 'WAIT_HIGHER_LOW_AND_FLOW': 76.0, 'WAIT_RETEST': 72.0},
        },
        'UNICORN_SNIPER_ICT': {
            'weights': (0.30, 0.22, 0.14, 0.08, 0.10, 0.08, 0.08),
            'horizon': '2–20 trading days',
            'action_scores': {'READY_TRIGGER': 100.0, 'READY_LIMIT': 92.0, 'WAIT_STRICT_CONFLUENCE': 76.0, 'WAIT_RETRACE': 72.0},
        },
    }
    if core_signals is None or core_signals.empty:
        empty = pd.DataFrame()
        empty.attrs['strategy_audit'] = pd.DataFrame({
            'strategy': list(specs), 'eligible_candidates': 0,
            'above_min_conviction': 0, 'included_primary': 0,
            'max_conviction': np.nan, 'ranking_state': 'NO_ELIGIBLE_STATUS',
        })
        empty.attrs['ai_audit'] = pd.DataFrame()
        empty.attrs['data_coverage_audit'] = pd.DataFrame()
        return empty

    eligible_status = {'EXECUTION_READY', 'READY_FOR_STOCKBIT_VERIFY', 'SIGNAL_READY', 'ENTRY_PLAN_READY', 'READY_FOR_PRICE_VERIFY'}
    for setup_name, spec in specs.items():
        setup_series = core_signals.get('setup', pd.Series(index=core_signals.index, dtype=object))
        local = core_signals[setup_series.eq(setup_name)].copy()
        status_series = local.get('status', pd.Series(index=local.index, dtype=object))
        local = local[status_series.isin(eligible_status)]
        for _, row in local.iterrows():
            def observed_numeric(*names: str) -> bool:
                for name in names:
                    value = pd.to_numeric(
                        pd.Series([row.get(name, np.nan)]),
                        errors='coerce',
                    ).iloc[0]
                    if np.isfinite(value):
                        return True
                return False

            def observed_field(name: str) -> bool:
                if name not in row.index:
                    return False
                value = row.get(name)
                if value is None:
                    return False
                try:
                    if bool(pd.isna(value)):
                        return False
                except (TypeError, ValueError):
                    pass
                return safe_text(value).strip() != ''

            def coverage_for(checks: list[bool]) -> float:
                return (
                    100.0 * sum(bool(value) for value in checks) / len(checks)
                    if checks else 0.0
                )

            status = safe_text(row.get('status'))
            action = safe_text(row.get('action'))
            quality = safe_number(row.get('quality_score'), 50.0)
            analyst = safe_number(row.get('analyst_fusion_score'), quality)
            structural = safe_number(row.get('structural_quality_score'), quality)
            confirmation = safe_number(row.get('confirmation_quality_score'), 60.0)
            silent_raw = safe_number(
                row.get('silent_accumulation_score', row.get('accumulation_score')),
                np.nan,
            )
            silent_confidence = safe_number(
                row.get(
                    'silent_accumulation_confidence',
                    row.get('silent_accumulation_data_coverage'),
                ),
                60.0 if np.isfinite(silent_raw) else 0.0,
            )
            silent_confidence = max(0.0, min(100.0, silent_confidence))
            effective_silent = (
                50.0
                + (silent_raw - 50.0) * silent_confidence / 100.0
                if np.isfinite(silent_raw)
                else 50.0
            )
            demand = safe_number(
                row.get('supply_demand_score'),
                effective_silent,
            )
            failure_risk = max(0.0, min(100.0, safe_number(row.get('failure_risk_score'), 0.0)))
            structure = max(0.0, min(100.0, 0.30*quality + 0.22*analyst + 0.20*structural + 0.16*demand + 0.12*confirmation - 0.20*failure_risk))
            distance = safe_number(row.get('distance_atr'), 99.0)
            extension = max(0.0, safe_number(row.get('extension_atr'), 0.0))
            action_score = spec['action_scores'].get(action, 45.0)
            status_score = {'EXECUTION_READY': 100.0, 'READY_FOR_STOCKBIT_VERIFY': 96.0, 'SIGNAL_READY': 86.0, 'ENTRY_PLAN_READY': 72.0, 'READY_FOR_PRICE_VERIFY': 82.0}.get(status, 50.0)
            proximity = max(0.0, 100.0 - 35.0*max(0.0, distance)) if np.isfinite(distance) else 30.0
            extension_score = max(0.0, 100.0 - 35.0*extension)
            timing = 0.38*action_score + 0.28*status_score + 0.20*proximity + 0.14*extension_score
            cmf_score = max(0.0, min(100.0, 50.0 + 500.0*safe_number(row.get('cmf20'), 0.0)))
            volume_score = (
                max(0.0, min(
                    100.0,
                    35.0 + 32.0 * safe_number(row.get('volume_ratio'), 0.0),
                ))
                if observed_numeric('volume_ratio', 'vol_ratio')
                else 50.0
            )
            flow = (
                0.40*demand
                + 0.30*effective_silent
                + 0.15*cmf_score
                + 0.15*volume_score
            )
            liquidity = (
                _focus_liquidity_score(row.get('adtv20_idr'))
                if observed_numeric('adtv20_idr')
                else 50.0
            )
            target = (
                _focus_target_score(
                    row.get('rr1'),
                    row.get('rr2'),
                    row.get('target_structure_valid', True),
                )
                if observed_numeric('rr1') and observed_numeric('rr2')
                else 50.0
            )
            completeness = safe_number(row.get('data_completeness_score'), 50.0)
            execution_conf = safe_number(row.get('execution_confidence_score'), completeness)
            data = 0.55*completeness + 0.45*execution_conf
            validation = safe_number(row.get('validation_gate_score'), 50.0)
            probability = safe_number(row.get('probability_estimate'), np.nan)
            if np.isfinite(probability):
                probability_pct = 100.0*probability if 0.0 <= probability <= 1.0 else probability
                validation = 0.55*validation + 0.45*max(0.0, min(100.0, probability_pct))
            component_coverages = (
                coverage_for([
                    observed_numeric('quality_score'),
                    observed_numeric('analyst_fusion_score'),
                    observed_numeric('structural_quality_score'),
                    observed_numeric('confirmation_quality_score'),
                    observed_numeric('supply_demand_score'),
                    observed_numeric('failure_risk_score'),
                ]),
                coverage_for([
                    bool(action), bool(status),
                    observed_numeric('distance_atr'),
                    observed_numeric('extension_atr'),
                ]),
                coverage_for([
                    observed_numeric('supply_demand_score'),
                    np.isfinite(silent_raw) and silent_confidence > 0.0,
                    observed_numeric('cmf20'),
                    observed_numeric('volume_ratio', 'vol_ratio'),
                ]),
                coverage_for([observed_numeric('adtv20_idr')]),
                coverage_for([
                    observed_numeric('rr1'),
                    observed_numeric('rr2'),
                    observed_field('target_structure_valid'),
                ]),
                coverage_for([
                    observed_numeric('data_completeness_score'),
                    observed_numeric('execution_confidence_score'),
                ]),
                coverage_for([
                    observed_numeric('validation_gate_score')
                    or observed_numeric('probability_estimate'),
                ]),
            )
            components = (structure, timing, flow, liquidity, target, data, validation)
            score = float(sum(w*max(0.0, min(100.0, c)) for w,c in zip(spec['weights'], components)))
            if observed_field('target_structure_valid') and not truthy(
                row.get('target_structure_valid'),
            ):
                score -= 15.0
            if observed_numeric('rr2') and safe_number(row.get('rr2'), np.nan) < 1.0:
                score -= 8.0
            raw_component_score = max(0.0, min(100.0, score))
            swing_component_coverage = float(sum(
                weight * coverage
                for weight, coverage in zip(
                    spec['weights'],
                    component_coverages,
                )
            ))
            pre_time = (
                50.0
                + swing_component_coverage / 100.0
                * (raw_component_score - 50.0)
            )
            pre_time = max(0.0, min(100.0, pre_time))
            minimum_component_coverage = safe_number(
                getattr(cfg, 'swing_min_component_coverage_pct', 65.0),
                65.0,
            )
            component_gate = (
                swing_component_coverage >= minimum_component_coverage
            )
            cycle_state = safe_text(row.get('time_cycle_state')).upper()
            cycle_weight = max(0.0, min(0.10, safe_number(row.get('time_cycle_effective_weight_pct'), 0.0)/100.0)) if cycle_state == 'VALIDATED' else 0.0
            cycle_alignment = max(0.0, min(100.0, safe_number(row.get('time_cycle_alignment_score'), 50.0)))
            final = round((1.0-cycle_weight)*pre_time + cycle_weight*cycle_alignment, 1)
            entry_plan_actions = {'WAIT_PULLBACK_CONFIRMATION','WAIT_STRICT_FLOW_CONFIRMATION','WAIT_RETEST','WAIT_CURRENT_RETEST_CONFIRMATION','WAIT_HIGHER_LOW_AND_FLOW','WAIT_STRICT_CONFLUENCE','WAIT_RETRACE'}
            decision = 'ENTRY_PLAN' if action in entry_plan_actions or status == 'ENTRY_PLAN_READY' else 'SETUP_READY'
            warnings_text = ' • '.join(part for part in (
                safe_text(row.get('signal_risk_warnings')),
                safe_text(row.get('evidence_warnings')),
                safe_text(row.get('blockers')),
                (
                    'Coverage komponen swing '
                    f'{swing_component_coverage:.0f}% '
                    f'(<{minimum_component_coverage:.0f}%)'
                    if not component_gate else ''
                ),
            ) if part)
            rows.append({
                'profit_rank': np.nan,
                'ticker': row.get('ticker'),
                'strategy': setup_name,
                'horizon': spec['horizon'],
                'decision_state': decision,
                'setup_status': status,
                'profit_conviction_score': final,
                'raw_component_score': round(raw_component_score, 1),
                'swing_component_coverage_pct': round(
                    swing_component_coverage, 1,
                ),
                'swing_score_state': (
                    'SCORED_SUFFICIENT'
                    if component_gate else
                    'DATA_INSUFFICIENT_NOT_PRODUCTION'
                ),
                'order_builder_eligible': bool(component_gate),
                'conviction_grade': _focus_grade(final),
                'pre_time_conviction_score': round(pre_time, 1),
                'time_cycle_alignment_score': round(cycle_alignment, 1),
                'time_cycle_effective_weight_pct': round(100.0*cycle_weight, 2),
                'time_cycle_adjustment': round(final-pre_time, 2),
                'time_cycle_score': safe_number(row.get('time_cycle_score'), np.nan),
                'time_cycle_confidence': safe_number(row.get('time_cycle_confidence'), np.nan),
                'time_cycle_state': safe_text(row.get('time_cycle_state')),
                'quick_buy_score': safe_number(row.get('quick_buy_score'), np.nan),
                'quick_buy_action': safe_text(row.get('quick_buy_action')),
                'best_buy_date': safe_text(row.get('best_buy_date')),
                'best_buy_window_start': safe_text(row.get('best_buy_window_start')),
                'best_buy_window_end': safe_text(row.get('best_buy_window_end')),
                'best_buy_score': safe_number(row.get('best_buy_score'), np.nan),
                'best_buy_confidence': safe_number(row.get('best_buy_confidence'), np.nan),
                'best_buy_entry_low': safe_number(row.get('best_buy_entry_low'), np.nan),
                'best_buy_entry_high': safe_number(row.get('best_buy_entry_high'), np.nan),
                'best_buy_trigger': safe_number(row.get('best_buy_trigger'), np.nan),
                'best_buy_stop_loss': safe_number(row.get('best_buy_stop_loss'), np.nan),
                'best_buy_tp1': safe_number(row.get('best_buy_tp1'), np.nan),
                'best_buy_tp2': safe_number(row.get('best_buy_tp2'), np.nan),
                'best_buy_rr1': safe_number(row.get('best_buy_rr1'), np.nan),
                'best_buy_rr2': safe_number(row.get('best_buy_rr2'), np.nan),
                'best_buy_target_basis': safe_text(row.get('best_buy_target_basis')),
                'best_buy_order_plan': safe_text(row.get('best_buy_order_plan')) or 'NO_ORDER',
                'best_buy_reason': safe_text(row.get('best_buy_reason')),
                'best_buy_no_trade_condition': safe_text(row.get('best_buy_no_trade_condition')),
                'best_buy_summary': safe_text(row.get('best_buy_summary')),
                'eoff_strength_label': safe_text(row.get('eoff_strength_label')),
                'eoff_reconstruction_score': safe_number(row.get('eoff_reconstruction_score'), np.nan),
                'eoff_signal_active': truthy(row.get('eoff_signal_active')),
                'eoff_direction_bias': safe_text(row.get('eoff_direction_bias')),
                'entry_low': safe_number(row.get('entry_low'), np.nan),
                'entry_high': safe_number(row.get('entry_high'), np.nan),
                'entry': safe_number(row.get('entry'), np.nan),
                'entry_type': safe_text(row.get('entry_type')),
                'action': safe_text(row.get('action')),
                'trigger': safe_number(row.get('trigger'), np.nan),
                'trigger_price': safe_number(row.get('trigger'), np.nan),
                'stockbit_trigger_price': safe_number(row.get('stockbit_trigger_price'), np.nan),
                'stockbit_limit_price': safe_number(row.get('stockbit_limit_price'), np.nan),
                'stockbit_order_price': safe_number(row.get('stockbit_order_price'), np.nan),
                'order_instruction': safe_text(row.get('order_instruction')),
                'execution_timing': safe_text(row.get('execution_timing')),
                'stop_loss': safe_number(row.get('stop_loss'), np.nan),
                'tp1': safe_number(row.get('tp1'), np.nan),
                'tp2': safe_number(row.get('tp2'), np.nan),
                'rr1': safe_number(row.get('rr1'), np.nan),
                'rr2': safe_number(row.get('rr2'), np.nan),
                'entry_plan_min_rr1': safe_number(row.get('entry_plan_min_rr1'), cfg.min_rr1),
                'entry_plan_min_rr2': safe_number(row.get('entry_plan_min_rr2'), cfg.min_rr2),
                'structure_score': round(structure, 1),
                'timing_score': round(timing, 1),
                'flow_score': round(flow, 1),
                'liquidity_score': round(liquidity, 1),
                'target_quality_score': round(target, 1),
                'data_quality_score': round(data, 1),
                'validation_score': round(validation, 1),
                'order_ready': truthy(row.get('autopilot_verified')),
                'stockbit_order_lots': int(safe_number(row.get('stockbit_order_lots'), 0.0)),
                'next_action': safe_text(row.get('order_instruction')) or action,
                'warnings': warnings_text,
                'conviction_basis': f'Structure {structure:.0f}; timing {timing:.0f}; flow {flow:.0f}; liquidity {liquidity:.0f}; target {target:.0f}; data {data:.0f}; validation {validation:.0f}',
                'market_regime': safe_text(row.get('market_regime')) or safe_text(row.get('regime')) or 'UNKNOWN',
                'stop_pct': (safe_number(row.get('entry'), np.nan)-safe_number(row.get('stop_loss'), np.nan))/safe_number(row.get('entry'), np.nan) if safe_number(row.get('entry'), np.nan)>safe_number(row.get('stop_loss'), np.nan)>0 else np.nan,
                'atr_pct': safe_number(row.get('atr_pct'), np.nan),
                'volume_ratio': safe_number(row.get('volume_ratio', row.get('vol_ratio')), np.nan),
                'rsi14': safe_number(row.get('rsi14'), np.nan),
                'adx14': safe_number(row.get('adx14'), np.nan),
                'cmf20': safe_number(row.get('cmf20'), np.nan),
                'roc60': safe_number(row.get('roc60'), np.nan),
                'distance_52w_high': safe_number(row.get('distance_52w_high'), np.nan),
                'relative_strength60': safe_number(row.get('relative_strength60'), np.nan),
                'silent_accumulation_score': safe_number(row.get('silent_accumulation_score', row.get('accumulation_score')), np.nan),
                'effective_silent_accumulation_score': round(
                    effective_silent, 1,
                ),
                'silent_accumulation_confidence': round(
                    silent_confidence, 1,
                ),
                'body_atr': safe_number(row.get('body_atr'), np.nan),
                'close_location': safe_number(row.get('close_location'), np.nan),
            })

    result = pd.DataFrame(rows)
    minimum = safe_number(getattr(cfg, 'profit_conviction_min_score', 68.0), 68.0)
    audit_rows=[]
    if result.empty:
        result.attrs['strategy_audit'] = pd.DataFrame({
            'strategy': list(specs), 'eligible_candidates': 0,
            'above_min_conviction': 0, 'included_primary': 0,
            'max_conviction': np.nan, 'ranking_state': 'NO_ELIGIBLE_STATUS',
        })
        result.attrs['ai_audit'] = pd.DataFrame()
        result.attrs['data_coverage_audit'] = pd.DataFrame()
        return result
    result = _attach_selector_overlay(result, selector_ranking)
    selector_attached = (
        isinstance(selector_ranking, pd.DataFrame)
        and not selector_ranking.empty
    )
    selector_coverage = pd.to_numeric(
        result.get(
            'technical_feature_coverage_pct',
            pd.Series(np.nan, index=result.index),
        ),
        errors='coerce',
    )
    minimum_selector_coverage = safe_number(
        getattr(
            cfg,
            'swing_min_selector_feature_coverage_pct',
            75.0,
        ),
        75.0,
    )
    if selector_attached:
        if 'selector_rank_eligible' in result:
            selector_gate = result['selector_rank_eligible'].map(truthy)
        else:
            selector_gate = pd.Series(True, index=result.index)
        selector_gate &= selector_coverage.fillna(0.0).ge(
            minimum_selector_coverage,
        )
    else:
        selector_gate = pd.Series(True, index=result.index)
    result['selector_production_gate'] = selector_gate.astype(bool)
    result['order_builder_eligible'] = (
        result['order_builder_eligible'].map(truthy)
        & result['selector_production_gate']
    )
    local_effective = pd.to_numeric(
        result.get(
            'effective_silent_accumulation_score',
            pd.Series(np.nan, index=result.index),
        ),
        errors='coerce',
    )
    selector_effective = pd.to_numeric(
        result.get(
            'effective_silent_accumulation_score_selector',
            pd.Series(np.nan, index=result.index),
        ),
        errors='coerce',
    )
    result['effective_silent_accumulation_score'] = (
        selector_effective.fillna(local_effective).fillna(50.0).round(1)
    )
    result.loc[
        ~result['selector_production_gate'],
        'swing_score_state',
    ] = 'SELECTOR_DATA_INSUFFICIENT_NOT_PRODUCTION'
    selection_score = pd.to_numeric(
        result.get(
            'swing_selection_score', pd.Series(np.nan, index=result.index),
        ), errors='coerce',
    )
    result['swing_selection_score'] = selection_score.fillna(
        pd.to_numeric(result['profit_conviction_score'], errors='coerce').fillna(0.0)
    ).round(1)
    result['core_priority_score'] = (
        0.70 * result['swing_selection_score']
        + 0.30 * pd.to_numeric(result['profit_conviction_score'], errors='coerce').fillna(0.0)
    ).round(1)
    result = attach_narrative_profiles(result, narrative_profiles)
    narrative_adjustment = pd.to_numeric(
        result.get(
            'narrative_swing_rank_adjustment',
            pd.Series(0.0, index=result.index),
        ),
        errors='coerce',
    ).fillna(0.0)
    result['core_priority_score_pre_narrative'] = (
        result['core_priority_score']
    )
    result['core_priority_score'] = (
        pd.to_numeric(result['core_priority_score'], errors='coerce')
        + narrative_adjustment
    ).clip(0.0, 100.0).round(1)
    narrative_block = result.get(
        'narrative_hard_block',
        pd.Series(False, index=result.index),
    ).fillna(False).map(truthy)
    result['order_builder_eligible'] = (
        result['order_builder_eligible'].map(truthy)
        & ~narrative_block
    )
    result['selection_before_setup'] = True
    raw=result.copy()
    coverage_columns = [
        'ticker', 'strategy', 'profit_conviction_score',
        'swing_component_coverage_pct', 'swing_score_state',
        'order_builder_eligible', 'selector_production_gate',
        'selector_rank_eligible', 'technical_feature_coverage_pct',
        'selector_data_state', 'selector_missing_feature_count',
        'selector_missing_features',
    ]
    coverage_audit = raw.loc[
        :, [column for column in coverage_columns if column in raw]
    ].copy()
    result=result[
        result['profit_conviction_score'].ge(minimum)
        & result['order_builder_eligible']
    ].copy()
    result=result.sort_values(
        ['core_priority_score','effective_silent_accumulation_score','profit_conviction_score','target_quality_score','liquidity_score'],
        ascending=[False,False,False,False,False],kind='stable',
    )
    alternatives=result.groupby('ticker')['strategy'].apply(lambda v:' • '.join(dict.fromkeys(map(str,v)))).to_dict() if not result.empty else {}
    result=result.drop_duplicates(['ticker','strategy'],keep='first').copy()
    if not result.empty:
        result['alternate_strategies']=result['ticker'].map(alternatives)
        result['candidate_id']=result.apply(lambda r:f"{safe_text(r.get('ticker'))}|{safe_text(r.get('strategy'))}",axis=1)
        result['strategy_rank']=result.groupby('strategy')['core_priority_score'].rank(method='first',ascending=False).astype(int)
        result=result.head(int(getattr(cfg,'profit_order_builder_limit',20))).reset_index(drop=True)
        result['profit_rank']=np.arange(1,len(result)+1)
    for strategy in specs:
        local_raw=raw[raw['strategy'].eq(strategy)]
        data_sufficient = local_raw[
            local_raw['order_builder_eligible'].map(truthy)
        ]
        local_above=local_raw[
            local_raw['profit_conviction_score'].ge(minimum)
            & local_raw['order_builder_eligible']
        ]
        local_final=result[result['strategy'].eq(strategy)] if not result.empty else pd.DataFrame()
        state=(
            'NO_ELIGIBLE_STATUS' if local_raw.empty
            else 'DATA_INSUFFICIENT' if data_sufficient.empty
            else 'BELOW_MIN_CONVICTION' if local_above.empty
            else 'OUTSIDE_TOP_LIMIT' if local_final.empty
            else 'INCLUDED'
        )
        audit_rows.append({
            'strategy':strategy,
            'eligible_candidates':len(local_raw),
            'data_sufficient_candidates':len(data_sufficient),
            'above_min_conviction':len(local_above),
            'included_primary':len(local_final),
            'max_conviction':round(
                safe_number(
                    local_raw['profit_conviction_score'].max(),
                    np.nan,
                ),
                1,
            ) if not local_raw.empty else np.nan,
            'ranking_state':state,
        })
    ai_cfg=LocalAIConfig(
        enabled=bool(getattr(cfg,'ai_enabled',True)),
        mode=safe_text(getattr(cfg,'ai_mode','HYBRID_GUARDED')) or 'HYBRID_GUARDED',
        max_weight=max(0.0,min(0.35,safe_number(getattr(cfg,'ai_max_weight',0.35),0.35))),
        min_training_events=max(12,int(safe_number(getattr(cfg,'ai_min_training_events',30),30))),
        min_strategy_events=max(8,int(safe_number(getattr(cfg,'ai_min_strategy_events',18),18))),
        knn_k=max(7,int(safe_number(getattr(cfg,'ai_knn_k',21),21))),
        memory_entry_window_bars=max(1,int(safe_number(getattr(cfg,'ai_memory_entry_window_bars',5),5))),
        memory_horizon_bars=max(5,int(safe_number(getattr(cfg,'ai_memory_horizon_bars',20),20))),
        min_expectancy_r=safe_number(getattr(cfg,'execution_ai_min_expectancy_r',0.05),0.05),
        max_oos_drawdown_pct=safe_number(getattr(cfg,'execution_ai_max_drawdown_pct',20.0),20.0),
        min_profit_factor=safe_number(getattr(cfg,'execution_ai_min_profit_factor',1.05),1.05),
    )
    if not result.empty:
        result, ai_audit=enrich_profit_ranking_with_ai(result,validation_events=validation_events,memory_events=ai_memory,config=ai_cfg)
        final_source = 'hybrid_conviction_score' if 'hybrid_conviction_score' in result.columns else 'profit_conviction_score'
        result['final_score'] = pd.to_numeric(result.get(final_source), errors='coerce').fillna(0.0).round(1)
        result['core_priority_score'] = (
            0.70 * pd.to_numeric(result.get('swing_selection_score'), errors='coerce').fillna(result['final_score'])
            + 0.30 * result['final_score']
            + pd.to_numeric(
                result.get(
                    'narrative_swing_rank_adjustment',
                    pd.Series(0.0, index=result.index),
                ),
                errors='coerce',
            ).fillna(0.0)
        ).clip(0.0, 100.0).round(1)
        result['_silent_rank'] = pd.to_numeric(
            result.get('effective_silent_accumulation_score'),
            errors='coerce',
        ).fillna(50.0)
        result = result.sort_values(
            ['core_priority_score','_silent_rank','final_score','target_quality_score','liquidity_score'],
            ascending=[False,False,False,False,False], kind='stable',
        ).drop(columns='_silent_rank').reset_index(drop=True)
        result['profit_rank'] = np.arange(1, len(result)+1)
        selector_selected_reason = result.get(
            'selector_selected_reason', pd.Series('', index=result.index),
        ).fillna('').astype(str)
        narrative_selected_reason = result.get(
            'narrative_primary_reason', pd.Series('', index=result.index),
        ).fillna('').astype(str)
        result['selected_reason'] = [
            ' • '.join(
                part for part in (technical, narrative)
                if safe_text(part)
            )
            for technical, narrative in zip(
                selector_selected_reason, narrative_selected_reason,
            )
        ]
        result['not_entry_reason'] = result.get(
            'warnings', pd.Series('', index=result.index),
        ).fillna('').astype(str)
        result.loc[result['not_entry_reason'].str.len().eq(0), 'not_entry_reason'] = (
            'Setup terdeteksi; tunggu execution gate dan harga trigger.'
        )
        trigger = pd.to_numeric(
            result.get('trigger_price', result.get('entry', pd.Series(np.nan, index=result.index))),
            errors='coerce',
        )
        result['trigger_waiting'] = [
            f'Tunggu trigger terkonfirmasi di {value:,.0f}'
            if np.isfinite(value) else 'Tunggu trigger harga dan volume terkonfirmasi'
            for value in trigger
        ]
        stop = pd.to_numeric(result.get('stop_loss'), errors='coerce')
        result['invalidation_reason'] = [
            f'Invalid bila struktur/penutupan menembus SL {value:,.0f}'
            if np.isfinite(value) else 'Invalidation belum lengkap'
            for value in stop
        ]
        selection_risks = result.get(
            'selector_selection_risks', pd.Series('', index=result.index),
        ).fillna('').astype(str)
        narrative_risks = result.get(
            'narrative_primary_risk', pd.Series('', index=result.index),
        ).fillna('').astype(str)
        narrative_blocks = result.get(
            'narrative_hard_block', pd.Series(False, index=result.index),
        ).fillna(False).map(truthy)
        result['primary_risk'] = [
            (
                safe_text(narrative)
                if blocked
                else safe_text(warning) or safe_text(risk)
                or safe_text(narrative)
                or 'Tidak ada risiko utama terstruktur'
            ).split(' • ')[0]
            for warning, risk, narrative, blocked in zip(
                result['warnings'], selection_risks,
                narrative_risks, narrative_blocks,
            )
        ]
    else:
        ai_audit=pd.DataFrame()
    result.attrs['strategy_audit']=pd.DataFrame(audit_rows)
    result.attrs['ai_audit']=ai_audit
    result.attrs['data_coverage_audit']=coverage_audit
    return result


def build_focus_daily_board(
    core_builder: pd.DataFrame | None,
    multibagger: pd.DataFrame | None,
    per_strategy: int=5,
) -> pd.DataFrame:
    """Create a daily board containing only Core Swing and Multibagger rows."""
    rows: list[dict[str, Any]]=[]
    if core_builder is not None and not core_builder.empty:
        score_col='core_priority_score' if 'core_priority_score' in core_builder else 'swing_selection_score' if 'swing_selection_score' in core_builder else 'final_score' if 'final_score' in core_builder else 'hybrid_conviction_score' if 'hybrid_conviction_score' in core_builder else 'profit_conviction_score'
        sort_cols=[score_col] + (['silent_accumulation_score'] if 'silent_accumulation_score' in core_builder else [])
        ranked=core_builder.sort_values(sort_cols,ascending=[False]*len(sort_cols),kind='stable').head(max(1,int(per_strategy)))
        for _,row in ranked.iterrows():
            rows.append({
                'category':'CORE_SWING',
                'strategy':row.get('strategy',row.get('active_setup','WAIT_SETUP')),
                'ticker':row.get('ticker'),
                'decision_state':row.get('decision_state',row.get('setup_status')),
                'status':row.get('setup_status'),
                'score':safe_number(row.get('final_score',row.get('hybrid_conviction_score',row.get('profit_conviction_score'))),np.nan),
                'entry':row.get('entry'),'stop_loss':row.get('stop_loss'),'tp1':row.get('tp1'),'tp2':row.get('tp2'),
                'rr1':row.get('rr1'),'rr2':row.get('rr2'),
                'next_action':row.get('next_action',row.get('setup_action',row.get('trigger_waiting'))),
                'best_buy_date':row.get('best_buy_date'),'eoff_strength_label':row.get('eoff_strength_label'),
                'silent_accumulation_score':safe_number(row.get('silent_accumulation_score'),0.0),
                'blockers':row.get('warnings',''),
            })
    if multibagger is not None and not multibagger.empty:
        eligible = multibagger.get(
            'research_eligible',
            pd.Series(False, index=multibagger.index),
        ).map(truthy)
        metric_gate = multibagger.get(
            'multibagger_metric_data_gate',
            pd.Series(False, index=multibagger.index),
        ).map(truthy)
        multibagger = multibagger.loc[eligible & metric_gate].copy()
    if multibagger is not None and not multibagger.empty:
        score_col='confidence_adjusted_multibagger_score' if 'confidence_adjusted_multibagger_score' in multibagger else 'capital_conviction_score' if 'capital_conviction_score' in multibagger else 'multibagger_score'
        silent_column = (
            'effective_silent_accumulation_score'
            if 'effective_silent_accumulation_score' in multibagger
            else 'silent_accumulation_score'
        )
        sort_cols=[score_col] + ([silent_column] if silent_column in multibagger else [])
        ranked=multibagger.sort_values(sort_cols,ascending=[False]*len(sort_cols),kind='stable').head(max(1,int(per_strategy)))
        for _,row in ranked.iterrows():
            rows.append({
                'category':'MULTIBAGGER','strategy':'MULTIBAGGER','ticker':row.get('ticker'),
                'decision_state':row.get('compounding_state'),'status':row.get('multibagger_status'),
                'score':safe_number(row.get(score_col),np.nan),'entry':row.get('entry'),
                'stop_loss':row.get('stop_loss'),'tp1':row.get('tp1'),'tp2':row.get('tp2'),
                'rr1':row.get('rr1'),'rr2':row.get('rr2'),'next_action':row.get('allocation_action',row.get('quick_buy_action')),
                'best_buy_date':row.get('best_buy_date'),'eoff_strength_label':row.get('eoff_strength_label'),
                'silent_accumulation_score':safe_number(
                    row.get(
                        'effective_silent_accumulation_score',
                        row.get('silent_accumulation_score'),
                    ),
                    50.0,
                ),
                'blockers':row.get('red_flags',''),
            })
    result=pd.DataFrame(rows)
    if not result.empty:
        result=result.sort_values(['score','silent_accumulation_score','category'],ascending=[False,False,True],kind='stable').reset_index(drop=True)
    return result


def build_focus_screens(
    prepared: Mapping[str, pd.DataFrame],
    fundamentals: pd.DataFrame | None=None,
    core_signals: pd.DataFrame | None=None,
    project_management: pd.DataFrame | None=None,
    news_review: pd.DataFrame | None=None,
    market_status: pd.DataFrame | None=None,
    narrative_events: pd.DataFrame | None=None,
    narrative_outcomes: pd.DataFrame | None=None,
    benchmark: pd.DataFrame | None=None,
    config: ScanConfig | None=None,
    validation_events: pd.DataFrame | None=None,
    ai_memory: pd.DataFrame | None=None,
) -> dict[str, pd.DataFrame]:
    """Build the two production focuses: Multibagger and Core Swing."""
    cfg=config or ScanConfig()
    silent_profiles = current_silent_profiles(prepared)
    selector_config = SelectorConfig(
        training_lookback_bars=max(220, int(safe_number(getattr(cfg, 'selector_training_lookback_bars', 620), 620))),
        anchor_step_bars=max(1, int(safe_number(getattr(cfg, 'selector_anchor_step_bars', 5), 5))),
        min_training_rows=max(60, int(safe_number(getattr(cfg, 'selector_min_training_rows', 180), 180))),
        min_evaluation_rows=max(30, int(safe_number(getattr(cfg, 'selector_min_eval_rows', 60), 60))),
        min_evaluation_dates=max(6, int(safe_number(getattr(cfg, 'selector_min_eval_dates', 12), 12))),
        min_evaluation_tickers=max(8, int(safe_number(getattr(cfg, 'selector_min_eval_tickers', 25), 25))),
        roundtrip_cost_pct=max(0.0, safe_number(getattr(cfg, 'selector_roundtrip_cost_pct', 0.0065), 0.0065)),
        market_impact_enabled=bool(getattr(cfg, 'selector_market_impact_enabled', True)),
        market_impact_very_illiquid_pct=max(0.0, safe_number(getattr(cfg, 'selector_impact_very_illiquid_pct', 0.0150), 0.0150)),
        market_impact_illiquid_pct=max(0.0, safe_number(getattr(cfg, 'selector_impact_illiquid_pct', 0.0075), 0.0075)),
        market_impact_medium_pct=max(0.0, safe_number(getattr(cfg, 'selector_impact_medium_pct', 0.0035), 0.0035)),
        market_impact_liquid_pct=max(0.0, safe_number(getattr(cfg, 'selector_impact_liquid_pct', 0.0015), 0.0015)),
        market_impact_very_liquid_pct=max(0.0, safe_number(getattr(cfg, 'selector_impact_very_liquid_pct', 0.0005), 0.0005)),
        max_ai_weight=max(0.0, min(0.30, safe_number(getattr(cfg, 'selector_max_ai_weight', 0.30), 0.30))),
        max_promotion_drawdown_pct=max(1.0, safe_number(getattr(cfg, 'selector_max_drawdown_pct', 20.0), 20.0)),
        cscv_slices=max(4, int(safe_number(getattr(cfg, 'selector_cscv_slices', 6), 6))),
        max_backtest_overfit_probability_pct=max(0.0, min(100.0, safe_number(getattr(cfg, 'selector_max_pbo_pct', 50.0), 50.0))),
        min_feature_coverage_pct=max(
            0.0,
            min(
                100.0,
                safe_number(
                    getattr(cfg, 'selector_min_feature_coverage_pct', 75.0),
                    75.0,
                ),
            ),
        ),
        min_model_feature_coverage_pct=max(
            0.0,
            min(
                100.0,
                safe_number(
                    getattr(
                        cfg,
                        'selector_min_model_feature_coverage_pct',
                        80.0,
                    ),
                    80.0,
                ),
            ),
        ),
    )
    sector_map: dict[str, str] = {}
    if isinstance(fundamentals, pd.DataFrame) and not fundamentals.empty and 'ticker' in fundamentals:
        for _, row in fundamentals.drop_duplicates('ticker', keep='last').iterrows():
            ticker = safe_text(row.get('ticker')).upper()
            sector = safe_text(row.get('sector'))
            if ticker and sector:
                sector_map[ticker] = sector
    selector, selector_audit, selector_panel = build_cross_sectional_selector(
        prepared, selector_config, silent_profiles, sector_map,
    )
    narrative_intelligence = build_narrative_intelligence(
        prepared=prepared,
        fundamentals=fundamentals,
        news_review=news_review,
        project_management=project_management,
        market_status=market_status,
        existing_events=narrative_events,
        existing_outcomes=narrative_outcomes,
        benchmark=benchmark,
        silent_profiles=silent_profiles,
        scan_config=cfg,
    )
    narrative_profiles = narrative_intelligence.get(
        'profiles', pd.DataFrame(),
    )
    core_radar = attach_setups_to_selector(selector, core_signals)
    core_radar = attach_narrative_profiles(
        core_radar, narrative_profiles,
    )
    if isinstance(core_radar, pd.DataFrame) and not core_radar.empty:
        base_swing = pd.to_numeric(
            core_radar.get(
                'swing_selection_score',
                pd.Series(50.0, index=core_radar.index),
            ),
            errors='coerce',
        ).fillna(50.0)
        narrative_adjustment = pd.to_numeric(
            core_radar.get(
                'narrative_swing_rank_adjustment',
                pd.Series(0.0, index=core_radar.index),
            ),
            errors='coerce',
        ).fillna(0.0)
        core_radar['swing_selection_score_pre_narrative'] = base_swing
        core_radar['swing_selection_score'] = (
            base_swing + narrative_adjustment
        ).clip(0.0, 100.0).round(1)
        narrative_block = core_radar.get(
            'narrative_hard_block',
            pd.Series(False, index=core_radar.index),
        ).fillna(False).map(truthy)
        if 'selector_rank_eligible' in core_radar:
            core_radar['selector_rank_eligible'] = (
                core_radar['selector_rank_eligible'].map(truthy)
                & ~narrative_block
            )
        technical_reason = core_radar.get(
            'selected_reason', pd.Series('', index=core_radar.index),
        ).fillna('').astype(str)
        narrative_reason = core_radar.get(
            'narrative_primary_reason',
            pd.Series('', index=core_radar.index),
        ).fillna('').astype(str)
        core_radar['selected_reason'] = [
            ' • '.join(
                part for part in (technical, narrative)
                if safe_text(part)
            )
            for technical, narrative in zip(
                technical_reason, narrative_reason,
            )
        ]
        core_radar['_narrative_block_sort'] = narrative_block.astype(int)
        core_radar['_selector_eligible_sort'] = (
            core_radar.get(
                'selector_rank_eligible',
                pd.Series(True, index=core_radar.index),
            ).fillna(False).map(truthy).astype(int)
        )
        core_radar = core_radar.sort_values(
            [
                '_selector_eligible_sort', '_narrative_block_sort',
                'swing_selection_score',
                'effective_silent_accumulation_score', 'ticker',
            ],
            ascending=[False, True, False, False, True],
            kind='stable',
            na_position='last',
        ).drop(
            columns=['_selector_eligible_sort', '_narrative_block_sort'],
        ).reset_index(drop=True)
        core_radar['swing_selection_rank'] = np.arange(
            1, len(core_radar) + 1,
        )
    multibagger=scan_multibagger_candidates(
        prepared,fundamentals,core_signals=core_signals,
        project_management=project_management,config=cfg,
        selector_ranking=selector,silent_profiles=silent_profiles,
        narrative_profiles=narrative_profiles,
    )
    multibagger_diagnostics = build_multibagger_diagnostic_views(
        multibagger,
        expected_ticker_count=len(prepared),
        limit=30,
    )
    if isinstance(multibagger, pd.DataFrame) and not multibagger.empty:
        research_mask = multibagger.get(
            'research_eligible',
            pd.Series(False, index=multibagger.index),
        ).fillna(False).astype(bool)
        research_multibagger = multibagger.loc[research_mask].copy()
        growth_compounder = research_multibagger[
            research_multibagger.get(
                'multibagger_lane',
                pd.Series('', index=research_multibagger.index),
            ).eq('GROWTH_COMPOUNDER')
        ].sort_values(
            [
                'growth_compounder_selection_score',
                'effective_silent_accumulation_score',
                'overall_research_confidence',
                'ticker',
            ],
            ascending=[False, False, False, True],
            kind='stable',
            na_position='last',
        ).reset_index(drop=True)
        turnaround = research_multibagger[
            research_multibagger.get(
                'multibagger_lane',
                pd.Series('', index=research_multibagger.index),
            ).eq('TURNAROUND_CYCLICAL')
        ].sort_values(
            [
                'turnaround_selection_score',
                'effective_silent_accumulation_score',
                'overall_research_confidence',
                'ticker',
            ],
            ascending=[False, False, False, True],
            kind='stable',
            na_position='last',
        ).reset_index(drop=True)
    else:
        research_multibagger = pd.DataFrame()
        growth_compounder = pd.DataFrame()
        turnaround = pd.DataFrame()
    core_builder=build_focus_order_builder(
        core_signals,config=cfg,validation_events=validation_events,
        ai_memory=ai_memory,selector_ranking=selector,
        narrative_profiles=narrative_profiles,
    )
    strategy_audit = core_builder.attrs.get('strategy_audit',pd.DataFrame())
    ai_audit = core_builder.attrs.get('ai_audit',pd.DataFrame())
    data_coverage_audit = core_builder.attrs.get(
        'data_coverage_audit',
        pd.DataFrame(),
    )
    # DataFrame attrs can contain DataFrames for internal audit. PyArrow cannot
    # serialise those attrs when Streamlit renders the ranking, so detach them
    # after they have been promoted to explicit result frames.
    core_builder = core_builder.copy()
    core_builder.attrs = {}
    return {
        'multibagger':multibagger,
        'multibagger_research':research_multibagger,
        'multibagger_growth_compounder':growth_compounder,
        'multibagger_turnaround':turnaround,
        'multibagger_coverage':multibagger_diagnostics['coverage'],
        'multibagger_gate_failures':multibagger_diagnostics['gate_failures'],
        'multibagger_growth_near_miss':multibagger_diagnostics['growth_near_miss'],
        'multibagger_turnaround_near_miss':multibagger_diagnostics['turnaround_near_miss'],
        'multibagger_growth_research_queue':multibagger_diagnostics['growth_research_queue'],
        'multibagger_turnaround_research_queue':multibagger_diagnostics['turnaround_research_queue'],
        'multibagger_data_pending':multibagger_diagnostics['data_pending'],
        'core_swing':core_radar,
        'stock_selector':selector,
        'selector_model_audit':selector_audit,
        'narrative_events':narrative_intelligence.get(
            'events', pd.DataFrame(),
        ),
        'narrative_event_outcomes':narrative_intelligence.get(
            'outcomes', pd.DataFrame(),
        ),
        'narrative_profiles':narrative_profiles,
        'narrative_engine_audit':narrative_intelligence.get(
            'audit', pd.DataFrame(),
        ),
        'selector_training_summary':pd.DataFrame([{
            'training_panel_rows':len(selector_panel),
            'training_panel_tickers':int(selector_panel['ticker'].nunique()) if not selector_panel.empty and 'ticker' in selector_panel else 0,
            'training_panel_dates':int(selector_panel['as_of'].nunique()) if not selector_panel.empty and 'as_of' in selector_panel else 0,
            'selector_version':selector.get('selector_version',pd.Series(dtype=str)).iloc[0] if not selector.empty else '',
        }]),
        'profit_order_builder':core_builder,
        'daily_opportunities':build_focus_daily_board(
            core_builder, research_multibagger,
        ),
        'profit_strategy_audit':strategy_audit,
        'profit_data_coverage_audit':data_coverage_audit,
        'ai_model_audit':ai_audit,
    }


__all__ = [
    'parse_project_management_csv', 'collect_automatic_forward_quality',
    'merge_project_management_reviews', 'allocate_multibagger_capital',
    'scan_multibagger_candidates', 'build_multibagger_diagnostic_views',
    'build_scanner_data_contract_audit',
    'build_focus_order_builder',
    'build_focus_daily_board', 'build_focus_screens',
]
