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

from scanner import Any, BinaryIO, Iterable, Mapping, ScanConfig, ThreadPoolExecutor, read_csv_input, safe_number, safe_text, silent_accumulation_metrics, truthy, as_completed, math, normalize_idx_ticker, np, pd


from ai_engine import LocalAIConfig, enrich_profit_ranking_with_ai, enrich_multibagger_with_peer_ai
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
            'ownership_pct': ownership if np.isfinite(ownership) else 1.0,
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
    return (pd.concat(evidence, ignore_index=True, sort=False) if evidence else pd.DataFrame(), pd.DataFrame(reports))


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
    out = pd.concat(frames, ignore_index=True, sort=False)
    if 'ticker' in out:
        out['ticker'] = out['ticker'].map(normalize_idx_ticker)
    return out


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
    ownership = max(0.0, min(1.0, safe_number(pm.get('project_ownership_pct'), 1.0)))
    probability = max(0.10, min(0.95, 0.35 + 0.004 * project_score + 0.15 * completion + 0.10 * funding))
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
    confidence = 'HIGH' if quorum and coverage >= 75 and expected_revenue > 0 and expected_ebitda > 0 else 'MEDIUM' if coverage >= 50 and (expected_revenue > 0 or capex > 0) else 'LOW'
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
    }





def _fundamental_records(fundamentals: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if fundamentals is None or fundamentals.empty or 'ticker' not in fundamentals:
        return {}
    return {str(row['ticker']): row.to_dict() for _, row in fundamentals.drop_duplicates('ticker', keep='last').iterrows()}















def _bounded_score(value: Any, maximum: float) -> float:
    """Normalize a non-negative component to a transparent 0-100 scale."""
    numeric = max(0.0, safe_number(value, 0.0))
    return float(max(0.0, min(100.0, 100.0 * numeric / max(1e-9, float(maximum)))))



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
        verified_projects = 0
        for _, item in project_rows.iterrows():
            stage = safe_text(item.get('project_stage')).upper()
            stage_score = stage_scores.get(stage, 20.0 if stage else 0.0)
            completion = 100.0 * max(0.0, min(1.0, safe_number(item.get('project_completion_pct'), 0.0)))
            funding = 100.0 * max(0.0, min(1.0, safe_number(item.get('funding_secured_pct'), 0.0)))
            offtake = 100.0 * max(0.0, min(1.0, safe_number(item.get('offtake_secured_pct'), 0.0)))
            ownership = 100.0 * max(0.0, min(1.0, safe_number(item.get('ownership_pct'), 1.0)))
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
            capex_total += capex
            expected_revenue += max(0.0, safe_number(item.get('expected_revenue_idr'), 0.0))
            expected_ebitda += max(0.0, safe_number(item.get('expected_ebitda_idr'), 0.0))
            metric_weight = capex if capex > 0 else 1.0
            weighted_completion += max(0.0, min(1.0, safe_number(item.get('project_completion_pct'), 0.0))) * metric_weight
            weighted_funding += max(0.0, min(1.0, safe_number(item.get('funding_secured_pct'), 0.0))) * metric_weight
            weighted_offtake += max(0.0, min(1.0, safe_number(item.get('offtake_secured_pct'), 0.0))) * metric_weight
            weighted_ownership += max(0.0, min(1.0, safe_number(item.get('ownership_pct'), 1.0))) * metric_weight
            metric_weight_total += metric_weight
            source_url = safe_text(item.get('source_url'))
            family = safe_text(item.get('source_family'))
            if not family and source_url:
                family, _ = _source_family(source_url)
            if family:
                source_families.add(family)
            if source_url:
                project_source_urls.append(source_url)
            name = safe_text(item.get('project_name'))
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
            'project_capex_idr': capex_total,
            'project_expected_revenue_idr': expected_revenue,
            'project_expected_ebitda_idr': expected_ebitda,
            'project_completion_pct': weighted_completion / metric_weight_total if metric_weight_total > 0 else np.nan,
            'project_funding_secured_pct': weighted_funding / metric_weight_total if metric_weight_total > 0 else np.nan,
            'project_offtake_secured_pct': weighted_offtake / metric_weight_total if metric_weight_total > 0 else np.nan,
            'project_ownership_pct': weighted_ownership / metric_weight_total if metric_weight_total > 0 else 1.0,
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
        'Growth': ('growth_score', 22.0),
        'Profitability': ('profitability_score', 18.0),
        'Cash-flow quality': ('earnings_quality_score', 18.0),
        'Valuation': ('valuation_score', 8.0),
        'Momentum': ('momentum_score', 12.0),
        'Smart-money proxy': ('accumulation_score', 10.0),
    }
    raw_weights: dict[object, float] = {}
    caps: dict[object, float] = {}
    eligible_rows: list[tuple[float, object]] = []
    minimum = max(0.0, min(100.0, cfg.multibagger_min_capital_conviction))

    for idx, row in out.iterrows():
        normalized = {label: _bounded_score(row.get(column), maximum) for label, (column, maximum) in pillar_columns.items()}
        data_integrity = _multibagger_data_integrity_score(row)
        solvency = _multibagger_solvency_strength(row)
        timing = _multibagger_technical_timing_score(row.get('technical_entry_state'))
        project_score = max(0.0, min(100.0, safe_number(row.get('project_pipeline_score'), 50.0)))
        management_score = max(0.0, min(100.0, safe_number(row.get('management_quality_score'), 50.0)))
        future_impact_score = max(0.0, min(100.0, safe_number(row.get('future_fundamental_impact_score'), 50.0)))
        project_coverage = max(0.0, min(100.0, safe_number(row.get('project_data_coverage_effective'), 0.0)))
        management_coverage = max(0.0, min(100.0, safe_number(row.get('management_data_coverage_effective'), 0.0)))
        project_weight = 0.07 * project_coverage / 100.0
        management_weight = 0.07 * management_coverage / 100.0
        impact_confidence = safe_text(row.get('future_impact_confidence')).upper()
        future_impact_weight = 0.08 * {'HIGH': 1.0, 'MEDIUM': 0.65, 'LOW': 0.30}.get(impact_confidence, 0.0)
        cycle_score = max(0.0, min(100.0, safe_number(row.get('multibagger_time_cycle_score'), 50.0)))
        cycle_confidence = max(0.0, min(100.0, safe_number(row.get('time_cycle_confidence'), 0.0)))
        cycle_samples = max(0, int(safe_number(row.get('cycle_validation_samples'), 0.0)))
        cycle_state = safe_text(row.get('time_cycle_state')).upper()
        cycle_cap = max(0.0, min(0.05, safe_number(getattr(cfg, 'time_cycle_multibagger_max_weight', 0.05), 0.05)))
        cycle_weight = (
            cycle_cap * cycle_confidence / 100.0 * min(1.0, cycle_samples / 12.0)
            if bool(getattr(cfg, 'time_cycle_enabled', True)) and cycle_state == 'VALIDATED' and cycle_confidence >= safe_number(getattr(cfg, 'time_cycle_min_confidence', 55.0), 55.0)
            else 0.0
        )
        base_weights = {
            'Growth': 0.15, 'Profitability': 0.11, 'Cash-flow quality': 0.18,
            'Solvency': 0.13, 'Valuation': 0.06, 'Data integrity': 0.08,
            'Momentum': 0.04, 'Smart-money proxy': 0.06, 'Timing': 0.03,
        }
        base_total_weight = sum(base_weights.values())
        base_scale = (1.0 - project_weight - management_weight - future_impact_weight - cycle_weight) / base_total_weight
        rule_conviction = round(
            base_scale * (
                base_weights['Growth'] * normalized['Growth']
                + base_weights['Profitability'] * normalized['Profitability']
                + base_weights['Cash-flow quality'] * normalized['Cash-flow quality']
                + base_weights['Solvency'] * solvency
                + base_weights['Valuation'] * normalized['Valuation']
                + base_weights['Data integrity'] * data_integrity
                + base_weights['Momentum'] * normalized['Momentum']
                + base_weights['Smart-money proxy'] * normalized['Smart-money proxy']
                + base_weights['Timing'] * timing
            )
            + project_weight * project_score
            + management_weight * management_score
            + future_impact_weight * future_impact_score
            + cycle_weight * cycle_score,
            1,
        )
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
        out.at[idx, 'project_capital_weight_pct'] = round(100.0 * project_weight, 2)
        out.at[idx, 'management_capital_weight_pct'] = round(100.0 * management_weight, 2)
        out.at[idx, 'future_impact_capital_weight_pct'] = round(100.0 * future_impact_weight, 2)
        out.at[idx, 'time_cycle_capital_weight_pct'] = round(100.0 * cycle_weight, 2)
        out.at[idx, 'multibagger_time_cycle_score'] = round(cycle_score, 1)
        strongest = sorted({**normalized, 'Data integrity': data_integrity, 'Solvency': solvency, 'Project pipeline': project_score, 'Management': management_score, 'Future impact': future_impact_score, 'Time-cycle': cycle_score}.items(), key=lambda item: item[1], reverse=True)[:3]
        strongest_text = ', '.join(f'{label} {value:.0f}' for label, value in strongest)
        ai_note = f"; peer-AI {ai_peer_score:.1f} weight {ai_weight*100:.1f}%" if ai_weight > 0 else ''
        cycle_note = f"; time-cycle {cycle_score:.1f} weight {cycle_weight*100:.1f}%" if cycle_weight > 0 else ''
        out.at[idx, 'allocation_reason'] = f'{tier}; conviction {conviction:.1f}/100 (rule {rule_conviction:.1f}){ai_note}{cycle_note}; strongest pillars: {strongest_text}'
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


def scan_multibagger_candidates(prepared: Mapping[str, pd.DataFrame], fundamentals: pd.DataFrame | None, core_signals: pd.DataFrame | None=None, project_management: pd.DataFrame | None=None, config: ScanConfig | None=None) -> pd.DataFrame:
    """Rank long-horizon growth/quality candidates; not a return guarantee."""
    cfg = config or ScanConfig()
    f_map = _fundamental_records(fundamentals)
    project_management_map = _project_management_records(project_management)
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
        coverage = safe_number(fund.get('fundamental_coverage'), 0.0)
        if coverage <= 0:
            continue
        row = frame.iloc[-1]
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
        accumulation, up_down = silent_accumulation_metrics(frame)
        growth_score = 0.0
        growth_score += 11.0 if revenue_growth >= 0.2 else 7.0 if revenue_growth >= 0.1 else 3.0 if revenue_growth >= 0 else 0.0
        growth_score += 11.0 if earnings_growth >= 0.25 else 7.0 if earnings_growth >= 0.12 else 3.0 if earnings_growth >= 0 else 0.0
        profitability_score = 0.0
        profitability_score += 7.0 if roe >= 0.2 else 4.5 if roe >= 0.12 else 2.0 if roe >= 0.08 else 0.0
        profitability_score += 3.0 if roa >= 0.08 else 1.5 if roa >= 0.04 else 0.0
        profitability_score += 4.0 if operating_margin >= 0.15 else 2.0 if operating_margin >= 0.08 else 0.0
        profitability_score += 4.0 if net_margin >= 0.12 else 2.0 if net_margin >= 0.06 else 0.0
        earnings_quality_score = 0.0
        earnings_quality_score += 5.0 if 0.8 <= cash_conversion <= 1.8 else 3.0 if cash_conversion >= 0.6 else 0.0
        earnings_quality_score += 4.0 if np.isfinite(fcf) and fcf > 0 else 0.0
        earnings_quality_score += 4.0 if positive_ocf_ratio >= 0.875 else 2.0 if positive_ocf_ratio >= 0.625 else 0.0
        earnings_quality_score += 3.0 if positive_earnings_ratio >= 0.875 else 1.5 if positive_earnings_ratio >= 0.625 else 0.0
        earnings_quality_score += 2.0 if np.isfinite(share_dilution) and share_dilution <= 0.02 else 1.0 if np.isfinite(share_dilution) and share_dilution <= 0.05 else 0.0
        balance_score = 0.0
        if not is_financial:
            balance_score += 4.0 if np.isfinite(debt_equity) and debt_equity <= 0.8 else 2.0 if np.isfinite(debt_equity) and debt_equity <= 1.5 else 0.0
            balance_score += 2.0 if np.isfinite(current_ratio) and current_ratio >= 1.5 else 1.0 if np.isfinite(current_ratio) and current_ratio >= 1.0 else 0.0
            balance_score += 2.0 if np.isfinite(cash_to_debt) and cash_to_debt >= 0.5 else 1.0 if np.isfinite(cash_to_debt) and cash_to_debt >= 0.2 else 0.0
            balance_score += 2.0 if np.isfinite(net_debt_ebitda) and net_debt_ebitda <= 1.5 else 1.0 if np.isfinite(net_debt_ebitda) and net_debt_ebitda <= 3.0 else 0.0
            balance_score += 2.0 if np.isfinite(interest_coverage) and interest_coverage >= 6.0 else 1.0 if np.isfinite(interest_coverage) and interest_coverage >= 3.0 else 0.0
        solvency_fields = (debt_equity, current_ratio, cash_to_debt)
        solvency_coverage = 100.0 * sum((np.isfinite(value) for value in solvency_fields)) / len(solvency_fields)
        valuation_score = 0.0
        valuation_score += 4.0 if np.isfinite(peg) and 0 < peg <= 1.5 else 2.0 if np.isfinite(peg) and 1.5 < peg <= 2.5 else 0.0
        valuation_score += 4.0 if np.isfinite(fcf_yield) and fcf_yield >= 0.04 else 2.0 if np.isfinite(fcf_yield) and fcf_yield > 0 else 0.0
        momentum_score = 0.0
        momentum_score += 4.0 if roc60 >= 0.15 else 2.5 if roc60 >= 0.07 else 0.0
        momentum_score += 3.0 if roc120 >= 0.25 else 2.0 if roc120 >= 0.12 else 0.0
        momentum_score += 2.0 if rs60 > 0 else 0.0
        momentum_score += 2.0 if dist_high >= -0.15 else 1.0 if dist_high >= -0.3 else 0.0
        momentum_score += 1.0 if close > safe_number(row.get('EMA200'), float('inf')) else 0.0
        accumulation_score = 0.0
        accumulation_score += 5.0 if accumulation >= 80 else 3.0 if accumulation >= 65 else 0.0
        accumulation_score += 3.0 if cmf_v > 0 and obv_up else 0.0
        accumulation_score += 2.0 if adtv >= 3000000000 else 1.0 if adtv >= 1000000000 else 0.0

        observed_project = safe_number(pm.get('project_pipeline_score_observed'), np.nan)
        observed_management = safe_number(pm.get('management_quality_score_observed'), np.nan)
        project_score = observed_project if np.isfinite(observed_project) else safe_number(forward_proxy.get('project_pipeline_score_proxy'), 45.0)
        management_score = observed_management if np.isfinite(observed_management) else safe_number(forward_proxy.get('management_quality_score_proxy'), 45.0)
        project_coverage = safe_number(pm.get('project_data_coverage'), 0.0) if np.isfinite(observed_project) else safe_number(forward_proxy.get('project_proxy_coverage'), 0.0)
        management_coverage = safe_number(pm.get('management_data_coverage'), 0.0) if np.isfinite(observed_management) else safe_number(forward_proxy.get('management_proxy_coverage'), 0.0)
        project_source = safe_text(pm.get('project_data_source')) if np.isfinite(observed_project) else 'AUTOMATIC_CAPEX_PROXY'
        management_source = safe_text(pm.get('management_data_source')) if np.isfinite(observed_management) else 'AUTOMATIC_OPERATING_TRACK_RECORD_PROXY'
        ceo_name = safe_text(pm.get('ceo_name_reviewed')) or safe_text(fund.get('ceo_name'))

        time_cycle = analyze_time_cycle(
            frame,
            TimeCycleConfig(
                min_bars=int(getattr(cfg, 'time_cycle_min_history_bars', 180)),
                lunar_enabled=bool(getattr(cfg, 'time_cycle_lunar_enabled', True)),
                eoff_enabled=bool(getattr(cfg, 'eoff_enabled', True)),
                eoff_ephemeris_enabled=bool(getattr(cfg, 'eoff_ephemeris_enabled', True)),
                eoff_min_fib_cluster=int(getattr(cfg, 'eoff_min_fib_cluster', 4)),
                eoff_aspect_orb_deg=float(getattr(cfg, 'eoff_aspect_orb_deg', 3.0)),
                eoff_require_astro_fib_confluence=bool(getattr(cfg, 'eoff_require_astro_fib_confluence', True)),
            ),
        ) if bool(getattr(cfg, 'time_cycle_enabled', True)) else {}
        multibagger_time_score = max(
            safe_number(time_cycle.get('bullish_timing_score'), 50.0),
            safe_number(time_cycle.get('continuation_timing_score'), 50.0),
        )
        base_total = growth_score + profitability_score + earnings_quality_score + balance_score + valuation_score + momentum_score + accumulation_score
        project_weight = 0.09 * max(0.0, min(1.0, project_coverage / 100.0))
        management_weight = 0.09 * max(0.0, min(1.0, management_coverage / 100.0))
        impact_confidence = safe_text(future_impact.get('future_impact_confidence')).upper()
        impact_coverage = {'HIGH': 1.0, 'MEDIUM': 0.65, 'LOW': 0.30}.get(impact_confidence, 0.0)
        impact_weight = 0.08 * impact_coverage
        base_weight = max(0.0, 1.0 - project_weight - management_weight - impact_weight)
        total = (
            base_weight * base_total
            + project_weight * project_score
            + management_weight * management_score
            + impact_weight * safe_number(future_impact.get('future_fundamental_impact_score'), 50.0)
        )
        severe_flags = any((flag in red_flags for flag in ('Margin bersih negatif', 'OCF negatif', 'DER tinggi')))
        governance_flags = safe_text(pm.get('management_governance_flags'))
        related_party_risk = safe_text(pm.get('management_related_party_risk')).upper()
        project_execution_flags = safe_text(pm.get('project_execution_flags'))
        severe_flags = bool(
            severe_flags or fundamental_conflicts or (np.isfinite(share_dilution) and share_dilution > 0.12)
            or governance_flags or related_party_risk == 'CRITICAL'
        )
        if severe_flags:
            total = min(total, 69.0)
        if np.isfinite(market_cap) and market_cap < 300000000000:
            total -= 4.0
        total = max(0.0, min(100.0, total))
        if total < 60:
            continue
        statement_current = bool(np.isfinite(statement_age_days) and 0 <= statement_age_days <= cfg.max_statement_age_days)
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
        forward_quality_coverage = min(100.0, 0.5 * project_coverage + 0.5 * management_coverage)
        forward_quality_score = (project_score + management_score) / 2.0
        forward_quality_gate = bool(
            forward_quality_coverage < 50.0
            or (project_score >= 50.0 and management_score >= 50.0 and not governance_flags and related_party_risk != 'CRITICAL')
        )
        if total >= 82 and fundamental_score_10 >= 8.0 and fundamental_data_gate and grade_a_gate and forward_quality_gate and (adtv >= 1500000000) and (not severe_flags) and (general_solvency_gate or bank_prudential_gate):
            status = 'MULTIBAGGER_A_CANDIDATE'
        elif total >= 72 and fundamental_score_10 >= 7.0 and fundamental_data_grade in {'A', 'B', 'C'} and coverage >= 60 and (not severe_flags):
            status = 'MULTIBAGGER_B_CANDIDATE'
        else:
            status = 'MULTIBAGGER_WATCHLIST'
        sig = signal_map.get(ticker, {})
        technical_entry_state = safe_text(sig.get('status')) or 'NO_ACTIVE_ENTRY_SETUP'
        actionable_entry_states = {
            'EXECUTION_READY', 'READY_FOR_STOCKBIT_VERIFY',
            'ENTRY_PLAN_READY', 'READY_FOR_PRICE_VERIFY',
        }
        if status == 'MULTIBAGGER_A_CANDIDATE' and technical_entry_state in actionable_entry_states:
            compounding_state = 'ACCUMULATE_NOW'
            review_action = 'ADD_GRADUALLY_WITH_CORE_ENTRY_PLAN'
        elif status == 'MULTIBAGGER_A_CANDIDATE':
            compounding_state = 'WAIT_ACCUMULATION_ZONE'
            review_action = 'KEEP_REALIZED_PROFIT_AS_COMPOUNDING_CASH'
        elif status == 'MULTIBAGGER_B_CANDIDATE' and technical_entry_state in actionable_entry_states and total >= 78:
            compounding_state = 'STARTER_NOW'
            review_action = 'OPEN_SMALL_STARTER_ONLY; ADD AFTER QUARTERLY CONFIRMATION'
        elif status == 'MULTIBAGGER_B_CANDIDATE':
            compounding_state = 'RESEARCH_AND_WAIT'
            review_action = 'VERIFY_QUARTERLY_TREND_BEFORE_ADDING'
        else:
            compounding_state = 'RESEARCH_ONLY'
            review_action = 'NO_COMPOUNDING_ALLOCATION'
        max_allocation = min(cfg.max_position_pct, 0.20 if status == 'MULTIBAGGER_A_CANDIDATE' else 0.12 if status == 'MULTIBAGGER_B_CANDIDATE' else 0.0)
        rows.append({'ticker': ticker, 'multibagger_status': status, 'multibagger_score': round(total, 1), 'growth_score': round(growth_score, 1), 'profitability_score': round(profitability_score, 1), 'earnings_quality_score': round(earnings_quality_score, 1), 'balance_sheet_score': round(balance_score, 1), 'valuation_score': round(valuation_score, 1), 'momentum_score': round(momentum_score, 1), 'accumulation_score': round(accumulation_score, 1), 'base_multibagger_score': round(base_total, 1), 'project_pipeline_score': round(project_score, 1), 'project_data_coverage_effective': round(project_coverage, 1), 'project_data_source': project_source, 'project_count': int(safe_number(pm.get('project_count'), 0.0)), 'project_names': safe_text(pm.get('project_names')), 'project_capex_idr': safe_number(pm.get('project_capex_idr'), 0.0), 'project_expected_revenue_idr': safe_number(pm.get('project_expected_revenue_idr'), 0.0), 'project_expected_ebitda_idr': safe_number(pm.get('project_expected_ebitda_idr'), 0.0), 'project_execution_flags': project_execution_flags, 'project_proxy_basis': safe_text(forward_proxy.get('project_proxy_basis')), 'management_quality_score': round(management_score, 1), 'management_data_coverage_effective': round(management_coverage, 1), 'management_data_source': management_source, 'ceo_name': ceo_name, 'ceo_title': safe_text(fund.get('ceo_title')), 'management_governance_flags': governance_flags, 'management_related_party_risk': related_party_risk, 'management_proxy_basis': safe_text(forward_proxy.get('management_proxy_basis')), 'forward_quality_score': round(forward_quality_score, 1), 'forward_quality_coverage': round(forward_quality_coverage, 1), 'forward_quality_gate': forward_quality_gate, 'future_fundamental_impact_score': future_impact.get('future_fundamental_impact_score'), 'future_impact_confidence': future_impact.get('future_impact_confidence'), 'future_impact_model': future_impact.get('future_impact_model'), 'future_impact_horizon': future_impact.get('future_impact_horizon'), 'future_revenue_uplift_bear_pct': future_impact.get('future_revenue_uplift_bear_pct'), 'future_revenue_uplift_base_pct': future_impact.get('future_revenue_uplift_base_pct'), 'future_revenue_uplift_bull_pct': future_impact.get('future_revenue_uplift_bull_pct'), 'future_ebitda_uplift_base_pct': future_impact.get('future_ebitda_uplift_base_pct'), 'future_net_profit_uplift_base_pct': future_impact.get('future_net_profit_uplift_base_pct'), 'future_fcf_pressure_idr': future_impact.get('future_fcf_pressure_idr'), 'future_net_debt_change_idr': future_impact.get('future_net_debt_change_idr'), 'future_net_debt_change_pct': future_impact.get('future_net_debt_change_pct'), 'project_success_probability_pct': future_impact.get('project_success_probability_pct'), 'project_source_families': safe_text(pm.get('project_source_families')), 'project_source_urls': safe_text(pm.get('project_source_urls')), 'project_source_quorum_verified': truthy(pm.get('project_source_quorum_verified')), 'management_source_urls': safe_text(pm.get('management_source_urls')), 'multibagger_time_cycle_score': round(multibagger_time_score, 1), 'time_cycle_score': time_cycle.get('time_cycle_score'), 'time_cycle_confidence': time_cycle.get('time_cycle_confidence'), 'time_cycle_state': time_cycle.get('time_cycle_state'), 'time_cycle_direction_bias': time_cycle.get('time_cycle_direction_bias'), 'time_cycle_phase': time_cycle.get('time_cycle_phase'), 'dominant_cycle_bars': time_cycle.get('dominant_cycle_bars'), 'cycle_historical_hit_rate': time_cycle.get('cycle_historical_hit_rate'), 'cycle_validation_samples': time_cycle.get('cycle_validation_samples'), 'next_reversal_window_start': time_cycle.get('next_reversal_window_start'), 'next_reversal_window_end': time_cycle.get('next_reversal_window_end'), 'bars_to_reversal_window': time_cycle.get('bars_to_reversal_window'), 'lunar_phase': time_cycle.get('lunar_phase'), 'lunar_days_to_major_marker': time_cycle.get('lunar_days_to_major_marker'), 'time_cycle_explanation': time_cycle.get('time_cycle_explanation'), 'quick_buy_state': time_cycle.get('quick_buy_state'), 'quick_buy_action': time_cycle.get('quick_buy_action'), 'best_buy_date': time_cycle.get('best_buy_date'), 'best_buy_date_basis': time_cycle.get('best_buy_date_basis'), 'best_buy_window_start': time_cycle.get('best_buy_window_start'), 'best_buy_window_end': time_cycle.get('best_buy_window_end'), 'best_buy_score': time_cycle.get('best_buy_score'), 'best_buy_confidence': time_cycle.get('best_buy_confidence'), 'best_buy_entry_low': time_cycle.get('best_buy_entry_low'), 'best_buy_entry_high': time_cycle.get('best_buy_entry_high'), 'best_buy_trigger': time_cycle.get('best_buy_trigger'), 'best_buy_stop_loss': time_cycle.get('best_buy_stop_loss'), 'best_buy_tp1': time_cycle.get('best_buy_tp1'), 'best_buy_tp2': time_cycle.get('best_buy_tp2'), 'best_buy_rr1': time_cycle.get('best_buy_rr1'), 'best_buy_rr2': time_cycle.get('best_buy_rr2'), 'best_buy_order_plan': time_cycle.get('best_buy_order_plan'), 'best_buy_reason': time_cycle.get('best_buy_reason'), 'best_buy_no_trade_condition': time_cycle.get('best_buy_no_trade_condition'), 'best_buy_summary': time_cycle.get('best_buy_summary'), 'eoff_state': time_cycle.get('eoff_state'), 'eoff_reconstruction_score': time_cycle.get('eoff_reconstruction_score'), 'eoff_strength_label': time_cycle.get('eoff_strength_label'), 'eoff_signal_active': time_cycle.get('eoff_signal_active'), 'eoff_direction_bias': time_cycle.get('eoff_direction_bias'), 'eoff_time_power_score': time_cycle.get('eoff_time_power_score'), 'eoff_price_power_score': time_cycle.get('eoff_price_power_score'), 'eoff_pattern_score': time_cycle.get('eoff_pattern_score'), 'eoff_momentum_score': time_cycle.get('eoff_momentum_score'), 'eoff_astro_score': time_cycle.get('eoff_astro_score'),'eoff_core_astro_score': time_cycle.get('eoff_core_astro_score'), 'eoff_adaptive_astro_score': time_cycle.get('eoff_adaptive_astro_score'), 'eoff_adaptive_total_weight_pct': time_cycle.get('eoff_adaptive_total_weight_pct'), 'eoff_adaptive_active_factors': time_cycle.get('eoff_adaptive_active_factors'), 'eoff_adaptive_validation_state': time_cycle.get('eoff_adaptive_validation_state'), 'eoff_validation_path': time_cycle.get('eoff_validation_path'), 'eoff_fib_cluster_count': time_cycle.get('eoff_fib_cluster_count'), 'eoff_fib_unique_anchor_count': time_cycle.get('eoff_fib_unique_anchor_count'), 'eoff_historical_hit_rate': time_cycle.get('eoff_historical_hit_rate'), 'eoff_historical_baseline_rate': time_cycle.get('eoff_historical_baseline_rate'), 'eoff_historical_lift': time_cycle.get('eoff_historical_lift'), 'eoff_confluence_historical_hit_rate': time_cycle.get('eoff_confluence_historical_hit_rate'), 'eoff_confluence_historical_events': time_cycle.get('eoff_confluence_historical_events'), 'eoff_confluence_historical_lift': time_cycle.get('eoff_confluence_historical_lift'), 'eoff_historical_events': time_cycle.get('eoff_historical_events'), 'eoff_reversal_date': time_cycle.get('eoff_reversal_date'), 'eoff_ephemeris_state': time_cycle.get('eoff_ephemeris_state'), 'eoff_ephemeris_date': time_cycle.get('eoff_ephemeris_date'), 'eoff_astro_events': time_cycle.get('eoff_astro_events'), 'eoff_active_aspects': time_cycle.get('eoff_active_aspects'), 'eoff_retrograde_planets': time_cycle.get('eoff_retrograde_planets'), 'eoff_retrograde_transition_events': time_cycle.get('eoff_retrograde_transition_events'), 'eoff_stationary_planets': time_cycle.get('eoff_stationary_planets'), 'eoff_ingress_events': time_cycle.get('eoff_ingress_events'), 'eoff_moon_declination_deg': time_cycle.get('eoff_moon_declination_deg'), 'eoff_moon_phase': time_cycle.get('eoff_moon_phase'), 'eoff_sun_sign': time_cycle.get('eoff_sun_sign'), 'eoff_sun_annual_cycle_bias': time_cycle.get('eoff_sun_annual_cycle_bias'), 'eoff_roadmap_json': time_cycle.get('eoff_roadmap_json'), 'eoff_internal_weight_pct': time_cycle.get('eoff_internal_weight_pct'), 'eoff_explanation': time_cycle.get('eoff_explanation'), 'fundamental_coverage': coverage, 'fundamental_score': fund.get('fundamental_score'), 'fundamental_score_10': fundamental_score_10, 'fundamental_reliability': fundamental_reliability, 'fundamental_data_grade': fundamental_data_grade, 'fundamental_source_count': source_count, 'fundamental_source_families': source_families, 'fundamental_history_quarters': history_quarters, 'fundamental_history_years': history_years, 'fundamental_history_coverage': history_coverage, 'fundamental_consensus_score': consensus_score, 'fundamental_conflicts': fundamental_conflicts, 'fundamental_official_reference': official_reference, 'fundamental_official_verified': official_verified, 'statement_age_days': statement_age_days, 'statement_current': statement_current, 'statement_age_state': 'CURRENT' if statement_current else 'UNKNOWN' if not np.isfinite(statement_age_days) else 'STALE', 'peg_valid_for_valuation': bool(np.isfinite(peg) and peg > 0), 'fundamental_data_gate': fundamental_data_gate, 'grade_a_gate': grade_a_gate, 'severe_fundamental_flags': severe_flags, 'revenue_growth': revenue_growth, 'earnings_growth': earnings_growth, 'roe': roe, 'roa': roa, 'net_margin': net_margin, 'debt_equity': debt_equity, 'current_ratio': current_ratio, 'cash_to_debt': cash_to_debt, 'operating_cash_flow': ocf, 'free_cash_flow': fcf, 'cash_conversion_ttm': cash_conversion, 'positive_ocf_ratio': positive_ocf_ratio, 'positive_earnings_ratio': positive_earnings_ratio, 'margin_stability': margin_stability, 'share_dilution_yoy': share_dilution, 'roic_proxy': roic_proxy, 'net_debt_ebitda': net_debt_ebitda, 'interest_coverage': interest_coverage, 'solvency_coverage': round(solvency_coverage, 1), 'fundamental_model': fundamental_model, 'car': car, 'npl_gross': npl_gross, 'ldr': ldr, 'bank_prudential_gate': bank_prudential_gate, 'peg_ratio': peg, 'fcf_yield': fcf_yield, 'market_cap': market_cap, 'last_price': close, 'roc60': roc60, 'roc120': roc120, 'relative_strength60': rs60, 'distance_52w_high': dist_high, 'silent_accumulation_score': accumulation, 'up_down_value_ratio20': up_down, 'adtv20_idr': adtv, 'active_setup': sig.get('setup', ''), 'technical_entry_state': technical_entry_state, 'entry': sig.get('entry', np.nan), 'stop_loss': sig.get('stop_loss', np.nan), 'tp1': sig.get('tp1', np.nan), 'tp2': sig.get('tp2', np.nan), 'compounding_state': compounding_state, 'review_action': review_action, 'profit_allocation_pct': 100.0 * cfg.multibagger_profit_allocation_pct, 'max_position_pct_equity': 100.0 * max_allocation, 'horizon': '12–36 months; quarterly review', 'red_flags': ' • '.join(part for part in (red_flags, fundamental_conflicts, governance_flags, project_execution_flags) if part), 'note': 'Bank grade A requires CAR/NPL/LDR history plus verified IDX/XBRL and multi-source consensus' if is_financial else 'Candidate ranking, not a forecast or guaranteed multiple'})
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
        rank = {'MULTIBAGGER_A_CANDIDATE': 0, 'MULTIBAGGER_B_CANDIDATE': 1, 'MULTIBAGGER_WATCHLIST': 2}
        result['_rank'] = result['multibagger_status'].map(rank).fillna(9)
        result = result.sort_values(['_rank', 'multibagger_score', 'adtv20_idr'], ascending=[True, False, False]).drop(columns='_rank').reset_index(drop=True)
        result = allocate_multibagger_capital(result, cfg)
        result['_capital_rank_sort'] = pd.to_numeric(result.get('capital_priority_rank'), errors='coerce').fillna(9999)
        result = result.sort_values(['_capital_rank_sort', 'capital_conviction_score', 'multibagger_score'], ascending=[True, False, False]).drop(columns='_capital_rank_sort').reset_index(drop=True)
    return result




















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


def build_focus_order_builder(
    core_signals: pd.DataFrame | None,
    config: ScanConfig | None=None,
    validation_events: pd.DataFrame | None=None,
    ai_memory: pd.DataFrame | None=None,
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
        return empty

    eligible_status = {'EXECUTION_READY', 'READY_FOR_STOCKBIT_VERIFY', 'SIGNAL_READY', 'ENTRY_PLAN_READY', 'READY_FOR_PRICE_VERIFY'}
    for setup_name, spec in specs.items():
        setup_series = core_signals.get('setup', pd.Series(index=core_signals.index, dtype=object))
        local = core_signals[setup_series.eq(setup_name)].copy()
        status_series = local.get('status', pd.Series(index=local.index, dtype=object))
        local = local[status_series.isin(eligible_status)]
        for _, row in local.iterrows():
            status = safe_text(row.get('status'))
            action = safe_text(row.get('action'))
            quality = safe_number(row.get('quality_score'), 0.0)
            analyst = safe_number(row.get('analyst_fusion_score'), quality)
            structural = safe_number(row.get('structural_quality_score'), quality)
            confirmation = safe_number(row.get('confirmation_quality_score'), 60.0)
            demand = safe_number(row.get('supply_demand_score'), safe_number(row.get('silent_accumulation_score'), 50.0))
            failure_risk = max(0.0, min(100.0, safe_number(row.get('failure_risk_score'), 0.0)))
            structure = max(0.0, min(100.0, 0.30*quality + 0.22*analyst + 0.20*structural + 0.16*demand + 0.12*confirmation - 0.20*failure_risk))
            distance = safe_number(row.get('distance_atr'), 99.0)
            extension = max(0.0, safe_number(row.get('extension_atr'), 0.0))
            action_score = spec['action_scores'].get(action, 45.0)
            status_score = {'EXECUTION_READY': 100.0, 'READY_FOR_STOCKBIT_VERIFY': 96.0, 'SIGNAL_READY': 86.0, 'ENTRY_PLAN_READY': 72.0, 'READY_FOR_PRICE_VERIFY': 82.0}.get(status, 50.0)
            proximity = max(0.0, 100.0 - 35.0*max(0.0, distance)) if np.isfinite(distance) else 30.0
            extension_score = max(0.0, 100.0 - 35.0*extension)
            timing = 0.38*action_score + 0.28*status_score + 0.20*proximity + 0.14*extension_score
            silent = safe_number(row.get('silent_accumulation_score'), 50.0)
            cmf_score = max(0.0, min(100.0, 50.0 + 500.0*safe_number(row.get('cmf20'), 0.0)))
            volume_score = max(0.0, min(100.0, 35.0 + 32.0*safe_number(row.get('volume_ratio'), 0.0)))
            flow = 0.40*demand + 0.30*silent + 0.15*cmf_score + 0.15*volume_score
            liquidity = _focus_liquidity_score(row.get('adtv20_idr'))
            target = _focus_target_score(row.get('rr1'), row.get('rr2'), row.get('target_structure_valid', True))
            completeness = safe_number(row.get('data_completeness_score'), 50.0)
            execution_conf = safe_number(row.get('execution_confidence_score'), completeness)
            data = 0.55*completeness + 0.45*execution_conf
            validation = safe_number(row.get('validation_gate_score'), 50.0)
            probability = safe_number(row.get('probability_estimate'), np.nan)
            if np.isfinite(probability):
                probability_pct = 100.0*probability if 0.0 <= probability <= 1.0 else probability
                validation = 0.55*validation + 0.45*max(0.0, min(100.0, probability_pct))
            components = (structure, timing, flow, liquidity, target, data, validation)
            score = float(sum(w*max(0.0, min(100.0, c)) for w,c in zip(spec['weights'], components)))
            if not truthy(row.get('target_structure_valid', True)):
                score -= 15.0
            if safe_number(row.get('rr2'), np.nan) < 1.0:
                score -= 8.0
            pre_time = max(0.0, min(100.0, score))
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
            ) if part)
            rows.append({
                'profit_rank': np.nan,
                'ticker': row.get('ticker'),
                'strategy': setup_name,
                'horizon': spec['horizon'],
                'decision_state': decision,
                'setup_status': status,
                'profit_conviction_score': final,
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
                'entry': safe_number(row.get('entry'), np.nan),
                'trigger_price': safe_number(row.get('stockbit_trigger_price', row.get('trigger')), np.nan),
                'stop_loss': safe_number(row.get('stop_loss'), np.nan),
                'tp1': safe_number(row.get('tp1'), np.nan),
                'tp2': safe_number(row.get('tp2'), np.nan),
                'rr1': safe_number(row.get('rr1'), np.nan),
                'rr2': safe_number(row.get('rr2'), np.nan),
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
        return result
    raw=result.copy()
    result=result[result['profit_conviction_score']>=minimum].copy()
    result=result.sort_values(['profit_conviction_score','target_quality_score','liquidity_score'],ascending=[False,False,False])
    alternatives=result.groupby('ticker')['strategy'].apply(lambda v:' • '.join(dict.fromkeys(map(str,v)))).to_dict() if not result.empty else {}
    result=result.drop_duplicates(['ticker','strategy'],keep='first').copy()
    if not result.empty:
        result['alternate_strategies']=result['ticker'].map(alternatives)
        result['candidate_id']=result.apply(lambda r:f"{safe_text(r.get('ticker'))}|{safe_text(r.get('strategy'))}",axis=1)
        result['strategy_rank']=result.groupby('strategy')['profit_conviction_score'].rank(method='first',ascending=False).astype(int)
        result=result.head(int(getattr(cfg,'profit_order_builder_limit',20))).reset_index(drop=True)
        result['profit_rank']=np.arange(1,len(result)+1)
    for strategy in specs:
        local_raw=raw[raw['strategy'].eq(strategy)]
        local_above=local_raw[local_raw['profit_conviction_score']>=minimum]
        local_final=result[result['strategy'].eq(strategy)] if not result.empty else pd.DataFrame()
        state='NO_ELIGIBLE_STATUS' if local_raw.empty else 'BELOW_MIN_CONVICTION' if local_above.empty else 'OUTSIDE_TOP_LIMIT' if local_final.empty else 'INCLUDED'
        audit_rows.append({'strategy':strategy,'eligible_candidates':len(local_raw),'above_min_conviction':len(local_above),'included_primary':len(local_final),'max_conviction':round(safe_number(local_raw['profit_conviction_score'].max(),np.nan),1) if not local_raw.empty else np.nan,'ranking_state':state})
    ai_cfg=LocalAIConfig(
        enabled=bool(getattr(cfg,'ai_enabled',True)),
        mode=safe_text(getattr(cfg,'ai_mode','HYBRID_GUARDED')) or 'HYBRID_GUARDED',
        max_weight=max(0.0,min(0.35,safe_number(getattr(cfg,'ai_max_weight',0.35),0.35))),
        min_training_events=max(12,int(safe_number(getattr(cfg,'ai_min_training_events',30),30))),
        min_strategy_events=max(8,int(safe_number(getattr(cfg,'ai_min_strategy_events',18),18))),
        knn_k=max(7,int(safe_number(getattr(cfg,'ai_knn_k',21),21))),
        memory_entry_window_bars=max(1,int(safe_number(getattr(cfg,'ai_memory_entry_window_bars',5),5))),
        memory_horizon_bars=max(5,int(safe_number(getattr(cfg,'ai_memory_horizon_bars',20),20))),
    )
    if not result.empty:
        result, ai_audit=enrich_profit_ranking_with_ai(result,validation_events=validation_events,memory_events=ai_memory,config=ai_cfg)
    else:
        ai_audit=pd.DataFrame()
    result.attrs['strategy_audit']=pd.DataFrame(audit_rows)
    result.attrs['ai_audit']=ai_audit
    return result


def build_focus_daily_board(
    core_builder: pd.DataFrame | None,
    multibagger: pd.DataFrame | None,
    per_strategy: int=5,
) -> pd.DataFrame:
    """Create a daily board containing only Core Swing and Multibagger rows."""
    rows: list[dict[str, Any]]=[]
    if core_builder is not None and not core_builder.empty:
        ranked=core_builder.sort_values('hybrid_conviction_score' if 'hybrid_conviction_score' in core_builder else 'profit_conviction_score',ascending=False).head(max(1,int(per_strategy)))
        for _,row in ranked.iterrows():
            rows.append({
                'category':'CORE_SWING','strategy':row.get('strategy'),'ticker':row.get('ticker'),
                'decision_state':row.get('decision_state'),'status':row.get('setup_status'),
                'score':safe_number(row.get('hybrid_conviction_score',row.get('profit_conviction_score')),np.nan),
                'entry':row.get('entry'),'stop_loss':row.get('stop_loss'),'tp1':row.get('tp1'),'tp2':row.get('tp2'),
                'rr1':row.get('rr1'),'rr2':row.get('rr2'),'next_action':row.get('next_action'),
                'best_buy_date':row.get('best_buy_date'),'eoff_strength_label':row.get('eoff_strength_label'),
                'blockers':row.get('warnings',''),
            })
    if multibagger is not None and not multibagger.empty:
        score_col='capital_conviction_score' if 'capital_conviction_score' in multibagger else 'multibagger_score'
        ranked=multibagger.sort_values(score_col,ascending=False).head(max(1,int(per_strategy)))
        for _,row in ranked.iterrows():
            rows.append({
                'category':'MULTIBAGGER','strategy':'MULTIBAGGER','ticker':row.get('ticker'),
                'decision_state':row.get('compounding_state'),'status':row.get('multibagger_status'),
                'score':safe_number(row.get(score_col),np.nan),'entry':row.get('entry'),
                'stop_loss':row.get('stop_loss'),'tp1':row.get('tp1'),'tp2':row.get('tp2'),
                'rr1':row.get('rr1'),'rr2':row.get('rr2'),'next_action':row.get('allocation_action',row.get('quick_buy_action')),
                'best_buy_date':row.get('best_buy_date'),'eoff_strength_label':row.get('eoff_strength_label'),
                'blockers':row.get('red_flags',''),
            })
    result=pd.DataFrame(rows)
    if not result.empty:
        result=result.sort_values(['score','category'],ascending=[False,True]).reset_index(drop=True)
    return result


def build_focus_screens(
    prepared: Mapping[str, pd.DataFrame],
    fundamentals: pd.DataFrame | None=None,
    core_signals: pd.DataFrame | None=None,
    project_management: pd.DataFrame | None=None,
    config: ScanConfig | None=None,
    validation_events: pd.DataFrame | None=None,
    ai_memory: pd.DataFrame | None=None,
) -> dict[str, pd.DataFrame]:
    """Build the two production focuses: Multibagger and Core Swing."""
    cfg=config or ScanConfig()
    multibagger=scan_multibagger_candidates(prepared,fundamentals,core_signals=core_signals,project_management=project_management,config=cfg)
    core_builder=build_focus_order_builder(core_signals,config=cfg,validation_events=validation_events,ai_memory=ai_memory)
    return {
        'multibagger':multibagger,
        'core_swing':core_builder,
        'profit_order_builder':core_builder,
        'daily_opportunities':build_focus_daily_board(core_builder,multibagger),
        'profit_strategy_audit':core_builder.attrs.get('strategy_audit',pd.DataFrame()),
        'ai_model_audit':core_builder.attrs.get('ai_audit',pd.DataFrame()),
    }


__all__ = [
    'parse_project_management_csv', 'collect_automatic_forward_quality',
    'merge_project_management_reviews', 'allocate_multibagger_capital',
    'scan_multibagger_candidates', 'build_focus_order_builder',
    'build_focus_daily_board', 'build_focus_screens',
]
