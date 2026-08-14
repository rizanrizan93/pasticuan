from pathlib import Path

path = Path("scanner.py")
text = path.read_text(encoding="utf-8")
old = '''        statement_shares = _num(row.get('history_shares_outstanding_latest'))\n        ksei_shares = _num(row.get('ksei_total_shares'))\n        ksei_shares_verified = _truthy(row.get('ksei_shares_verified', False))\n        ksei_shares_asof = _as_jakarta_naive_timestamp(\n            row.get('ksei_shares_observed_at') or row.get('ksei_shares_checked_at')\n        )\n'''
new = '''        statement_shares = _num(row.get('history_shares_outstanding_latest'))\n        ksei_shares = _num(row.get('ksei_total_shares'))\n        # KSEI profile ingestion historically persisted verification as\n        # ksei_source_verified, while valuation expected ksei_shares_verified.\n        # They describe the same parsed official profile lineage for the share\n        # count; accept either flag without relaxing the positive/freshness gates.\n        ksei_shares_verified = (\n            _truthy(row.get('ksei_shares_verified', False))\n            or _truthy(row.get('ksei_source_verified', False))\n        )\n        ksei_shares_asof = _as_jakarta_naive_timestamp(\n            row.get('ksei_shares_observed_at')\n            or row.get('ksei_shares_checked_at')\n            or row.get('source_checked_at')\n        )\n'''
if old in text:
    text = text.replace(old, new, 1)
elif "_truthy(row.get('ksei_source_verified', False))" not in text:
    raise SystemExit("KSEI valuation provenance anchor not found")
path.write_text(text, encoding="utf-8")
