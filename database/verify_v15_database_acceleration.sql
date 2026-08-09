select table_name, column_name
from information_schema.columns
where table_schema='public'
  and (
    (table_name='ohlcv_daily_cache' and column_name in ('payload_compact','payload_codec','compact_bar_count','compact_hash'))
    or
    (table_name='scanner_feature_cache' and column_name in ('ticker','last_bar_date','feature_state','scanner_version','feature_schema_version','payload'))
  )
order by table_name, column_name;
