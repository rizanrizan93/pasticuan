-- Read-only verification for v9.8.3 evidence/hash integrity.
select
    to_regprocedure('public.scanner_semantic_hash_v2(jsonb)') is not null as hash_function_exists,
    has_function_privilege('anon', 'public.scanner_semantic_hash_v2(jsonb)', 'EXECUTE') as anon_execute,
    has_function_privilege('authenticated', 'public.scanner_semantic_hash_v2(jsonb)', 'EXECUTE') as authenticated_execute,
    has_function_privilege('service_role', 'public.scanner_semantic_hash_v2(jsonb)', 'EXECUTE') as service_role_execute;

select 'fundamental_cache' as table_name, count(*) as hash_mismatch
from public.fundamental_cache
where payload is not null
  and content_hash is distinct from public.scanner_semantic_hash_v2(payload)
union all
select 'fundamental_history_cache', count(*)
from public.fundamental_history_cache
where payload is not null
  and content_hash is distinct from public.scanner_semantic_hash_v2(payload)
union all
select 'forward_quality_cache', count(*)
from public.forward_quality_cache
where payload is not null
  and content_hash is distinct from public.scanner_semantic_hash_v2(payload)
union all
select 'scanner_feature_cache', count(*)
from public.scanner_feature_cache
where payload is not null
  and content_hash is distinct from public.scanner_semantic_hash_v2(payload)
order by table_name;
