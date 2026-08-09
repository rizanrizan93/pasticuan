-- IDX Super Scanner v9.8.3: database evidence and semantic-hash integrity.
-- The hash token mirrors scanner_database._semantic_hash V2 and is tolerant
-- of PostgreSQL JSONB normalising a Python integer to an equivalent float.

begin;

create or replace function public.scanner_semantic_token_v2(payload jsonb)
returns text
language plpgsql
immutable
strict
parallel safe
security invoker
set search_path = pg_catalog, extensions
as $$
declare
    kind text;
    output text;
    scalar_text text;
    child record;
begin
    kind := jsonb_typeof(payload);
    if kind is null or kind = 'null' then
        return 'Z';
    elsif kind = 'object' then
        output := '{';
        for child in
            select key, value
            from jsonb_each(payload)
            where key <> all (array[
                'as_of', 'scan_id', 'snapshot_id', 'created_at', 'updated_at',
                'database_source_state', 'database_source_checked_at',
                'source_checked_at', 'source_fetched_at', 'fundamental_fetched_at',
                'generated_at', 'last_seen_at', 'age_days', 'refresh_state',
                'refresh_reason', 'next_check_at'
            ])
              and key !~ '(_checked_at|_fetched_at|_generated_at)$'
            order by key collate "C"
        loop
            output := output
                || 'S' || octet_length(convert_to(child.key, 'UTF8'))::text
                || ':' || child.key
                || public.scanner_semantic_token_v2(child.value);
        end loop;
        return output || '}';
    elsif kind = 'array' then
        output := '[';
        for child in
            select value
            from jsonb_array_elements(payload) with ordinality as item(value, position)
            order by position
        loop
            output := output || public.scanner_semantic_token_v2(child.value);
        end loop;
        return output || ']';
    elsif kind = 'boolean' then
        return case when (payload #>> '{}')::boolean then 'T' else 'F' end;
    elsif kind = 'number' then
        return 'N'
            || encode(float8send((payload #>> '{}')::double precision), 'hex')
            || ';';
    end if;

    scalar_text := payload #>> '{}';
    return 'S' || octet_length(convert_to(scalar_text, 'UTF8'))::text
        || ':' || scalar_text;
end;
$$;

create or replace function public.scanner_semantic_hash_v2(payload jsonb)
returns text
language sql
immutable
strict
parallel safe
security invoker
set search_path = pg_catalog, extensions
as $$
    select encode(
        extensions.digest(
            convert_to(public.scanner_semantic_token_v2(payload), 'UTF8'),
            'sha256'
        ),
        'hex'
    )
$$;

comment on function public.scanner_semantic_hash_v2(jsonb) is
    'Internal V2 semantic hash: JSONB numeric-container tolerant; transport timestamps excluded.';

revoke execute on function public.scanner_semantic_token_v2(jsonb) from public, anon, authenticated;
revoke execute on function public.scanner_semantic_hash_v2(jsonb) from public, anon, authenticated;
grant execute on function public.scanner_semantic_token_v2(jsonb) to service_role;
grant execute on function public.scanner_semantic_hash_v2(jsonb) to service_role;

update public.fundamental_cache
set content_hash = public.scanner_semantic_hash_v2(payload)
where payload is not null;

update public.fundamental_history_cache
set content_hash = public.scanner_semantic_hash_v2(payload)
where payload is not null;

update public.forward_quality_cache
set content_hash = public.scanner_semantic_hash_v2(payload)
where payload is not null;

update public.scanner_feature_cache
set content_hash = public.scanner_semantic_hash_v2(payload)
where payload is not null;

commit;
