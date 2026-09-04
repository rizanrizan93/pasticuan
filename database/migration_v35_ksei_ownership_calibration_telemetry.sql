-- v35: Persist decision-time official KSEI registration/composition context.
-- Additive and telemetry-only.  No historical backfill is performed because
-- current KSEI facts must not be fabricated as evidence seen by older scans.
-- KSEI scripless composition is explicitly NOT regulatory free float.

alter table public.multibagger_snapshots
    add column if not exists ownership_ksei_total_shares numeric,
    add column if not exists ownership_ksei_scripless_pct numeric,
    add column if not exists ownership_ksei_local_pct numeric,
    add column if not exists ownership_ksei_foreign_pct numeric,
    add column if not exists ownership_ksei_context_coverage_pct numeric,
    add column if not exists ownership_ksei_observed_on text,
    add column if not exists ownership_ksei_source_url text,
    add column if not exists ownership_ksei_source_authority text,
    add column if not exists ownership_ksei_official_verified boolean,
    add column if not exists ownership_ksei_provenance_state text,
    add column if not exists ownership_ksei_context_state text;

alter table public.pasticuan_calibration_snapshots
    add column if not exists ownership_ksei_total_shares numeric,
    add column if not exists ownership_ksei_scripless_pct numeric,
    add column if not exists ownership_ksei_local_pct numeric,
    add column if not exists ownership_ksei_foreign_pct numeric,
    add column if not exists ownership_ksei_context_coverage_pct numeric,
    add column if not exists ownership_ksei_observed_on text,
    add column if not exists ownership_ksei_source_url text,
    add column if not exists ownership_ksei_source_authority text,
    add column if not exists ownership_ksei_official_verified boolean,
    add column if not exists ownership_ksei_provenance_state text,
    add column if not exists ownership_ksei_context_state text;

create or replace function public.capture_pasticuan_calibration_ownership_trigger()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
declare
    v_scan_id text := nullif(btrim(coalesce(new.scan_id, '')), '');
    v_ticker text := nullif(upper(btrim(coalesce(new.ticker, ''))), '');
    v_calibration_id text;
begin
    if v_scan_id is null or v_ticker is null then
        return new;
    end if;

    v_calibration_id := v_scan_id || '|' || v_ticker;

    insert into public.pasticuan_calibration_snapshots (
        calibration_id, scan_id, ticker, scanner_version, scan_as_of,
        captured_at, last_captured_at, capture_version,
        has_multibagger, multibagger_snapshot_id,
        ownership_public_insiders_held_pct,
        ownership_public_institutions_held_pct,
        ownership_public_institutions_float_held_pct,
        ownership_public_institutions_count,
        ownership_public_context_coverage_pct,
        ownership_public_source_period,
        ownership_public_observed_on,
        ownership_public_source_authority,
        ownership_public_official_verified,
        ownership_public_context_provenance_state,
        ownership_public_context_state,
        ownership_ksei_total_shares,
        ownership_ksei_scripless_pct,
        ownership_ksei_local_pct,
        ownership_ksei_foreign_pct,
        ownership_ksei_context_coverage_pct,
        ownership_ksei_observed_on,
        ownership_ksei_source_url,
        ownership_ksei_source_authority,
        ownership_ksei_official_verified,
        ownership_ksei_provenance_state,
        ownership_ksei_context_state
    ) values (
        v_calibration_id, v_scan_id, v_ticker,
        nullif(new.model_version, ''), new.as_of,
        now(), now(), 'v35', true, new.snapshot_id,
        new.ownership_public_insiders_held_pct,
        new.ownership_public_institutions_held_pct,
        new.ownership_public_institutions_float_held_pct,
        new.ownership_public_institutions_count,
        new.ownership_public_context_coverage_pct,
        nullif(new.ownership_public_source_period, ''),
        nullif(new.ownership_public_observed_on, ''),
        nullif(new.ownership_public_source_authority, ''),
        new.ownership_public_official_verified,
        nullif(new.ownership_public_context_provenance_state, ''),
        nullif(new.ownership_public_context_state, ''),
        new.ownership_ksei_total_shares,
        new.ownership_ksei_scripless_pct,
        new.ownership_ksei_local_pct,
        new.ownership_ksei_foreign_pct,
        new.ownership_ksei_context_coverage_pct,
        nullif(new.ownership_ksei_observed_on, ''),
        nullif(new.ownership_ksei_source_url, ''),
        nullif(new.ownership_ksei_source_authority, ''),
        new.ownership_ksei_official_verified,
        nullif(new.ownership_ksei_provenance_state, ''),
        nullif(new.ownership_ksei_context_state, '')
    )
    on conflict (scan_id, ticker) do update set
        scanner_version = coalesce(excluded.scanner_version, public.pasticuan_calibration_snapshots.scanner_version),
        scan_as_of = coalesce(excluded.scan_as_of, public.pasticuan_calibration_snapshots.scan_as_of),
        last_captured_at = now(),
        capture_version = 'v35',
        has_multibagger = public.pasticuan_calibration_snapshots.has_multibagger or excluded.has_multibagger,
        multibagger_snapshot_id = coalesce(excluded.multibagger_snapshot_id, public.pasticuan_calibration_snapshots.multibagger_snapshot_id),
        ownership_public_insiders_held_pct = coalesce(excluded.ownership_public_insiders_held_pct, public.pasticuan_calibration_snapshots.ownership_public_insiders_held_pct),
        ownership_public_institutions_held_pct = coalesce(excluded.ownership_public_institutions_held_pct, public.pasticuan_calibration_snapshots.ownership_public_institutions_held_pct),
        ownership_public_institutions_float_held_pct = coalesce(excluded.ownership_public_institutions_float_held_pct, public.pasticuan_calibration_snapshots.ownership_public_institutions_float_held_pct),
        ownership_public_institutions_count = coalesce(excluded.ownership_public_institutions_count, public.pasticuan_calibration_snapshots.ownership_public_institutions_count),
        ownership_public_context_coverage_pct = coalesce(excluded.ownership_public_context_coverage_pct, public.pasticuan_calibration_snapshots.ownership_public_context_coverage_pct),
        ownership_public_source_period = coalesce(excluded.ownership_public_source_period, public.pasticuan_calibration_snapshots.ownership_public_source_period),
        ownership_public_observed_on = coalesce(excluded.ownership_public_observed_on, public.pasticuan_calibration_snapshots.ownership_public_observed_on),
        ownership_public_source_authority = coalesce(excluded.ownership_public_source_authority, public.pasticuan_calibration_snapshots.ownership_public_source_authority),
        ownership_public_official_verified = coalesce(excluded.ownership_public_official_verified, public.pasticuan_calibration_snapshots.ownership_public_official_verified),
        ownership_public_context_provenance_state = coalesce(excluded.ownership_public_context_provenance_state, public.pasticuan_calibration_snapshots.ownership_public_context_provenance_state),
        ownership_public_context_state = coalesce(excluded.ownership_public_context_state, public.pasticuan_calibration_snapshots.ownership_public_context_state),
        ownership_ksei_total_shares = coalesce(excluded.ownership_ksei_total_shares, public.pasticuan_calibration_snapshots.ownership_ksei_total_shares),
        ownership_ksei_scripless_pct = coalesce(excluded.ownership_ksei_scripless_pct, public.pasticuan_calibration_snapshots.ownership_ksei_scripless_pct),
        ownership_ksei_local_pct = coalesce(excluded.ownership_ksei_local_pct, public.pasticuan_calibration_snapshots.ownership_ksei_local_pct),
        ownership_ksei_foreign_pct = coalesce(excluded.ownership_ksei_foreign_pct, public.pasticuan_calibration_snapshots.ownership_ksei_foreign_pct),
        ownership_ksei_context_coverage_pct = coalesce(excluded.ownership_ksei_context_coverage_pct, public.pasticuan_calibration_snapshots.ownership_ksei_context_coverage_pct),
        ownership_ksei_observed_on = coalesce(excluded.ownership_ksei_observed_on, public.pasticuan_calibration_snapshots.ownership_ksei_observed_on),
        ownership_ksei_source_url = coalesce(excluded.ownership_ksei_source_url, public.pasticuan_calibration_snapshots.ownership_ksei_source_url),
        ownership_ksei_source_authority = coalesce(excluded.ownership_ksei_source_authority, public.pasticuan_calibration_snapshots.ownership_ksei_source_authority),
        ownership_ksei_official_verified = coalesce(excluded.ownership_ksei_official_verified, public.pasticuan_calibration_snapshots.ownership_ksei_official_verified),
        ownership_ksei_provenance_state = coalesce(excluded.ownership_ksei_provenance_state, public.pasticuan_calibration_snapshots.ownership_ksei_provenance_state),
        ownership_ksei_context_state = coalesce(excluded.ownership_ksei_context_state, public.pasticuan_calibration_snapshots.ownership_ksei_context_state);

    return new;
end;
$$;

revoke all on function public.capture_pasticuan_calibration_ownership_trigger()
    from public, anon, authenticated, service_role;

drop trigger if exists trg_capture_pasticuan_calibration_ownership on public.multibagger_snapshots;
create trigger trg_capture_pasticuan_calibration_ownership
after insert or update on public.multibagger_snapshots
for each row execute function public.capture_pasticuan_calibration_ownership_trigger();

comment on column public.multibagger_snapshots.ownership_ksei_context_coverage_pct is
'Runtime-used official KSEI registration/composition context coverage; not regulatory free float or beneficial ownership.';
comment on column public.pasticuan_calibration_snapshots.ownership_ksei_context_coverage_pct is
'Immutable decision-time official KSEI registration/composition context coverage; no historical backfill.';
