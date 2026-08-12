begin;
create or replace function public.scanner_free_tier_housekeeping_trigger()
returns trigger
language plpgsql
security definer
set search_path=public
as $$
begin
  perform public.scanner_free_tier_housekeeping();
  return null;
exception when others then
  -- Storage maintenance must never break scan persistence.
  return null;
end;
$$;

drop trigger if exists trg_scanner_free_tier_housekeeping on public.scan_runs;
create trigger trg_scanner_free_tier_housekeeping
after insert on public.scan_runs
for each statement
execute function public.scanner_free_tier_housekeeping_trigger();

revoke all on function public.scanner_free_tier_housekeeping_trigger() from public;
grant execute on function public.scanner_free_tier_housekeeping_trigger() to service_role;
commit;
