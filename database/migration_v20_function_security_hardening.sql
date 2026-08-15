-- Super Scanner security hardening for the official-cache preservation trigger.
begin;

alter function public.guard_fundamental_cache_official_no_downgrade() set search_path = '';
revoke all on function public.guard_fundamental_cache_official_no_downgrade() from public, anon, authenticated;
grant execute on function public.guard_fundamental_cache_official_no_downgrade() to service_role;

commit;
