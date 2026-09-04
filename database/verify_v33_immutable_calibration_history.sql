-- v33 verification: structural, trigger, access, and seed checks.

select table_name
from information_schema.tables
where table_schema = 'public'
  and table_name in ('pasticuan_calibration_snapshots', 'pasticuan_calibration_outcomes')
order by table_name;

select tc.constraint_name, tc.constraint_type,
       string_agg(kcu.column_name, ',' order by kcu.ordinal_position) as columns
from information_schema.table_constraints tc
left join information_schema.key_column_usage kcu
  on tc.constraint_name = kcu.constraint_name
 and tc.table_schema = kcu.table_schema
 and tc.table_name = kcu.table_name
where tc.table_schema = 'public'
  and tc.table_name = 'pasticuan_calibration_snapshots'
  and tc.constraint_type in ('PRIMARY KEY', 'UNIQUE')
group by tc.constraint_name, tc.constraint_type
order by tc.constraint_name;

select event_object_table, trigger_name, event_manipulation
from information_schema.triggers
where trigger_schema = 'public'
  and trigger_name like 'trg_capture_pasticuan_calibration_%'
order by event_object_table, trigger_name, event_manipulation;

select c.relname as table_name, c.relrowsecurity as rls_enabled
from pg_class c
join pg_namespace n on n.oid = c.relnamespace
where n.nspname = 'public'
  and c.relname in ('pasticuan_calibration_snapshots', 'pasticuan_calibration_outcomes')
order by c.relname;

select grantee, table_name, string_agg(privilege_type, ',' order by privilege_type) privileges
from information_schema.role_table_grants
where table_schema = 'public'
  and table_name in ('pasticuan_calibration_snapshots', 'pasticuan_calibration_outcomes')
group by grantee, table_name
order by table_name, grantee;

select count(*) as calibration_rows,
       count(distinct scan_id) as scans,
       count(*) filter (where has_multibagger) as with_multibagger,
       count(*) filter (where has_technical) as with_technical,
       count(*) filter (where has_fundamental) as with_fundamental
from public.pasticuan_calibration_snapshots;
