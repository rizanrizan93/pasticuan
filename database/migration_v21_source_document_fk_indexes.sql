-- Super Scanner database migration v21
-- Add covering indexes for source-document foreign keys flagged by the
-- Supabase performance advisor. These indexes are idempotent and do not
-- change row-level-security or data-access semantics.

create index if not exists idx_corporate_events_source_document_id
    on public.corporate_events (source_document_id);

create index if not exists idx_financial_periods_document_id
    on public.financial_periods (document_id);

create index if not exists idx_management_roles_source_document_id
    on public.management_roles (source_document_id);

create index if not exists idx_ownership_events_source_document_id
    on public.ownership_events (source_document_id);
