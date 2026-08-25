-- Fast metadata aggregation for the paper-practice subjects list.
-- The A-Level bank is ~56k rows; streaming every row to the API (even paged)
-- took ~10s and surfaced as "failed to fetch" / a practice page that never
-- opened. This RPC aggregates in the DB and returns a few thousand grouped
-- counts as a single JSON value (no PostgREST row cap), in <1s once indexed.

create index if not exists questions_meta_idx
  on public.questions (level, subject, type, exam_year, variant, topic);

create or replace function public.paper_practice_meta(p_level text)
returns jsonb
language sql
stable
as $$
  select coalesce(jsonb_agg(t), '[]'::jsonb)
  from (
    select subject,
           type::text as qtype,
           exam_year,
           variant,
           topic,
           count(*)::int as n
    from public.questions
    where level = p_level
    group by subject, type, exam_year, variant, topic
  ) t
$$;

-- let PostgREST pick up the new function immediately
notify pgrst, 'reload schema';
