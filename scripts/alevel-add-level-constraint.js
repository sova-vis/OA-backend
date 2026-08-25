/**
 * One-off migration: include `level` in the questions composite unique so A-Level
 * and O-Level content with the same subject/year/paper/variant/question_number
 * can coexist without collision. Additive + reversible; existing O-Level rows
 * (all level='olevel') remain unique under the wider key.
 *
 *   node scripts/alevel-add-level-constraint.js          # apply
 *   node scripts/alevel-add-level-constraint.js --revert # roll back
 */
require("dotenv").config();
const { Client } = require("pg");

const OLD = "questions_subject_exam_year_session_paper_variant_question__key";
const NEW = "questions_subj_yr_sess_paper_var_qnum_level_key";
const COLS_OLD = "(subject, exam_year, session, paper, variant, question_number)";
const COLS_NEW = "(subject, exam_year, session, paper, variant, question_number, level)";

(async () => {
  const revert = process.argv.includes("--revert");
  const c = new Client({ connectionString: process.env.DATABASE_URL });
  await c.connect();
  await c.query("BEGIN");
  try {
    if (revert) {
      await c.query(`ALTER TABLE public.questions DROP CONSTRAINT IF EXISTS ${NEW}`);
      await c.query(`ALTER TABLE public.questions ADD CONSTRAINT ${OLD} UNIQUE ${COLS_OLD}`);
    } else {
      await c.query(`ALTER TABLE public.questions DROP CONSTRAINT IF EXISTS ${OLD}`);
      await c.query(`ALTER TABLE public.questions ADD CONSTRAINT ${NEW} UNIQUE ${COLS_NEW}`);
    }
    await c.query("COMMIT");
    console.log(revert ? "REVERTED to 6-col unique" : "APPLIED: 7-col unique incl level");
  } catch (e) {
    await c.query("ROLLBACK");
    console.log("FAILED (rolled back):", e.message);
    process.exitCode = 1;
  }
  const r = await c.query(
    "select conname, pg_get_constraintdef(oid) def from pg_constraint where conrelid='public.questions'::regclass and contype='u'"
  );
  r.rows.forEach((x) => console.log("  ", x.conname, "=>", x.def));
  await c.end();
})().catch((e) => { console.log("ERR", e.message); process.exitCode = 1; });
