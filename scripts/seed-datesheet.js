// Seed the exam datesheet for Oct/Nov 2026 (O Level) with CONFIRMED dates,
// parsed from Cambridge International's official November 2026 Zone 4 timetable.
// (Zone calendar dates are identical across zones; only Morning/Afternoon varies.)
// Idempotent. Usage: node ./scripts/seed-datesheet.js
require('dotenv').config();
const { Client } = require('pg');

const SESSION = 'Oct/Nov 2026';

// subject, syllabus_code, component, date (YYYY-MM-DD), slot, duration
const ROWS = [
  ['Islamiyat', '2058', 'Paper 1', '2026-09-28', 'Morning', '1h 30m'],
  ['Chemistry', '5070', 'Paper 2', '2026-09-28', 'Morning', '1h 45m'],
  ['Pakistan Studies', '2059', 'Paper 1', '2026-09-29', 'Afternoon', '1h 30m'],
  ['English Language', '1123', 'Paper 1', '2026-09-29', 'Morning', '2h'],
  ['Islamiyat', '2058', 'Paper 2', '2026-09-30', null, '1h 30m'],
  ['Additional Mathematics', '4037', 'Paper 1', '2026-09-30', null, '2h'],
  ['Pakistan Studies', '2059', 'Paper 2', '2026-10-01', 'Afternoon', '1h 30m'],
  ['Biology', '5090', 'Paper 2', '2026-10-01', 'Morning', '1h 45m'],
  ['English Language', '1123', 'Paper 2', '2026-10-05', 'Morning', '2h'],
  ['Additional Mathematics', '4037', 'Paper 2', '2026-10-06', 'Afternoon', '2h'],
  ['Business', '7115', 'Paper 1', '2026-10-06', 'Morning', '1h 30m'],
  ['Physics', '5054', 'Paper 2', '2026-10-07', null, '1h 45m'],
  ['Mathematics', '4024', 'Paper 1', '2026-10-08', 'Morning', '2h'],
  ['Accounting', '7707', 'Paper 2', '2026-10-08', 'Afternoon', '1h 45m'],
  ['Computer Science', '2210', 'Paper 1', '2026-10-09', 'Afternoon', '1h 45m'],
  ['Biology', '5090', 'Paper 3', '2026-10-13', 'Morning', '1h 30m'],
  ['Biology', '5090', 'Paper 4', '2026-10-13', 'Morning', '1h'],
  ['Mathematics', '4024', 'Paper 2', '2026-10-14', null, '2h'],
  ['Chemistry', '5070', 'Paper 3', '2026-10-15', 'Morning', '1h 30m'],
  ['Chemistry', '5070', 'Paper 4', '2026-10-15', 'Morning', '1h'],
  ['Business', '7115', 'Paper 2', '2026-10-16', 'Morning', '1h 30m'],
  ['Computer Science', '2210', 'Paper 2', '2026-10-19', 'Afternoon', '1h 45m'],
  ['Physics', '5054', 'Paper 3', '2026-10-20', 'Morning', '1h 30m'],
  ['Physics', '5054', 'Paper 4', '2026-10-20', 'Morning', '1h'],
  ['Economics', '2281', 'Paper 2', '2026-10-20', 'Afternoon', '2h 15m'],
  ['Economics', '2281', 'Paper 1', '2026-11-04', null, '45m'],
  ['Physics', '5054', 'Paper 1', '2026-11-05', null, '1h'],
  ['Accounting', '7707', 'Paper 1', '2026-11-06', 'Afternoon', '1h 15m'],
  ['Biology', '5090', 'Paper 1', '2026-11-10', 'Morning', '1h'],
  ['Chemistry', '5070', 'Paper 1', '2026-11-12', 'Morning', '1h'],
];

async function main() {
  const c = new Client({ connectionString: process.env.DATABASE_URL, ssl: { rejectUnauthorized: false } });
  await c.connect();
  await c.query('DELETE FROM exam_timetable WHERE exam_session = $1', [SESSION]);
  for (const [subject, code, component, date, slot, duration] of ROWS) {
    await c.query(
      `INSERT INTO exam_timetable
         (exam_session, level, syllabus_code, subject, component, exam_date, session_slot, duration, status)
       VALUES ($1, 'O', $2, $3, $4, $5, $6, $7, 'confirmed')`,
      [SESSION, code, subject, component, date, slot, duration]
    );
  }
  console.log(`Seeded ${ROWS.length} confirmed O Level papers for ${SESSION}.`);
  await c.end();
}

main().catch((e) => { console.error(e.message); process.exit(1); });
