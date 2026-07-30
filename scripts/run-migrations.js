require('dotenv').config();
const fs = require('fs');
const path = require('path');
const { Client } = require('pg');

const MIGRATIONS_DIR = path.join(__dirname, '..', 'migrations');

async function main() {
  const client = new Client({ connectionString: process.env.DATABASE_URL, ssl: { rejectUnauthorized: false } });
  await client.connect();
  console.log('Connected to database.');

  const files = fs.readdirSync(MIGRATIONS_DIR).filter(f => f.endsWith('.sql')).sort();
  for (const file of files) {
    const sql = fs.readFileSync(path.join(MIGRATIONS_DIR, file), 'utf8');
    console.log(`Running ${file} ...`);
    try {
      await client.query(sql);
      console.log(`  OK: ${file}`);
    } catch (err) {
      console.error(`  FAILED: ${file}`);
      console.error(`  ${err.message}`);
      await client.end();
      process.exit(1);
    }
  }

  await client.end();
  console.log('All migrations applied.');
}

main().catch(err => {
  console.error('Migration runner crashed:', err.message);
  process.exit(1);
});
