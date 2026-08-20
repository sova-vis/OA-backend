/**
 * Minimal test runner — the project has no test framework, and the handwritten
 * pipeline needs assertions badly enough to justify 60 lines over a new dep.
 */

type Test = { name: string; fn: () => void | Promise<void> };

const tests: Test[] = [];
let currentGroup = '';

export function group(name: string): void {
  currentGroup = name;
}

export function test(name: string, fn: () => void | Promise<void>): void {
  tests.push({ name: currentGroup ? `${currentGroup} › ${name}` : name, fn });
}

export class AssertionError extends Error {}

function fail(message: string): never {
  throw new AssertionError(message);
}

export function ok(value: unknown, message?: string): void {
  if (!value) fail(message || `expected truthy, got ${JSON.stringify(value)}`);
}

export function equal<T>(actual: T, expected: T, message?: string): void {
  if (actual !== expected) {
    fail(`${message ? message + ': ' : ''}expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
}

export function deepEqual(actual: unknown, expected: unknown, message?: string): void {
  const a = JSON.stringify(actual);
  const b = JSON.stringify(expected);
  if (a !== b) fail(`${message ? message + ': ' : ''}expected ${b}, got ${a}`);
}

export function includes(haystack: string, needle: string, message?: string): void {
  if (!String(haystack).toLowerCase().includes(needle.toLowerCase())) {
    fail(`${message ? message + ': ' : ''}expected "${haystack}" to include "${needle}"`);
  }
}

export async function throws(fn: () => unknown | Promise<unknown>, needle?: string): Promise<void> {
  try {
    await fn();
  } catch (error) {
    if (needle) includes(error instanceof Error ? error.message : String(error), needle);
    return;
  }
  fail(`expected a throw${needle ? ` containing "${needle}"` : ''}`);
}

export async function run(): Promise<void> {
  let passed = 0;
  const failures: Array<{ name: string; error: unknown }> = [];

  for (const t of tests) {
    try {
      await t.fn();
      passed += 1;
      console.log(`  PASS  ${t.name}`);
    } catch (error) {
      failures.push({ name: t.name, error });
      console.log(`  FAIL  ${t.name}`);
      console.log(`        ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  console.log(`\n${passed}/${tests.length} passed`);
  if (failures.length > 0) {
    console.log(`\n${failures.length} failing:`);
    for (const failure of failures) {
      console.log(`  - ${failure.name}`);
      if (!(failure.error instanceof AssertionError) && failure.error instanceof Error) {
        console.log(`    ${failure.error.stack}`);
      }
    }
    process.exitCode = 1;
  }
}
