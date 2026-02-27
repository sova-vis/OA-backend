/**
 * Environment Variables Configuration
 */

import dotenv from 'dotenv';
import { z } from 'zod';

dotenv.config();

const trimString = (value: unknown) =>
  typeof value === 'string' ? value.trim() : value;

const envSchema = z.object({
  NODE_ENV: z.enum(['development', 'production', 'test']).default('development'),
  PORT: z.preprocess(trimString, z.string()).default('3001'),
  
  // Supabase
  SUPABASE_URL: z.preprocess(trimString, z.string().url()),
  SUPABASE_SERVICE_ROLE_KEY: z.preprocess(trimString, z.string().min(1, 'Supabase service role key is required')),
  SUPABASE_JWKS_URL: z.preprocess(trimString, z.string().url()),
  SUPABASE_JWT_ISSUER: z.preprocess(trimString, z.string().url()),
  
  // Database (Supabase connection string)
  DATABASE_URL: z.preprocess(trimString, z.string().url()),
  
  // CORS
  FRONTEND_URL: z.preprocess(trimString, z.string()).default('http://localhost:3000'),
  
  // Clerk (optional for development)
  CLERK_SECRET_KEY: z.preprocess(trimString, z.string().optional()),
  CLERK_JWT_KEY: z.preprocess(trimString, z.string().optional()),
  NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY: z.preprocess(trimString, z.string().optional()),
  
  // Google Drive OAuth
  GOOGLE_CLIENT_ID: z.preprocess(trimString, z.string().optional()),
  GOOGLE_CLIENT_SECRET: z.preprocess(trimString, z.string().optional()),
  GOOGLE_REDIRECT_URI: z.preprocess(trimString, z.string().optional()),
  GOOGLE_REFRESH_TOKEN: z.preprocess(trimString, z.string().optional()),
  GOOGLE_DRIVE_FOLDER_ID: z.preprocess(trimString, z.string().optional()),
});

export type Env = z.infer<typeof envSchema>;

let env: Env;

try {
  env = envSchema.parse(process.env);
} catch (error) {
  if (error instanceof z.ZodError) {
    console.error('❌ Invalid environment variables:');
    console.error(error.errors);
    process.exit(1);
  }
  throw error;
}

export { env };
