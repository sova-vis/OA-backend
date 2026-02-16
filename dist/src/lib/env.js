"use strict";
/**
 * Environment Variables Configuration
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.env = void 0;
const dotenv_1 = __importDefault(require("dotenv"));
const zod_1 = require("zod");
dotenv_1.default.config();
const envSchema = zod_1.z.object({
    NODE_ENV: zod_1.z.enum(['development', 'production', 'test']).default('development'),
    PORT: zod_1.z.string().default('3001'),
    // Supabase
    SUPABASE_URL: zod_1.z.string().url(),
    SUPABASE_SERVICE_ROLE_KEY: zod_1.z.string().min(1, 'Supabase service role key is required'),
    SUPABASE_JWKS_URL: zod_1.z.string().url(),
    SUPABASE_JWT_ISSUER: zod_1.z.string().url(),
    // Database (Supabase connection string)
    DATABASE_URL: zod_1.z.string().url(),
    // CORS
    FRONTEND_URL: zod_1.z.string().url().default('http://localhost:3000'),
    // Google Drive OAuth
    GOOGLE_CLIENT_ID: zod_1.z.string().optional(),
    GOOGLE_CLIENT_SECRET: zod_1.z.string().optional(),
    GOOGLE_REDIRECT_URI: zod_1.z.string().optional(),
    GOOGLE_REFRESH_TOKEN: zod_1.z.string().optional(),
    GOOGLE_DRIVE_FOLDER_ID: zod_1.z.string().optional(),
});
let env;
try {
    exports.env = env = envSchema.parse(process.env);
}
catch (error) {
    if (error instanceof zod_1.z.ZodError) {
        console.error('❌ Invalid environment variables:');
        console.error(error.errors);
        process.exit(1);
    }
    throw error;
}
