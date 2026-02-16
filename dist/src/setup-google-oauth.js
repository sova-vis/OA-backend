"use strict";
/**
 * One-time setup script to authorize Google Drive access using OAuth 2.0
 *
 * Run this script once to get your refresh token:
 * npm run setup:google
 */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
const googleDrive_1 = require("./lib/googleDrive");
const readline = __importStar(require("readline"));
async function setupGoogleOAuth() {
    console.log('\n🔐 Google Drive OAuth Setup\n');
    console.log('This script will help you authorize access to your Google Drive.\n');
    // Step 1: Generate authorization URL
    const authUrl = (0, googleDrive_1.getAuthUrl)();
    console.log('Step 1: Visit this URL to authorize access:');
    console.log('\n' + authUrl + '\n');
    console.log('After authorizing, Google will redirect you to a URL.');
    console.log('Copy the ENTIRE redirect URL from your browser.\n');
    // Step 2: Get the authorization code from user
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    });
    rl.question('Paste the redirect URL here: ', async (redirectUrl) => {
        try {
            // Extract code from URL
            const url = new URL(redirectUrl);
            const code = url.searchParams.get('code');
            if (!code) {
                console.error('\n❌ Error: Could not find authorization code in URL');
                console.error('Make sure you pasted the complete redirect URL\n');
                rl.close();
                process.exit(1);
            }
            // Step 3: Exchange code for tokens
            console.log('\n⏳ Exchanging authorization code for tokens...\n');
            const tokens = await (0, googleDrive_1.getTokensFromCode)(code);
            // Step 4: Display refresh token
            console.log('✅ Success! Add this to your .env file:\n');
            console.log('GOOGLE_REFRESH_TOKEN=' + tokens.refresh_token + '\n');
            if (tokens.access_token) {
                console.log('Your access token (valid for ~1 hour):');
                console.log(tokens.access_token.substring(0, 50) + '...\n');
            }
            console.log('⚠️  Important: Keep your refresh token secret!');
            console.log('Add it to .env and never commit it to version control.\n');
            rl.close();
        }
        catch (error) {
            console.error('\n❌ Error getting tokens:', error);
            rl.close();
            process.exit(1);
        }
    });
}
// Run the setup
setupGoogleOAuth().catch((error) => {
    console.error('Fatal error:', error);
    process.exit(1);
});
