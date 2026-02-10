/**
 * One-time setup script to authorize Google Drive access using OAuth 2.0
 * 
 * Run this script once to get your refresh token:
 * npm run setup:google
 */

import { getAuthUrl, getTokensFromCode } from './lib/googleDrive';
import * as readline from 'readline';

async function setupGoogleOAuth() {
  console.log('\n🔐 Google Drive OAuth Setup\n');
  console.log('This script will help you authorize access to your Google Drive.\n');

  // Step 1: Generate authorization URL
  const authUrl = getAuthUrl();
  
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
      const tokens = await getTokensFromCode(code);

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
    } catch (error) {
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
