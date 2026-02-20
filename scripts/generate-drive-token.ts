/**
 * Script to generate a new Google Drive refresh token
 * Run this once to get a new refresh token after changing OAuth credentials
 */

import { google } from 'googleapis';
import * as readline from 'readline';
import * as dotenv from 'dotenv';

// Load environment variables
dotenv.config();

const oauth2Client = new google.auth.OAuth2(
  process.env.GOOGLE_CLIENT_ID,
  process.env.GOOGLE_CLIENT_SECRET,
  process.env.GOOGLE_REDIRECT_URI || 'http://localhost:3001/oauth/callback'
);

const SCOPES = [
  'https://www.googleapis.com/auth/drive.readonly',
  'https://www.googleapis.com/auth/drive.file',
];

// Generate auth URL
const authUrl = oauth2Client.generateAuthUrl({
  access_type: 'offline',
  scope: SCOPES,
  prompt: 'consent', // Force to get refresh token
});

console.log('\n=== Google Drive OAuth Setup ===\n');
console.log('1. Open this URL in your browser:\n');
console.log(authUrl);
console.log('\n2. Authorize the app');
console.log('3. Copy the code from the URL (after "code=")');
console.log('4. Paste it here:\n');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

rl.question('Enter the authorization code: ', async (code) => {
  try {
    const { tokens } = await oauth2Client.getToken(code);
    
    console.log('\n✅ Success! Add this to your .env file:\n');
    console.log(`GOOGLE_REFRESH_TOKEN=${tokens.refresh_token}`);
    console.log('\n');
    
  } catch (error) {
    console.error('Error getting tokens:', error);
  }
  
  rl.close();
  process.exit(0);
});
