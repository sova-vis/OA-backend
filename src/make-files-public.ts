/**
 * Script to make all files in Google Drive folder publicly accessible
 * Run this once: npm run make-public
 */

import { listFilesInFolder, makeFilePublic } from './lib/googleDrive';
import { env } from './lib/env';

async function makeAllFilesPublic() {
  try {
    const folderId = env.GOOGLE_DRIVE_FOLDER_ID;
    
    if (!folderId) {
      console.error('❌ GOOGLE_DRIVE_FOLDER_ID is not set in .env');
      process.exit(1);
    }

    console.log('📂 Fetching files from Google Drive...\n');
    const files = await listFilesInFolder(folderId);
    
    console.log(`Found ${files.length} files. Making them publicly accessible...\n`);
    
    let successCount = 0;
    let errorCount = 0;

    for (const file of files) {
      try {
        await makeFilePublic(file.id);
        console.log(`✅ ${file.name}`);
        successCount++;
      } catch (error) {
        console.error(`❌ ${file.name} - Error: ${error}`);
        errorCount++;
      }
    }

    console.log(`\n✅ Done! ${successCount} files made public, ${errorCount} errors`);
  } catch (error) {
    console.error('❌ Fatal error:', error);
    process.exit(1);
  }
}

makeAllFilesPublic();
