"use strict";
/**
 * Script to make all files in Google Drive folder publicly accessible
 * Run this once: npm run make-public
 */
Object.defineProperty(exports, "__esModule", { value: true });
const googleDrive_1 = require("./lib/googleDrive");
const env_1 = require("./lib/env");
async function makeAllFilesPublic() {
    try {
        const folderId = env_1.env.GOOGLE_DRIVE_FOLDER_ID;
        if (!folderId) {
            console.error('❌ GOOGLE_DRIVE_FOLDER_ID is not set in .env');
            process.exit(1);
        }
        console.log('📂 Fetching files from Google Drive...\n');
        const files = await (0, googleDrive_1.listFilesInFolder)(folderId);
        console.log(`Found ${files.length} files. Making them publicly accessible...\n`);
        let successCount = 0;
        let errorCount = 0;
        for (const file of files) {
            try {
                await (0, googleDrive_1.makeFilePublic)(file.id);
                console.log(`✅ ${file.name}`);
                successCount++;
            }
            catch (error) {
                console.error(`❌ ${file.name} - Error: ${error}`);
                errorCount++;
            }
        }
        console.log(`\n✅ Done! ${successCount} files made public, ${errorCount} errors`);
    }
    catch (error) {
        console.error('❌ Fatal error:', error);
        process.exit(1);
    }
}
makeAllFilesPublic();
