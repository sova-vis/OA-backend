"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = require("express");
const googleDrive_1 = require("./lib/googleDrive");
const env_1 = require("./lib/env");
const router = (0, express_1.Router)();
/**
 * GET /papers/drive/list
 * List all papers from Google Drive folder (recursively gets all PDFs)
 */
router.get('/drive/list', async (req, res) => {
    try {
        const folderId = env_1.env.GOOGLE_DRIVE_FOLDER_ID;
        if (!folderId) {
            return res.status(400).json({
                error: 'Google Drive folder ID not configured'
            });
        }
        console.log('📂 Getting all PDFs recursively...');
        const allPDFs = await (0, googleDrive_1.getAllPDFsRecursive)(folderId);
        console.log(`📄 Found ${allPDFs.length} PDF files`);
        // Transform to include URLs
        const filesWithUrls = allPDFs.map((file) => ({
            id: file.id,
            name: file.name,
            path: file.path,
            fullPath: file.fullPath,
            size: file.size,
            modifiedTime: file.modifiedTime,
            viewUrl: `/papers/view/${file.id}`,
            downloadUrl: `/papers/download/${file.id}`,
            embedUrl: `/papers/view/${file.id}`,
            mimeType: 'application/pdf',
        }));
        res.json({ files: filesWithUrls });
    }
    catch (error) {
        console.error('Error listing papers:', error);
        res.status(500).json({ error: 'Failed to fetch papers from Google Drive' });
    }
});
/**
 * GET /papers/browse/:folderId
 * Browse a specific folder (for navigation) - returns immediate children only
 */
router.get('/browse/:folderId?', async (req, res) => {
    try {
        const folderId = req.params.folderId || env_1.env.GOOGLE_DRIVE_FOLDER_ID;
        if (!folderId) {
            return res.status(400).json({
                error: 'Folder ID required'
            });
        }
        console.log(`📂 Browsing folder: ${folderId}`);
        const items = await (0, googleDrive_1.listFoldersAndFiles)(folderId);
        // Helper function to detect folder type based on name
        const detectFolderType = (name) => {
            // Check if it's a year (4 digits)
            if (/^\d{4}$/.test(name))
                return 'year';
            // Check if it's a month/session (contains month names or session patterns)
            const monthPatterns = /may|june|march|october|november|feb|jan|apr|jul|aug|sep|dec|m\/j|o\/n|f\/m/i;
            if (monthPatterns.test(name))
                return 'month';
            // Check if it's a category folder
            const categoryPatterns = /past\s*papers?|syllabus|reference|topical|notes/i;
            if (categoryPatterns.test(name))
                return 'category';
            // Otherwise assume it's a subject
            return 'subject';
        };
        // Transform items to include URLs for files and type detection for folders
        const transformedItems = items.map(item => {
            if (item.isFolder) {
                const folderType = detectFolderType(item.name);
                return {
                    id: item.id,
                    name: item.name,
                    isFolder: true,
                    folderType,
                    mimeType: item.mimeType,
                };
            }
            else {
                return {
                    id: item.id,
                    name: item.name,
                    isFolder: false,
                    mimeType: item.mimeType,
                    size: item.size,
                    modifiedTime: item.modifiedTime,
                    viewUrl: `/papers/view/${item.id}`,
                    downloadUrl: `/papers/download/${item.id}`,
                    embedUrl: `/papers/view/${item.id}`,
                };
            }
        });
        // Sort items: folders first (subjects, then categories, then years, then months), then files
        const sortOrder = { subject: 0, category: 1, year: 2, month: 3, unknown: 4 };
        transformedItems.sort((a, b) => {
            if (a.isFolder && !b.isFolder)
                return -1;
            if (!a.isFolder && b.isFolder)
                return 1;
            if (a.isFolder && b.isFolder) {
                const aType = a.folderType || 'unknown';
                const bType = b.folderType || 'unknown';
                const aOrder = sortOrder[aType] || 5;
                const bOrder = sortOrder[bType] || 5;
                if (aOrder !== bOrder)
                    return aOrder - bOrder;
            }
            return a.name.localeCompare(b.name);
        });
        console.log(`✅ Found ${transformedItems.length} items (${transformedItems.filter(i => i.isFolder).length} folders, ${transformedItems.filter(i => !i.isFolder).length} files)`);
        res.json({
            folderId,
            items: transformedItems
        });
    }
    catch (error) {
        console.error('Error browsing folder:', error);
        res.status(500).json({ error: 'Failed to browse folder' });
    }
});
/**
 * GET /papers/view/:fileId
 * Stream file for viewing (embedded PDF viewer)
 */
router.get('/view/:fileId', async (req, res) => {
    try {
        const { fileId } = req.params;
        // Get file metadata first
        const metadata = await (0, googleDrive_1.getFileMetadata)(fileId);
        // Set appropriate headers
        res.setHeader('Content-Type', metadata.mimeType);
        res.setHeader('Content-Disposition', `inline; filename="${metadata.name}"`);
        // Stream the file
        const stream = await (0, googleDrive_1.getFileStream)(fileId);
        stream.pipe(res);
    }
    catch (error) {
        console.error('Error viewing file:', error);
        res.status(500).json({ error: 'Failed to view file' });
    }
});
/**
 * GET /papers/download/:fileId
 * Stream file for download
 */
router.get('/download/:fileId', async (req, res) => {
    try {
        const { fileId } = req.params;
        // Get file metadata first
        const metadata = await (0, googleDrive_1.getFileMetadata)(fileId);
        // Set headers for download
        res.setHeader('Content-Type', metadata.mimeType);
        res.setHeader('Content-Disposition', `attachment; filename="${metadata.name}"`);
        // Stream the file
        const stream = await (0, googleDrive_1.getFileStream)(fileId);
        stream.pipe(res);
    }
    catch (error) {
        console.error('Error downloading file:', error);
        res.status(500).json({ error: 'Failed to download file' });
    }
});
/**
 * GET /papers/drive/search?q=searchTerm
 * Search papers by name
 */
router.get('/drive/search', async (req, res) => {
    try {
        const { q } = req.query;
        const folderId = env_1.env.GOOGLE_DRIVE_FOLDER_ID;
        if (!folderId) {
            return res.status(400).json({
                error: 'Google Drive folder ID not configured'
            });
        }
        if (!q || typeof q !== 'string') {
            return res.status(400).json({ error: 'Search term is required' });
        }
        const files = await (0, googleDrive_1.searchFilesByName)(folderId, q);
        const filesWithUrls = files.map((file) => ({
            id: file.id,
            name: file.name,
            size: file.size,
            modifiedTime: file.modifiedTime,
            viewUrl: `/papers/view/${file.id}`,
            downloadUrl: `/papers/download/${file.id}`,
            embedUrl: `/papers/view/${file.id}`,
            mimeType: file.mimeType,
        }));
        res.json({ files: filesWithUrls });
    }
    catch (error) {
        console.error('Error searching papers:', error);
        res.status(500).json({ error: 'Failed to search papers' });
    }
});
/**
 * GET /papers/drive/:fileId
 * Get metadata for a specific file
 */
router.get('/drive/:fileId', async (req, res) => {
    try {
        const { fileId } = req.params;
        const file = await (0, googleDrive_1.getFileMetadata)(fileId);
        res.json({
            id: file.id,
            name: file.name,
            size: file.size,
            modifiedTime: file.modifiedTime,
            viewUrl: `/papers/view/${file.id}`,
            downloadUrl: `/papers/download/${file.id}`,
            embedUrl: `/papers/view/${file.id}`,
            mimeType: file.mimeType,
        });
    }
    catch (error) {
        console.error('Error getting file metadata:', error);
        res.status(500).json({ error: 'Failed to get file metadata' });
    }
});
/**
 * GET /papers/organized
 * Get papers organized by subject, year, session
 */
router.get('/organized', async (req, res) => {
    try {
        const folderId = env_1.env.GOOGLE_DRIVE_FOLDER_ID;
        if (!folderId) {
            return res.status(400).json({
                error: 'Google Drive folder ID not configured'
            });
        }
        const files = await (0, googleDrive_1.listFilesInFolder)(folderId);
        // Organize files by parsing filenames
        // Expected format: "SubjectCode_Year_Session_Paper_Variant.pdf"
        // Example: "0580_2023_MJ_P1_12.pdf" or "Physics_2022_ON_P4_41.pdf"
        const organized = {};
        files.forEach((file) => {
            const name = file.name;
            // Try to parse the filename
            // Pattern: Subject_Year_Session_Paper_Variant
            const parts = name.replace('.pdf', '').split('_');
            if (parts.length >= 5) {
                const [subject, year, session, paper, variant] = parts;
                if (!organized[subject]) {
                    organized[subject] = {};
                }
                if (!organized[subject][year]) {
                    organized[subject][year] = {};
                }
                if (!organized[subject][year][session]) {
                    organized[subject][year][session] = [];
                }
                organized[subject][year][session].push({
                    id: file.id,
                    name: file.name,
                    paper,
                    variant,
                    viewUrl: `/papers/view/${file.id}`,
                    downloadUrl: `/papers/download/${file.id}`,
                    embedUrl: `/papers/view/${file.id}`,
                    size: file.size,
                });
            }
        });
        res.json({ papers: organized });
    }
    catch (error) {
        console.error('Error organizing papers:', error);
        res.status(500).json({ error: 'Failed to organize papers' });
    }
});
exports.default = router;
