"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.getAuthUrl = getAuthUrl;
exports.getTokensFromCode = getTokensFromCode;
exports.listFilesInFolder = listFilesInFolder;
exports.listFoldersAndFiles = listFoldersAndFiles;
exports.getAllPDFsRecursive = getAllPDFsRecursive;
exports.getFileMetadata = getFileMetadata;
exports.makeFilePublic = makeFilePublic;
exports.searchFilesByName = searchFilesByName;
exports.getFileByName = getFileByName;
exports.getEmbedViewerUrl = getEmbedViewerUrl;
exports.getDirectDownloadUrl = getDirectDownloadUrl;
exports.getFileStream = getFileStream;
const googleapis_1 = require("googleapis");
const env_1 = require("./env");
// Initialize Google Drive API with OAuth2
const oauth2Client = new googleapis_1.google.auth.OAuth2(env_1.env.GOOGLE_CLIENT_ID, env_1.env.GOOGLE_CLIENT_SECRET, env_1.env.GOOGLE_REDIRECT_URI);
// Set credentials if refresh token is available
if (env_1.env.GOOGLE_REFRESH_TOKEN) {
    oauth2Client.setCredentials({
        refresh_token: env_1.env.GOOGLE_REFRESH_TOKEN,
    });
}
const drive = googleapis_1.google.drive({ version: 'v3', auth: oauth2Client });
/**
 * Generate OAuth authorization URL for initial setup
 * Run this once to get the authorization URL
 */
function getAuthUrl() {
    const scopes = [
        'https://www.googleapis.com/auth/drive.readonly',
        'https://www.googleapis.com/auth/drive.file',
    ];
    return oauth2Client.generateAuthUrl({
        access_type: 'offline',
        scope: scopes,
        prompt: 'consent', // Force to get refresh token
    });
}
/**
 * Exchange authorization code for tokens
 * Use this after user authorizes via the auth URL
 */
async function getTokensFromCode(code) {
    const { tokens } = await oauth2Client.getToken(code);
    oauth2Client.setCredentials(tokens);
    return tokens;
}
/**
 * List all files in a Google Drive folder
 */
async function listFilesInFolder(folderId) {
    try {
        const response = await drive.files.list({
            q: `'${folderId}' in parents and trashed=false`,
            fields: 'files(id, name, mimeType, webViewLink, webContentLink, size, modifiedTime)',
            orderBy: 'name',
        });
        return response.data.files || [];
    }
    catch (error) {
        console.error('Error listing files from Google Drive:', error);
        throw new Error('Failed to fetch files from Google Drive');
    }
}
/**
 * List all files AND folders in a folder with type distinction
 */
async function listFoldersAndFiles(folderId) {
    try {
        const response = await drive.files.list({
            q: `'${folderId}' in parents and trashed=false`,
            fields: 'files(id, name, mimeType, webViewLink, size, modifiedTime)',
            orderBy: 'name',
        });
        const items = response.data.files || [];
        return items.map(item => ({
            id: item.id,
            name: item.name,
            mimeType: item.mimeType,
            isFolder: item.mimeType === 'application/vnd.google-apps.folder',
            size: item.size,
            modifiedTime: item.modifiedTime,
        }));
    }
    catch (error) {
        console.error('Error listing folder contents:', error);
        throw new Error('Failed to list folder contents');
    }
}
/**
 * Recursively get all PDF files from folder and subfolders with full path
 */
async function getAllPDFsRecursive(folderId, path = []) {
    try {
        const items = await listFoldersAndFiles(folderId);
        const allFiles = [];
        for (const item of items) {
            if (item.isFolder) {
                // Recursively get files from subfolders
                const subFiles = await getAllPDFsRecursive(item.id, [...path, item.name]);
                allFiles.push(...subFiles);
            }
            else if (item.mimeType === 'application/pdf') {
                // Add PDF file with its path
                allFiles.push({
                    id: item.id,
                    name: item.name,
                    path: [...path, item.name],
                    fullPath: [...path, item.name].join(' / '),
                    size: item.size,
                    modifiedTime: item.modifiedTime,
                });
            }
        }
        return allFiles;
    }
    catch (error) {
        console.error('Error getting PDFs recursively:', error);
        throw new Error('Failed to get PDFs from folder structure');
    }
}
/**
 * Get file metadata from Google Drive
 */
async function getFileMetadata(fileId) {
    try {
        const response = await drive.files.get({
            fileId,
            fields: 'id, name, mimeType, webViewLink, webContentLink, size, modifiedTime',
        });
        return response.data;
    }
    catch (error) {
        console.error('Error getting file metadata:', error);
        throw new Error('Failed to get file metadata from Google Drive');
    }
}
/**
 * Generate a shareable link for a file (makes it publicly viewable)
 */
async function makeFilePublic(fileId) {
    try {
        // Make file publicly accessible
        await drive.permissions.create({
            fileId,
            requestBody: {
                role: 'reader',
                type: 'anyone',
            },
        });
        // Get the file metadata with the public link
        const file = await getFileMetadata(fileId);
        return file.webViewLink;
    }
    catch (error) {
        console.error('Error making file public:', error);
        throw new Error('Failed to make file public');
    }
}
/**
 * Search files by name pattern in a folder
 */
async function searchFilesByName(folderId, searchTerm) {
    try {
        const response = await drive.files.list({
            q: `'${folderId}' in parents and name contains '${searchTerm}' and trashed=false`,
            fields: 'files(id, name, mimeType, webViewLink, webContentLink, size, modifiedTime)',
            orderBy: 'name',
        });
        return response.data.files || [];
    }
    catch (error) {
        console.error('Error searching files:', error);
        throw new Error('Failed to search files');
    }
}
/**
 * Get file by exact name
 */
async function getFileByName(folderId, fileName) {
    try {
        const response = await drive.files.list({
            q: `'${folderId}' in parents and name='${fileName}' and trashed=false`,
            fields: 'files(id, name, mimeType, webViewLink, webContentLink, size, modifiedTime)',
        });
        const files = response.data.files || [];
        return files.length > 0 ? files[0] : null;
    }
    catch (error) {
        console.error('Error getting file by name:', error);
        return null;
    }
}
/**
 * Generate embedded viewer URL for PDF files
 */
function getEmbedViewerUrl(fileId) {
    return `https://drive.google.com/file/d/${fileId}/preview`;
}
/**
 * Generate direct download URL
 */
function getDirectDownloadUrl(fileId) {
    return `https://drive.google.com/uc?export=download&id=${fileId}`;
}
/**
 * Stream file content from Google Drive
 * Returns a readable stream of the file
 */
async function getFileStream(fileId) {
    try {
        const response = await drive.files.get({ fileId, alt: 'media' }, { responseType: 'stream' });
        return response.data;
    }
    catch (error) {
        console.error('Error streaming file:', error);
        throw new Error('Failed to stream file from Google Drive');
    }
}
