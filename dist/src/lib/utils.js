"use strict";
/**
 * Server Utilities
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.generateCode = generateCode;
exports.isValidEmail = isValidEmail;
/**
 * Generate random code
 */
function generateCode(length = 6) {
    const chars = '0123456789';
    let code = '';
    for (let i = 0; i < length; i++) {
        code += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return code;
}
/**
 * Validate email format
 */
function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}
