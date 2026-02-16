"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = require("express");
const adminService_1 = require("./services/adminService");
const router = (0, express_1.Router)();
// Hardcoded admin credentials (keep for simple prototype admin access)
const ADMIN_EMAIL = 'admin@gmail.com';
const ADMIN_PASSWORD = '123';
// Admin login
router.post('/login', (req, res) => {
    const { email, password } = req.body;
    if (email === ADMIN_EMAIL && password === ADMIN_PASSWORD) {
        return res.json({ token: 'admin-session-token', role: 'admin' });
    }
    return res.status(401).json({ error: 'Invalid credentials' });
});
// Middleware
function requireAdmin(req, res, next) {
    const token = req.headers['authorization'];
    if (token === 'admin-session-token')
        return next();
    return res.status(403).json({ error: 'Forbidden' });
}
// Add teacher using Supabase Admin
router.post('/add-teacher', requireAdmin, async (req, res) => {
    const { email, password, name } = req.body;
    if (!email || !password || !name)
        return res.status(400).json({ error: 'Missing fields' });
    try {
        const user = await (0, adminService_1.createTeacherAccount)(email, password, name);
        return res.json({ message: 'Teacher created successfully', userId: user.id });
    }
    catch (error) {
        console.error("Create teacher error:", error);
        return res.status(500).json({ error: error.message || 'Failed to create teacher' });
    }
});
// Update user profile (admin only)
// Update user profile (admin only) - Placeholder for future implementation using Supabase
router.put('/update-profile/:id', requireAdmin, (req, res) => {
    return res.status(501).json({ error: 'Not implemented yet' });
});
exports.default = router;
