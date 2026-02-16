"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const cors_1 = __importDefault(require("cors"));
require("dotenv/config");
const express_1 = __importDefault(require("express"));
const auth_routes_1 = __importDefault(require("./auth.routes"));
const admin_routes_1 = __importDefault(require("./admin.routes"));
const papers_routes_1 = __importDefault(require("./papers.routes"));
const rag_routes_1 = __importDefault(require("./rag.routes"));
const content_routes_1 = __importDefault(require("./content.routes"));
const app = (0, express_1.default)();
const PORT = process.env.PORT || 3001;
// Basic middleware
app.use((0, cors_1.default)({
    origin: 'http://localhost:3000',
    credentials: true
}));
app.use(express_1.default.json());
app.use(express_1.default.urlencoded({ extended: true }));
// Auth API
app.use('/auth', auth_routes_1.default);
// Admin API
app.use('/admin', admin_routes_1.default);
// Papers API
app.use('/papers', papers_routes_1.default);
// Content API (navigation/search)
app.use('/content', content_routes_1.default);
// RAG API
app.use('/rag', rag_routes_1.default);
// Health check
app.get('/health', (req, res) => {
    res.json({ status: 'ok', timestamp: new Date().toISOString() });
});
// Root route
app.get('/', (req, res) => {
    res.send('Welcome to the Propel backend API!');
});
// Start server
app.listen(PORT, () => {
    console.log(`🚀 Backend server running on http://localhost:${PORT}`);
    console.log(`📊 Health check: http://localhost:${PORT}/health`);
});
exports.default = app;
