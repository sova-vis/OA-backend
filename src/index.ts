import cors from 'cors';

import 'dotenv/config';
import express, { Express, Request, Response } from 'express';
import authRoutes from './auth.routes';
import adminRoutes from './admin.routes';
import papersRoutes from './papers.routes';
import ragRoutes from './rag.routes';
import contentRoutes from './content.routes';

const app: Express = express();
const PORT = process.env.PORT || 3001;

// Basic middleware
app.use(cors({
  origin: 'http://localhost:3000',
  credentials: true
}));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Auth API
app.use('/auth', authRoutes);

// Admin API
app.use('/admin', adminRoutes);

// Papers API
app.use('/papers', papersRoutes);

// Content API (navigation/search)
app.use('/content', contentRoutes);

// RAG API
app.use('/rag', ragRoutes);

// Health check
app.get('/health', (req: Request, res: Response) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Root route
app.get('/', (req: Request, res: Response) => {
  res.send('Welcome to the Propel backend API!');
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Backend server running on http://localhost:${PORT}`);
  console.log(`📊 Health check: http://localhost:${PORT}/health`);
});

export default app;
