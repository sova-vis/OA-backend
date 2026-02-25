import { Router, Request, Response } from 'express';
import { createTeacherAccount } from './services/adminService';
import { AuthenticatedRequest, clerkAuth, requireRole } from './lib/clerkAuth';

const router = Router();

// Deprecated legacy endpoint kept for backward compatibility
router.post('/login', (req: Request, res: Response) => {
  return res.status(410).json({
    error: 'Admin login is now handled by Clerk',
    message: 'Use /sign-in and ensure your profile role is admin',
  });
});

// Add teacher using Supabase Admin
router.post('/add-teacher', clerkAuth, requireRole('admin'), async (req: AuthenticatedRequest, res: Response) => {
  const { email, password, name } = req.body;
  if (!email || !password || !name) return res.status(400).json({ error: 'Missing fields' });

  try {
    const user = await createTeacherAccount(email, password, name);
    return res.json({ message: 'Teacher created successfully', userId: user.id });
  } catch (error: any) {
    console.error("Create teacher error:", error);
    return res.status(500).json({ error: error.message || 'Failed to create teacher' });
  }
});

// Update user profile (admin only)
// Update user profile (admin only) - Placeholder for future implementation using Supabase
router.put('/update-profile/:id', clerkAuth, requireRole('admin'), (req: Request, res: Response) => {
  return res.status(501).json({ error: 'Not implemented yet' });
});

export default router;
