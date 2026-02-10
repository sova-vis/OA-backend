import { Router, Request, Response } from 'express';
import { supabase } from './lib/supabase';

const router = Router();

// Signup (students/teachers)
router.post('/signup', async (req: Request, res: Response) => {
  const { email, password, name, school, city, country } = req.body;
  const { data, error } = await supabase.auth.admin.createUser({
    email,
    password,
    user_metadata: { name, school, city, country, role: 'student' },
  });
  if (error) return res.status(400).json({ error: error.message });
  return res.json({ user: data.user });
});

// Login (students/teachers)
router.post('/login', async (req: Request, res: Response) => {
  const { email, password } = req.body;
  console.log('POST /login called');
  console.log('Login attempt:', { email });
  try {
    const { data, error } = await supabase.auth.signInWithPassword({ email, password });
    if (error) {
      console.error('Supabase login error:', error.message);
      return res.status(401).json({ error: error.message });
    }
    console.log('Supabase login success:', { user: data.user?.id, session: !!data.session });

    // Fetch user role from profiles table
    const { data: profile, error: profileError } = await supabase
      .from('profiles')
      .select('role, full_name')
      .eq('id', data.user.id)
      .single();

    if (profileError) {
      console.error('Profile fetch error:', profileError);
      return res.status(500).json({ error: 'Failed to fetch user profile' });
    }

    return res.json({
      user: data.user,
      session: data.session,
      role: profile?.role || 'student',
      name: profile?.full_name
    });
  } catch (err) {
    console.error('Unexpected error in /login:', err);
    return res.status(500).json({ error: 'Internal server error' });
  }
});

// Verify JWT (for protected routes)
router.get('/verify', async (req: Request, res: Response) => {
  const token = req.headers['authorization']?.split(' ')[1];
  if (!token) return res.status(401).json({ error: 'No token provided' });
  const { data, error } = await supabase.auth.getUser(token);
  if (error) return res.status(401).json({ error: error.message });
  return res.json({ user: data.user });
});

export default router;
