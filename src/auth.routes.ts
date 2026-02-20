import { Router, Request, Response } from 'express';
import { supabase } from './lib/supabase';
import { env } from './lib/env';

const router = Router();

// Signup (students/teachers)
router.post('/signup', async (req: Request, res: Response) => {
  const { email, password, name, school, city, country } = req.body;
  
  // Use proper signup with email confirmation
  const { data, error } = await supabase.auth.signUp({
    email,
    password,
    options: {
      emailRedirectTo: `${env.FRONTEND_URL}/auth/callback`,
      data: {
        full_name: name,
        school: school,
        city: city,
        country: country,
      }
    }
  });
  
  if (error) return res.status(400).json({ error: error.message });
  
  // Create profile record if user was created
  if (data.user) {
    const { error: profileError } = await supabase
      .from('profiles')
      .insert({
        id: data.user.id,
        full_name: name,
        role: 'student',
        region_school: `${school}, ${city}, ${country}`,
      });
    
    if (profileError) {
      console.error('Profile creation error:', profileError);
    }
  }
  
  return res.json({ 
    user: data.user,
    message: 'Signup successful! Please check your email to verify your account.'
  });
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
    let { data: profile, error: profileError } = await supabase
      .from('profiles')
      .select('role, full_name, onboarding_complete')
      .eq('id', data.user.id)
      .single();

    // If profile doesn't exist, create it from user metadata
    if (profileError && profileError.code === 'PGRST116') {
      console.log('Profile not found, creating from user metadata');
      const metadata = data.user.user_metadata || {};
      const hasRegionSchool = metadata.region_school && metadata.region_school !== '';
      
      const { data: newProfile, error: createError } = await supabase
        .from('profiles')
        .insert({
          id: data.user.id,
          full_name: metadata.full_name || metadata.name || '',
          role: 'student',
          region_school: metadata.region_school || '',
          // If region_school exists, user filled manual form, skip onboarding
          onboarding_complete: hasRegionSchool,
          level: 'O Level',
        })
        .select()
        .single();

      if (createError) {
        console.error('Error creating profile:', createError);
        return res.status(500).json({ error: 'Failed to create user profile' });
      }

      profile = newProfile;
    } else if (profileError) {
      console.error('Profile fetch error:', profileError);
      return res.status(500).json({ error: 'Failed to fetch user profile' });
    }

    return res.json({
      user: data.user,
      session: data.session,
      role: profile?.role || 'student',
      name: profile?.full_name,
      onboarding_complete: profile?.onboarding_complete || false,
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

// Google OAuth callback
router.post('/google', async (req: Request, res: Response) => {
  const { token } = req.body;
  
  try {
    const { data, error } = await supabase.auth.getUser(token);
    if (error) return res.status(401).json({ error: error.message });
    
    // Get or create profile
    const { data: profile, error: profileError } = await supabase
      .from('profiles')
      .select('role, full_name, onboarding_complete')
      .eq('id', data.user.id)
      .single();
    
    if (profileError && profileError.code === 'PGRST116') {
      // Profile doesn't exist, create it - needs onboarding since Google doesn't provide detailed info
      const { error: insertError } = await supabase
        .from('profiles')
        .insert({
          id: data.user.id,
          full_name: data.user.user_metadata.full_name || data.user.user_metadata.name || '',
          role: 'student',
          onboarding_complete: false, // Google users need onboarding
        });
      
      if (insertError) {
        console.error('Profile creation error:', insertError);
      }
      
      return res.json({
        user: data.user,
        role: 'student',
        name: data.user.user_metadata.full_name || 'User',
        needsOnboarding: true
      });
    }
    
    return res.json({
      user: data.user,
      role: profile?.role || 'student',
      name: profile?.full_name || 'User',
      needsOnboarding: !profile?.onboarding_complete
    });
  } catch (err) {
    console.error('Google auth error:', err);
    return res.status(500).json({ error: 'Internal server error' });
  }
});

export default router;
