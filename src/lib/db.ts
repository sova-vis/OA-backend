// Simple in-memory DB for demonstration. Replace with real DB in production.
export type UserRole = 'admin' | 'teacher' | 'student';

export interface UserProfile {
  id: string;
  email: string;
  password: string; // hashed in real DB
  name: string;
  role: UserRole;
}

// Initial admin user (hardcoded, not exposed to frontend)
export const users: UserProfile[] = [
  {
    id: 'admin-1',
    email: 'admin@propel.com',
    password: 'SuperSecretAdmin123', // In production, hash this!
    name: 'Admin',
    role: 'admin',
  },
];

export function findUserByEmail(email: string) {
  return users.find(u => u.email === email);
}

export function addUser(user: UserProfile) {
  users.push(user);
}

export function updateUser(id: string, updates: Partial<UserProfile>) {
  const user = users.find(u => u.id === id);
  if (user) Object.assign(user, updates);
  return user;
}
