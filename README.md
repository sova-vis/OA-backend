# Backend - Express API

This is the backend API server built with Express and PostgreSQL.

## Structure

```
backend/
├── src/
│   ├── routes/           # Route definitions
│   ├── controllers/      # HTTP request handlers
│   ├── services/         # Business logic
│   ├── repositories/     # Database access layer
│   ├── models/           # Database schemas (TODO)
│   ├── middleware/       # Auth, error handling
│   ├── validators/       # Request validation (TODO)
│   ├── lib/              # Utilities (db, logger, env)
│   ├── storage/          # File storage (TODO)
│   └── index.ts          # Entry point
├── package.json
├── tsconfig.json
└── README.md
```

## Environment Variables

Create a `.env` file in the backend directory:

```env
NODE_ENV=development
PORT=3001

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=education_platform
DB_USER=postgres
DB_PASSWORD=your_password

# JWT
JWT_SECRET=your-super-secret-jwt-key-at-least-32-characters-long
JWT_EXPIRES_IN=7d

# CORS
FRONTEND_URL=http://localhost:3000
```

## Getting Started

### Install Dependencies
```bash
cd backend
npm install
```

### Setup Database
```bash
# Create PostgreSQL database
createdb education_platform

# Run migrations (TODO: Add migration tool)
```

### Development
```bash
npm run dev
```
Runs on http://localhost:3001

### Build
```bash
npm run build
npm start
```

## API Endpoints

### Authentication
- `POST /auth/register` - Register new user
- `POST /auth/login` - Login user
- `POST /auth/logout` - Logout user
- `GET /auth/me` - Get current user

### Users
- `GET /users/:id` - Get user profile
- `PUT /users/:id` - Update user profile
- `DELETE /users/:id` - Delete user (admin only)

### Content
- `GET /content/past-papers` - Get all past papers
- `GET /content/past-papers/:id` - Get past paper by ID
- `GET /content/topicals` - Get all topicals
- `GET /content/topicals/:id` - Get topical by ID
- `GET /content/papers/:paperId/questions` - Get questions for a paper

### Attempts
- `POST /attempts` - Create attempt
- `GET /attempts` - Get user attempts
- `GET /attempts/:id` - Get attempt by ID
- `PUT /attempts/:id` - Update attempt

### Bookmarks
- `GET /bookmarks` - Get user bookmarks
- `POST /bookmarks` - Create bookmark
- `DELETE /bookmarks/:id` - Delete bookmark

### Progress
- `GET /progress` - Get user progress
- `GET /progress/stats` - Get user stats

### Admin (admin role only)
- `GET /admin/users` - Get all users
- `GET /admin/content` - Get all content
- `GET /admin/teachers` - Get all teachers
- `PUT /admin/users/:id/role` - Update user role

### Teacher (teacher role only)
- `GET /teacher/sessions` - Get teacher sessions
- `GET /teacher/sessions/:id` - Get session by ID
- `POST /teacher/availability` - Set availability
- `GET /teacher/availability` - Get availability

## Architecture

### Layers

1. **Routes** (`routes/`) - Define endpoints and map to controllers
2. **Controllers** (`controllers/`) - Handle HTTP requests, validate input, call services
3. **Services** (`services/`) - Business logic, data processing
4. **Repositories** (`repositories/`) - Database queries, data access
5. **Models** (`models/`) - Database schemas (TODO: Add Prisma/Drizzle)

### Request Flow

```
Client Request
    ↓
Route Handler
    ↓
Middleware (auth, validation)
    ↓
Controller (validate, parse)
    ↓
Service (business logic)
    ↓
Repository (database query)
    ↓
Database
```

## Importing Shared Types

```typescript
import { User, UserRole, PastPaper } from '@shared/types';
```

## Deployment

Deploy to Railway, Render, or any Node.js hosting:

```bash
npm run build
npm start
```

Make sure to set all environment variables in your hosting platform.
