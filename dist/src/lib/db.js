"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.users = void 0;
exports.findUserByEmail = findUserByEmail;
exports.addUser = addUser;
exports.updateUser = updateUser;
// Initial admin user (hardcoded, not exposed to frontend)
exports.users = [
    {
        id: 'admin-1',
        email: 'admin@propel.com',
        password: 'SuperSecretAdmin123', // In production, hash this!
        name: 'Admin',
        role: 'admin',
    },
];
function findUserByEmail(email) {
    return exports.users.find(u => u.email === email);
}
function addUser(user) {
    exports.users.push(user);
}
function updateUser(id, updates) {
    const user = exports.users.find(u => u.id === id);
    if (user)
        Object.assign(user, updates);
    return user;
}
