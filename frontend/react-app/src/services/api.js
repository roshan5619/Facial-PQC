import axios from 'axios';

// const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
export const API = process.env.REACT_APP_BACKEND_URL;

const api = axios.create({
  baseURL: API,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add token to requests if available
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

export const authAPI = {
  // Face login
  login: async (imageFile) => {
    const formData = new FormData();
    formData.append('image', imageFile);
    return api.post('/api/auth/login', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },

  // Logout
  logout: () => api.post('/api/auth/logout'),

  // Verify token
  verifyToken: () => api.get('/api/auth/verify'),
};

export const registrationAPI = {
  // User self-registration
  register: async (userData) => {
    const formData = new FormData();
    formData.append('username', userData.username);
    formData.append('email', userData.email);
    formData.append('full_name', userData.full_name);
    formData.append('image1', userData.image1);
    formData.append('image2', userData.image2);
    formData.append('image3', userData.image3);
    
    return api.post('/api/registration/register', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
};

export const adminAPI = {
  // Get all users
  getUsers: () => api.get('/api/admin/users'),

  // Get access logs
  getLogs: (skip = 0, limit = 50) => 
    api.get(`/api/admin/logs?skip=${skip}&limit=${limit}`),

  // Get statistics
  getStats: () => api.get('/api/admin/stats'),
};

export const healthAPI = {
  // Health check
  check: () => api.get('/health'),
  
  // Detailed health check
  checkDetailed: () => api.get('/health/detailed'),
};

export default api;