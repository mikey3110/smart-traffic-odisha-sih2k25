import { ApiService } from './apiService';
import { User, LoginForm, UserRole, Permission } from '@/types';

export interface LoginResponse {
  user: User;
  token: string;
  refreshToken: string;
}

export interface RegisterForm {
  name: string;
  email: string;
  password: string;
  confirmPassword: string;
  role: UserRole;
}

class AuthService {
  // Login user
  async login(credentials: LoginForm): Promise<LoginResponse> {
    const response = await ApiService.post<LoginResponse>('/auth/login', credentials);
    return response;
  }

  // Register user
  async register(userData: RegisterForm): Promise<LoginResponse> {
    const response = await ApiService.post<LoginResponse>('/auth/register', userData);
    return response;
  }

  // Get current user
  async getCurrentUser(): Promise<User> {
    const response = await ApiService.get<User>('/auth/me');
    return response;
  }

  // Refresh token
  async refreshToken(): Promise<{ token: string; refreshToken: string }> {
    const refreshToken = localStorage.getItem('refreshToken');
    const response = await ApiService.post<{ token: string; refreshToken: string }>('/auth/refresh', {
      refreshToken
    });
    return response;
  }

  // Logout user
  async logout(): Promise<void> {
    try {
      await ApiService.post('/auth/logout');
    } catch (error) {
      // Even if logout fails on server, clear local storage
      console.error('Logout error:', error);
    }
  }

  // Change password
  async changePassword(currentPassword: string, newPassword: string): Promise<void> {
    await ApiService.post('/auth/change-password', {
      currentPassword,
      newPassword
    });
  }

  // Reset password request
  async requestPasswordReset(email: string): Promise<void> {
    await ApiService.post('/auth/forgot-password', { email });
  }

  // Reset password with token
  async resetPassword(token: string, newPassword: string): Promise<void> {
    await ApiService.post('/auth/reset-password', {
      token,
      newPassword
    });
  }

  // Update user profile
  async updateProfile(userData: Partial<User>): Promise<User> {
    const response = await ApiService.put<User>('/auth/profile', userData);
    return response;
  }

  // Upload avatar
  async uploadAvatar(file: File): Promise<{ avatarUrl: string }> {
    const response = await ApiService.upload<{ avatarUrl: string }>('/auth/avatar', file);
    return response;
  }

  // Check if user is authenticated
  isAuthenticated(): boolean {
    const token = localStorage.getItem('authToken') || sessionStorage.getItem('authToken');
    return !!token;
  }

  // Get stored token
  getToken(): string | null {
    return localStorage.getItem('authToken') || sessionStorage.getItem('authToken');
  }

  // Mock user for development
  getMockUser(): User {
    return {
      id: '1',
      name: 'John Doe',
      email: 'john.doe@example.com',
      role: UserRole.ADMIN,
      avatar: 'https://via.placeholder.com/150',
      permissions: [
        Permission.VIEW_DASHBOARD,
        Permission.MANAGE_TRAFFIC,
        Permission.CONFIGURE_SYSTEM,
        Permission.VIEW_ANALYTICS,
        Permission.MANAGE_USERS
      ],
      lastLogin: new Date()
    };
  }

  // Mock login for development
  async mockLogin(credentials: LoginForm): Promise<LoginResponse> {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    if (credentials.email === 'admin@example.com' && credentials.password === 'admin123') {
      return {
        user: this.getMockUser(),
        token: 'mock-jwt-token',
        refreshToken: 'mock-refresh-token'
      };
    }
    
    throw new Error('Invalid credentials');
  }
}

export const authService = new AuthService();
