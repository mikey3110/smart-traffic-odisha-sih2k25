import React, { createContext, useContext, useReducer, useEffect, ReactNode } from 'react';
import { User, Theme, Notification, SystemConfig, AppError, LoadingState, AppContextType, UserRole, Permission } from '@/types';
import { authService } from '@/services/authService';
import { notificationService } from '@/services/notificationService';
import { configService } from '@/services/configService';
import { toast } from 'react-hot-toast';

// Initial state
interface AppState {
  user: User | null;
  theme: Theme;
  notifications: Notification[];
  systemConfig: SystemConfig;
  loading: LoadingState;
  isAuthenticated: boolean;
}

const initialState: AppState = {
  user: null,
  theme: {
    name: 'sap_horizon',
    colors: {
      primary: '#0070f3',
      secondary: '#6c757d',
      success: '#28a745',
      warning: '#ffc107',
      error: '#dc3545',
      info: '#17a2b8',
      background: '#ffffff',
      surface: '#f8f9fa',
      text: '#212529',
      textSecondary: '#6c757d'
    },
    spacing: {
      xs: '0.25rem',
      sm: '0.5rem',
      md: '1rem',
      lg: '1.5rem',
      xl: '3rem'
    },
    typography: {
      fontFamily: 'SAP-icons, "72", "72full", Arial, Helvetica, sans-serif',
      fontSize: {
        xs: '0.75rem',
        sm: '0.875rem',
        md: '1rem',
        lg: '1.125rem',
        xl: '1.25rem'
      }
    }
  },
  notifications: [],
  systemConfig: {
    simulation: {
      enabled: true,
      stepSize: 1,
      endTime: 3600
    },
    trafficLights: {
      minPhaseDuration: 5,
      maxPhaseDuration: 60,
      updateInterval: 1
    },
    dataExport: {
      enabled: true,
      interval: 10,
      format: 'json'
    },
    notifications: {
      enabled: true,
      email: false,
      push: true
    }
  },
  loading: {
    isLoading: false
  },
  isAuthenticated: false
};

// Action types
type AppAction =
  | { type: 'SET_LOADING'; payload: LoadingState }
  | { type: 'SET_USER'; payload: User | null }
  | { type: 'SET_THEME'; payload: Theme }
  | { type: 'ADD_NOTIFICATION'; payload: Notification }
  | { type: 'REMOVE_NOTIFICATION'; payload: string }
  | { type: 'MARK_NOTIFICATION_READ'; payload: string }
  | { type: 'UPDATE_SYSTEM_CONFIG'; payload: Partial<SystemConfig> }
  | { type: 'SET_AUTHENTICATED'; payload: boolean }
  | { type: 'CLEAR_NOTIFICATIONS' };

// Reducer
function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'SET_LOADING':
      return { ...state, loading: action.payload };
    
    case 'SET_USER':
      return { ...state, user: action.payload };
    
    case 'SET_THEME':
      return { ...state, theme: action.payload };
    
    case 'ADD_NOTIFICATION':
      return {
        ...state,
        notifications: [action.payload, ...state.notifications].slice(0, 50) // Keep only last 50
      };
    
    case 'REMOVE_NOTIFICATION':
      return {
        ...state,
        notifications: state.notifications.filter(n => n.id !== action.payload)
      };
    
    case 'MARK_NOTIFICATION_READ':
      return {
        ...state,
        notifications: state.notifications.map(n =>
          n.id === action.payload ? { ...n, read: true } : n
        )
      };
    
    case 'UPDATE_SYSTEM_CONFIG':
      return {
        ...state,
        systemConfig: { ...state.systemConfig, ...action.payload }
      };
    
    case 'SET_AUTHENTICATED':
      return { ...state, isAuthenticated: action.payload };
    
    case 'CLEAR_NOTIFICATIONS':
      return { ...state, notifications: [] };
    
    default:
      return state;
  }
}

// Context
const AppContext = createContext<AppContextType | undefined>(undefined);

// Provider component
interface AppProviderProps {
  children: ReactNode;
}

export function AppProvider({ children }: AppProviderProps) {
  const [state, dispatch] = useReducer(appReducer, initialState);

  // Initialize app
  useEffect(() => {
    const initializeApp = async () => {
      try {
        dispatch({ type: 'SET_LOADING', payload: { isLoading: true } });
        
        // Check if user is already logged in
        const token = localStorage.getItem('authToken');
        if (token) {
          const user = await authService.getCurrentUser();
          dispatch({ type: 'SET_USER', payload: user });
          dispatch({ type: 'SET_AUTHENTICATED', payload: true });
        }
        
        // Load system configuration
        const config = await configService.getSystemConfig();
        dispatch({ type: 'UPDATE_SYSTEM_CONFIG', payload: config });
        
        // Load notifications
        const notifications = await notificationService.getNotifications();
        dispatch({ type: 'CLEAR_NOTIFICATIONS' });
        notifications.forEach(notification => {
          dispatch({ type: 'ADD_NOTIFICATION', payload: notification });
        });
        
      } catch (error) {
        console.error('Failed to initialize app:', error);
        toast.error('Failed to initialize application');
      } finally {
        dispatch({ type: 'SET_LOADING', payload: { isLoading: false } });
      }
    };

    initializeApp();
  }, []);

  // Context methods
  const login = async (credentials: { email: string; password: string; rememberMe: boolean }) => {
    try {
      dispatch({ type: 'SET_LOADING', payload: { isLoading: true } });
      
      const { user, token } = await authService.login(credentials);
      
      if (credentials.rememberMe) {
        localStorage.setItem('authToken', token);
      } else {
        sessionStorage.setItem('authToken', token);
      }
      
      dispatch({ type: 'SET_USER', payload: user });
      dispatch({ type: 'SET_AUTHENTICATED', payload: true });
      
      toast.success(`Welcome back, ${user.name}!`);
      
    } catch (error: any) {
      const errorMessage = error.message || 'Login failed';
      toast.error(errorMessage);
      throw error;
    } finally {
      dispatch({ type: 'SET_LOADING', payload: { isLoading: false } });
    }
  };

  const logout = async () => {
    try {
      await authService.logout();
      
      localStorage.removeItem('authToken');
      sessionStorage.removeItem('authToken');
      
      dispatch({ type: 'SET_USER', payload: null });
      dispatch({ type: 'SET_AUTHENTICATED', payload: false });
      dispatch({ type: 'CLEAR_NOTIFICATIONS' });
      
      toast.success('Logged out successfully');
      
    } catch (error) {
      console.error('Logout error:', error);
      toast.error('Logout failed');
    }
  };

  const updateTheme = (theme: Theme) => {
    dispatch({ type: 'SET_THEME', payload: theme });
    localStorage.setItem('theme', JSON.stringify(theme));
  };

  const addNotification = (notification: Omit<Notification, 'id' | 'timestamp'>) => {
    const newNotification: Notification = {
      ...notification,
      id: Date.now().toString(),
      timestamp: new Date()
    };
    
    dispatch({ type: 'ADD_NOTIFICATION', payload: newNotification });
    
    // Show toast based on notification type
    switch (notification.type) {
      case 'success':
        toast.success(notification.message);
        break;
      case 'error':
        toast.error(notification.message);
        break;
      case 'warning':
        toast(notification.message, { icon: '⚠️' });
        break;
      case 'info':
        toast(notification.message, { icon: 'ℹ️' });
        break;
      default:
        toast(notification.message);
    }
  };

  const removeNotification = (id: string) => {
    dispatch({ type: 'REMOVE_NOTIFICATION', payload: id });
  };

  const markNotificationAsRead = (id: string) => {
    dispatch({ type: 'MARK_NOTIFICATION_READ', payload: id });
  };

  const updateSystemConfig = (config: Partial<SystemConfig>) => {
    dispatch({ type: 'UPDATE_SYSTEM_CONFIG', payload: config });
    configService.updateSystemConfig(config);
  };

  // Check permissions
  const hasPermission = (permission: Permission): boolean => {
    if (!state.user) return false;
    return state.user.permissions.includes(permission);
  };

  const hasRole = (role: UserRole): boolean => {
    if (!state.user) return false;
    return state.user.role === role;
  };

  const contextValue: AppContextType = {
    ...state,
    login,
    logout,
    updateTheme,
    addNotification,
    removeNotification,
    markNotificationAsRead,
    updateSystemConfig,
    hasPermission,
    hasRole
  };

  return (
    <AppContext.Provider value={contextValue}>
      {children}
    </AppContext.Provider>
  );
}

// Custom hook to use the context
export function useApp() {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
}

// HOC for protected routes
interface ProtectedRouteProps {
  children: ReactNode;
  requiredPermission?: Permission;
  requiredRole?: UserRole;
  fallback?: ReactNode;
}

export function ProtectedRoute({ 
  children, 
  requiredPermission, 
  requiredRole, 
  fallback = <div>Access Denied</div> 
}: ProtectedRouteProps) {
  const { user, hasPermission, hasRole } = useApp();
  
  if (!user) {
    return <div>Please log in</div>;
  }
  
  if (requiredPermission && !hasPermission(requiredPermission)) {
    return <>{fallback}</>;
  }
  
  if (requiredRole && !hasRole(requiredRole)) {
    return <>{fallback}</>;
  }
  
  return <>{children}</>;
}
