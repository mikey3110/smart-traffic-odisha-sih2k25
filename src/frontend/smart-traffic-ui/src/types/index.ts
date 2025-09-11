// Core application types
export interface User {
  id: string;
  name: string;
  email: string;
  role: UserRole;
  avatar?: string;
  permissions: Permission[];
  lastLogin?: Date;
}

export enum UserRole {
  ADMIN = 'admin',
  OPERATOR = 'operator',
  VIEWER = 'viewer'
}

export enum Permission {
  VIEW_DASHBOARD = 'view_dashboard',
  MANAGE_TRAFFIC = 'manage_traffic',
  CONFIGURE_SYSTEM = 'configure_system',
  VIEW_ANALYTICS = 'view_analytics',
  MANAGE_USERS = 'manage_users'
}

// Traffic management types
export interface TrafficLight {
  id: string;
  name: string;
  location: {
    lat: number;
    lng: number;
  };
  status: TrafficLightStatus;
  currentPhase: number;
  phaseDuration: number;
  program: string;
  lastUpdate: Date;
  vehicleCount: number;
  waitingTime: number;
}

export enum TrafficLightStatus {
  NORMAL = 'normal',
  MAINTENANCE = 'maintenance',
  ERROR = 'error',
  OFFLINE = 'offline'
}

export interface Vehicle {
  id: string;
  type: VehicleType;
  position: {
    lat: number;
    lng: number;
  };
  speed: number;
  lane: string;
  route: string[];
  waitingTime: number;
  co2Emission: number;
  fuelConsumption: number;
  timestamp: Date;
}

export enum VehicleType {
  PASSENGER = 'passenger',
  TRUCK = 'truck',
  BUS = 'bus',
  MOTORCYCLE = 'motorcycle',
  EMERGENCY = 'emergency'
}

export interface Intersection {
  id: string;
  name: string;
  location: {
    lat: number;
    lng: number;
  };
  trafficLights: TrafficLight[];
  totalVehicles: number;
  waitingVehicles: number;
  averageSpeed: number;
  throughput: number;
  lastUpdate: Date;
}

// Analytics and metrics types
export interface PerformanceMetrics {
  timestamp: Date;
  totalVehicles: number;
  runningVehicles: number;
  waitingVehicles: number;
  totalWaitingTime: number;
  averageSpeed: number;
  totalCo2Emission: number;
  totalFuelConsumption: number;
  throughput: number;
}

export interface ChartData {
  labels: string[];
  datasets: ChartDataset[];
}

export interface ChartDataset {
  label: string;
  data: number[];
  backgroundColor?: string | string[];
  borderColor?: string | string[];
  borderWidth?: number;
  fill?: boolean;
  tension?: number;
}

// Notification types
export interface Notification {
  id: string;
  type: NotificationType;
  title: string;
  message: string;
  timestamp: Date;
  read: boolean;
  priority: NotificationPriority;
  actionUrl?: string;
}

export enum NotificationType {
  INFO = 'info',
  SUCCESS = 'success',
  WARNING = 'warning',
  ERROR = 'error',
  ALERT = 'alert'
}

export enum NotificationPriority {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

// Configuration types
export interface SystemConfig {
  simulation: {
    enabled: boolean;
    stepSize: number;
    endTime: number;
  };
  trafficLights: {
    minPhaseDuration: number;
    maxPhaseDuration: number;
    updateInterval: number;
  };
  dataExport: {
    enabled: boolean;
    interval: number;
    format: string;
  };
  notifications: {
    enabled: boolean;
    email: boolean;
    push: boolean;
  };
}

// API response types
export interface ApiResponse<T> {
  success: boolean;
  data: T;
  message?: string;
  error?: string;
  timestamp: Date;
}

export interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  };
}

// WebSocket message types
export interface WebSocketMessage {
  type: string;
  payload: any;
  timestamp: Date;
}

export interface RealtimeData {
  intersections: Intersection[];
  vehicles: Vehicle[];
  metrics: PerformanceMetrics;
  alerts: Notification[];
}

// Form types
export interface LoginForm {
  email: string;
  password: string;
  rememberMe: boolean;
}

export interface TrafficLightConfigForm {
  id: string;
  name: string;
  minPhaseDuration: number;
  maxPhaseDuration: number;
  program: string;
  enabled: boolean;
}

export interface SystemSettingsForm {
  simulation: {
    enabled: boolean;
    stepSize: number;
    endTime: number;
  };
  notifications: {
    enabled: boolean;
    email: boolean;
    push: boolean;
  };
  dataExport: {
    enabled: boolean;
    interval: number;
    format: string;
  };
}

// Filter and search types
export interface FilterOptions {
  dateRange?: {
    start: Date;
    end: Date;
  };
  vehicleTypes?: VehicleType[];
  trafficLightStatus?: TrafficLightStatus[];
  intersectionIds?: string[];
}

export interface SortOptions {
  field: string;
  direction: 'asc' | 'desc';
}

export interface TableColumn {
  key: string;
  label: string;
  sortable?: boolean;
  filterable?: boolean;
  width?: string;
  align?: 'left' | 'center' | 'right';
  render?: (value: any, row: any) => React.ReactNode;
}

// Theme types
export interface Theme {
  name: string;
  colors: {
    primary: string;
    secondary: string;
    success: string;
    warning: string;
    error: string;
    info: string;
    background: string;
    surface: string;
    text: string;
    textSecondary: string;
  };
  spacing: {
    xs: string;
    sm: string;
    md: string;
    lg: string;
    xl: string;
  };
  typography: {
    fontFamily: string;
    fontSize: {
      xs: string;
      sm: string;
      md: string;
      lg: string;
      xl: string;
    };
  };
}

// Error types
export interface AppError {
  code: string;
  message: string;
  details?: any;
  timestamp: Date;
  stack?: string;
}

// Loading states
export interface LoadingState {
  isLoading: boolean;
  error?: AppError;
  progress?: number;
}

// Context types
export interface AppContextType {
  user: User | null;
  theme: Theme;
  notifications: Notification[];
  systemConfig: SystemConfig;
  loading: LoadingState;
  login: (credentials: LoginForm) => Promise<void>;
  logout: () => void;
  updateTheme: (theme: Theme) => void;
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp'>) => void;
  removeNotification: (id: string) => void;
  markNotificationAsRead: (id: string) => void;
  updateSystemConfig: (config: Partial<SystemConfig>) => void;
}

// Hook types
export interface UseApiOptions {
  enabled?: boolean;
  refetchInterval?: number;
  onSuccess?: (data: any) => void;
  onError?: (error: AppError) => void;
}

export interface UseRealtimeOptions {
  enabled?: boolean;
  reconnectInterval?: number;
  onMessage?: (message: WebSocketMessage) => void;
  onError?: (error: Event) => void;
}
