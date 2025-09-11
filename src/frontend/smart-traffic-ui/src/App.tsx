import React, { Suspense, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';
import { Toaster } from 'react-hot-toast';
import { AppProvider, useApp } from '@/contexts/AppContext';
import { DashboardLayout } from '@/components/layout/DashboardLayout';
import { ErrorBoundary } from '@/components/common/ErrorBoundary';
import { LoadingSpinner } from '@/components/common/LoadingSpinner';
import { LoginPage } from '@/pages/LoginPage';
import { Dashboard } from '@/components/dashboard/Dashboard';
import { TrafficLightsPage } from '@/pages/TrafficLightsPage';
import { AnalyticsPage } from '@/pages/AnalyticsPage';
import { ConfigurationPage } from '@/pages/ConfigurationPage';
import { NotFoundPage } from '@/pages/NotFoundPage';
import '@/styles/global.scss';

// Lazy load pages for better performance
const TrafficLightsPageLazy = React.lazy(() => import('@/pages/TrafficLightsPage'));
const AnalyticsPageLazy = React.lazy(() => import('@/pages/AnalyticsPage'));
const ConfigurationPageLazy = React.lazy(() => import('@/pages/ConfigurationPage'));

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
      refetchOnWindowFocus: false,
      refetchOnReconnect: true
    },
    mutations: {
      retry: 1,
      retryDelay: 1000
    }
  }
});

// Main App Routes Component
function AppRoutes() {
  const { isAuthenticated, loading } = useApp();

  if (loading.isLoading) {
    return <LoadingSpinner />;
  }

  if (!isAuthenticated) {
    return (
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="*" element={<Navigate to="/login" replace />} />
      </Routes>
    );
  }

  return (
    <DashboardLayout>
      <Suspense fallback={<LoadingSpinner />}>
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/traffic" element={<Navigate to="/traffic/lights" replace />} />
          <Route path="/traffic/lights" element={<TrafficLightsPageLazy />} />
          <Route path="/traffic/intersections" element={<div>Intersections Page</div>} />
          <Route path="/traffic/vehicles" element={<div>Vehicles Page</div>} />
          <Route path="/analytics" element={<Navigate to="/analytics/performance" replace />} />
          <Route path="/analytics/performance" element={<AnalyticsPageLazy />} />
          <Route path="/analytics/reports" element={<div>Reports Page</div>} />
          <Route path="/analytics/trends" element={<div>Trends Page</div>} />
          <Route path="/simulation" element={<div>Simulation Page</div>} />
          <Route path="/configuration" element={<Navigate to="/configuration/system" replace />} />
          <Route path="/configuration/system" element={<ConfigurationPageLazy />} />
          <Route path="/configuration/users" element={<div>User Management Page</div>} />
          <Route path="/configuration/alerts" element={<div>Alerts Page</div>} />
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </Suspense>
    </DashboardLayout>
  );
}

// Main App Component
function App() {
  useEffect(() => {
    // Initialize app theme
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
      try {
        const theme = JSON.parse(savedTheme);
        document.documentElement.setAttribute('data-theme', theme.name);
      } catch (error) {
        console.error('Failed to load saved theme:', error);
      }
    }

    // Set initial theme
    document.documentElement.setAttribute('data-theme', 'sap_horizon');
  }, []);

  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <AppProvider>
          <Router>
            <div className="app">
              <AppRoutes />
              
              {/* Global Toast Notifications */}
              <Toaster
                position="top-right"
                toastOptions={{
                  duration: 4000,
                  style: {
                    background: 'var(--sapBackgroundColor)',
                    color: 'var(--sapTextColor)',
                    border: '1px solid var(--sapBorderColor)',
                    borderRadius: '8px',
                    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)'
                  },
                  success: {
                    iconTheme: {
                      primary: 'var(--sapPositiveColor)',
                      secondary: 'white'
                    }
                  },
                  error: {
                    iconTheme: {
                      primary: 'var(--sapNegativeColor)',
                      secondary: 'white'
                    }
                  },
                  loading: {
                    iconTheme: {
                      primary: 'var(--sapHighlightColor)',
                      secondary: 'white'
                    }
                  }
                }}
              />
            </div>
          </Router>
        </AppProvider>
      </QueryClientProvider>
    </ErrorBoundary>
  );
}

export default App;
