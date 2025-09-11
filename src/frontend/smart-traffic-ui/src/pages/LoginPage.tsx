import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import {
  Card,
  CardHeader,
  Text,
  Title,
  Input,
  Button,
  CheckBox,
  MessageStrip,
  MessageStripDesign,
  Icon,
  Toolbar,
  ToolbarSpacer
} from '@ui5/webcomponents-react';
import { useApp } from '@/contexts/AppContext';
import { LoginForm } from '@/types';
import { authService } from '@/services/authService';
import './LoginPage.scss';

export function LoginPage() {
  const navigate = useNavigate();
  const { login, isAuthenticated, loading } = useApp();
  const [formData, setFormData] = useState<LoginForm>({
    email: '',
    password: '',
    rememberMe: false
  });
  const [errors, setErrors] = useState<Partial<LoginForm>>({});
  const [showPassword, setShowPassword] = useState(false);

  // Redirect if already authenticated
  useEffect(() => {
    if (isAuthenticated) {
      navigate('/dashboard');
    }
  }, [isAuthenticated, navigate]);

  const handleInputChange = (field: keyof LoginForm, value: string | boolean) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    // Clear error when user starts typing
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: undefined }));
    }
  };

  const validateForm = (): boolean => {
    const newErrors: Partial<LoginForm> = {};

    if (!formData.email) {
      newErrors.email = 'Email is required';
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = 'Email is invalid';
    }

    if (!formData.password) {
      newErrors.password = 'Password is required';
    } else if (formData.password.length < 6) {
      newErrors.password = 'Password must be at least 6 characters';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }

    try {
      await login(formData);
      navigate('/dashboard');
    } catch (error: any) {
      console.error('Login failed:', error);
      // Error is handled by the context
    }
  };

  const handleDemoLogin = async () => {
    try {
      await login({
        email: 'admin@example.com',
        password: 'admin123',
        rememberMe: false
      });
      navigate('/dashboard');
    } catch (error) {
      console.error('Demo login failed:', error);
    }
  };

  return (
    <div className="login-page">
      <div className="login-background">
        <div className="background-pattern" />
        <div className="background-overlay" />
      </div>

      <div className="login-container">
        <motion.div
          className="login-card"
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: 'easeOut' }}
        >
          <Card>
            <CardHeader>
              <div className="login-header">
                <div className="logo">
                  <Icon name="traffic-light" size="L" />
                  <Title level="H1">Smart Traffic</Title>
                </div>
                <Text>Traffic Management System</Text>
              </div>
            </CardHeader>

            <div className="login-content">
              <form onSubmit={handleSubmit} className="login-form">
                <div className="form-group">
                  <Input
                    type="email"
                    placeholder="Email address"
                    value={formData.email}
                    onInput={(e) => handleInputChange('email', e.target.value)}
                    className={errors.email ? 'error' : ''}
                    disabled={loading.isLoading}
                  />
                  {errors.email && (
                    <MessageStrip
                      design={MessageStripDesign.Negative}
                      className="error-message"
                    >
                      {errors.email}
                    </MessageStrip>
                  )}
                </div>

                <div className="form-group">
                  <Input
                    type={showPassword ? 'text' : 'password'}
                    placeholder="Password"
                    value={formData.password}
                    onInput={(e) => handleInputChange('password', e.target.value)}
                    className={errors.password ? 'error' : ''}
                    disabled={loading.isLoading}
                  />
                  <Button
                    icon={showPassword ? 'hide' : 'show'}
                    design="Transparent"
                    onClick={() => setShowPassword(!showPassword)}
                    className="password-toggle"
                  />
                  {errors.password && (
                    <MessageStrip
                      design={MessageStripDesign.Negative}
                      className="error-message"
                    >
                      {errors.password}
                    </MessageStrip>
                  )}
                </div>

                <div className="form-options">
                  <CheckBox
                    text="Remember me"
                    checked={formData.rememberMe}
                    onChange={(e) => handleInputChange('rememberMe', e.target.checked)}
                    disabled={loading.isLoading}
                  />
                  <Button
                    design="Transparent"
                    onClick={() => {/* Handle forgot password */}}
                    className="forgot-password"
                  >
                    Forgot password?
                  </Button>
                </div>

                <Button
                  type="submit"
                  design="Emphasized"
                  className="login-button"
                  disabled={loading.isLoading}
                >
                  {loading.isLoading ? (
                    <>
                      <Icon name="loading" />
                      Signing in...
                    </>
                  ) : (
                    'Sign In'
                  )}
                </Button>
              </form>

              <div className="login-divider">
                <span>or</span>
              </div>

              <Button
                design="Transparent"
                onClick={handleDemoLogin}
                className="demo-button"
                disabled={loading.isLoading}
              >
                <Icon name="play" />
                Try Demo
              </Button>

              <div className="login-footer">
                <Text>Demo credentials:</Text>
                <Text>Email: admin@example.com</Text>
                <Text>Password: admin123</Text>
              </div>
            </div>
          </Card>
        </motion.div>

        <motion.div
          className="login-info"
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <div className="info-content">
            <Title level="H2">Welcome to Smart Traffic Management</Title>
            <Text>
              Monitor and control traffic in real-time with our advanced AI-powered system.
              Optimize traffic flow, reduce congestion, and improve safety across your city.
            </Text>
            
            <div className="features">
              <div className="feature">
                <Icon name="traffic-light" />
                <div>
                  <Text className="feature-title">Real-time Control</Text>
                  <Text>Monitor and control traffic lights in real-time</Text>
                </div>
              </div>
              
              <div className="feature">
                <Icon name="analytics" />
                <div>
                  <Text className="feature-title">Advanced Analytics</Text>
                  <Text>Get insights with comprehensive traffic analytics</Text>
                </div>
              </div>
              
              <div className="feature">
                <Icon name="simulation" />
                <div>
                  <Text className="feature-title">Simulation</Text>
                  <Text>Test scenarios with our traffic simulation engine</Text>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </div>

      <div className="login-footer">
        <Toolbar>
          <Text>© 2024 Smart Traffic Management System</Text>
          <ToolbarSpacer />
          <Text>Version 2.1.0</Text>
        </Toolbar>
      </div>
    </div>
  );
}
