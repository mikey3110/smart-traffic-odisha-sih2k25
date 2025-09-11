import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Card,
  CardHeader,
  Text,
  Title,
  Toolbar,
  ToolbarSpacer,
  Button,
  Icon,
  Switch,
  Label,
  Badge
} from '@ui5/webcomponents-react';
import { useApp } from '@/contexts/AppContext';
import { TrafficOverview } from './TrafficOverview';
import { PerformanceMetrics } from './PerformanceMetrics';
import { TrafficLightsGrid } from './TrafficLightsGrid';
import { RealTimeChart } from './RealTimeChart';
import { AlertsPanel } from './AlertsPanel';
import { useWebSocket } from '@/services/websocketService';
import { trafficService } from '@/services/trafficService';
import { PerformanceMetrics as PerformanceMetricsType } from '@/types';
import './Dashboard.scss';

export function Dashboard() {
  const { systemConfig, updateSystemConfig } = useApp();
  const [metrics, setMetrics] = useState<PerformanceMetricsType | null>(null);
  const [loading, setLoading] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(5000);

  // WebSocket connection for real-time updates
  const { isConnected, subscribe } = useWebSocket({
    enabled: true,
    onMessage: (message) => {
      if (message.type === 'metrics_update') {
        setMetrics(message.payload);
      }
    }
  });

  // Fetch initial data
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const currentMetrics = await trafficService.getCurrentMetrics();
        setMetrics(currentMetrics);
      } catch (error) {
        console.error('Failed to fetch dashboard data:', error);
        // Use mock data in case of error
        setMetrics({
          timestamp: new Date(),
          totalVehicles: 156,
          runningVehicles: 142,
          waitingVehicles: 14,
          totalWaitingTime: 234.5,
          averageSpeed: 28.3,
          totalCo2Emission: 45.2,
          totalFuelConsumption: 23.1,
          throughput: 89
        });
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  // Auto-refresh functionality
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(async () => {
      try {
        const currentMetrics = await trafficService.getCurrentMetrics();
        setMetrics(currentMetrics);
      } catch (error) {
        console.error('Failed to refresh metrics:', error);
      }
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval]);

  // Subscribe to real-time updates
  useEffect(() => {
    const unsubscribe = subscribe('realtime_data', (data) => {
      if (data.metrics) {
        setMetrics(data.metrics);
      }
    });

    return unsubscribe;
  }, [subscribe]);

  const handleRefresh = async () => {
    setLoading(true);
    try {
      const currentMetrics = await trafficService.getCurrentMetrics();
      setMetrics(currentMetrics);
    } catch (error) {
      console.error('Failed to refresh data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSimulationToggle = (enabled: boolean) => {
    updateSystemConfig({
      simulation: { ...systemConfig.simulation, enabled }
    });
  };

  return (
    <div className="dashboard">
      {/* Header */}
      <div className="dashboard-header">
        <div className="header-left">
          <Title level="H1">Traffic Management Dashboard</Title>
          <Text>Real-time traffic control and monitoring</Text>
        </div>
        <div className="header-right">
          <div className="connection-status">
            <Icon name={isConnected ? 'connected' : 'disconnected'} />
            <Text>{isConnected ? 'Connected' : 'Disconnected'}</Text>
          </div>
          <div className="simulation-control">
            <Label>Simulation</Label>
            <Switch
              checked={systemConfig.simulation.enabled}
              onChange={(e) => handleSimulationToggle(e.target.checked)}
            />
          </div>
          <Button
            icon="refresh"
            design="Transparent"
            onClick={handleRefresh}
            disabled={loading}
          >
            Refresh
          </Button>
        </div>
      </div>

      {/* Main Content */}
      <div className="dashboard-content">
        {/* Top Row - Overview and Metrics */}
        <div className="dashboard-row">
          <motion.div
            className="overview-card"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <TrafficOverview metrics={metrics} loading={loading} />
          </motion.div>
          
          <motion.div
            className="metrics-card"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <PerformanceMetrics metrics={metrics} loading={loading} />
          </motion.div>
        </div>

        {/* Middle Row - Charts and Alerts */}
        <div className="dashboard-row">
          <motion.div
            className="chart-card"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Card>
              <CardHeader
                titleText="Real-time Performance"
                subtitleText="Live traffic metrics and trends"
              >
                <div className="card-actions">
                  <Badge colorScheme="8">Live</Badge>
                  <Button
                    icon="full-screen"
                    design="Transparent"
                    onClick={() => {/* Open full screen */}}
                  />
                </div>
              </CardHeader>
              <div className="card-content">
                <RealTimeChart metrics={metrics} />
              </div>
            </Card>
          </motion.div>

          <motion.div
            className="alerts-card"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <AlertsPanel />
          </motion.div>
        </div>

        {/* Bottom Row - Traffic Lights Grid */}
        <motion.div
          className="traffic-lights-section"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          <TrafficLightsGrid />
        </motion.div>
      </div>

      {/* Auto-refresh Controls */}
      <div className="dashboard-controls">
        <Toolbar>
          <div className="controls-left">
            <Label>Auto-refresh</Label>
            <Switch
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            {autoRefresh && (
              <>
                <Label>Interval</Label>
                <select
                  value={refreshInterval}
                  onChange={(e) => setRefreshInterval(Number(e.target.value))}
                  className="interval-select"
                >
                  <option value={1000}>1 second</option>
                  <option value={5000}>5 seconds</option>
                  <option value={10000}>10 seconds</option>
                  <option value={30000}>30 seconds</option>
                </select>
              </>
            )}
          </div>
          <ToolbarSpacer />
          <div className="controls-right">
            <Text>Last updated: {metrics?.timestamp.toLocaleTimeString() || 'Never'}</Text>
          </div>
        </Toolbar>
      </div>
    </div>
  );
}
