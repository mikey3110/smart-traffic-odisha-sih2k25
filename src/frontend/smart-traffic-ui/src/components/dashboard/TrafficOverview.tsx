import React from "react";
import { motion } from "framer-motion";
import {
  Card,
  CardHeader,
  Text,
  Title,
  Icon,
  Badge,
  ProgressIndicator,
} from "@ui5/webcomponents-react";
import { PerformanceMetrics } from "@/types";
import "./TrafficOverview.scss";

interface TrafficOverviewProps {
  metrics: PerformanceMetrics | null;
  loading: boolean;
}

export function TrafficOverview({ metrics, loading }: TrafficOverviewProps) {
  if (loading) {
    return (
      <Card className="traffic-overview">
        <CardHeader titleText="Traffic Overview">
          <ProgressIndicator value={undefined} />
        </CardHeader>
        <div className="card-content">
          <div className="loading-state">
            <Text>Loading traffic data...</Text>
          </div>
        </div>
      </Card>
    );
  }

  if (!metrics) {
    return (
      <Card className="traffic-overview">
        <CardHeader titleText="Traffic Overview">
          <Badge colorScheme="1">No Data</Badge>
        </CardHeader>
        <div className="card-content">
          <div className="empty-state">
            <Icon name="traffic-light" />
            <Text>No traffic data available</Text>
          </div>
        </div>
      </Card>
    );
  }

  const efficiency =
    metrics.totalVehicles > 0
      ? Math.round((metrics.runningVehicles / metrics.totalVehicles) * 100)
      : 0;

  const averageWaitingTime =
    metrics.totalVehicles > 0
      ? Math.round(metrics.totalWaitingTime / metrics.totalVehicles)
      : 0;

  const getEfficiencyColor = (efficiency: number) => {
    if (efficiency >= 80) return "var(--sapPositiveColor)";
    if (efficiency >= 60) return "var(--sapCriticalColor)";
    return "var(--sapNegativeColor)";
  };

  const getEfficiencyStatus = (efficiency: number) => {
    if (efficiency >= 80) return "Excellent";
    if (efficiency >= 60) return "Good";
    return "Poor";
  };

  return (
    <Card className="traffic-overview">
      <CardHeader titleText="Traffic Overview">
        <div className="header-actions">
          <Badge colorScheme="8">Live</Badge>
          <Text className="timestamp">
            {metrics.timestamp.toLocaleTimeString()}
          </Text>
        </div>
      </CardHeader>

      <div className="card-content">
        {/* Key Metrics */}
        <div className="metrics-grid">
          <motion.div
            className="metric-item"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3 }}
          >
            <div className="metric-icon">
              <Icon name="car" />
            </div>
            <div className="metric-content">
              <Text className="metric-value">{metrics.totalVehicles}</Text>
              <Text className="metric-label">Total Vehicles</Text>
            </div>
          </motion.div>

          <motion.div
            className="metric-item"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3, delay: 0.1 }}
          >
            <div className="metric-icon">
              <Icon name="accelerated" />
            </div>
            <div className="metric-content">
              <Text className="metric-value">{metrics.runningVehicles}</Text>
              <Text className="metric-label">Moving</Text>
            </div>
          </motion.div>

          <motion.div
            className="metric-item"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3, delay: 0.2 }}
          >
            <div className="metric-icon">
              <Icon name="stop" />
            </div>
            <div className="metric-content">
              <Text className="metric-value">{metrics.waitingVehicles}</Text>
              <Text className="metric-label">Waiting</Text>
            </div>
          </motion.div>

          <motion.div
            className="metric-item"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3, delay: 0.3 }}
          >
            <div className="metric-icon">
              <Icon name="speed" />
            </div>
            <div className="metric-content">
              <Text className="metric-value">
                {metrics.averageSpeed.toFixed(1)}
              </Text>
              <Text className="metric-label">Avg Speed (km/h)</Text>
            </div>
          </motion.div>
        </div>

        {/* Efficiency Indicator */}
        <div className="efficiency-section">
          <div className="efficiency-header">
            <Text>Traffic Efficiency</Text>
            <Text
              className="efficiency-status"
              style={{ color: getEfficiencyColor(efficiency) }}
            >
              {getEfficiencyStatus(efficiency)}
            </Text>
          </div>
          <div className="efficiency-bar">
            <div
              className="efficiency-fill"
              style={{
                width: `${efficiency}%`,
                backgroundColor: getEfficiencyColor(efficiency),
              }}
            />
          </div>
          <Text className="efficiency-percentage">{efficiency}%</Text>
        </div>

        {/* Additional Stats */}
        <div className="additional-stats">
          <div className="stat-item">
            <Icon name="time" />
            <div className="stat-content">
              <Text className="stat-value">{averageWaitingTime}s</Text>
              <Text className="stat-label">Avg Wait Time</Text>
            </div>
          </div>

          <div className="stat-item">
            <Icon name="trending-up" />
            <div className="stat-content">
              <Text className="stat-value">{metrics.throughput}</Text>
              <Text className="stat-label">Throughput</Text>
            </div>
          </div>

          <div className="stat-item">
            <Icon name="environment" />
            <div className="stat-content">
              <Text className="stat-value">
                {metrics.totalCo2Emission.toFixed(1)}
              </Text>
              <Text className="stat-label">CO2 (kg)</Text>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
}
