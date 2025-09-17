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
import { PerformanceMetrics as PerformanceMetricsType } from "@/types";
import "./PerformanceMetrics.scss";

interface PerformanceMetricsProps {
  metrics: PerformanceMetricsType | null;
  loading: boolean;
}

export function PerformanceMetrics({
  metrics,
  loading,
}: PerformanceMetricsProps) {
  if (loading) {
    return (
      <Card className="performance-metrics">
        <CardHeader titleText="Performance Metrics">
          <ProgressIndicator value={undefined} />
        </CardHeader>
        <div className="card-content">
          <div className="loading-state">
            <Text>Loading performance data...</Text>
          </div>
        </div>
      </Card>
    );
  }

  if (!metrics) {
    return (
      <Card className="performance-metrics">
        <CardHeader titleText="Performance Metrics">
          <Badge colorScheme="1">No Data</Badge>
        </CardHeader>
        <div className="card-content">
          <div className="empty-state">
            <Icon name="chart-axis" />
            <Text>No performance data available</Text>
          </div>
        </div>
      </Card>
    );
  }

  const getEfficiencyScore = () => {
    const totalVehicles = metrics.totalVehicles;
    const runningVehicles = metrics.runningVehicles;
    const waitingVehicles = metrics.waitingVehicles;

    if (totalVehicles === 0) return 0;

    // Calculate efficiency based on running vs waiting vehicles
    const efficiency = (runningVehicles / totalVehicles) * 100;
    return Math.round(efficiency);
  };

  const getEfficiencyColor = (score: number) => {
    if (score >= 80) return "var(--sapPositiveColor)";
    if (score >= 60) return "var(--sapCriticalColor)";
    return "var(--sapNegativeColor)";
  };

  const getEfficiencyStatus = (score: number) => {
    if (score >= 80) return "Excellent";
    if (score >= 60) return "Good";
    return "Needs Improvement";
  };

  const efficiencyScore = getEfficiencyScore();

  return (
    <Card className="performance-metrics">
      <CardHeader titleText="Performance Metrics">
        <div className="header-actions">
          <Badge colorScheme="8">Live</Badge>
          <Text className="timestamp">
            {metrics.timestamp.toLocaleTimeString()}
          </Text>
        </div>
      </CardHeader>

      <div className="card-content">
        {/* Efficiency Score */}
        <div className="efficiency-section">
          <div className="efficiency-header">
            <Text>Traffic Efficiency</Text>
            <Text
              className="efficiency-score"
              style={{ color: getEfficiencyColor(efficiencyScore) }}
            >
              {efficiencyScore}%
            </Text>
          </div>
          <div className="efficiency-bar">
            <div
              className="efficiency-fill"
              style={{
                width: `${efficiencyScore}%`,
                backgroundColor: getEfficiencyColor(efficiencyScore),
              }}
            />
          </div>
          <Text className="efficiency-status">
            {getEfficiencyStatus(efficiencyScore)}
          </Text>
        </div>

        {/* Key Performance Indicators */}
        <div className="kpi-grid">
          <motion.div
            className="kpi-item"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3 }}
          >
            <div className="kpi-icon">
              <Icon name="time" />
            </div>
            <div className="kpi-content">
              <Text className="kpi-value">
                {metrics.totalWaitingTime.toFixed(1)}s
              </Text>
              <Text className="kpi-label">Total Wait Time</Text>
            </div>
          </motion.div>

          <motion.div
            className="kpi-item"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3, delay: 0.1 }}
          >
            <div className="kpi-icon">
              <Icon name="trending-up" />
            </div>
            <div className="kpi-content">
              <Text className="kpi-value">{metrics.throughput}</Text>
              <Text className="kpi-label">Throughput</Text>
            </div>
          </motion.div>

          <motion.div
            className="kpi-item"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3, delay: 0.2 }}
          >
            <div className="kpi-icon">
              <Icon name="environment" />
            </div>
            <div className="kpi-content">
              <Text className="kpi-value">
                {metrics.totalCo2Emission.toFixed(1)}kg
              </Text>
              <Text className="kpi-label">CO2 Emissions</Text>
            </div>
          </motion.div>

          <motion.div
            className="kpi-item"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3, delay: 0.3 }}
          >
            <div className="kpi-icon">
              <Icon name="fuel" />
            </div>
            <div className="kpi-content">
              <Text className="kpi-value">
                {metrics.totalFuelConsumption.toFixed(1)}L
              </Text>
              <Text className="kpi-label">Fuel Consumption</Text>
            </div>
          </motion.div>
        </div>

        {/* Performance Trends */}
        <div className="trends-section">
          <Text className="section-title">Performance Trends</Text>
          <div className="trend-items">
            <div className="trend-item">
              <Icon name="arrow-up" />
              <Text>Average Speed: {metrics.averageSpeed.toFixed(1)} km/h</Text>
            </div>
            <div className="trend-item">
              <Icon name="arrow-down" />
              <Text>Waiting Vehicles: {metrics.waitingVehicles}</Text>
            </div>
            <div className="trend-item">
              <Icon name="arrow-up" />
              <Text>Running Vehicles: {metrics.runningVehicles}</Text>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
}
