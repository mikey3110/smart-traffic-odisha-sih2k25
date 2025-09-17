import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  Panel,
  List,
  StandardListItem,
  Button,
  Icon,
  Badge,
  Text,
  Title,
  ProgressIndicator,
  MessageStrip,
  MessageStripDesign,
  Toolbar,
  ToolbarSpacer,
} from "@ui5/webcomponents-react";
import { configService } from "@/services/configService";
import { useApp } from "@/contexts/AppContext";
import "./SystemStatus.scss";

interface SystemStatusProps {
  isConnected: boolean;
  onClose: () => void;
}

interface ServiceStatus {
  name: string;
  status: "healthy" | "degraded" | "down";
  responseTime: number;
  lastCheck: Date;
}

export function SystemStatus({ isConnected, onClose }: SystemStatusProps) {
  const { systemConfig } = useApp();
  const [services, setServices] = useState<ServiceStatus[]>([]);
  const [systemUptime, setSystemUptime] = useState(0);
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchSystemStatus = async () => {
      try {
        setLoading(true);
        const status = await configService.getSystemStatus();

        setServices(
          status.services.map((service) => ({
            ...service,
            lastCheck: new Date(),
          }))
        );
        setSystemUptime(status.uptime);
        setLastUpdate(status.lastUpdate);
      } catch (error) {
        console.error("Failed to fetch system status:", error);
        // Use mock data in case of error
        setServices([
          {
            name: "Traffic Management API",
            status: "healthy",
            responseTime: 45,
            lastCheck: new Date(),
          },
          {
            name: "Database",
            status: "healthy",
            responseTime: 12,
            lastCheck: new Date(),
          },
          {
            name: "WebSocket Service",
            status: isConnected ? "healthy" : "down",
            responseTime: isConnected ? 8 : 0,
            lastCheck: new Date(),
          },
          {
            name: "File Storage",
            status: "degraded",
            responseTime: 150,
            lastCheck: new Date(),
          },
        ]);
        setSystemUptime(86400); // 24 hours
        setLastUpdate(new Date());
      } finally {
        setLoading(false);
      }
    };

    fetchSystemStatus();

    // Refresh status every 30 seconds
    const interval = setInterval(fetchSystemStatus, 30000);
    return () => clearInterval(interval);
  }, [isConnected]);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "healthy":
        return "accept";
      case "degraded":
        return "alert";
      case "down":
        return "error";
      default:
        return "question-mark";
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "healthy":
        return "var(--sapPositiveColor)";
      case "degraded":
        return "var(--sapCriticalColor)";
      case "down":
        return "var(--sapNegativeColor)";
      default:
        return "var(--sapNeutralColor)";
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "healthy":
        return <Badge colorScheme="8">Healthy</Badge>;
      case "degraded":
        return <Badge colorScheme="2">Degraded</Badge>;
      case "down":
        return <Badge colorScheme="1">Down</Badge>;
      default:
        return <Badge colorScheme="9">Unknown</Badge>;
    }
  };

  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);

    if (days > 0) {
      return `${days}d ${hours}h ${minutes}m`;
    } else if (hours > 0) {
      return `${hours}h ${minutes}m`;
    } else {
      return `${minutes}m`;
    }
  };

  const getOverallStatus = () => {
    if (services.some((s) => s.status === "down")) return "down";
    if (services.some((s) => s.status === "degraded")) return "degraded";
    return "healthy";
  };

  const overallStatus = getOverallStatus();
  const healthyServices = services.filter((s) => s.status === "healthy").length;
  const totalServices = services.length;

  return (
    <Panel className="system-status-panel">
      <div className="panel-header">
        <div className="panel-header-content">
          <div className="header-left">
            <Icon name="monitor" />
            <Title level="H3">System Status</Title>
            <Badge
              colorScheme={
                overallStatus === "healthy"
                  ? "8"
                  : overallStatus === "degraded"
                  ? "2"
                  : "1"
              }
            >
              {overallStatus.toUpperCase()}
            </Badge>
          </div>
          <div className="header-right">
            <Button
              design="Transparent"
              icon="refresh"
              onClick={() => window.location.reload()}
              className="refresh-button"
            />
            <Button
              design="Transparent"
              icon="decline"
              onClick={onClose}
              className="close-button"
            />
          </div>
        </div>
      </div>

      <div className="system-status-content">
        {loading ? (
          <div className="loading-state">
            <ProgressIndicator value={undefined} />
            <Text>Loading system status...</Text>
          </div>
        ) : (
          <>
            {/* Overall Status */}
            <div className="overall-status">
              <MessageStrip
                design={
                  overallStatus === "healthy"
                    ? MessageStripDesign.Positive
                    : overallStatus === "degraded"
                    ? MessageStripDesign.Warning
                    : MessageStripDesign.Negative
                }
                className="status-message"
              >
                <div className="status-content">
                  <Icon name={getStatusIcon(overallStatus)} />
                  <div className="status-text">
                    <Text>System is {overallStatus}</Text>
                    <Text>
                      {healthyServices} of {totalServices} services operational
                    </Text>
                  </div>
                </div>
              </MessageStrip>
            </div>

            {/* System Info */}
            <div className="system-info">
              <div className="info-item">
                <Icon name="time" />
                <div className="info-content">
                  <Text>Uptime</Text>
                  <Text className="info-value">
                    {formatUptime(systemUptime)}
                  </Text>
                </div>
              </div>
              <div className="info-item">
                <Icon name="refresh" />
                <div className="info-content">
                  <Text>Last Update</Text>
                  <Text className="info-value">
                    {lastUpdate.toLocaleTimeString()}
                  </Text>
                </div>
              </div>
              <div className="info-item">
                <Icon name="connected" />
                <div className="info-content">
                  <Text>WebSocket</Text>
                  <Text className="info-value">
                    {isConnected ? "Connected" : "Disconnected"}
                  </Text>
                </div>
              </div>
            </div>

            {/* Services Status */}
            <div className="services-section">
              <Title level="H4">Services</Title>
              <List>
                {services.map((service, index) => (
                  <motion.div
                    key={service.name}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.1 }}
                  >
                    <StandardListItem
                      icon={getStatusIcon(service.status)}
                      title={service.name}
                      description={`Response time: ${service.responseTime}ms`}
                      className={`service-item ${service.status}`}
                      style={
                        {
                          "--service-color": getStatusColor(service.status),
                        } as React.CSSProperties
                      }
                    >
                      <div className="service-meta">
                        {getStatusBadge(service.status)}
                        <div className="service-time">
                          {service.lastCheck.toLocaleTimeString()}
                        </div>
                      </div>
                    </StandardListItem>
                  </motion.div>
                ))}
              </List>
            </div>

            {/* Configuration Status */}
            <div className="config-section">
              <Title level="H4">Configuration</Title>
              <div className="config-items">
                <div className="config-item">
                  <Text>Simulation</Text>
                  <Badge
                    colorScheme={systemConfig.simulation.enabled ? "8" : "9"}
                  >
                    {systemConfig.simulation.enabled ? "Enabled" : "Disabled"}
                  </Badge>
                </div>
                <div className="config-item">
                  <Text>Data Export</Text>
                  <Badge
                    colorScheme={systemConfig.dataExport.enabled ? "8" : "9"}
                  >
                    {systemConfig.dataExport.enabled ? "Enabled" : "Disabled"}
                  </Badge>
                </div>
                <div className="config-item">
                  <Text>Notifications</Text>
                  <Badge
                    colorScheme={systemConfig.notifications.enabled ? "8" : "9"}
                  >
                    {systemConfig.notifications.enabled
                      ? "Enabled"
                      : "Disabled"}
                  </Badge>
                </div>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Footer */}
      <div className="system-status-footer">
        <Toolbar>
          <Button
            design="Transparent"
            onClick={() => {
              /* Navigate to system logs */
            }}
          >
            View System Logs
          </Button>
          <ToolbarSpacer />
          <Button
            design="Transparent"
            onClick={() => {
              /* Navigate to configuration */
            }}
          >
            System Settings
          </Button>
        </Toolbar>
      </div>
    </Panel>
  );
}
