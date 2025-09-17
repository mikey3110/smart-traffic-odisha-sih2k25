import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Card,
  CardHeader,
  Text,
  Title,
  Icon,
  Badge,
  Button,
  Toolbar,
  ToolbarSpacer,
  ProgressIndicator,
  MessageStrip,
  MessageStripDesign,
} from "@ui5/webcomponents-react";
import { TrafficLight, TrafficLightStatus } from "@/types";
import { trafficService } from "@/services/trafficService";
import { useWebSocket } from "@/services/websocketService";
import "./TrafficLightsGrid.scss";

interface TrafficLightsGridProps {
  className?: string;
}

export function TrafficLightsGrid({ className }: TrafficLightsGridProps) {
  const [trafficLights, setTrafficLights] = useState<TrafficLight[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedLight, setSelectedLight] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");

  // WebSocket connection for real-time updates
  const { isConnected, subscribe } = useWebSocket({
    enabled: true,
    onMessage: (message) => {
      if (message.type === "traffic_lights_update") {
        setTrafficLights(message.payload);
      }
    },
  });

  // Fetch traffic lights data
  useEffect(() => {
    const fetchTrafficLights = async () => {
      try {
        setLoading(true);
        const lights = await trafficService.getTrafficLights();
        setTrafficLights(lights);
      } catch (error) {
        console.error("Failed to fetch traffic lights:", error);
        // Use mock data in case of error
        setTrafficLights(trafficService.getMockTrafficLights());
      } finally {
        setLoading(false);
      }
    };

    fetchTrafficLights();
  }, []);

  // Subscribe to real-time updates
  useEffect(() => {
    const unsubscribe = subscribe("realtime_data", (data) => {
      if (data.trafficLights) {
        setTrafficLights(data.trafficLights);
      }
    });

    return unsubscribe;
  }, [subscribe]);

  const getStatusIcon = (status: TrafficLightStatus) => {
    switch (status) {
      case "normal":
        return "traffic-light";
      case "maintenance":
        return "wrench";
      case "error":
        return "error";
      case "offline":
        return "disconnected";
      default:
        return "question-mark";
    }
  };

  const getStatusColor = (status: TrafficLightStatus) => {
    switch (status) {
      case "normal":
        return "var(--sapPositiveColor)";
      case "maintenance":
        return "var(--sapCriticalColor)";
      case "error":
        return "var(--sapNegativeColor)";
      case "offline":
        return "var(--sapNeutralColor)";
      default:
        return "var(--sapNeutralColor)";
    }
  };

  const getStatusBadge = (status: TrafficLightStatus) => {
    switch (status) {
      case "normal":
        return <Badge colorScheme="8">Normal</Badge>;
      case "maintenance":
        return <Badge colorScheme="2">Maintenance</Badge>;
      case "error":
        return <Badge colorScheme="1">Error</Badge>;
      case "offline":
        return <Badge colorScheme="9">Offline</Badge>;
      default:
        return <Badge colorScheme="9">Unknown</Badge>;
    }
  };

  const getPhaseColor = (phase: number) => {
    switch (phase) {
      case 0: // Red
        return "#dc3545";
      case 1: // Yellow
        return "#ffc107";
      case 2: // Green
        return "#28a745";
      default:
        return "#6c757d";
    }
  };

  const getPhaseName = (phase: number) => {
    switch (phase) {
      case 0:
        return "Red";
      case 1:
        return "Yellow";
      case 2:
        return "Green";
      default:
        return "Unknown";
    }
  };

  const handleLightClick = (lightId: string) => {
    setSelectedLight(selectedLight === lightId ? null : lightId);
  };

  const handleControlLight = async (
    lightId: string,
    phase: number,
    duration: number
  ) => {
    try {
      await trafficService.controlTrafficLight(lightId, phase, duration);
      // Refresh data after control
      const lights = await trafficService.getTrafficLights();
      setTrafficLights(lights);
    } catch (error) {
      console.error("Failed to control traffic light:", error);
    }
  };

  const getOverallStatus = () => {
    const normalCount = trafficLights.filter(
      (light) => light.status === "normal"
    ).length;
    const errorCount = trafficLights.filter(
      (light) => light.status === "error"
    ).length;
    const offlineCount = trafficLights.filter(
      (light) => light.status === "offline"
    ).length;

    if (errorCount > 0)
      return { status: "error", count: errorCount, label: "Errors" };
    if (offlineCount > 0)
      return { status: "warning", count: offlineCount, label: "Offline" };
    return { status: "success", count: normalCount, label: "Normal" };
  };

  const overallStatus = getOverallStatus();

  if (loading) {
    return (
      <Card className={`traffic-lights-grid ${className || ""}`}>
        <CardHeader titleText="Traffic Lights">
          <ProgressIndicator value={undefined} />
        </CardHeader>
        <div className="card-content">
          <div className="loading-state">
            <Text>Loading traffic lights...</Text>
          </div>
        </div>
      </Card>
    );
  }

  return (
    <Card className={`traffic-lights-grid ${className || ""}`}>
      <CardHeader titleText="Traffic Lights">
        <div className="header-actions">
          <div className="status-summary">
            <MessageStrip
              design={
                overallStatus.status === "success"
                  ? MessageStripDesign.Positive
                  : overallStatus.status === "warning"
                  ? MessageStripDesign.Warning
                  : MessageStripDesign.Negative
              }
            >
              {overallStatus.count} {overallStatus.label}
            </MessageStrip>
          </div>
          <div className="view-controls">
            <Button
              icon={viewMode === "grid" ? "grid" : "list"}
              design="Transparent"
              onClick={() => setViewMode(viewMode === "grid" ? "list" : "grid")}
            />
          </div>
        </div>
      </CardHeader>

      <div className="card-content">
        {trafficLights.length === 0 ? (
          <div className="empty-state">
            <Icon name="traffic-light" />
            <Text>No traffic lights found</Text>
            <Text>Check your connection or configuration</Text>
          </div>
        ) : (
          <div className={`lights-container ${viewMode}`}>
            <AnimatePresence>
              {trafficLights.map((light, index) => (
                <motion.div
                  key={light.id}
                  className={`traffic-light-card ${
                    selectedLight === light.id ? "selected" : ""
                  }`}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3, delay: index * 0.1 }}
                  onClick={() => handleLightClick(light.id)}
                >
                  <div className="light-header">
                    <div className="light-info">
                      <Icon
                        name={getStatusIcon(light.status)}
                        style={{ color: getStatusColor(light.status) }}
                      />
                      <div className="light-details">
                        <Text className="light-name">{light.name}</Text>
                        <Text className="light-location">
                          {light.location.lat.toFixed(4)},{" "}
                          {light.location.lng.toFixed(4)}
                        </Text>
                      </div>
                    </div>
                    {getStatusBadge(light.status)}
                  </div>

                  <div className="light-status">
                    <div className="phase-indicator">
                      <div
                        className="phase-circle"
                        style={{
                          backgroundColor: getPhaseColor(light.currentPhase),
                        }}
                      />
                      <Text className="phase-text">
                        {getPhaseName(light.currentPhase)}
                      </Text>
                    </div>
                    <div className="phase-timer">
                      <Text className="timer-text">{light.phaseDuration}s</Text>
                    </div>
                  </div>

                  <div className="light-metrics">
                    <div className="metric">
                      <Icon name="car" />
                      <Text>{light.vehicleCount}</Text>
                    </div>
                    <div className="metric">
                      <Icon name="time" />
                      <Text>{light.waitingTime.toFixed(1)}s</Text>
                    </div>
                  </div>

                  {selectedLight === light.id && (
                    <motion.div
                      className="light-controls"
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.3 }}
                    >
                      <div className="control-buttons">
                        <Button
                          design="Emphasized"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleControlLight(light.id, 0, 30);
                          }}
                        >
                          Red
                        </Button>
                        <Button
                          design="Emphasized"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleControlLight(light.id, 1, 5);
                          }}
                        >
                          Yellow
                        </Button>
                        <Button
                          design="Emphasized"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleControlLight(light.id, 2, 30);
                          }}
                        >
                          Green
                        </Button>
                      </div>
                    </motion.div>
                  )}
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>

      <div className="card-footer">
        <Toolbar>
          <div className="footer-left">
            <Text>Total: {trafficLights.length} lights</Text>
          </div>
          <ToolbarSpacer />
          <div className="footer-right">
            <Text>Last updated: {new Date().toLocaleTimeString()}</Text>
          </div>
        </Toolbar>
      </div>
    </Card>
  );
}
