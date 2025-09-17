import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  Card,
  CardHeader,
  Text,
  Title,
  Icon,
  Badge,
  List,
  StandardListItem,
  Button,
  MessageStrip,
  MessageStripDesign,
  ProgressIndicator,
} from "@ui5/webcomponents-react";
import { Notification, NotificationType, NotificationPriority } from "@/types";
import "./AlertsPanel.scss";

interface AlertsPanelProps {
  className?: string;
}

export function AlertsPanel({ className }: AlertsPanelProps) {
  const [alerts, setAlerts] = useState<Notification[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Simulate loading alerts
    const loadAlerts = async () => {
      setLoading(true);

      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 1000));

      // Mock alerts data
      const mockAlerts: Notification[] = [
        {
          id: "1",
          type: NotificationType.WARNING,
          title: "High Traffic Volume",
          message:
            "Traffic volume at Main Street intersection is 40% above normal",
          timestamp: new Date(Date.now() - 5 * 60 * 1000), // 5 minutes ago
          read: false,
          priority: NotificationPriority.HIGH,
        },
        {
          id: "2",
          type: NotificationType.INFO,
          title: "System Update",
          message: "Traffic light optimization algorithm updated successfully",
          timestamp: new Date(Date.now() - 15 * 60 * 1000), // 15 minutes ago
          read: true,
          priority: NotificationPriority.MEDIUM,
        },
        {
          id: "3",
          type: NotificationType.ERROR,
          title: "Connection Lost",
          message: "Lost connection to intersection camera at Oak Avenue",
          timestamp: new Date(Date.now() - 30 * 60 * 1000), // 30 minutes ago
          read: false,
          priority: NotificationPriority.CRITICAL,
        },
        {
          id: "4",
          type: NotificationType.SUCCESS,
          title: "Optimization Complete",
          message: "Traffic flow optimization completed for downtown area",
          timestamp: new Date(Date.now() - 45 * 60 * 1000), // 45 minutes ago
          read: true,
          priority: NotificationPriority.LOW,
        },
      ];

      setAlerts(mockAlerts);
      setLoading(false);
    };

    loadAlerts();
  }, []);

  const getAlertIcon = (type: NotificationType) => {
    switch (type) {
      case NotificationType.ERROR:
        return "error";
      case NotificationType.WARNING:
        return "alert";
      case NotificationType.SUCCESS:
        return "accept";
      case NotificationType.INFO:
        return "information";
      default:
        return "bell";
    }
  };

  const getAlertColor = (type: NotificationType) => {
    switch (type) {
      case NotificationType.ERROR:
        return "var(--sapNegativeColor)";
      case NotificationType.WARNING:
        return "var(--sapCriticalColor)";
      case NotificationType.SUCCESS:
        return "var(--sapPositiveColor)";
      case NotificationType.INFO:
        return "var(--sapNeutralColor)";
      default:
        return "var(--sapNeutralColor)";
    }
  };

  const getPriorityBadge = (priority: NotificationPriority) => {
    switch (priority) {
      case NotificationPriority.CRITICAL:
        return <Badge colorScheme="1">Critical</Badge>;
      case NotificationPriority.HIGH:
        return <Badge colorScheme="2">High</Badge>;
      case NotificationPriority.MEDIUM:
        return <Badge colorScheme="3">Medium</Badge>;
      case NotificationPriority.LOW:
        return <Badge colorScheme="8">Low</Badge>;
      default:
        return <Badge colorScheme="9">Unknown</Badge>;
    }
  };

  const getMessageStripDesign = (type: NotificationType) => {
    switch (type) {
      case NotificationType.ERROR:
        return MessageStripDesign.Negative;
      case NotificationType.WARNING:
        return MessageStripDesign.Warning;
      case NotificationType.SUCCESS:
        return MessageStripDesign.Positive;
      case NotificationType.INFO:
        return MessageStripDesign.Information;
      default:
        return MessageStripDesign.Information;
    }
  };

  const formatTimestamp = (timestamp: Date) => {
    const now = new Date();
    const diff = now.getTime() - timestamp.getTime();
    const minutes = Math.floor(diff / (1000 * 60));
    const hours = Math.floor(diff / (1000 * 60 * 60));

    if (minutes < 1) return "Just now";
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    return timestamp.toLocaleDateString();
  };

  const unreadCount = alerts.filter((alert) => !alert.read).length;
  const criticalCount = alerts.filter(
    (alert) => alert.priority === NotificationPriority.CRITICAL && !alert.read
  ).length;

  const handleMarkAsRead = (alertId: string) => {
    setAlerts((prev) =>
      prev.map((alert) =>
        alert.id === alertId ? { ...alert, read: true } : alert
      )
    );
  };

  const handleMarkAllAsRead = () => {
    setAlerts((prev) => prev.map((alert) => ({ ...alert, read: true })));
  };

  if (loading) {
    return (
      <Card className={`alerts-panel ${className || ""}`}>
        <CardHeader titleText="System Alerts">
          <ProgressIndicator value={undefined} />
        </CardHeader>
        <div className="card-content">
          <div className="loading-state">
            <Text>Loading alerts...</Text>
          </div>
        </div>
      </Card>
    );
  }

  return (
    <Card className={`alerts-panel ${className || ""}`}>
      <CardHeader titleText="System Alerts">
        <div className="header-actions">
          {unreadCount > 0 && (
            <Badge colorScheme="1">{unreadCount} Unread</Badge>
          )}
          {criticalCount > 0 && (
            <Badge colorScheme="1">{criticalCount} Critical</Badge>
          )}
          <Button
            design="Transparent"
            icon="refresh"
            onClick={() => window.location.reload()}
          />
        </div>
      </CardHeader>

      <div className="card-content">
        {alerts.length === 0 ? (
          <div className="empty-state">
            <Icon name="bell" />
            <Text>No alerts at this time</Text>
          </div>
        ) : (
          <>
            {/* Critical Alerts */}
            {criticalCount > 0 && (
              <div className="critical-alerts">
                <Text className="section-title">Critical Alerts</Text>
                {alerts
                  .filter(
                    (alert) =>
                      alert.priority === NotificationPriority.CRITICAL &&
                      !alert.read
                  )
                  .map((alert, index) => (
                    <motion.div
                      key={alert.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.3, delay: index * 0.1 }}
                    >
                      <MessageStrip
                        design={getMessageStripDesign(alert.type)}
                        className="alert-item critical"
                      >
                        <div className="alert-content">
                          <div className="alert-header">
                            <Icon name={getAlertIcon(alert.type)} />
                            <Text className="alert-title">{alert.title}</Text>
                            <Text className="alert-time">
                              {formatTimestamp(alert.timestamp)}
                            </Text>
                          </div>
                          <Text className="alert-message">{alert.message}</Text>
                          <div className="alert-actions">
                            <Button
                              design="Transparent"
                              onClick={() => handleMarkAsRead(alert.id)}
                            >
                              Mark as Read
                            </Button>
                          </div>
                        </div>
                      </MessageStrip>
                    </motion.div>
                  ))}
              </div>
            )}

            {/* All Alerts */}
            <div className="all-alerts">
              <div className="alerts-header">
                <Text className="section-title">All Alerts</Text>
                {unreadCount > 0 && (
                  <Button design="Transparent" onClick={handleMarkAllAsRead}>
                    Mark All as Read
                  </Button>
                )}
              </div>

              <List>
                {alerts.map((alert, index) => (
                  <motion.div
                    key={alert.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.05 }}
                  >
                    <StandardListItem
                      icon={getAlertIcon(alert.type)}
                      title={alert.title}
                      description={alert.message}
                      className={`alert-item ${alert.read ? "read" : "unread"}`}
                      style={
                        {
                          "--alert-color": getAlertColor(alert.type),
                        } as React.CSSProperties
                      }
                    >
                      <div className="alert-meta">
                        {getPriorityBadge(alert.priority)}
                        <div className="alert-time">
                          {formatTimestamp(alert.timestamp)}
                        </div>
                        {!alert.read && (
                          <Button
                            design="Transparent"
                            onClick={() => handleMarkAsRead(alert.id)}
                          >
                            Mark as Read
                          </Button>
                        )}
                      </div>
                    </StandardListItem>
                  </motion.div>
                ))}
              </List>
            </div>
          </>
        )}
      </div>
    </Card>
  );
}
