import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ShellBar,
  ShellBarItem,
  Avatar,
  Badge,
  Popover,
  List,
  StandardListItem,
  Button,
  Icon,
  Switch,
  Toolbar,
  ToolbarSpacer,
  ToolbarSeparator,
} from "@ui5/webcomponents-react";
import { useApp } from "@/contexts/AppContext";
import { Notification, NotificationType } from "@/types";
import { Sidebar } from "./Sidebar";
import { NotificationPanel } from "./NotificationPanel";
import { SystemStatus } from "./SystemStatus";
import { useWebSocket } from "@/services/websocketService";
import "./DashboardLayout.scss";

interface DashboardLayoutProps {
  children: React.ReactNode;
}

export function DashboardLayout({ children }: DashboardLayoutProps) {
  const { user, notifications, addNotification, markNotificationAsRead } =
    useApp();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [showNotifications, setShowNotifications] = useState(false);
  const [showSystemStatus, setShowSystemStatus] = useState(false);
  const [unreadCount, setUnreadCount] = useState(0);

  // WebSocket connection for real-time updates
  const { isConnected, subscribe } = useWebSocket({
    enabled: true,
    onMessage: (message) => {
      if (message.type === "notification") {
        addNotification(message.payload);
      }
    },
  });

  // Update unread count
  useEffect(() => {
    const count = notifications.filter((n) => !n.read).length;
    setUnreadCount(count);
  }, [notifications]);

  // Subscribe to real-time data updates
  useEffect(() => {
    const unsubscribe = subscribe("realtime_data", (data) => {
      // Handle real-time data updates
      console.log("Real-time data received:", data);
    });

    return unsubscribe;
  }, [subscribe]);

  const handleNotificationClick = (notification: Notification) => {
    markNotificationAsRead(notification.id);
    if (notification.actionUrl) {
      // Navigate to action URL
      window.location.href = notification.actionUrl;
    }
  };

  const handleMarkAllAsRead = () => {
    notifications.forEach((notification) => {
      if (!notification.read) {
        markNotificationAsRead(notification.id);
      }
    });
  };

  const getNotificationIcon = (type: NotificationType) => {
    switch (type) {
      case "success":
        return "accept";
      case "error":
        return "error";
      case "warning":
        return "alert";
      case "alert":
        return "bell";
      default:
        return "information";
    }
  };

  const getNotificationColor = (type: NotificationType) => {
    switch (type) {
      case "success":
        return "var(--sapPositiveColor)";
      case "error":
        return "var(--sapNegativeColor)";
      case "warning":
        return "var(--sapCriticalColor)";
      case "alert":
        return "var(--sapHighlightColor)";
      default:
        return "var(--sapNeutralColor)";
    }
  };

  return (
    <div className="dashboard-layout">
      {/* Sidebar */}
      <motion.div
        className={`sidebar-container ${sidebarCollapsed ? "collapsed" : ""}`}
        initial={false}
        animate={{ width: sidebarCollapsed ? 60 : 280 }}
        transition={{ duration: 0.3, ease: "easeInOut" }}
      >
        <Sidebar
          collapsed={sidebarCollapsed}
          onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
        />
      </motion.div>

      {/* Main Content */}
      <div className="main-content">
        {/* Header */}
        <ShellBar
          className="dashboard-header"
          primaryTitle="Smart Traffic Management"
          secondaryTitle="Real-time Traffic Control Dashboard"
          logo={<Icon name="traffic-light" />}
          showNotifications
          showProductSwitch
          showCoPilot
          onLogoClick={() => setSidebarCollapsed(!sidebarCollapsed)}
        >
          {/* Notifications */}
          <ShellBarItem
            icon="bell"
            text="Notifications"
            onClick={() => setShowNotifications(!showNotifications)}
          >
            {unreadCount > 0 && (
              <Badge
                colorScheme="8"
                style={{ position: "absolute", top: "8px", right: "8px" }}
              >
                {unreadCount}
              </Badge>
            )}
          </ShellBarItem>

          {/* System Status */}
          <ShellBarItem
            icon="monitor"
            text="System Status"
            onClick={() => setShowSystemStatus(!showSystemStatus)}
          >
            <Badge
              colorScheme={isConnected ? "8" : "1"}
              style={{ position: "absolute", top: "8px", right: "8px" }}
            >
              {isConnected ? "ON" : "OFF"}
            </Badge>
          </ShellBarItem>

          {/* User Menu */}
          <ShellBarItem
            icon="customer"
            text={user?.name || "User"}
            onClick={() => {
              /* Handle user menu */
            }}
          >
            <Avatar initials={user?.name?.charAt(0) || "U"} size="S" />
          </ShellBarItem>
        </ShellBar>

        {/* Content Area */}
        <motion.div
          className="content-area"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          {children}
        </motion.div>

        {/* Footer */}
        <footer className="dashboard-footer">
          <Toolbar>
            <div className="footer-left">
              <span>© 2024 Smart Traffic Management System</span>
              <ToolbarSeparator />
              <span>Version 2.1.0</span>
            </div>
            <ToolbarSpacer />
            <div className="footer-right">
              <span
                className={`connection-status ${
                  isConnected ? "connected" : "disconnected"
                }`}
              >
                <Icon name={isConnected ? "connected" : "disconnected"} />
                {isConnected ? "Connected" : "Disconnected"}
              </span>
              <ToolbarSeparator />
              <span>Last updated: {new Date().toLocaleTimeString()}</span>
            </div>
          </Toolbar>
        </footer>
      </div>

      {/* Notification Panel */}
      <AnimatePresence>
        {showNotifications && (
          <motion.div
            className="notification-panel-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setShowNotifications(false)}
          >
            <motion.div
              className="notification-panel"
              initial={{ x: 400, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: 400, opacity: 0 }}
              transition={{ duration: 0.3 }}
              onClick={(e) => e.stopPropagation()}
            >
              <NotificationPanel
                notifications={notifications}
                onNotificationClick={handleNotificationClick}
                onMarkAllAsRead={handleMarkAllAsRead}
                onClose={() => setShowNotifications(false)}
              />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* System Status Panel */}
      <AnimatePresence>
        {showSystemStatus && (
          <motion.div
            className="system-status-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setShowSystemStatus(false)}
          >
            <motion.div
              className="system-status-panel"
              initial={{ x: 400, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: 400, opacity: 0 }}
              transition={{ duration: 0.3 }}
              onClick={(e) => e.stopPropagation()}
            >
              <SystemStatus
                isConnected={isConnected}
                onClose={() => setShowSystemStatus(false)}
              />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
