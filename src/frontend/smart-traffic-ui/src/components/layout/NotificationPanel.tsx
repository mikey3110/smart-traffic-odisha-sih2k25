import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Panel,
  List,
  StandardListItem,
  Button,
  Icon,
  Badge,
  Toolbar,
  ToolbarSpacer,
  Text,
  Title,
  MessageStrip,
  MessageStripDesign,
} from "@ui5/webcomponents-react";
import { Notification, NotificationType, NotificationPriority } from "@/types";
import { formatDistanceToNow } from "date-fns";
import "./NotificationPanel.scss";

interface NotificationPanelProps {
  notifications: Notification[];
  onNotificationClick: (notification: Notification) => void;
  onMarkAllAsRead: () => void;
  onClose: () => void;
}

export function NotificationPanel({
  notifications,
  onNotificationClick,
  onMarkAllAsRead,
  onClose,
}: NotificationPanelProps) {
  const unreadNotifications = notifications.filter((n) => !n.read);
  const readNotifications = notifications.filter((n) => n.read);

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

  const getPriorityBadge = (priority: NotificationPriority) => {
    switch (priority) {
      case "critical":
        return <Badge colorScheme="1">Critical</Badge>;
      case "high":
        return <Badge colorScheme="2">High</Badge>;
      case "medium":
        return <Badge colorScheme="8">Medium</Badge>;
      case "low":
        return <Badge colorScheme="9">Low</Badge>;
      default:
        return null;
    }
  };

  const formatNotificationTime = (timestamp: Date) => {
    return formatDistanceToNow(timestamp, { addSuffix: true });
  };

  return (
    <Panel className="notification-panel">
      <div className="panel-header">
        <div className="panel-header-content">
          <div className="header-left">
            <Icon name="bell" />
            <Title level="H3">Notifications</Title>
            {unreadNotifications.length > 0 && (
              <Badge colorScheme="8">{unreadNotifications.length}</Badge>
            )}
          </div>
          <div className="header-right">
            {unreadNotifications.length > 0 && (
              <Button
                design="Transparent"
                onClick={onMarkAllAsRead}
                className="mark-all-read"
              >
                Mark all as read
              </Button>
            )}
            <Button
              design="Transparent"
              icon="decline"
              onClick={onClose}
              className="close-button"
            />
          </div>
        </div>
      </div>

      <div className="notification-content">
        {notifications.length === 0 ? (
          <div className="empty-state">
            <Icon name="bell" />
            <Text>No notifications</Text>
            <Text>You're all caught up!</Text>
          </div>
        ) : (
          <>
            {/* Unread Notifications */}
            {unreadNotifications.length > 0 && (
              <div className="notification-section">
                <div className="section-header">
                  <Text>Unread ({unreadNotifications.length})</Text>
                </div>
                <List>
                  <AnimatePresence>
                    {unreadNotifications.map((notification) => (
                      <motion.div
                        key={notification.id}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -20 }}
                        transition={{ duration: 0.2 }}
                      >
                        <StandardListItem
                          icon={getNotificationIcon(notification.type)}
                          title={notification.title}
                          description={notification.message}
                          onClick={() => onNotificationClick(notification)}
                          className={`notification-item unread ${notification.priority}`}
                          style={
                            {
                              "--notification-color": getNotificationColor(
                                notification.type
                              ),
                            } as React.CSSProperties
                          }
                        >
                          <div className="notification-meta">
                            <div className="notification-badges">
                              {getPriorityBadge(notification.priority)}
                            </div>
                            <div className="notification-time">
                              {formatNotificationTime(notification.timestamp)}
                            </div>
                          </div>
                        </StandardListItem>
                      </motion.div>
                    ))}
                  </AnimatePresence>
                </List>
              </div>
            )}

            {/* Read Notifications */}
            {readNotifications.length > 0 && (
              <div className="notification-section">
                <div className="section-header">
                  <Text>Earlier ({readNotifications.length})</Text>
                </div>
                <List>
                  <AnimatePresence>
                    {readNotifications.slice(0, 10).map((notification) => (
                      <motion.div
                        key={notification.id}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -20 }}
                        transition={{ duration: 0.2 }}
                      >
                        <StandardListItem
                          icon={getNotificationIcon(notification.type)}
                          title={notification.title}
                          description={notification.message}
                          onClick={() => onNotificationClick(notification)}
                          className={`notification-item read ${notification.priority}`}
                          style={
                            {
                              "--notification-color": getNotificationColor(
                                notification.type
                              ),
                            } as React.CSSProperties
                          }
                        >
                          <div className="notification-meta">
                            <div className="notification-badges">
                              {getPriorityBadge(notification.priority)}
                            </div>
                            <div className="notification-time">
                              {formatNotificationTime(notification.timestamp)}
                            </div>
                          </div>
                        </StandardListItem>
                      </motion.div>
                    ))}
                  </AnimatePresence>
                </List>
              </div>
            )}
          </>
        )}
      </div>

      {/* Footer */}
      <div className="notification-footer">
        <Button
          design="Transparent"
          onClick={() => {
            /* Navigate to all notifications */
          }}
        >
          View all notifications
        </Button>
      </div>
    </Panel>
  );
}
