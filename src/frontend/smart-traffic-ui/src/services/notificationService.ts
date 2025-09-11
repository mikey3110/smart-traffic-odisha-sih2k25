import { ApiService } from './apiService';
import { Notification, NotificationType, NotificationPriority } from '@/types';

class NotificationService {
  // Get all notifications
  async getNotifications(): Promise<Notification[]> {
    const response = await ApiService.get<Notification[]>('/notifications');
    return response;
  }

  // Get unread notifications
  async getUnreadNotifications(): Promise<Notification[]> {
    const response = await ApiService.get<Notification[]>('/notifications/unread');
    return response;
  }

  // Mark notification as read
  async markAsRead(id: string): Promise<void> {
    await ApiService.patch(`/notifications/${id}/read`, {});
  }

  // Mark all notifications as read
  async markAllAsRead(): Promise<void> {
    await ApiService.patch('/notifications/read-all', {});
  }

  // Delete notification
  async deleteNotification(id: string): Promise<void> {
    await ApiService.delete(`/notifications/${id}`);
  }

  // Create notification
  async createNotification(notification: Omit<Notification, 'id' | 'timestamp'>): Promise<Notification> {
    const response = await ApiService.post<Notification>('/notifications', notification);
    return response;
  }

  // Update notification preferences
  async updatePreferences(preferences: {
    email: boolean;
    push: boolean;
    types: NotificationType[];
  }): Promise<void> {
    await ApiService.put('/notifications/preferences', preferences);
  }

  // Mock notifications for development
  getMockNotifications(): Notification[] {
    return [
      {
        id: '1',
        type: NotificationType.INFO,
        title: 'System Update',
        message: 'Traffic management system has been updated to version 2.1.0',
        timestamp: new Date(Date.now() - 1000 * 60 * 30), // 30 minutes ago
        read: false,
        priority: NotificationPriority.MEDIUM
      },
      {
        id: '2',
        type: NotificationType.WARNING,
        title: 'High Traffic Alert',
        message: 'Traffic congestion detected at Main Street intersection',
        timestamp: new Date(Date.now() - 1000 * 60 * 15), // 15 minutes ago
        read: false,
        priority: NotificationPriority.HIGH
      },
      {
        id: '3',
        type: NotificationType.SUCCESS,
        title: 'Optimization Complete',
        message: 'Traffic light optimization completed successfully',
        timestamp: new Date(Date.now() - 1000 * 60 * 5), // 5 minutes ago
        read: true,
        priority: NotificationPriority.LOW
      },
      {
        id: '4',
        type: NotificationType.ERROR,
        title: 'Connection Lost',
        message: 'Lost connection to traffic light #3. Attempting to reconnect...',
        timestamp: new Date(Date.now() - 1000 * 60 * 2), // 2 minutes ago
        read: false,
        priority: NotificationPriority.CRITICAL
      },
      {
        id: '5',
        type: NotificationType.ALERT,
        title: 'Emergency Vehicle',
        message: 'Emergency vehicle approaching Main Street. Priority route activated.',
        timestamp: new Date(Date.now() - 1000 * 30), // 30 seconds ago
        read: false,
        priority: NotificationPriority.CRITICAL,
        actionUrl: '/traffic/lights/1'
      }
    ];
  }

  // Create mock notification
  createMockNotification(
    type: NotificationType,
    title: string,
    message: string,
    priority: NotificationPriority = NotificationPriority.MEDIUM
  ): Notification {
    return {
      id: Date.now().toString(),
      type,
      title,
      message,
      timestamp: new Date(),
      read: false,
      priority
    };
  }
}

export const notificationService = new NotificationService();
