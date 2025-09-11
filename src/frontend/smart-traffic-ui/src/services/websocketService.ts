import { WebSocketMessage, RealtimeData, UseRealtimeOptions } from '@/types';

type WebSocketStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

class WebSocketService {
  private ws: WebSocket | null = null;
  private status: WebSocketStatus = 'disconnected';
  private reconnectInterval: number = 5000;
  private maxReconnectAttempts: number = 10;
  private reconnectAttempts: number = 0;
  private messageHandlers: Map<string, ((data: any) => void)[]> = new Map();
  private statusHandlers: ((status: WebSocketStatus) => void)[] = [];
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private url: string;

  constructor() {
    this.url = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws';
  }

  // Connect to WebSocket
  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }

      this.status = 'connecting';
      this.notifyStatusHandlers();

      try {
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
          this.status = 'connected';
          this.reconnectAttempts = 0;
          this.notifyStatusHandlers();
          this.startHeartbeat();
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data);
            this.handleMessage(message);
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
          }
        };

        this.ws.onclose = (event) => {
          this.status = 'disconnected';
          this.notifyStatusHandlers();
          this.stopHeartbeat();

          if (!event.wasClean && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect();
          }
        };

        this.ws.onerror = (error) => {
          this.status = 'error';
          this.notifyStatusHandlers();
          console.error('WebSocket error:', error);
          reject(error);
        };

      } catch (error) {
        this.status = 'error';
        this.notifyStatusHandlers();
        reject(error);
      }
    });
  }

  // Disconnect from WebSocket
  disconnect(): void {
    this.stopHeartbeat();
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    this.status = 'disconnected';
    this.notifyStatusHandlers();
  }

  // Send message
  send(message: WebSocketMessage): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket is not connected. Message not sent:', message);
    }
  }

  // Subscribe to message type
  subscribe(type: string, handler: (data: any) => void): () => void {
    if (!this.messageHandlers.has(type)) {
      this.messageHandlers.set(type, []);
    }
    this.messageHandlers.get(type)!.push(handler);

    // Return unsubscribe function
    return () => {
      const handlers = this.messageHandlers.get(type);
      if (handlers) {
        const index = handlers.indexOf(handler);
        if (index > -1) {
          handlers.splice(index, 1);
        }
      }
    };
  }

  // Subscribe to status changes
  onStatusChange(handler: (status: WebSocketStatus) => void): () => void {
    this.statusHandlers.push(handler);
    return () => {
      const index = this.statusHandlers.indexOf(handler);
      if (index > -1) {
        this.statusHandlers.splice(index, 1);
      }
    };
  }

  // Get current status
  getStatus(): WebSocketStatus {
    return this.status;
  }

  // Check if connected
  isConnected(): boolean {
    return this.status === 'connected' && this.ws?.readyState === WebSocket.OPEN;
  }

  // Private methods
  private handleMessage(message: WebSocketMessage): void {
    const handlers = this.messageHandlers.get(message.type);
    if (handlers) {
      handlers.forEach(handler => {
        try {
          handler(message.payload);
        } catch (error) {
          console.error('Error in message handler:', error);
        }
      });
    }
  }

  private notifyStatusHandlers(): void {
    this.statusHandlers.forEach(handler => {
      try {
        handler(this.status);
      } catch (error) {
        console.error('Error in status handler:', error);
      }
    });
  }

  private scheduleReconnect(): void {
    setTimeout(() => {
      this.reconnectAttempts++;
      console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
      this.connect().catch(() => {
        // Reconnect failed, will try again
      });
    }, this.reconnectInterval);
  }

  private startHeartbeat(): void {
    this.heartbeatInterval = setInterval(() => {
      if (this.isConnected()) {
        this.send({
          type: 'ping',
          payload: { timestamp: new Date().toISOString() },
          timestamp: new Date()
        });
      }
    }, 30000); // Send ping every 30 seconds
  }

  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }
}

// Singleton instance
export const websocketService = new WebSocketService();

// React hook for WebSocket
export function useWebSocket(options: UseRealtimeOptions = {}) {
  const { enabled = true, reconnectInterval = 5000, onMessage, onError } = options;

  React.useEffect(() => {
    if (!enabled) return;

    const connect = async () => {
      try {
        await websocketService.connect();
      } catch (error) {
        if (onError) {
          onError(error as Event);
        }
      }
    };

    connect();

    return () => {
      websocketService.disconnect();
    };
  }, [enabled, onError]);

  React.useEffect(() => {
    if (onMessage) {
      const unsubscribe = websocketService.subscribe('realtime_data', onMessage);
      return unsubscribe;
    }
  }, [onMessage]);

  return {
    status: websocketService.getStatus(),
    isConnected: websocketService.isConnected(),
    send: websocketService.send.bind(websocketService),
    subscribe: websocketService.subscribe.bind(websocketService)
  };
}

// Import React for the hook
import React from 'react';
