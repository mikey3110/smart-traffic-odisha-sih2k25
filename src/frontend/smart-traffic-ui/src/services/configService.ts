import { ApiService } from "./apiService";
import { SystemConfig } from "@/types";

class ConfigService {
  // Get system configuration
  async getSystemConfig(): Promise<SystemConfig> {
    const response = await ApiService.get<SystemConfig>("/config/system");
    return response;
  }

  // Update system configuration
  async updateSystemConfig(
    config: Partial<SystemConfig>
  ): Promise<SystemConfig> {
    const response = await ApiService.put<SystemConfig>(
      "/config/system",
      config
    );
    return response;
  }

  // Get user preferences
  async getUserPreferences(): Promise<{
    theme: string;
    language: string;
    timezone: string;
    notifications: {
      email: boolean;
      push: boolean;
      types: string[];
    };
  }> {
    const response = await ApiService.get<{
      theme: string;
      language: string;
      timezone: string;
      notifications: {
        email: boolean;
        push: boolean;
        types: string[];
      };
    }>("/config/user-preferences");
    return response;
  }

  // Update user preferences
  async updateUserPreferences(preferences: {
    theme?: string;
    language?: string;
    timezone?: string;
    notifications?: {
      email: boolean;
      push: boolean;
      types: string[];
    };
  }): Promise<void> {
    await ApiService.put("/config/user-preferences", preferences);
  }

  // Get available themes
  async getAvailableThemes(): Promise<
    Array<{ name: string; displayName: string; description: string }>
  > {
    const response = await ApiService.get<
      Array<{ name: string; displayName: string; description: string }>
    >("/config/themes");
    return response;
  }

  // Get system status
  async getSystemStatus(): Promise<{
    status: "online" | "offline" | "maintenance";
    uptime: number;
    lastUpdate: Date;
    services: Array<{
      name: string;
      status: "healthy" | "degraded" | "down";
      responseTime: number;
    }>;
  }> {
    const response = await ApiService.get<{
      status: "online" | "offline" | "maintenance";
      uptime: number;
      lastUpdate: Date;
      services: Array<{
        name: string;
        status: "healthy" | "degraded" | "down";
        responseTime: number;
      }>;
    }>("/config/status");
    return response;
  }

  // Export configuration
  async exportConfig(): Promise<Blob> {
    const response = await ApiService.get("/config/export", {
      responseType: "blob",
    });
    return response as Blob;
  }

  // Import configuration
  async importConfig(file: File): Promise<void> {
    await ApiService.upload("/config/import", file);
  }

  // Reset configuration to defaults
  async resetToDefaults(): Promise<SystemConfig> {
    const response = await ApiService.post<SystemConfig>("/config/reset");
    return response;
  }

  // Mock system configuration for development
  getMockSystemConfig(): SystemConfig {
    return {
      simulation: {
        enabled: true,
        stepSize: 1,
        endTime: 3600,
      },
      trafficLights: {
        minPhaseDuration: 5,
        maxPhaseDuration: 60,
        updateInterval: 1,
      },
      dataExport: {
        enabled: true,
        interval: 10,
        format: "json",
      },
      notifications: {
        enabled: true,
        email: false,
        push: true,
      },
    };
  }

  // Mock user preferences for development
  getMockUserPreferences() {
    return {
      theme: "sap_horizon",
      language: "en",
      timezone: "UTC",
      notifications: {
        email: false,
        push: true,
        types: ["info", "warning", "error", "alert"],
      },
    };
  }

  // Mock available themes
  getMockAvailableThemes() {
    return [
      {
        name: "sap_horizon",
        displayName: "SAP Horizon",
        description: "Modern SAP design language",
      },
      {
        name: "sap_horizon_dark",
        displayName: "SAP Horizon Dark",
        description: "Dark theme with SAP Horizon design",
      },
      {
        name: "sap_fiori_3",
        displayName: "SAP Fiori 3",
        description: "Classic SAP Fiori design",
      },
      {
        name: "sap_fiori_3_dark",
        displayName: "SAP Fiori 3 Dark",
        description: "Dark theme with SAP Fiori 3 design",
      },
    ];
  }

  // Mock system status
  getMockSystemStatus() {
    return {
      status: "online" as const,
      uptime: 86400, // 24 hours
      lastUpdate: new Date(),
      services: [
        {
          name: "Traffic Management API",
          status: "healthy" as const,
          responseTime: 45,
        },
        {
          name: "Database",
          status: "healthy" as const,
          responseTime: 12,
        },
        {
          name: "WebSocket Service",
          status: "healthy" as const,
          responseTime: 8,
        },
        {
          name: "File Storage",
          status: "degraded" as const,
          responseTime: 150,
        },
      ],
    };
  }
}

export const configService = new ConfigService();
