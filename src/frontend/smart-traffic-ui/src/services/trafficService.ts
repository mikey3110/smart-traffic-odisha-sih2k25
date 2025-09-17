import { ApiService } from "./apiService";
import {
  TrafficLight,
  Vehicle,
  Intersection,
  PerformanceMetrics,
  FilterOptions,
  PaginatedResponse,
  TrafficLightConfigForm,
} from "@/types";

class TrafficService {
  // Traffic Lights
  async getTrafficLights(): Promise<TrafficLight[]> {
    const response = await ApiService.get<TrafficLight[]>("/traffic/lights");
    return response;
  }

  async getTrafficLight(id: string): Promise<TrafficLight> {
    const response = await ApiService.get<TrafficLight>(
      `/traffic/lights/${id}`
    );
    return response;
  }

  async updateTrafficLight(
    id: string,
    config: TrafficLightConfigForm
  ): Promise<TrafficLight> {
    const response = await ApiService.put<TrafficLight>(
      `/traffic/lights/${id}`,
      config
    );
    return response;
  }

  async controlTrafficLight(
    id: string,
    phase: number,
    duration: number
  ): Promise<void> {
    await ApiService.post(`/traffic/lights/${id}/control`, { phase, duration });
  }

  async getTrafficLightStatus(
    id: string
  ): Promise<{ status: string; phase: number; duration: number }> {
    const response = await ApiService.get<{
      status: string;
      phase: number;
      duration: number;
    }>(`/traffic/lights/${id}/status`);
    return response;
  }

  // Vehicles
  async getVehicles(filters?: FilterOptions): Promise<Vehicle[]> {
    const params = new URLSearchParams();
    if (filters?.dateRange) {
      params.append("startDate", filters.dateRange.start.toISOString());
      params.append("endDate", filters.dateRange.end.toISOString());
    }
    if (filters?.vehicleTypes?.length) {
      params.append("vehicleTypes", filters.vehicleTypes.join(","));
    }
    if (filters?.intersectionIds?.length) {
      params.append("intersectionIds", filters.intersectionIds.join(","));
    }

    const response = await ApiService.get<Vehicle[]>(
      `/traffic/vehicles?${params.toString()}`
    );
    return response;
  }

  async getVehicle(id: string): Promise<Vehicle> {
    const response = await ApiService.get<Vehicle>(`/traffic/vehicles/${id}`);
    return response;
  }

  async getVehiclesByIntersection(intersectionId: string): Promise<Vehicle[]> {
    const response = await ApiService.get<Vehicle[]>(
      `/traffic/intersections/${intersectionId}/vehicles`
    );
    return response;
  }

  // Intersections
  async getIntersections(): Promise<Intersection[]> {
    const response = await ApiService.get<Intersection[]>(
      "/traffic/intersections"
    );
    return response;
  }

  async getIntersection(id: string): Promise<Intersection> {
    const response = await ApiService.get<Intersection>(
      `/traffic/intersections/${id}`
    );
    return response;
  }

  async getIntersectionMetrics(
    id: string,
    timeRange?: { start: Date; end: Date }
  ): Promise<PerformanceMetrics[]> {
    const params = new URLSearchParams();
    if (timeRange) {
      params.append("startDate", timeRange.start.toISOString());
      params.append("endDate", timeRange.end.toISOString());
    }

    const response = await ApiService.get<PerformanceMetrics[]>(
      `/traffic/intersections/${id}/metrics?${params.toString()}`
    );
    return response;
  }

  // Performance Metrics
  async getPerformanceMetrics(timeRange?: {
    start: Date;
    end: Date;
  }): Promise<PerformanceMetrics[]> {
    const params = new URLSearchParams();
    if (timeRange) {
      params.append("startDate", timeRange.start.toISOString());
      params.append("endDate", timeRange.end.toISOString());
    }

    const response = await ApiService.get<PerformanceMetrics[]>(
      `/traffic/metrics?${params.toString()}`
    );
    return response;
  }

  async getCurrentMetrics(): Promise<PerformanceMetrics> {
    const response = await ApiService.get<PerformanceMetrics>(
      "/traffic/metrics/current"
    );
    return response;
  }

  // Analytics
  async getTrafficAnalytics(filters?: FilterOptions): Promise<{
    totalVehicles: number;
    averageSpeed: number;
    totalWaitingTime: number;
    totalCo2Emission: number;
    throughput: number;
    peakHours: { hour: number; count: number }[];
    vehicleTypeDistribution: { type: string; count: number }[];
    intersectionPerformance: { id: string; name: string; efficiency: number }[];
  }> {
    const params = new URLSearchParams();
    if (filters?.dateRange) {
      params.append("startDate", filters.dateRange.start.toISOString());
      params.append("endDate", filters.dateRange.end.toISOString());
    }

    const response = await ApiService.get<{
      totalVehicles: number;
      averageSpeed: number;
      totalWaitingTime: number;
      totalCo2Emission: number;
      throughput: number;
      peakHours: { hour: number; count: number }[];
      vehicleTypeDistribution: { type: string; count: number }[];
      intersectionPerformance: {
        id: string;
        name: string;
        efficiency: number;
      }[];
    }>(`/traffic/analytics?${params.toString()}`);
    return response;
  }

  // Reports
  async generateReport(
    type: "summary" | "detailed" | "export",
    filters?: FilterOptions
  ): Promise<Blob> {
    const params = new URLSearchParams();
    params.append("type", type);
    if (filters?.dateRange) {
      params.append("startDate", filters.dateRange.start.toISOString());
      params.append("endDate", filters.dateRange.end.toISOString());
    }

    const response = await ApiService.get(
      `/traffic/reports?${params.toString()}`,
      {
        responseType: "blob",
      }
    );

    return response as Blob;
  }

  // Mock data for development
  getMockTrafficLights(): TrafficLight[] {
    return [
      {
        id: "1",
        name: "Main Street & First Ave",
        location: { lat: 40.7128, lng: -74.006 },
        status: "normal" as any,
        currentPhase: 0,
        phaseDuration: 30,
        program: "adaptive",
        lastUpdate: new Date(),
        vehicleCount: 15,
        waitingTime: 45,
      },
      {
        id: "2",
        name: "Second Street & Oak Ave",
        location: { lat: 40.7589, lng: -73.9851 },
        status: "normal" as any,
        currentPhase: 2,
        phaseDuration: 25,
        program: "adaptive",
        lastUpdate: new Date(),
        vehicleCount: 8,
        waitingTime: 20,
      },
    ];
  }

  getMockVehicles(): Vehicle[] {
    return [
      {
        id: "1",
        type: "passenger" as any,
        position: { lat: 40.7128, lng: -74.006 },
        speed: 25.5,
        lane: "north_approach_0",
        route: ["north_approach", "center_junction", "south_exit"],
        waitingTime: 5.2,
        co2Emission: 0.1,
        fuelConsumption: 0.05,
        timestamp: new Date(),
      },
      {
        id: "2",
        type: "truck" as any,
        position: { lat: 40.7589, lng: -73.9851 },
        speed: 18.0,
        lane: "east_approach_0",
        route: ["east_approach", "center_junction", "west_exit"],
        waitingTime: 12.8,
        co2Emission: 0.3,
        fuelConsumption: 0.15,
        timestamp: new Date(),
      },
    ];
  }

  getMockIntersections(): Intersection[] {
    return [
      {
        id: "1",
        name: "Main Intersection",
        location: { lat: 40.7128, lng: -74.006 },
        trafficLights: this.getMockTrafficLights(),
        totalVehicles: 23,
        waitingVehicles: 8,
        averageSpeed: 22.5,
        throughput: 45,
        lastUpdate: new Date(),
      },
    ];
  }

  getMockPerformanceMetrics(): PerformanceMetrics[] {
    const metrics: PerformanceMetrics[] = [];
    const now = new Date();

    for (let i = 0; i < 24; i++) {
      const timestamp = new Date(now.getTime() - (23 - i) * 60 * 60 * 1000);
      metrics.push({
        timestamp,
        totalVehicles: Math.floor(Math.random() * 100) + 50,
        runningVehicles: Math.floor(Math.random() * 80) + 40,
        waitingVehicles: Math.floor(Math.random() * 20) + 5,
        totalWaitingTime: Math.random() * 1000 + 500,
        averageSpeed: Math.random() * 20 + 15,
        totalCo2Emission: Math.random() * 100 + 50,
        totalFuelConsumption: Math.random() * 50 + 25,
        throughput: Math.random() * 50 + 30,
      });
    }

    return metrics;
  }
}

export const trafficService = new TrafficService();
