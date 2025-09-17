import { useState, useEffect, useCallback } from "react";
import { trafficService } from "../services/trafficService";

export interface MapIntersection {
  id: string;
  name: string;
  position: [number, number]; // [lat, lng]
  signalState: "red" | "yellow" | "green" | "unknown";
  vehicleCount: number;
  lastUpdate: Date;
  status: "normal" | "warning" | "error";
}

export interface MapData {
  intersections: MapIntersection[];
  lastUpdate: Date;
  loading: boolean;
  error: string | null;
}

interface UseMapDataOptions {
  enabled?: boolean;
  refreshInterval?: number;
  onError?: (error: Error) => void;
}

export function useMapData(options: UseMapDataOptions = {}) {
  const {
    enabled = true,
    refreshInterval = 30000, // 30 seconds
    onError,
  } = options;

  const [data, setData] = useState<MapData>({
    intersections: [],
    lastUpdate: new Date(),
    loading: false,
    error: null,
  });

  const fetchMapData = useCallback(async () => {
    if (!enabled) return;

    setData((prev) => ({ ...prev, loading: true, error: null }));

    try {
      // Fetch traffic status and vehicle counts in parallel
      const [trafficLights, vehicleCounts] = await Promise.all([
        trafficService.getTrafficLights().catch(() => []),
        fetch("/api/cv/counts")
          .then((res) => res.json())
          .catch(() => []),
      ]);

      // Transform data into map format
      const intersections: MapIntersection[] = trafficLights.map((light) => {
        // Find corresponding vehicle count
        const vehicleCount =
          vehicleCounts.find((count: any) => count.intersectionId === light.id)
            ?.vehicleCount || 0;

        // Determine signal state
        let signalState: "red" | "yellow" | "green" | "unknown" = "unknown";
        if (light.status === "normal") {
          // Map phase to signal state (simplified)
          if (light.currentPhase === 0 || light.currentPhase === 2) {
            signalState = "green";
          } else if (light.currentPhase === 1 || light.currentPhase === 3) {
            signalState = "yellow";
          } else {
            signalState = "red";
          }
        }

        // Determine status based on vehicle count and waiting time
        let status: "normal" | "warning" | "error" = "normal";
        if (vehicleCount > 20 || light.waitingTime > 60) {
          status = "warning";
        }
        if (light.status === "error" || light.status === "offline") {
          status = "error";
        }

        return {
          id: light.id,
          name: light.name,
          position: [light.location.lat, light.location.lng],
          signalState,
          vehicleCount,
          lastUpdate: light.lastUpdate,
          status,
        };
      });

      setData({
        intersections,
        lastUpdate: new Date(),
        loading: false,
        error: null,
      });
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Failed to fetch map data";
      setData((prev) => ({
        ...prev,
        loading: false,
        error: errorMessage,
      }));

      if (onError) {
        onError(error instanceof Error ? error : new Error(errorMessage));
      }
    }
  }, [enabled, onError]);

  // Initial fetch
  useEffect(() => {
    fetchMapData();
  }, [fetchMapData]);

  // Set up polling
  useEffect(() => {
    if (!enabled || refreshInterval <= 0) return;

    const interval = setInterval(fetchMapData, refreshInterval);
    return () => clearInterval(interval);
  }, [fetchMapData, enabled, refreshInterval]);

  // Manual refresh function
  const refresh = useCallback(() => {
    fetchMapData();
  }, [fetchMapData]);

  return {
    ...data,
    refresh,
  };
}
