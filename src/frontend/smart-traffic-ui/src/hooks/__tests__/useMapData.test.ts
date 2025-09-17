import { renderHook, waitFor } from "@testing-library/react";
import { useMapData } from "../useMapData";
import { trafficService } from "../../services/trafficService";
import { TrafficLightStatus } from "../../types";

// Mock the traffic service
jest.mock("../../services/trafficService");
const mockTrafficService = trafficService as jest.Mocked<typeof trafficService>;

// Mock fetch
global.fetch = jest.fn();
const mockFetch = fetch as jest.MockedFunction<typeof fetch>;

describe("useMapData", () => {
  const mockTrafficLights = [
    {
      id: "intersection_1",
      name: "Main Street & First Avenue",
      location: { lat: 20.2961, lng: 85.8245 },
      status: TrafficLightStatus.NORMAL,
      currentPhase: 0,
      phaseDuration: 30,
      program: "0",
      lastUpdate: new Date("2024-01-15T10:30:00Z"),
      vehicleCount: 0,
      waitingTime: 45,
    },
    {
      id: "intersection_2",
      name: "Second Street & Park Avenue",
      location: { lat: 20.3, lng: 85.83 },
      status: TrafficLightStatus.NORMAL,
      currentPhase: 2,
      phaseDuration: 30,
      program: "0",
      lastUpdate: new Date("2024-01-15T10:30:00Z"),
      vehicleCount: 0,
      waitingTime: 30,
    },
  ];

  const mockVehicleCounts = [
    { intersectionId: "intersection_1", vehicleCount: 12 },
    { intersectionId: "intersection_2", vehicleCount: 8 },
  ];

  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();

    mockTrafficService.getTrafficLights.mockResolvedValue(mockTrafficLights);
    mockFetch.mockResolvedValue({
      json: () => Promise.resolve(mockVehicleCounts),
    } as Response);
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it("should fetch and transform data correctly", async () => {
    const { result } = renderHook(() => useMapData());

    expect(result.current.loading).toBe(true);
    expect(result.current.intersections).toEqual([]);

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.intersections).toHaveLength(2);
    expect(result.current.intersections[0]).toEqual({
      id: "intersection_1",
      name: "Main Street & First Avenue",
      position: [20.2961, 85.8245],
      signalState: "green",
      vehicleCount: 12,
      lastUpdate: mockTrafficLights[0].lastUpdate,
      status: "normal",
    });
    expect(result.current.error).toBeNull();
  });

  it("should handle API errors gracefully", async () => {
    const errorMessage = "Network error";
    mockTrafficService.getTrafficLights.mockRejectedValue(
      new Error(errorMessage)
    );

    const onError = jest.fn();
    const { result } = renderHook(() => useMapData({ onError }));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.error).toBe(errorMessage);
    expect(result.current.intersections).toEqual([]);
    expect(onError).toHaveBeenCalledWith(expect.any(Error));
  });

  it("should determine signal state correctly", async () => {
    const trafficLightsWithDifferentPhases = [
      { ...mockTrafficLights[0], currentPhase: 0 }, // green
      { ...mockTrafficLights[1], currentPhase: 1 }, // yellow
    ];

    mockTrafficService.getTrafficLights.mockResolvedValue(
      trafficLightsWithDifferentPhases
    );

    const { result } = renderHook(() => useMapData());

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.intersections[0].signalState).toBe("green");
    expect(result.current.intersections[1].signalState).toBe("yellow");
  });

  it("should determine status based on vehicle count and waiting time", async () => {
    const trafficLightsWithHighCounts = [
      { ...mockTrafficLights[0], status: TrafficLightStatus.MAINTENANCE }, // warning
      { ...mockTrafficLights[1], status: TrafficLightStatus.ERROR }, // error
    ];

    mockTrafficService.getTrafficLights.mockResolvedValue(
      trafficLightsWithHighCounts
    );
    mockFetch.mockResolvedValue({
      json: () =>
        Promise.resolve([
          { intersectionId: "intersection_1", vehicleCount: 25 },
          { intersectionId: "intersection_2", vehicleCount: 5 },
        ]),
    } as Response);

    const { result } = renderHook(() => useMapData());

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.intersections[0].status).toBe("warning");
    expect(result.current.intersections[1].status).toBe("error");
  });

  it("should not fetch data when disabled", () => {
    renderHook(() => useMapData({ enabled: false }));

    expect(mockTrafficService.getTrafficLights).not.toHaveBeenCalled();
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("should poll data at specified interval", async () => {
    const { result } = renderHook(() => useMapData({ refreshInterval: 1000 }));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    // Fast-forward time to trigger polling
    jest.advanceTimersByTime(1000);

    await waitFor(() => {
      expect(mockTrafficService.getTrafficLights).toHaveBeenCalledTimes(2);
    });
  });

  it("should provide manual refresh function", async () => {
    const { result } = renderHook(() => useMapData());

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    // Call refresh manually
    result.current.refresh();

    await waitFor(() => {
      expect(mockTrafficService.getTrafficLights).toHaveBeenCalledTimes(2);
    });
  });

  it("should handle missing vehicle count data", async () => {
    mockFetch.mockResolvedValue({
      json: () => Promise.resolve([]),
    } as Response);

    const { result } = renderHook(() => useMapData());

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.intersections[0].vehicleCount).toBe(0);
    expect(result.current.intersections[1].vehicleCount).toBe(0);
  });

  it("should handle fetch errors for vehicle counts", async () => {
    mockFetch.mockRejectedValue(new Error("Fetch error"));

    const { result } = renderHook(() => useMapData());

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    // Should still return traffic lights with 0 vehicle count
    expect(result.current.intersections).toHaveLength(2);
    expect(result.current.intersections[0].vehicleCount).toBe(0);
  });
});
