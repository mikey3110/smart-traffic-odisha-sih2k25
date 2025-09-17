import React from "react";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { TrafficMap } from "../TrafficMap";
import { useMapData } from "../../../hooks/useMapData";

// Mock the useMapData hook
jest.mock("../../../hooks/useMapData");
const mockUseMapData = useMapData as jest.MockedFunction<typeof useMapData>;

// Mock the CameraFeedOverlay component
jest.mock("../CameraFeedOverlay", () => ({
  CameraFeedOverlay: ({ intersection, onClose }: any) => (
    <div data-testid="camera-feed-overlay">
      <span>Camera for {intersection.name}</span>
      <button onClick={onClose}>Close</button>
    </div>
  ),
}));

describe("TrafficMap", () => {
  const mockIntersections = [
    {
      id: "intersection_1",
      name: "Main Street & First Avenue",
      position: [20.2961, 85.8245] as [number, number],
      signalState: "green" as const,
      vehicleCount: 12,
      lastUpdate: new Date("2024-01-15T10:30:00Z"),
      status: "normal" as const,
    },
    {
      id: "intersection_2",
      name: "Second Street & Park Avenue",
      position: [20.3, 85.83] as [number, number],
      signalState: "red" as const,
      vehicleCount: 8,
      lastUpdate: new Date("2024-01-15T10:30:00Z"),
      status: "warning" as const,
    },
  ];

  beforeEach(() => {
    mockUseMapData.mockReturnValue({
      intersections: mockIntersections,
      lastUpdate: new Date("2024-01-15T10:30:00Z"),
      loading: false,
      error: null,
      refresh: jest.fn(),
    });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it("renders map container with correct props", () => {
    render(<TrafficMap />);

    expect(screen.getByTestId("map-container")).toBeInTheDocument();
    expect(screen.getByTestId("tile-layer")).toBeInTheDocument();
  });

  it("renders traffic markers for each intersection", () => {
    render(<TrafficMap />);

    const markers = screen.getAllByTestId("marker");
    expect(markers).toHaveLength(2);
  });

  it("displays loading state when data is loading", () => {
    mockUseMapData.mockReturnValue({
      intersections: [],
      lastUpdate: new Date(),
      loading: true,
      error: null,
      refresh: jest.fn(),
    });

    render(<TrafficMap />);

    expect(screen.getByText("Loading traffic data...")).toBeInTheDocument();
  });

  it("displays error state when there is an error", () => {
    mockUseMapData.mockReturnValue({
      intersections: [],
      lastUpdate: new Date(),
      loading: false,
      error: "Failed to fetch data",
      refresh: jest.fn(),
    });

    render(<TrafficMap />);

    expect(
      screen.getByText("Failed to load map data: Failed to fetch data")
    ).toBeInTheDocument();
    expect(screen.getByText("Retry")).toBeInTheDocument();
  });

  it("calls onIntersectionClick when marker is clicked", () => {
    const mockOnIntersectionClick = jest.fn();
    render(<TrafficMap onIntersectionClick={mockOnIntersectionClick} />);

    const markers = screen.getAllByTestId("marker");
    fireEvent.click(markers[0]);

    expect(mockOnIntersectionClick).toHaveBeenCalledWith(mockIntersections[0]);
  });

  it("shows camera feed overlay when intersection is clicked", async () => {
    render(<TrafficMap showCameraFeeds={true} />);

    const markers = screen.getAllByTestId("marker");
    fireEvent.click(markers[0]);

    await waitFor(() => {
      expect(screen.getByTestId("camera-feed-overlay")).toBeInTheDocument();
      expect(
        screen.getByText("Camera for Main Street & First Avenue")
      ).toBeInTheDocument();
    });
  });

  it("hides camera feed overlay when close button is clicked", async () => {
    render(<TrafficMap showCameraFeeds={true} />);

    const markers = screen.getAllByTestId("marker");
    fireEvent.click(markers[0]);

    await waitFor(() => {
      expect(screen.getByTestId("camera-feed-overlay")).toBeInTheDocument();
    });

    const closeButton = screen.getByText("Close");
    fireEvent.click(closeButton);

    await waitFor(() => {
      expect(
        screen.queryByTestId("camera-feed-overlay")
      ).not.toBeInTheDocument();
    });
  });

  it("displays map legend", () => {
    render(<TrafficMap />);

    expect(screen.getByText("Legend")).toBeInTheDocument();
    expect(screen.getByText("Green Signal")).toBeInTheDocument();
    expect(screen.getByText("Yellow Signal")).toBeInTheDocument();
    expect(screen.getByText("Red Signal")).toBeInTheDocument();
    expect(screen.getByText("Error/Offline")).toBeInTheDocument();
  });

  it("calls refresh when refresh button is clicked", () => {
    const mockRefresh = jest.fn();
    mockUseMapData.mockReturnValue({
      intersections: mockIntersections,
      lastUpdate: new Date("2024-01-15T10:30:00Z"),
      loading: false,
      error: null,
      refresh: mockRefresh,
    });

    render(<TrafficMap />);

    const refreshButton = screen.getByRole("button", { name: /refresh/i });
    fireEvent.click(refreshButton);

    expect(mockRefresh).toHaveBeenCalled();
  });

  it("displays last update time", () => {
    const lastUpdate = new Date("2024-01-15T10:30:00Z");
    mockUseMapData.mockReturnValue({
      intersections: mockIntersections,
      lastUpdate,
      loading: false,
      error: null,
      refresh: jest.fn(),
    });

    render(<TrafficMap />);

    expect(
      screen.getByText(`Last update: ${lastUpdate.toLocaleTimeString()}`)
    ).toBeInTheDocument();
  });

  it("applies custom className", () => {
    const { container } = render(<TrafficMap className="custom-class" />);

    expect(
      container.querySelector(".traffic-map.custom-class")
    ).toBeInTheDocument();
  });

  it("uses custom height when provided", () => {
    render(<TrafficMap height="600px" />);

    const mapContainer = screen.getByTestId("map-container").parentElement;
    expect(mapContainer).toHaveStyle({ height: "600px" });
  });
});
