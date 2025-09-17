import "@testing-library/jest-dom";
import React from "react";

// Mock window.matchMedia
Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: jest.fn().mockImplementation((query) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(), // deprecated
    removeListener: jest.fn(), // deprecated
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});

// Mock IntersectionObserver
global.IntersectionObserver = class IntersectionObserver {
  root: Element | null = null;
  rootMargin: string = "0px";
  thresholds: ReadonlyArray<number> = [0];

  constructor() {}
  disconnect() {}
  observe() {}
  unobserve() {}
  takeRecords() {
    return [];
  }
} as any;

// Mock ResizeObserver
global.ResizeObserver = class ResizeObserver {
  constructor() {}
  disconnect() {}
  observe() {}
  unobserve() {}
};

// Mock Leaflet
jest.mock("leaflet", () => ({
  map: jest.fn(() => ({
    setView: jest.fn(),
    addLayer: jest.fn(),
    removeLayer: jest.fn(),
    on: jest.fn(),
    off: jest.fn(),
    invalidateSize: jest.fn(),
  })),
  tileLayer: jest.fn(() => ({
    addTo: jest.fn(),
  })),
  marker: jest.fn(() => ({
    addTo: jest.fn(),
    bindPopup: jest.fn(),
    on: jest.fn(),
  })),
  divIcon: jest.fn(() => ({})),
  Icon: {
    Default: {
      mergeOptions: jest.fn(),
    },
  },
}));

// Mock react-leaflet
jest.mock("react-leaflet", () => ({
  MapContainer: ({ children, ...props }: any) =>
    React.createElement(
      "div",
      { "data-testid": "map-container", ...props },
      children
    ),
  TileLayer: ({ ...props }: any) =>
    React.createElement("div", { "data-testid": "tile-layer", ...props }),
  Marker: ({ children, ...props }: any) =>
    React.createElement("div", { "data-testid": "marker", ...props }, children),
  Popup: ({ children, ...props }: any) =>
    React.createElement("div", { "data-testid": "popup", ...props }, children),
  useMap: () => ({
    setView: jest.fn(),
    addLayer: jest.fn(),
    removeLayer: jest.fn(),
  }),
}));

// Mock framer-motion
jest.mock("framer-motion", () => ({
  motion: {
    div: ({ children, ...props }: any) =>
      React.createElement("div", props, children),
    span: ({ children, ...props }: any) =>
      React.createElement("span", props, children),
  },
  AnimatePresence: ({ children }: any) => children,
}));

// Mock environment variables
Object.defineProperty(global, "import", {
  value: {
    meta: {
      env: {
        VITE_API_BASE_URL: "http://localhost:8000",
        VITE_WS_URL: "ws://localhost:8000",
      },
    },
  },
  writable: true,
});

// Mock console methods to reduce noise in tests
const originalError = console.error;
const originalWarn = console.warn;

beforeAll(() => {
  console.error = (...args) => {
    if (
      typeof args[0] === "string" &&
      args[0].includes("Warning: ReactDOM.render is no longer supported")
    ) {
      return;
    }
    originalError.call(console, ...args);
  };

  console.warn = (...args) => {
    if (
      typeof args[0] === "string" &&
      (args[0].includes("componentWillReceiveProps") ||
        args[0].includes("componentWillMount"))
    ) {
      return;
    }
    originalWarn.call(console, ...args);
  };
});

afterAll(() => {
  console.error = originalError;
  console.warn = originalWarn;
});
