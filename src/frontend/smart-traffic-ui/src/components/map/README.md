# Traffic Map Components

This directory contains React components for real-time traffic visualization using Leaflet maps and camera feed integration.

## Components

### TrafficMap
A comprehensive traffic map component that displays intersections with real-time traffic data.

**Features:**
- Interactive Leaflet map with OpenStreetMap tiles
- Custom traffic light markers with signal states
- Real-time vehicle count display
- Clickable intersections with detailed popups
- Camera feed overlay integration
- Map legend and controls
- Responsive design

**Props:**
```typescript
interface TrafficMapProps {
  className?: string;
  center?: [number, number]; // [lat, lng] - default: Bhubaneswar, Odisha
  zoom?: number; // default: 13
  height?: string; // default: '500px'
  showCameraFeeds?: boolean; // default: true
  onIntersectionClick?: (intersection: MapIntersection) => void;
}
```

**Usage:**
```tsx
<TrafficMap 
  height="400px"
  showCameraFeeds={true}
  onIntersectionClick={(intersection) => {
    console.log('Intersection clicked:', intersection);
  }}
/>
```

### CameraFeedOverlay
A modal overlay component that displays camera feeds for selected intersections.

**Features:**
- Multiple camera feed thumbnails
- Fullscreen video display
- Real-time camera status indicators
- Intersection information panel
- Responsive grid layout
- Animated transitions

**Props:**
```typescript
interface CameraFeedOverlayProps {
  intersection: MapIntersection;
  onClose: () => void;
  className?: string;
}
```

**Usage:**
```tsx
<CameraFeedOverlay
  intersection={selectedIntersection}
  onClose={() => setSelectedIntersection(null)}
/>
```

## Hooks

### useMapData
A custom hook for fetching and managing map data from the backend APIs.

**Features:**
- Automatic data polling (30-second intervals)
- Error handling and retry logic
- Loading states
- Manual refresh capability
- TypeScript support

**API Endpoints:**
- `GET /traffic/lights` - Traffic light status
- `GET /api/cv/counts` - Vehicle counts from computer vision

**Usage:**
```tsx
const {
  intersections,
  loading,
  error,
  lastUpdate,
  refresh
} = useMapData({
  enabled: true,
  refreshInterval: 30000,
  onError: (error) => console.error('Map data error:', error)
});
```

## Data Types

### MapIntersection
```typescript
interface MapIntersection {
  id: string;
  name: string;
  position: [number, number]; // [lat, lng]
  signalState: 'red' | 'yellow' | 'green' | 'unknown';
  vehicleCount: number;
  lastUpdate: Date;
  status: 'normal' | 'warning' | 'error';
}
```

### MapData
```typescript
interface MapData {
  intersections: MapIntersection[];
  lastUpdate: Date;
  loading: boolean;
  error: string | null;
}
```

## Styling

Components use SCSS modules for styling:
- `TrafficMap.scss` - Map container and marker styles
- `CameraFeedOverlay.scss` - Camera feed modal styles

### Custom Marker Styling
Traffic light markers are styled with custom CSS:
- Color-coded by signal state (green, yellow, red)
- Vehicle count badges
- Hover effects and animations
- Status indicators (normal, warning, error)

## Configuration

### Map Center
Default center is set to Bhubaneswar, Odisha coordinates:
```typescript
center = [20.2961, 85.8245] // [lat, lng]
```

### Camera Feed URLs
Camera feeds use mock RTSP URLs:
```
rtsp://camera-server.com/{intersectionId}/{position}
```

### Refresh Intervals
- Map data: 30 seconds
- Camera feeds: Real-time (when available)

## Dependencies

- `leaflet` - Map library
- `react-leaflet` - React Leaflet bindings
- `@types/leaflet` - TypeScript definitions
- `framer-motion` - Animations
- `@ui5/webcomponents-react` - UI components

## Browser Support

- Modern browsers with ES6+ support
- WebGL support for smooth map rendering
- WebRTC support for camera feeds (optional)

## Performance Considerations

- Map markers are optimized for performance
- Camera feeds use image placeholders to reduce bandwidth
- Automatic cleanup of event listeners
- Debounced refresh operations
- Lazy loading of map tiles

## Error Handling

- Graceful fallback for failed API calls
- Retry logic with exponential backoff
- User-friendly error messages
- Offline state indicators
