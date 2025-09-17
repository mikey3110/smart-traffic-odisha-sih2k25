import React, { useEffect, useRef, useState } from "react";
import { MapContainer, TileLayer, Marker, Popup, useMap } from "react-leaflet";
import L from "leaflet";
import { motion } from "framer-motion";
import {
  Card,
  CardHeader,
  Text,
  Title,
  Icon,
  Badge,
  Button,
  ProgressIndicator,
} from "@ui5/webcomponents-react";
import { useMapData, MapIntersection } from "@/hooks/useMapData";
import { CameraFeedOverlay } from "./CameraFeedOverlay";
import "./TrafficMap.scss";

// Fix for default markers in react-leaflet
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png",
  iconUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png",
  shadowUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png",
});

interface TrafficMapProps {
  className?: string;
  center?: [number, number];
  zoom?: number;
  height?: string;
  showCameraFeeds?: boolean;
  onIntersectionClick?: (intersection: MapIntersection) => void;
}

// Custom marker component for traffic lights
function TrafficLightMarker({
  intersection,
  onClick,
}: {
  intersection: MapIntersection;
  onClick?: (intersection: MapIntersection) => void;
}) {
  const map = useMap();

  const getMarkerIcon = (intersection: MapIntersection) => {
    const { signalState, status, vehicleCount } = intersection;

    let color = "#666";
    if (status === "error") color = "#d32f2f";
    else if (status === "warning") color = "#f57c00";
    else if (signalState === "green") color = "#388e3c";
    else if (signalState === "yellow") color = "#fbc02d";
    else if (signalState === "red") color = "#d32f2f";

    const iconHtml = `
      <div class="traffic-marker" style="background-color: ${color}">
        <div class="signal-light ${signalState}"></div>
        <div class="vehicle-count">${vehicleCount}</div>
      </div>
    `;

    return L.divIcon({
      html: iconHtml,
      className: "custom-traffic-marker",
      iconSize: [30, 30],
      iconAnchor: [15, 15],
      popupAnchor: [0, -15],
    });
  };

  return (
    <Marker
      position={intersection.position}
      icon={getMarkerIcon(intersection)}
      eventHandlers={{
        click: () => onClick?.(intersection),
      }}
    >
      <Popup>
        <div className="intersection-popup">
          <h3>{intersection.name}</h3>
          <div className="popup-details">
            <div className="detail-item">
              <Icon name="traffic-light" />
              <span>Signal: {intersection.signalState.toUpperCase()}</span>
            </div>
            <div className="detail-item">
              <Icon name="car" />
              <span>Vehicles: {intersection.vehicleCount}</span>
            </div>
            <div className="detail-item">
              <Icon name="time" />
              <span>
                Updated: {intersection.lastUpdate.toLocaleTimeString()}
              </span>
            </div>
            <div className="detail-item">
              <Badge
                colorScheme={
                  intersection.status === "normal"
                    ? "8"
                    : intersection.status === "warning"
                    ? "2"
                    : "1"
                }
              >
                {intersection.status.toUpperCase()}
              </Badge>
            </div>
          </div>
        </div>
      </Popup>
    </Marker>
  );
}

// Map controls component
function MapControls({
  onRefresh,
  loading,
  lastUpdate,
}: {
  onRefresh: () => void;
  loading: boolean;
  lastUpdate: Date;
}) {
  return (
    <div className="map-controls">
      <Button
        design="Transparent"
        icon="refresh"
        onClick={onRefresh}
        disabled={loading}
        className="refresh-button"
      />
      <div className="last-update">
        <Text>Last update: {lastUpdate.toLocaleTimeString()}</Text>
      </div>
    </div>
  );
}

export function TrafficMap({
  className = "",
  center = [20.2961, 85.8245], // Bhubaneswar, Odisha coordinates
  zoom = 13,
  height = "500px",
  showCameraFeeds = true,
  onIntersectionClick,
}: TrafficMapProps) {
  const mapRef = useRef<L.Map>(null);
  const [selectedIntersection, setSelectedIntersection] =
    useState<MapIntersection | null>(null);

  const { intersections, loading, error, lastUpdate, refresh } = useMapData({
    enabled: true,
    refreshInterval: 30000, // 30 seconds
    onError: (error) => {
      console.error("Map data error:", error);
    },
  });

  const handleIntersectionClick = (intersection: MapIntersection) => {
    setSelectedIntersection(intersection);
    onIntersectionClick?.(intersection);
  };

  const handleRefresh = () => {
    refresh();
  };

  if (error) {
    return (
      <Card className={`traffic-map error ${className}`}>
        <CardHeader titleText="Traffic Map">
          <Badge colorScheme="1">Error</Badge>
        </CardHeader>
        <div className="error-state">
          <Icon name="error" />
          <Text>Failed to load map data: {error}</Text>
          <Button design="Emphasized" onClick={handleRefresh}>
            Retry
          </Button>
        </div>
      </Card>
    );
  }

  return (
    <Card className={`traffic-map ${className}`}>
      <CardHeader titleText="Traffic Map">
        <div className="header-actions">
          <Badge colorScheme="8">Live</Badge>
          <MapControls
            onRefresh={handleRefresh}
            loading={loading}
            lastUpdate={lastUpdate}
          />
        </div>
      </CardHeader>

      <div className="map-container" style={{ height }}>
        {loading && (
          <div className="map-loading">
            <ProgressIndicator value={undefined} />
            <Text>Loading traffic data...</Text>
          </div>
        )}

        <MapContainer
          center={center}
          zoom={zoom}
          style={{ height: "100%", width: "100%" }}
          ref={mapRef}
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          {intersections.map((intersection) => (
            <TrafficLightMarker
              key={intersection.id}
              intersection={intersection}
              onClick={handleIntersectionClick}
            />
          ))}
        </MapContainer>

        {/* Camera Feed Overlay */}
        {showCameraFeeds && selectedIntersection && (
          <motion.div
            className="camera-overlay"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            transition={{ duration: 0.3 }}
          >
            <CameraFeedOverlay
              intersection={selectedIntersection}
              onClose={() => setSelectedIntersection(null)}
            />
          </motion.div>
        )}

        {/* Map Legend */}
        <div className="map-legend">
          <div className="legend-title">
            <Text>Legend</Text>
          </div>
          <div className="legend-items">
            <div className="legend-item">
              <div className="legend-color green"></div>
              <Text>Green Signal</Text>
            </div>
            <div className="legend-item">
              <div className="legend-color yellow"></div>
              <Text>Yellow Signal</Text>
            </div>
            <div className="legend-item">
              <div className="legend-color red"></div>
              <Text>Red Signal</Text>
            </div>
            <div className="legend-item">
              <div className="legend-color error"></div>
              <Text>Error/Offline</Text>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
}
