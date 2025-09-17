import React, { useState } from "react";
import { motion } from "framer-motion";
import {
  Card,
  CardHeader,
  Text,
  Title,
  Button,
  Badge,
} from "@ui5/webcomponents-react";
import { TrafficMap } from "@/components/map/TrafficMap";
import { MapIntersection } from "@/hooks/useMapData";
import "./MapDemo.scss";

export function MapDemo() {
  const [selectedIntersection, setSelectedIntersection] =
    useState<MapIntersection | null>(null);
  const [mapCenter, setMapCenter] = useState<[number, number]>([
    20.2961, 85.8245,
  ]); // Bhubaneswar
  const [mapZoom, setMapZoom] = useState(13);

  const handleIntersectionClick = (intersection: MapIntersection) => {
    setSelectedIntersection(intersection);
    console.log("Selected intersection:", intersection);
  };

  const resetMap = () => {
    setMapCenter([20.2961, 85.8245]);
    setMapZoom(13);
    setSelectedIntersection(null);
  };

  const focusOnDelhi = () => {
    setMapCenter([28.6139, 77.209]); // Delhi
    setMapZoom(12);
  };

  const focusOnMumbai = () => {
    setMapCenter([19.076, 72.8777]); // Mumbai
    setMapZoom(12);
  };

  return (
    <div className="map-demo">
      <div className="demo-header">
        <div className="header-content">
          <Title level="H1">Traffic Map Demo</Title>
          <Text>Interactive traffic visualization with real-time data</Text>
        </div>
        <div className="header-actions">
          <Button design="Transparent" onClick={resetMap}>
            Reset to Bhubaneswar
          </Button>
          <Button design="Transparent" onClick={focusOnDelhi}>
            Focus on Delhi
          </Button>
          <Button design="Transparent" onClick={focusOnMumbai}>
            Focus on Mumbai
          </Button>
        </div>
      </div>

      <div className="demo-content">
        <div className="map-section">
          <TrafficMap
            center={mapCenter}
            zoom={mapZoom}
            height="600px"
            showCameraFeeds={true}
            onIntersectionClick={handleIntersectionClick}
            className="demo-map"
          />
        </div>

        {selectedIntersection && (
          <motion.div
            className="selected-info"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            transition={{ duration: 0.3 }}
          >
            <Card>
              <CardHeader titleText="Selected Intersection">
                <Badge colorScheme="8">Selected</Badge>
              </CardHeader>
              <div className="intersection-details">
                <div className="detail-row">
                  <Text className="label">Name:</Text>
                  <Text className="value">{selectedIntersection.name}</Text>
                </div>
                <div className="detail-row">
                  <Text className="label">Signal State:</Text>
                  <Text className="value">
                    {selectedIntersection.signalState.toUpperCase()}
                  </Text>
                </div>
                <div className="detail-row">
                  <Text className="label">Vehicle Count:</Text>
                  <Text className="value">
                    {selectedIntersection.vehicleCount}
                  </Text>
                </div>
                <div className="detail-row">
                  <Text className="label">Status:</Text>
                  <Badge
                    colorScheme={
                      selectedIntersection.status === "normal"
                        ? "8"
                        : selectedIntersection.status === "warning"
                        ? "2"
                        : "1"
                    }
                  >
                    {selectedIntersection.status.toUpperCase()}
                  </Badge>
                </div>
                <div className="detail-row">
                  <Text className="label">Last Update:</Text>
                  <Text className="value">
                    {selectedIntersection.lastUpdate.toLocaleString()}
                  </Text>
                </div>
                <div className="detail-row">
                  <Text className="label">Coordinates:</Text>
                  <Text className="value">
                    {selectedIntersection.position[0].toFixed(4)},{" "}
                    {selectedIntersection.position[1].toFixed(4)}
                  </Text>
                </div>
              </div>
            </Card>
          </motion.div>
        )}
      </div>

      <div className="demo-features">
        <Card>
          <CardHeader titleText="Map Features">
            <Badge colorScheme="8">Live Demo</Badge>
          </CardHeader>
          <div className="features-grid">
            <div className="feature-item">
              <Text className="feature-title">Real-time Data</Text>
              <Text className="feature-description">
                Live traffic light states and vehicle counts updated every 30
                seconds
              </Text>
            </div>
            <div className="feature-item">
              <Text className="feature-title">Interactive Markers</Text>
              <Text className="feature-description">
                Click on traffic light markers to view detailed information and
                camera feeds
              </Text>
            </div>
            <div className="feature-item">
              <Text className="feature-title">Camera Integration</Text>
              <Text className="feature-description">
                View live camera feeds from multiple angles at each intersection
              </Text>
            </div>
            <div className="feature-item">
              <Text className="feature-title">Responsive Design</Text>
              <Text className="feature-description">
                Optimized for desktop, tablet, and mobile devices
              </Text>
            </div>
            <div className="feature-item">
              <Text className="feature-title">Multiple Cities</Text>
              <Text className="feature-description">
                Switch between different Indian cities to view traffic patterns
              </Text>
            </div>
            <div className="feature-item">
              <Text className="feature-title">Status Indicators</Text>
              <Text className="feature-description">
                Color-coded markers show signal states and traffic conditions
              </Text>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
