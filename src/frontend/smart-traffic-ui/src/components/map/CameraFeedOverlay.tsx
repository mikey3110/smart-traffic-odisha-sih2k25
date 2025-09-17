import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Card,
  CardHeader,
  Text,
  Title,
  Icon,
  Button,
  Badge,
  ProgressIndicator,
} from "@ui5/webcomponents-react";
import { MapIntersection } from "@/hooks/useMapData";
import "./CameraFeedOverlay.scss";

interface CameraFeedOverlayProps {
  intersection: MapIntersection;
  onClose: () => void;
  className?: string;
}

interface CameraFeed {
  id: string;
  name: string;
  url: string;
  position: "north" | "south" | "east" | "west";
  status: "online" | "offline" | "error";
  lastFrame: string;
}

export function CameraFeedOverlay({
  intersection,
  onClose,
  className = "",
}: CameraFeedOverlayProps) {
  const [feeds, setFeeds] = useState<CameraFeed[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedFeed, setSelectedFeed] = useState<CameraFeed | null>(null);
  const [fullscreen, setFullscreen] = useState(false);

  useEffect(() => {
    loadCameraFeeds();
  }, [intersection.id]);

  const loadCameraFeeds = async () => {
    setLoading(true);
    try {
      // Simulate API call to get camera feeds for intersection
      const mockFeeds: CameraFeed[] = [
        {
          id: `${intersection.id}_north`,
          name: "North Approach",
          url: `rtsp://camera-server.com/${intersection.id}/north`,
          position: "north",
          status: "online",
          lastFrame: `https://camera-server.com/${intersection.id}/north/latest.jpg`,
        },
        {
          id: `${intersection.id}_south`,
          name: "South Approach",
          url: `rtsp://camera-server.com/${intersection.id}/south`,
          position: "south",
          status: "online",
          lastFrame: `https://camera-server.com/${intersection.id}/south/latest.jpg`,
        },
        {
          id: `${intersection.id}_east`,
          name: "East Approach",
          url: `rtsp://camera-server.com/${intersection.id}/east`,
          position: "east",
          status: "offline",
          lastFrame: "",
        },
        {
          id: `${intersection.id}_west`,
          name: "West Approach",
          url: `rtsp://camera-server.com/${intersection.id}/west`,
          position: "west",
          status: "online",
          lastFrame: `https://camera-server.com/${intersection.id}/west/latest.jpg`,
        },
      ];

      setFeeds(mockFeeds);
      setSelectedFeed(mockFeeds[0]); // Select first available feed
    } catch (error) {
      console.error("Failed to load camera feeds:", error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "online":
        return "video";
      case "offline":
        return "video-off";
      case "error":
        return "error";
      default:
        return "question-mark";
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "online":
        return "var(--sapPositiveColor)";
      case "offline":
        return "var(--sapNeutralColor)";
      case "error":
        return "var(--sapNegativeColor)";
      default:
        return "var(--sapNeutralColor)";
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "online":
        return <Badge colorScheme="8">Online</Badge>;
      case "offline":
        return <Badge colorScheme="9">Offline</Badge>;
      case "error":
        return <Badge colorScheme="1">Error</Badge>;
      default:
        return <Badge colorScheme="9">Unknown</Badge>;
    }
  };

  const handleFeedSelect = (feed: CameraFeed) => {
    if (feed.status === "online") {
      setSelectedFeed(feed);
    }
  };

  const toggleFullscreen = () => {
    setFullscreen(!fullscreen);
  };

  if (loading) {
    return (
      <Card className={`camera-feed-overlay loading ${className}`}>
        <CardHeader titleText="Camera Feeds">
          <ProgressIndicator value={undefined} />
        </CardHeader>
        <div className="loading-state">
          <Text>Loading camera feeds...</Text>
        </div>
      </Card>
    );
  }

  return (
    <AnimatePresence>
      <motion.div
        className={`camera-feed-overlay ${
          fullscreen ? "fullscreen" : ""
        } ${className}`}
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.9 }}
        transition={{ duration: 0.3 }}
      >
        <Card className="camera-card">
          <CardHeader titleText={`Camera Feeds - ${intersection.name}`}>
            <div className="header-actions">
              <Button
                design="Transparent"
                icon={fullscreen ? "exit-full-screen" : "full-screen"}
                onClick={toggleFullscreen}
              />
              <Button design="Transparent" icon="decline" onClick={onClose} />
            </div>
          </CardHeader>

          <div className="camera-content">
            {/* Main Video Display */}
            <div className="main-video">
              {selectedFeed ? (
                <div className="video-container">
                  {selectedFeed.status === "online" ? (
                    <div className="video-frame">
                      <img
                        src={selectedFeed.lastFrame}
                        alt={`${selectedFeed.name} camera feed`}
                        className="video-image"
                        onError={(e) => {
                          // Fallback to placeholder if image fails to load
                          (e.target as HTMLImageElement).src =
                            "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzIwIiBoZWlnaHQ9IjI0MCIgdmlld0JveD0iMCAwIDMyMCAyNDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSIzMjAiIGhlaWdodD0iMjQwIiBmaWxsPSIjZjVmNWY1Ii8+Cjx0ZXh0IHg9IjE2MCIgeT0iMTIwIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmaWxsPSIjOTk5IiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTQiPkNhbWVyYSBmZWVkIHVuYXZhaWxhYmxlPC90ZXh0Pgo8L3N2Zz4K";
                        }}
                      />
                      <div className="video-overlay">
                        <div className="video-info">
                          <Text className="feed-name">{selectedFeed.name}</Text>
                          {getStatusBadge(selectedFeed.status)}
                        </div>
                        <div className="video-controls">
                          <Button
                            design="Transparent"
                            icon="refresh"
                            onClick={() => {
                              // Refresh the image
                              const img = document.querySelector(
                                ".video-image"
                              ) as HTMLImageElement;
                              if (img) {
                                img.src =
                                  selectedFeed.lastFrame + "?t=" + Date.now();
                              }
                            }}
                          />
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="video-offline">
                      <Icon name={getStatusIcon(selectedFeed.status)} />
                      <Text>Camera Offline</Text>
                    </div>
                  )}
                </div>
              ) : (
                <div className="no-feed">
                  <Icon name="video-off" />
                  <Text>No camera feed selected</Text>
                </div>
              )}
            </div>

            {/* Camera Feed Thumbnails */}
            <div className="feed-thumbnails">
              <div className="thumbnails-header">
                <Text>Available Cameras</Text>
              </div>
              <div className="thumbnails-grid">
                {feeds.map((feed) => (
                  <motion.div
                    key={feed.id}
                    className={`thumbnail ${
                      selectedFeed?.id === feed.id ? "selected" : ""
                    } ${feed.status}`}
                    onClick={() => handleFeedSelect(feed)}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <div className="thumbnail-image">
                      {feed.status === "online" && feed.lastFrame ? (
                        <img
                          src={feed.lastFrame}
                          alt={`${feed.name} thumbnail`}
                          className="thumbnail-img"
                        />
                      ) : (
                        <div className="thumbnail-placeholder">
                          <Icon name={getStatusIcon(feed.status)} />
                        </div>
                      )}
                      <div className="thumbnail-overlay">
                        <Icon name={getStatusIcon(feed.status)} />
                      </div>
                    </div>
                    <div className="thumbnail-info">
                      <Text className="thumbnail-name">{feed.name}</Text>
                      <div className="thumbnail-status">
                        <div
                          className="status-indicator"
                          style={{
                            backgroundColor: getStatusColor(feed.status),
                          }}
                        />
                        <Text className="status-text">{feed.status}</Text>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>

            {/* Intersection Info */}
            <div className="intersection-info">
              <div className="info-section">
                <Text className="section-title">Intersection Details</Text>
                <div className="info-grid">
                  <div className="info-item">
                    <Icon name="traffic-light" />
                    <div className="info-content">
                      <Text className="info-label">Signal State</Text>
                      <Text className="info-value">
                        {intersection.signalState.toUpperCase()}
                      </Text>
                    </div>
                  </div>
                  <div className="info-item">
                    <Icon name="car" />
                    <div className="info-content">
                      <Text className="info-label">Vehicle Count</Text>
                      <Text className="info-value">
                        {intersection.vehicleCount}
                      </Text>
                    </div>
                  </div>
                  <div className="info-item">
                    <Icon name="time" />
                    <div className="info-content">
                      <Text className="info-label">Last Update</Text>
                      <Text className="info-value">
                        {intersection.lastUpdate.toLocaleTimeString()}
                      </Text>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </Card>
      </motion.div>
    </AnimatePresence>
  );
}
