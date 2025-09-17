import React from "react";
import { motion } from "framer-motion";
import { ProgressIndicator, Text, Icon } from "@ui5/webcomponents-react";
import "./LoadingSpinner.scss";

interface LoadingSpinnerProps {
  size?: "small" | "medium" | "large";
  text?: string;
  overlay?: boolean;
  className?: string;
}

export function LoadingSpinner({
  size = "medium",
  text = "Loading...",
  overlay = false,
  className = "",
}: LoadingSpinnerProps) {
  const sizeClasses = {
    small: "spinner-small",
    medium: "spinner-medium",
    large: "spinner-large",
  };

  const content = (
    <div className={`loading-spinner ${sizeClasses[size]} ${className}`}>
      <motion.div
        className="spinner-container"
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.3 }}
      >
        <motion.div
          className="spinner"
          animate={{ rotate: 360 }}
          transition={{
            duration: 1,
            repeat: Infinity,
            ease: "linear",
          }}
        >
          <Icon name="loading" />
        </motion.div>

        {text && (
          <motion.div
            className="spinner-text"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.1 }}
          >
            <Text>{text}</Text>
          </motion.div>
        )}
      </motion.div>
    </div>
  );

  if (overlay) {
    return (
      <div className="loading-overlay">
        <motion.div
          className="overlay-backdrop"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3 }}
        />
        <div className="overlay-content">{content}</div>
      </div>
    );
  }

  return content;
}

// Full page loading spinner
export function FullPageLoadingSpinner({
  text = "Loading...",
}: {
  text?: string;
}) {
  return (
    <div className="full-page-loading">
      <motion.div
        className="loading-content"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <motion.div
          className="logo-container"
          animate={{
            scale: [1, 1.1, 1],
            rotate: [0, 5, -5, 0],
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        >
          <Icon name="traffic-light" />
        </motion.div>

        <motion.div
          className="loading-spinner-container"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <LoadingSpinner size="large" text={text} />
        </motion.div>

        <motion.div
          className="loading-message"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          <Text>Smart Traffic Management System</Text>
          <Text>Initializing...</Text>
        </motion.div>
      </motion.div>
    </div>
  );
}

// Inline loading spinner
export function InlineLoadingSpinner({ text }: { text?: string }) {
  return (
    <div className="inline-loading">
      <motion.div
        className="inline-spinner"
        animate={{ rotate: 360 }}
        transition={{
          duration: 1,
          repeat: Infinity,
          ease: "linear",
        }}
      >
        <Icon name="loading" />
      </motion.div>
      {text && <Text className="inline-text">{text}</Text>}
    </div>
  );
}

// Button loading spinner
export function ButtonLoadingSpinner() {
  return (
    <motion.div
      className="button-spinner"
      animate={{ rotate: 360 }}
      transition={{
        duration: 1,
        repeat: Infinity,
        ease: "linear",
      }}
    >
      <Icon name="loading" />
    </motion.div>
  );
}

// Skeleton loading components
export function SkeletonCard({ className = "" }: { className?: string }) {
  return (
    <motion.div
      className={`skeleton-card ${className}`}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
    >
      <div className="skeleton-header">
        <div className="skeleton-line skeleton-title" />
        <div className="skeleton-line skeleton-subtitle" />
      </div>
      <div className="skeleton-content">
        <div className="skeleton-line skeleton-text" />
        <div className="skeleton-line skeleton-text" />
        <div className="skeleton-line skeleton-text short" />
      </div>
    </motion.div>
  );
}

export function SkeletonTable({
  rows = 5,
  className = "",
}: {
  rows?: number;
  className?: string;
}) {
  return (
    <motion.div
      className={`skeleton-table ${className}`}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
    >
      <div className="skeleton-table-header">
        {Array.from({ length: 4 }).map((_, index) => (
          <div key={index} className="skeleton-line skeleton-header-cell" />
        ))}
      </div>
      <div className="skeleton-table-body">
        {Array.from({ length: rows }).map((_, rowIndex) => (
          <div key={rowIndex} className="skeleton-table-row">
            {Array.from({ length: 4 }).map((_, cellIndex) => (
              <div key={cellIndex} className="skeleton-line skeleton-cell" />
            ))}
          </div>
        ))}
      </div>
    </motion.div>
  );
}

// Loading states for different components
export function ChartLoadingSkeleton() {
  return (
    <div className="chart-loading-skeleton">
      <div className="chart-header">
        <div className="skeleton-line skeleton-title" />
        <div className="skeleton-line skeleton-subtitle" />
      </div>
      <div className="chart-content">
        <div className="skeleton-chart">
          <div className="skeleton-bars">
            {Array.from({ length: 8 }).map((_, index) => (
              <motion.div
                key={index}
                className="skeleton-bar"
                initial={{ height: 0 }}
                animate={{ height: "100%" }}
                transition={{
                  duration: 0.5,
                  delay: index * 0.1,
                  repeat: Infinity,
                  repeatType: "reverse",
                }}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export function ListLoadingSkeleton({ items = 3 }: { items?: number }) {
  return (
    <div className="list-loading-skeleton">
      {Array.from({ length: items }).map((_, index) => (
        <motion.div
          key={index}
          className="skeleton-list-item"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.3, delay: index * 0.1 }}
        >
          <div className="skeleton-avatar" />
          <div className="skeleton-content">
            <div className="skeleton-line skeleton-title" />
            <div className="skeleton-line skeleton-text" />
          </div>
        </motion.div>
      ))}
    </div>
  );
}
