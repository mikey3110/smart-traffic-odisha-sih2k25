import React, { Component, ErrorInfo, ReactNode } from "react";
import {
  Card,
  CardHeader,
  Text,
  Title,
  Button,
  Icon,
  MessageStrip,
  MessageStripDesign,
} from "@ui5/webcomponents-react";
import { AppError } from "@/types";
import "./ErrorBoundary.scss";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: AppError) => void;
}

interface State {
  hasError: boolean;
  error: AppError | null;
  errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    // Update state so the next render will show the fallback UI
    return {
      hasError: true,
      error: {
        code: "COMPONENT_ERROR",
        message: error.message,
        timestamp: new Date(),
        stack: error.stack,
      },
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Log error to console
    console.error("ErrorBoundary caught an error:", error, errorInfo);

    // Update state with error info
    this.setState({
      error: {
        code: "COMPONENT_ERROR",
        message: error.message,
        timestamp: new Date(),
        stack: error.stack,
        details: errorInfo,
      },
      errorInfo,
    });

    // Call onError callback if provided
    if (this.props.onError) {
      this.props.onError({
        code: "COMPONENT_ERROR",
        message: error.message,
        timestamp: new Date(),
        stack: error.stack,
        details: errorInfo,
      });
    }

    // Log to external service (e.g., Sentry)
    this.logErrorToService(error, errorInfo);
  }

  private logErrorToService = (error: Error, errorInfo: ErrorInfo) => {
    // Here you would typically send the error to an external service
    // For example, Sentry, LogRocket, or your own error tracking service
    console.log("Logging error to external service:", {
      error: error.message,
      stack: error.stack,
      componentStack: errorInfo.componentStack,
      timestamp: new Date().toISOString(),
    });
  };

  private handleRetry = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  private handleReload = () => {
    window.location.reload();
  };

  private handleReport = () => {
    if (this.state.error) {
      // Here you would typically open a bug report form or send the error details
      const errorReport = {
        message: this.state.error.message,
        stack: this.state.error.stack,
        componentStack: this.state.errorInfo?.componentStack,
        timestamp: this.state.error.timestamp.toISOString(),
        userAgent: navigator.userAgent,
        url: window.location.href,
      };

      console.log("Error report:", errorReport);

      // You could send this to your backend or open a support ticket
      alert(
        "Error report generated. Please contact support with this information."
      );
    }
  };

  render() {
    if (this.state.hasError) {
      // Custom fallback UI
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // Default error UI
      return (
        <div className="error-boundary">
          <Card className="error-card">
            <CardHeader>
              <div className="error-header">
                <Icon name="error" />
                <Title level="H2">Something went wrong</Title>
              </div>
            </CardHeader>

            <div className="error-content">
              <MessageStrip
                design={MessageStripDesign.Negative}
                className="error-message"
              >
                <div className="error-details">
                  <Text className="error-title">
                    An unexpected error occurred
                  </Text>
                  <Text className="error-description">
                    We're sorry, but something went wrong. Our team has been
                    notified and is working to fix the issue.
                  </Text>
                </div>
              </MessageStrip>

              {process.env.NODE_ENV === "development" && this.state.error && (
                <div className="error-debug">
                  <Title level="H4">Error Details (Development)</Title>
                  <div className="debug-info">
                    <Text className="debug-label">Message:</Text>
                    <Text className="debug-value">
                      {this.state.error.message}
                    </Text>
                  </div>
                  <div className="debug-info">
                    <Text className="debug-label">Code:</Text>
                    <Text className="debug-value">{this.state.error.code}</Text>
                  </div>
                  <div className="debug-info">
                    <Text className="debug-label">Timestamp:</Text>
                    <Text className="debug-value">
                      {this.state.error.timestamp.toLocaleString()}
                    </Text>
                  </div>
                  {this.state.error.stack && (
                    <div className="debug-info">
                      <Text className="debug-label">Stack Trace:</Text>
                      <pre className="debug-stack">
                        {this.state.error.stack}
                      </pre>
                    </div>
                  )}
                </div>
              )}

              <div className="error-actions">
                <Button
                  design="Emphasized"
                  onClick={this.handleRetry}
                  className="retry-button"
                >
                  <Icon name="refresh" />
                  Try Again
                </Button>

                <Button
                  design="Transparent"
                  onClick={this.handleReload}
                  className="reload-button"
                >
                  <Icon name="refresh" />
                  Reload Page
                </Button>

                <Button
                  design="Transparent"
                  onClick={this.handleReport}
                  className="report-button"
                >
                  <Icon name="message" />
                  Report Issue
                </Button>
              </div>

              <div className="error-help">
                <Text>
                  If the problem persists, please contact support or try
                  refreshing the page.
                </Text>
              </div>
            </div>
          </Card>
        </div>
      );
    }

    return this.props.children;
  }
}

// HOC for wrapping components with error boundary
export function withErrorBoundary<P extends object>(
  Component: React.ComponentType<P>,
  errorBoundaryProps?: Omit<Props, "children">
) {
  const WrappedComponent = (props: P) => (
    <ErrorBoundary {...errorBoundaryProps}>
      <Component {...props} />
    </ErrorBoundary>
  );

  WrappedComponent.displayName = `withErrorBoundary(${
    Component.displayName || Component.name
  })`;

  return WrappedComponent;
}

// Hook for error handling
export function useErrorHandler() {
  const [error, setError] = React.useState<AppError | null>(null);

  const resetError = React.useCallback(() => {
    setError(null);
  }, []);

  const handleError = React.useCallback((error: Error | AppError) => {
    const appError: AppError = {
      code: "error" in error && "code" in error ? error.code : "UNKNOWN_ERROR",
      message: error.message,
      timestamp: new Date(),
      stack: error.stack,
    };

    setError(appError);
  }, []);

  React.useEffect(() => {
    if (error) {
      // Log error to console
      console.error("Error caught by useErrorHandler:", error);

      // You could also send to external error tracking service here
    }
  }, [error]);

  return { error, handleError, resetError };
}
