import React, { useRef, useEffect, useState } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  TimeScale
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import 'chartjs-adapter-date-fns';
import { PerformanceMetrics } from '@/types';
import './RealTimeChart.scss';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  TimeScale
);

interface RealTimeChartProps {
  metrics: PerformanceMetrics | null;
}

interface ChartDataPoint {
  timestamp: Date;
  totalVehicles: number;
  runningVehicles: number;
  waitingVehicles: number;
  averageSpeed: number;
  throughput: number;
}

export function RealTimeChart({ metrics }: RealTimeChartProps) {
  const chartRef = useRef<ChartJS>(null);
  const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
  const [selectedMetric, setSelectedMetric] = useState<string>('totalVehicles');

  // Update chart data when new metrics arrive
  useEffect(() => {
    if (metrics) {
      const newDataPoint: ChartDataPoint = {
        timestamp: metrics.timestamp,
        totalVehicles: metrics.totalVehicles,
        runningVehicles: metrics.runningVehicles,
        waitingVehicles: metrics.waitingVehicles,
        averageSpeed: metrics.averageSpeed,
        throughput: metrics.throughput
      };

      setChartData(prev => {
        const updated = [...prev, newDataPoint];
        // Keep only last 50 data points
        return updated.slice(-50);
      });
    }
  }, [metrics]);

  // Chart configuration
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      intersect: false,
      mode: 'index' as const,
    },
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          usePointStyle: true,
          padding: 20,
          font: {
            size: 12
          }
        }
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: 'white',
        bodyColor: 'white',
        borderColor: 'rgba(255, 255, 255, 0.1)',
        borderWidth: 1,
        cornerRadius: 8,
        displayColors: true,
        callbacks: {
          title: (context: any) => {
            const date = new Date(context[0].parsed.x);
            return date.toLocaleTimeString();
          },
          label: (context: any) => {
            const label = context.dataset.label || '';
            const value = context.parsed.y;
            return `${label}: ${value}`;
          }
        }
      }
    },
    scales: {
      x: {
        type: 'time' as const,
        time: {
          displayFormats: {
            minute: 'HH:mm',
            hour: 'HH:mm'
          }
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)',
          drawBorder: false
        },
        ticks: {
          color: 'var(--sapTextColor)',
          font: {
            size: 11
          }
        }
      },
      y: {
        beginAtZero: true,
        grid: {
          color: 'rgba(0, 0, 0, 0.1)',
          drawBorder: false
        },
        ticks: {
          color: 'var(--sapTextColor)',
          font: {
            size: 11
          }
        }
      }
    },
    elements: {
      point: {
        radius: 3,
        hoverRadius: 6,
        borderWidth: 2
      },
      line: {
        tension: 0.4,
        borderWidth: 2
      }
    },
    animation: {
      duration: 750,
      easing: 'easeInOutQuart' as const
    }
  };

  // Get chart data based on selected metric
  const getChartData = () => {
    const labels = chartData.map(point => point.timestamp);
    
    const datasets = [
      {
        label: 'Total Vehicles',
        data: chartData.map(point => point.totalVehicles),
        borderColor: 'rgb(0, 112, 243)',
        backgroundColor: 'rgba(0, 112, 243, 0.1)',
        fill: true,
        tension: 0.4
      },
      {
        label: 'Running Vehicles',
        data: chartData.map(point => point.runningVehicles),
        borderColor: 'rgb(40, 167, 69)',
        backgroundColor: 'rgba(40, 167, 69, 0.1)',
        fill: false,
        tension: 0.4
      },
      {
        label: 'Waiting Vehicles',
        data: chartData.map(point => point.waitingVehicles),
        borderColor: 'rgb(255, 193, 7)',
        backgroundColor: 'rgba(255, 193, 7, 0.1)',
        fill: false,
        tension: 0.4
      }
    ];

    return {
      labels,
      datasets: selectedMetric === 'all' ? datasets : datasets.filter(d => 
        selectedMetric === 'totalVehicles' ? d.label === 'Total Vehicles' :
        selectedMetric === 'runningVehicles' ? d.label === 'Running Vehicles' :
        selectedMetric === 'waitingVehicles' ? d.label === 'Waiting Vehicles' :
        false
      )
    };
  };

  const metricOptions = [
    { value: 'all', label: 'All Metrics' },
    { value: 'totalVehicles', label: 'Total Vehicles' },
    { value: 'runningVehicles', label: 'Running Vehicles' },
    { value: 'waitingVehicles', label: 'Waiting Vehicles' },
    { value: 'averageSpeed', label: 'Average Speed' },
    { value: 'throughput', label: 'Throughput' }
  ];

  if (chartData.length === 0) {
    return (
      <div className="real-time-chart">
        <div className="chart-header">
          <div className="chart-title">
            <h3>Real-time Performance</h3>
            <p>Live traffic metrics and trends</p>
          </div>
          <div className="chart-controls">
            <select
              value={selectedMetric}
              onChange={(e) => setSelectedMetric(e.target.value)}
              className="metric-select"
            >
              {metricOptions.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
        </div>
        <div className="chart-container">
          <div className="empty-state">
            <div className="empty-icon">📊</div>
            <p>No data available</p>
            <p>Waiting for traffic data...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="real-time-chart">
      <div className="chart-header">
        <div className="chart-title">
          <h3>Real-time Performance</h3>
          <p>Live traffic metrics and trends</p>
        </div>
        <div className="chart-controls">
          <select
            value={selectedMetric}
            onChange={(e) => setSelectedMetric(e.target.value)}
            className="metric-select"
          >
            {metricOptions.map(option => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
          <div className="chart-legend">
            <div className="legend-item">
              <div className="legend-color" style={{ backgroundColor: 'rgb(0, 112, 243)' }}></div>
              <span>Total</span>
            </div>
            <div className="legend-item">
              <div className="legend-color" style={{ backgroundColor: 'rgb(40, 167, 69)' }}></div>
              <span>Running</span>
            </div>
            <div className="legend-item">
              <div className="legend-color" style={{ backgroundColor: 'rgb(255, 193, 7)' }}></div>
              <span>Waiting</span>
            </div>
          </div>
        </div>
      </div>
      
      <div className="chart-container">
        <Line
          ref={chartRef}
          data={getChartData()}
          options={chartOptions}
        />
      </div>
      
      <div className="chart-footer">
        <div className="chart-stats">
          <div className="stat">
            <span className="stat-label">Data Points:</span>
            <span className="stat-value">{chartData.length}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Last Update:</span>
            <span className="stat-value">
              {chartData[chartData.length - 1]?.timestamp.toLocaleTimeString() || 'Never'}
            </span>
          </div>
          <div className="stat">
            <span className="stat-label">Status:</span>
            <span className="stat-value live">● Live</span>
          </div>
        </div>
      </div>
    </div>
  );
}
