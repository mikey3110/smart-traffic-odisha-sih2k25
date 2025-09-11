import { useState, useEffect } from 'react'
import axios from 'axios'
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js'
import { Bar } from 'react-chartjs-2'
import './App.css'

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend)

const API_BASE_URL = 'http://localhost:8000'

function App() {
  const [trafficData, setTrafficData] = useState({})
  const [signalData, setSignalData] = useState({})
  const [intersections, setIntersections] = useState([])
  const [selectedIntersection, setSelectedIntersection] = useState('junction-1')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Fetch traffic data
  const fetchTrafficData = async (intersectionId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/traffic/status/${intersectionId}`)
      setTrafficData(response.data.data || {})
    } catch (err) {
      console.error('Error fetching traffic data:', err)
      setError('Failed to fetch traffic data')
    }
  }

  // Fetch signal data
  const fetchSignalData = async (intersectionId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/signal/status/${intersectionId}`)
      setSignalData(response.data.data || {})
    } catch (err) {
      console.error('Error fetching signal data:', err)
    }
  }

  // Fetch all intersections
  const fetchIntersections = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/intersections`)
      setIntersections(response.data.intersections || [])
    } catch (err) {
      console.error('Error fetching intersections:', err)
    }
  }

  // Auto-refresh data every 5 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      fetchTrafficData(selectedIntersection)
      fetchSignalData(selectedIntersection)
    }, 5000)

    return () => clearInterval(interval)
  }, [selectedIntersection])

  // Initial data fetch
  useEffect(() => {
    fetchIntersections()
    fetchTrafficData(selectedIntersection)
    fetchSignalData(selectedIntersection)
  }, [])

  // Chart data for traffic counts
  const trafficChartData = {
    labels: ['North Lane', 'South Lane', 'East Lane', 'West Lane'],
    datasets: [
      {
        label: 'Vehicle Count',
        data: [
          trafficData.lane_counts?.north_lane || 0,
          trafficData.lane_counts?.south_lane || 0,
          trafficData.lane_counts?.east_lane || 0,
          trafficData.lane_counts?.west_lane || 0
        ],
        backgroundColor: [
          'rgba(255, 99, 132, 0.8)',
          'rgba(54, 162, 235, 0.8)',
          'rgba(255, 205, 86, 0.8)',
          'rgba(75, 192, 192, 0.8)'
        ],
        borderColor: [
          'rgba(255, 99, 132, 1)',
          'rgba(54, 162, 235, 1)',
          'rgba(255, 205, 86, 1)',
          'rgba(75, 192, 192, 1)'
        ],
        borderWidth: 1
      }
    ]
  }

  // Chart data for signal timings
  const signalChartData = {
    labels: ['North Lane', 'South Lane', 'East Lane', 'West Lane'],
    datasets: [
      {
        label: 'Signal Timing (seconds)',
        data: [
          signalData.optimized_timings?.north_lane || 30,
          signalData.optimized_timings?.south_lane || 30,
          signalData.optimized_timings?.east_lane || 30,
          signalData.optimized_timings?.west_lane || 30
        ],
        backgroundColor: [
          'rgba(34, 197, 94, 0.8)',
          'rgba(34, 197, 94, 0.8)',
          'rgba(34, 197, 94, 0.8)',
          'rgba(34, 197, 94, 0.8)'
        ],
        borderColor: [
          'rgba(34, 197, 94, 1)',
          'rgba(34, 197, 94, 1)',
          'rgba(34, 197, 94, 1)',
          'rgba(34, 197, 94, 1)'
        ],
        borderWidth: 1
      }
    ]
  }

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Real-time Traffic Data'
      }
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>🚦 Smart Traffic Management System</h1>
        <p>Real-time AI-powered traffic signal optimization</p>
      </header>

      <div className="dashboard">
        <div className="controls">
          <div className="intersection-selector">
            <label htmlFor="intersection">Select Intersection:</label>
            <select 
              id="intersection"
              value={selectedIntersection} 
              onChange={(e) => setSelectedIntersection(e.target.value)}
            >
              {intersections.map(intersection => (
                <option key={intersection} value={intersection}>
                  {intersection.replace('-', ' ').toUpperCase()}
                </option>
              ))}
            </select>
          </div>
          <div className="status-indicator">
            <span className={`status-dot ${trafficData.timestamp ? 'online' : 'offline'}`}></span>
            <span>{trafficData.timestamp ? 'Live Data' : 'No Data'}</span>
          </div>
        </div>

        {error && (
          <div className="error-message">
            ⚠️ {error}
          </div>
        )}

        <div className="charts-grid">
          <div className="chart-container">
            <h3>📊 Vehicle Counts by Lane</h3>
            <Bar data={trafficChartData} options={chartOptions} />
          </div>

          <div className="chart-container">
            <h3>🚦 Signal Timings (Optimized)</h3>
            <Bar data={signalChartData} options={chartOptions} />
          </div>
        </div>

        <div className="info-grid">
          <div className="info-card">
            <h4>📈 Traffic Statistics</h4>
            <div className="stats">
              <div className="stat">
                <span className="stat-label">Total Vehicles:</span>
                <span className="stat-value">
                  {Object.values(trafficData.lane_counts || {}).reduce((a, b) => a + b, 0)}
                </span>
              </div>
              <div className="stat">
                <span className="stat-label">Average Speed:</span>
                <span className="stat-value">{trafficData.avg_speed || 'N/A'} km/h</span>
              </div>
              <div className="stat">
                <span className="stat-label">Weather:</span>
                <span className="stat-value">{trafficData.weather_condition || 'N/A'}</span>
              </div>
              <div className="stat">
                <span className="stat-label">Last Updated:</span>
                <span className="stat-value">
                  {trafficData.ingested_at ? new Date(trafficData.ingested_at).toLocaleTimeString() : 'N/A'}
                </span>
              </div>
            </div>
          </div>

          <div className="info-card">
            <h4>🤖 AI Optimization</h4>
            <div className="stats">
              <div className="stat">
                <span className="stat-label">Status:</span>
                <span className="stat-value">{signalData.status || 'Default'}</span>
              </div>
              <div className="stat">
                <span className="stat-label">Confidence Score:</span>
                <span className="stat-value">
                  {signalData.confidence_score ? `${(signalData.confidence_score * 100).toFixed(1)}%` : 'N/A'}
                </span>
              </div>
              <div className="stat">
                <span className="stat-label">Expected Improvement:</span>
                <span className="stat-value">
                  {signalData.expected_improvement ? `${signalData.expected_improvement.toFixed(1)}%` : 'N/A'}
                </span>
              </div>
              <div className="stat">
                <span className="stat-label">Optimized At:</span>
                <span className="stat-value">
                  {signalData.optimized_at ? new Date(signalData.optimized_at).toLocaleTimeString() : 'N/A'}
                </span>
              </div>
            </div>
          </div>
        </div>

        <div className="footer">
          <p>🔄 Auto-refreshing every 5 seconds | 🚦 Smart Traffic Management System v1.0</p>
        </div>
      </div>
    </div>
  )
}

export default App
