import React from 'react';

function App() {
  return (
    <div style={{ 
      padding: '50px', 
      textAlign: 'center', 
      fontFamily: 'Arial, sans-serif',
      backgroundColor: '#f0f0f0',
      minHeight: '100vh'
    }}>
      <h1 style={{ color: '#333', fontSize: '3rem', marginBottom: '20px' }}>
        🚦 Smart Traffic Management System
      </h1>
      
      <div style={{ 
        backgroundColor: 'white', 
        padding: '30px', 
        borderRadius: '10px', 
        boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
        maxWidth: '800px',
        margin: '0 auto'
      }}>
        <h2 style={{ color: '#0070f3', marginBottom: '20px' }}>✅ System Status</h2>
        
        <div style={{ marginBottom: '20px' }}>
          <p><strong>Backend API:</strong> <a href="http://localhost:8000" target="_blank" style={{ color: '#0070f3' }}>http://localhost:8000</a></p>
          <p><strong>API Docs:</strong> <a href="http://localhost:8000/docs" target="_blank" style={{ color: '#0070f3' }}>http://localhost:8000/docs</a></p>
          <p><strong>Health Check:</strong> <a href="http://localhost:8000/health" target="_blank" style={{ color: '#0070f3' }}>http://localhost:8000/health</a></p>
        </div>

        <div style={{ marginBottom: '30px' }}>
          <h3 style={{ color: '#28a745' }}>🎯 Features Available:</h3>
          <ul style={{ textAlign: 'left', listStyle: 'none', padding: 0 }}>
            <li>✅ Real-time traffic monitoring</li>
            <li>✅ ML-powered optimization</li>
            <li>✅ SUMO simulation</li>
            <li>✅ Performance analytics</li>
            <li>✅ API endpoints</li>
          </ul>
        </div>

        <button 
          onClick={() => window.open('http://localhost:8000/docs', '_blank')}
          style={{
            padding: '15px 30px',
            fontSize: '18px',
            backgroundColor: '#0070f3',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            cursor: 'pointer',
            margin: '10px',
            fontWeight: 'bold'
          }}
        >
          View API Documentation
        </button>
      </div>
    </div>
  );
}

export default App;