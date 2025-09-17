describe('Smart Traffic Dashboard E2E Tests', () => {
  beforeEach(() => {
    // Mock API responses
    cy.mockApi('GET', '/traffic/lights', {
      lights: [
        {
          id: 'intersection_1',
          name: 'Main Street & First Avenue',
          location: { lat: 20.2961, lng: 85.8245 },
          status: 'normal',
          currentPhase: 0,
          waitingTime: 45,
          lastUpdate: '2024-01-15T10:30:00Z',
        },
        {
          id: 'intersection_2',
          name: 'Second Street & Park Avenue',
          location: { lat: 20.3000, lng: 85.8300 },
          status: 'normal',
          currentPhase: 2,
          waitingTime: 30,
          lastUpdate: '2024-01-15T10:30:00Z',
        },
      ],
    });

    cy.mockApi('GET', '/api/cv/counts', [
      { intersectionId: 'intersection_1', vehicleCount: 12 },
      { intersectionId: 'intersection_2', vehicleCount: 8 },
    ]);

    cy.mockApi('GET', '/traffic/analytics', {
      totalVehicles: 20,
      averageSpeed: 25.5,
      totalWaitingTime: 75.0,
      totalCo2Emission: 5.2,
      throughput: 45,
      peakHours: [
        { hour: 8, count: 45 },
        { hour: 9, count: 52 },
        { hour: 17, count: 48 },
        { hour: 18, count: 55 },
      ],
      vehicleTypeDistribution: [
        { type: 'car', count: 15 },
        { type: 'motorcycle', count: 3 },
        { type: 'truck', count: 2 },
      ],
      intersectionPerformance: [
        { id: 'intersection_1', name: 'Main Street & First Avenue', efficiency: 85 },
        { id: 'intersection_2', name: 'Second Street & Park Avenue', efficiency: 92 },
      ],
    });

    cy.mockApi('GET', '/config/status', {
      status: 'online',
      uptime: 86400,
      lastUpdate: '2024-01-15T10:30:00Z',
      services: [
        { name: 'Traffic Management API', status: 'healthy', responseTime: 45 },
        { name: 'Database', status: 'healthy', responseTime: 12 },
        { name: 'WebSocket Service', status: 'healthy', responseTime: 8 },
        { name: 'File Storage', status: 'healthy', responseTime: 25 },
      ],
    });

    cy.visit('/');
  });

  it('should load the dashboard successfully', () => {
    cy.get('[data-testid="dashboard"]').should('be.visible');
    cy.get('h1').should('contain', 'Smart Traffic Management System');
  });

  it('should display traffic map with markers', () => {
    cy.waitForMap();
    cy.get('[data-testid="marker"]').should('have.length', 2);
  });

  it('should show intersection details when marker is clicked', () => {
    cy.waitForMap();
    cy.get('[data-testid="marker"]').first().click();
    cy.get('[data-testid="popup"]').should('be.visible');
    cy.get('[data-testid="popup"]').should('contain', 'Main Street & First Avenue');
  });

  it('should display performance metrics', () => {
    cy.get('[data-testid="performance-metrics"]').should('be.visible');
    cy.get('[data-testid="performance-metrics"]').should('contain', 'Total Vehicles');
    cy.get('[data-testid="performance-metrics"]').should('contain', 'Average Speed');
    cy.get('[data-testid="performance-metrics"]').should('contain', 'Total Waiting Time');
  });

  it('should show system status', () => {
    cy.get('[data-testid="system-status"]').should('be.visible');
    cy.get('[data-testid="system-status"]').should('contain', 'System Status');
    cy.get('[data-testid="system-status"]').should('contain', 'online');
  });

  it('should display analytics charts', () => {
    cy.get('[data-testid="analytics-charts"]').should('be.visible');
    cy.get('[data-testid="peak-hours-chart"]').should('be.visible');
    cy.get('[data-testid="vehicle-distribution-chart"]').should('be.visible');
  });

  it('should refresh data when refresh button is clicked', () => {
    cy.get('[data-testid="refresh-button"]').click();
    cy.wait('@mockApi');
    cy.get('[data-testid="last-update"]').should('be.visible');
  });

  it('should handle camera feed overlay', () => {
    cy.waitForMap();
    cy.get('[data-testid="marker"]').first().click();
    cy.get('[data-testid="camera-feed-overlay"]').should('be.visible');
    cy.get('[data-testid="camera-feed-overlay"]').should('contain', 'Camera for Main Street & First Avenue');
    
    cy.get('[data-testid="camera-feed-overlay"]').within(() => {
      cy.get('button').contains('Close').click();
    });
    cy.get('[data-testid="camera-feed-overlay"]').should('not.exist');
  });

  it('should display map legend', () => {
    cy.waitForMap();
    cy.get('[data-testid="map-legend"]').should('be.visible');
    cy.get('[data-testid="map-legend"]').should('contain', 'Legend');
    cy.get('[data-testid="map-legend"]').should('contain', 'Green Signal');
    cy.get('[data-testid="map-legend"]').should('contain', 'Yellow Signal');
    cy.get('[data-testid="map-legend"]').should('contain', 'Red Signal');
  });

  it('should be responsive on mobile viewport', () => {
    cy.viewport(375, 667);
    cy.get('[data-testid="dashboard"]').should('be.visible');
    cy.waitForMap();
    cy.get('[data-testid="marker"]').should('have.length', 2);
  });

  it('should handle API errors gracefully', () => {
    cy.intercept('GET', '/traffic/lights', { statusCode: 500, body: { error: 'Server Error' } }).as('apiError');
    cy.visit('/');
    cy.wait('@apiError');
    cy.get('[data-testid="error-message"]').should('be.visible');
    cy.get('[data-testid="retry-button"]').should('be.visible');
  });

  it('should show loading states', () => {
    cy.intercept('GET', '/traffic/lights', { delay: 2000, fixture: 'traffic-lights.json' }).as('slowApi');
    cy.visit('/');
    cy.get('[data-testid="loading"]').should('be.visible');
    cy.wait('@slowApi');
    cy.get('[data-testid="loading"]').should('not.exist');
  });
});