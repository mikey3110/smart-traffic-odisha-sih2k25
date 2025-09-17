import React from 'react';
import { TrafficMap } from '../../src/components/map/TrafficMap';

describe('TrafficMap Component', () => {
  const mockIntersections = [
    {
      id: 'intersection_1',
      name: 'Main Street & First Avenue',
      position: [20.2961, 85.8245] as [number, number],
      signalState: 'green' as const,
      vehicleCount: 12,
      lastUpdate: new Date('2024-01-15T10:30:00Z'),
      status: 'normal' as const,
    },
    {
      id: 'intersection_2',
      name: 'Second Street & Park Avenue',
      position: [20.3000, 85.8300] as [number, number],
      signalState: 'red' as const,
      vehicleCount: 8,
      lastUpdate: new Date('2024-01-15T10:30:00Z'),
      status: 'warning' as const,
    },
  ];

  beforeEach(() => {
    // Mock the useMapData hook
    cy.intercept('GET', '/traffic/lights', { body: { lights: [] } }).as('getTrafficLights');
    cy.intercept('GET', '/api/cv/counts', { body: [] }).as('getVehicleCounts');
  });

  it('renders with default props', () => {
    cy.mount(<TrafficMap />);
    cy.get('[data-testid="map-container"]').should('be.visible');
    cy.get('[data-testid="tile-layer"]').should('be.visible');
  });

  it('renders with custom height', () => {
    cy.mount(<TrafficMap height="600px" />);
    cy.get('[data-testid="map-container"]').parent().should('have.css', 'height', '600px');
  });

  it('renders with custom className', () => {
    cy.mount(<TrafficMap className="custom-class" />);
    cy.get('.traffic-map.custom-class').should('exist');
  });

  it('shows loading state', () => {
    cy.intercept('GET', '/traffic/lights', { delay: 2000, body: { lights: [] } }).as('slowApi');
    cy.mount(<TrafficMap />);
    cy.get('[data-testid="loading"]').should('be.visible');
    cy.wait('@slowApi');
    cy.get('[data-testid="loading"]').should('not.exist');
  });

  it('shows error state', () => {
    cy.intercept('GET', '/traffic/lights', { statusCode: 500, body: { error: 'Server Error' } }).as('apiError');
    cy.mount(<TrafficMap />);
    cy.wait('@apiError');
    cy.get('[data-testid="error-message"]').should('be.visible');
    cy.get('[data-testid="retry-button"]').should('be.visible');
  });

  it('displays map legend', () => {
    cy.mount(<TrafficMap />);
    cy.get('[data-testid="map-legend"]').should('be.visible');
    cy.get('[data-testid="map-legend"]').should('contain', 'Legend');
    cy.get('[data-testid="map-legend"]').should('contain', 'Green Signal');
    cy.get('[data-testid="map-legend"]').should('contain', 'Yellow Signal');
    cy.get('[data-testid="map-legend"]').should('contain', 'Red Signal');
  });

  it('calls onIntersectionClick when marker is clicked', () => {
    const onIntersectionClick = cy.stub();
    cy.mount(<TrafficMap onIntersectionClick={onIntersectionClick} />);
    
    // Mock the useMapData hook to return test data
    cy.window().then((win) => {
      win.useMapData = () => ({
        intersections: mockIntersections,
        lastUpdate: new Date(),
        loading: false,
        error: null,
        refresh: cy.stub(),
      });
    });

    cy.get('[data-testid="marker"]').first().click();
    cy.wrap(onIntersectionClick).should('have.been.calledWith', mockIntersections[0]);
  });

  it('shows camera feed overlay when enabled', () => {
    cy.mount(<TrafficMap showCameraFeeds={true} />);
    
    // Mock the useMapData hook
    cy.window().then((win) => {
      win.useMapData = () => ({
        intersections: mockIntersections,
        lastUpdate: new Date(),
        loading: false,
        error: null,
        refresh: cy.stub(),
      });
    });

    cy.get('[data-testid="marker"]').first().click();
    cy.get('[data-testid="camera-feed-overlay"]').should('be.visible');
    cy.get('[data-testid="camera-feed-overlay"]').should('contain', 'Camera for Main Street & First Avenue');
  });

  it('hides camera feed overlay when close button is clicked', () => {
    cy.mount(<TrafficMap showCameraFeeds={true} />);
    
    // Mock the useMapData hook
    cy.window().then((win) => {
      win.useMapData = () => ({
        intersections: mockIntersections,
        lastUpdate: new Date(),
        loading: false,
        error: null,
        refresh: cy.stub(),
      });
    });

    cy.get('[data-testid="marker"]').first().click();
    cy.get('[data-testid="camera-feed-overlay"]').should('be.visible');
    
    cy.get('[data-testid="camera-feed-overlay"]').within(() => {
      cy.get('button').contains('Close').click();
    });
    
    cy.get('[data-testid="camera-feed-overlay"]').should('not.exist');
  });

  it('displays last update time', () => {
    const lastUpdate = new Date('2024-01-15T10:30:00Z');
    cy.mount(<TrafficMap />);
    
    // Mock the useMapData hook
    cy.window().then((win) => {
      win.useMapData = () => ({
        intersections: mockIntersections,
        lastUpdate,
        loading: false,
        error: null,
        refresh: cy.stub(),
      });
    });

    cy.get('[data-testid="last-update"]').should('contain', lastUpdate.toLocaleTimeString());
  });

  it('calls refresh when refresh button is clicked', () => {
    const refresh = cy.stub();
    cy.mount(<TrafficMap />);
    
    // Mock the useMapData hook
    cy.window().then((win) => {
      win.useMapData = () => ({
        intersections: mockIntersections,
        lastUpdate: new Date(),
        loading: false,
        error: null,
        refresh,
      });
    });

    cy.get('[data-testid="refresh-button"]').click();
    cy.wrap(refresh).should('have.been.called');
  });
});
