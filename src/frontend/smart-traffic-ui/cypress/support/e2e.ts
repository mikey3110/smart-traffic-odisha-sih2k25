// Import commands.js using ES2015 syntax:
import './commands';

// Alternatively you can use CommonJS syntax:
// require('./commands')

// Hide fetch/XHR requests from command log
Cypress.on('window:before:load', (win) => {
  // Mock IntersectionObserver
  win.IntersectionObserver = class IntersectionObserver {
    constructor() {}
    disconnect() {}
    observe() {}
    unobserve() {}
  };

  // Mock ResizeObserver
  win.ResizeObserver = class ResizeObserver {
    constructor() {}
    disconnect() {}
    observe() {}
    unobserve() {}
  };
});

// Mock Leaflet for E2E tests
Cypress.on('window:before:load', (win) => {
  win.L = {
    map: () => ({
      setView: cy.stub(),
      addLayer: cy.stub(),
      removeLayer: cy.stub(),
      on: cy.stub(),
      off: cy.stub(),
      invalidateSize: cy.stub(),
    }),
    tileLayer: () => ({
      addTo: cy.stub(),
    }),
    marker: () => ({
      addTo: cy.stub(),
      bindPopup: cy.stub(),
      on: cy.stub(),
    }),
    divIcon: () => ({}),
    Icon: {
      Default: {
        mergeOptions: cy.stub(),
      },
    },
  };
});
