/// <reference types="cypress" />

declare global {
  namespace Cypress {
    interface Chainable {
      /**
       * Custom command to select DOM element by data-cy attribute.
       * @example cy.dataCy('greeting')
       */
      dataCy(value: string): Chainable<JQuery<HTMLElement>>
      
      /**
       * Custom command to mock API responses
       * @example cy.mockApi('GET', '/api/traffic/lights', { lights: [] })
       */
      mockApi(method: string, url: string, response: any): Chainable<void>
      
      /**
       * Custom command to wait for map to load
       * @example cy.waitForMap()
       */
      waitForMap(): Chainable<void>
    }
  }
}

Cypress.Commands.add('dataCy', (value) => {
  return cy.get(`[data-cy=${value}]`)
})

Cypress.Commands.add('mockApi', (method, url, response) => {
  cy.intercept(method, url, {
    statusCode: 200,
    body: response,
  }).as('mockApi')
})

Cypress.Commands.add('waitForMap', () => {
  cy.get('[data-testid="map-container"]', { timeout: 10000 }).should('be.visible')
  cy.get('[data-testid="tile-layer"]', { timeout: 10000 }).should('be.visible')
})
