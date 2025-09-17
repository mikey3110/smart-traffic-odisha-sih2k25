# Testing Guide - Smart Traffic Management Frontend

This guide covers all testing strategies and best practices for the Smart Traffic Management System frontend.

## 🧪 Testing Strategy

The project uses a comprehensive testing approach with multiple layers:

1. **Unit Tests** - Individual component testing
2. **Integration Tests** - Component interaction testing
3. **E2E Tests** - Full user journey testing
4. **Component Tests** - Isolated component testing

## 🔧 Test Setup

### Prerequisites

- Node.js 18+
- npm 9+
- Chrome/Chromium (for Cypress)

### Installation

```bash
# Install dependencies
npm install

# Run all tests
npm run test:all
```

## 📝 Unit Tests

### Framework

- **Jest** - Test runner and assertion library
- **React Testing Library** - Component testing utilities
- **@testing-library/jest-dom** - Custom matchers

### Running Unit Tests

```bash
# Run all unit tests
npm run test

# Run in watch mode
npm run test:watch

# Run with coverage
npm run test:coverage
```

### Writing Unit Tests

Example test for a component:

```tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { TrafficMap } from './TrafficMap';

describe('TrafficMap', () => {
  it('renders map container', () => {
    render(<TrafficMap />);
    expect(screen.getByTestId('map-container')).toBeInTheDocument();
  });

  it('handles intersection click', () => {
    const onIntersectionClick = jest.fn();
    render(<TrafficMap onIntersectionClick={onIntersectionClick} />);
    
    fireEvent.click(screen.getByTestId('marker'));
    expect(onIntersectionClick).toHaveBeenCalled();
  });
});
```

### Test Utilities

Custom test utilities in `src/setupTests.ts`:

- Mock implementations for external libraries
- Global test configuration
- Custom matchers

## 🌐 E2E Tests

### Framework

- **Cypress** - End-to-end testing framework
- **@cypress/react** - Component testing support

### Running E2E Tests

```bash
# Run E2E tests headlessly
npm run test:e2e

# Open Cypress Test Runner
npm run test:e2e:open
```

### Writing E2E Tests

Example E2E test:

```tsx
describe('Dashboard E2E Tests', () => {
  beforeEach(() => {
    // Mock API responses
    cy.intercept('GET', '/traffic/lights', { fixture: 'traffic-lights.json' });
    cy.visit('/');
  });

  it('loads dashboard successfully', () => {
    cy.get('[data-testid="dashboard"]').should('be.visible');
    cy.get('h1').should('contain', 'Smart Traffic Management System');
  });

  it('displays traffic map', () => {
    cy.get('[data-testid="map-container"]').should('be.visible');
    cy.get('[data-testid="marker"]').should('have.length', 2);
  });
});
```

### E2E Test Structure

```
cypress/
├── e2e/                 # End-to-end tests
│   └── dashboard.cy.ts
├── component/           # Component tests
│   └── TrafficMap.cy.tsx
├── support/             # Test utilities
│   ├── commands.ts
│   └── e2e.ts
└── fixtures/            # Test data
    └── traffic-lights.json
```

## 🧩 Component Tests

### Running Component Tests

```bash
# Run component tests
npm run test:component

# Open component test runner
npm run test:component:open
```

### Writing Component Tests

Example component test:

```tsx
import { mount } from '@cypress/react';
import { TrafficMap } from './TrafficMap';

describe('TrafficMap Component', () => {
  it('renders with default props', () => {
    mount(<TrafficMap />);
    cy.get('[data-testid="map-container"]').should('be.visible');
  });

  it('handles prop changes', () => {
    mount(<TrafficMap height="600px" />);
    cy.get('[data-testid="map-container"]').parent().should('have.css', 'height', '600px');
  });
});
```

## 📊 Test Coverage

### Coverage Reports

```bash
# Generate coverage report
npm run test:coverage
```

Coverage reports are generated in:
- `coverage/lcov-report/index.html` - HTML report
- `coverage/lcov.info` - LCOV format

### Coverage Thresholds

Current coverage thresholds:
- Branches: 70%
- Functions: 70%
- Lines: 70%
- Statements: 70%

### Improving Coverage

1. **Identify uncovered code**:
   ```bash
   npm run test:coverage
   open coverage/lcov-report/index.html
   ```

2. **Write tests for uncovered code**:
   - Edge cases
   - Error conditions
   - Different code paths

3. **Mock external dependencies**:
   ```tsx
   jest.mock('@/services/trafficService');
   ```

## 🎯 Testing Best Practices

### Component Testing

1. **Test behavior, not implementation**:
   ```tsx
   // Good
   expect(screen.getByText('Loading...')).toBeInTheDocument();
   
   // Bad
   expect(component.state.isLoading).toBe(true);
   ```

2. **Use data-testid for stable selectors**:
   ```tsx
   <div data-testid="map-container">...</div>
   ```

3. **Test user interactions**:
   ```tsx
   fireEvent.click(screen.getByRole('button'));
   fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'Test' } });
   ```

### E2E Testing

1. **Use realistic test data**:
   ```tsx
   cy.fixture('traffic-lights.json').then((data) => {
     cy.intercept('GET', '/traffic/lights', data);
   });
   ```

2. **Test complete user journeys**:
   ```tsx
   it('user can view traffic data', () => {
     cy.visit('/');
     cy.get('[data-testid="map-container"]').should('be.visible');
     cy.get('[data-testid="marker"]').first().click();
     cy.get('[data-testid="popup"]').should('be.visible');
   });
   ```

3. **Use page object pattern**:
   ```tsx
   class DashboardPage {
     visit() {
       cy.visit('/');
     }
     
     getMap() {
       return cy.get('[data-testid="map-container"]');
     }
   }
   ```

### Mocking

1. **Mock external APIs**:
   ```tsx
   jest.mock('@/services/trafficService', () => ({
     getTrafficLights: jest.fn().mockResolvedValue(mockData),
   }));
   ```

2. **Mock browser APIs**:
   ```tsx
   Object.defineProperty(window, 'matchMedia', {
     value: jest.fn().mockImplementation(query => ({
       matches: false,
       media: query,
       onchange: null,
     })),
   });
   ```

## 🚀 CI/CD Testing

### GitHub Actions

Tests run automatically on:
- Pull requests
- Push to main branch
- Manual triggers

### Test Pipeline

1. **Lint and Type Check**
2. **Unit Tests**
3. **E2E Tests**
4. **Component Tests**
5. **Coverage Report**

### Local CI Simulation

```bash
# Run all checks locally
npm run lint
npm run test:coverage
npm run test:e2e
npm run test:component
```

## 🐛 Debugging Tests

### Unit Test Debugging

```bash
# Run specific test
npm test -- --testNamePattern="TrafficMap"

# Run with verbose output
npm test -- --verbose

# Debug mode
node --inspect-brk node_modules/.bin/jest --runInBand
```

### E2E Test Debugging

```bash
# Open Cypress with debug mode
npm run test:e2e:open

# Run specific test
npx cypress run --spec "cypress/e2e/dashboard.cy.ts"
```

### Common Issues

1. **Tests timing out**:
   - Increase timeout
   - Check for async operations
   - Use proper wait conditions

2. **Elements not found**:
   - Check selector specificity
   - Wait for elements to load
   - Use data-testid attributes

3. **Mock not working**:
   - Check mock placement
   - Verify mock implementation
   - Clear mocks between tests

## 📋 Test Checklist

### Before Writing Tests

- [ ] Understand component behavior
- [ ] Identify test scenarios
- [ ] Plan test data requirements
- [ ] Consider edge cases

### While Writing Tests

- [ ] Test happy path
- [ ] Test error conditions
- [ ] Test edge cases
- [ ] Test user interactions
- [ ] Verify accessibility

### After Writing Tests

- [ ] Tests pass consistently
- [ ] Coverage meets thresholds
- [ ] Tests are maintainable
- [ ] Documentation is updated

## 🆘 Troubleshooting

### Common Problems

1. **Tests failing randomly**:
   - Check for race conditions
   - Use proper async/await
   - Clear state between tests

2. **Slow test execution**:
   - Optimize test setup
   - Use parallel execution
   - Mock heavy operations

3. **Coverage not updating**:
   - Check file paths
   - Verify test execution
   - Clear coverage cache

### Getting Help

1. Check test logs
2. Review test documentation
3. Check GitHub Actions logs
4. Contact development team

---

**Testing Guide v1.0 - Smart India Hackathon 2025**
