# Smart Traffic Management System - Frontend

A modern, production-ready React frontend for the Smart Traffic Management System built with TypeScript, Vite, and SAP UI5 WebComponents.

## 🚀 Features

- **Real-time Traffic Visualization** - Interactive Leaflet maps with live traffic data
- **Camera Feed Integration** - Live camera feeds from traffic intersections
- **Responsive Dashboard** - Modern UI with SAP UI5 WebComponents
- **TypeScript Support** - Full type safety and IntelliSense
- **Production Ready** - Docker, Kubernetes, and CI/CD pipeline
- **Comprehensive Testing** - Unit, integration, and E2E tests
- **Component Library** - Storybook for component development

## 🛠️ Tech Stack

- **Framework**: React 18 + TypeScript
- **Build Tool**: Vite 4
- **UI Library**: SAP UI5 WebComponents
- **Maps**: Leaflet + React-Leaflet
- **Styling**: SCSS + CSS Modules
- **Testing**: Jest + React Testing Library + Cypress
- **Documentation**: Storybook
- **Deployment**: Docker + Kubernetes

## 📦 Quick Start

### Prerequisites

- Node.js 18+ 
- npm 9+
- Docker (optional)
- Kubernetes (optional)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd smart-traffic-odisha-sih2k25/src/frontend/smart-traffic-ui

# Install dependencies
npm install

# Start development server
npm run dev
```

The application will be available at `http://localhost:3000`

## 🏗️ Development

### Available Scripts

```bash
# Development
npm run dev              # Start development server
npm run build            # Build for production
npm run preview          # Preview production build

# Testing
npm run test             # Run unit tests
npm run test:watch       # Run tests in watch mode
npm run test:coverage    # Run tests with coverage
npm run test:e2e         # Run E2E tests
npm run test:component   # Run component tests
npm run test:all         # Run all tests

# Code Quality
npm run lint             # Run ESLint
npm run lint:fix         # Fix ESLint issues
npm run format           # Format code with Prettier
npm run format:check     # Check code formatting

# Documentation
npm run storybook        # Start Storybook
npm run build-storybook  # Build Storybook

# Deployment
npm run docker:build     # Build Docker image
npm run docker:run       # Run Docker container
npm run docker:compose   # Run with Docker Compose
npm run k8s:deploy       # Deploy to Kubernetes
npm run k8s:delete       # Delete from Kubernetes
npm run k8s:status       # Check Kubernetes status
```

### Project Structure

```
src/
├── components/           # React components
│   ├── common/          # Shared components
│   ├── dashboard/       # Dashboard components
│   ├── layout/          # Layout components
│   └── map/             # Map-related components
├── contexts/            # React contexts
├── hooks/               # Custom React hooks
├── pages/               # Page components
├── services/            # API services
├── styles/              # Global styles
├── types/               # TypeScript type definitions
└── utils/               # Utility functions
```

## 🧪 Testing

### Unit Tests

Unit tests are written with Jest and React Testing Library:

```bash
npm run test
```

### E2E Tests

End-to-end tests are written with Cypress:

```bash
npm run test:e2e
```

### Component Tests

Component tests are also written with Cypress:

```bash
npm run test:component
```

## 📚 Documentation

### Storybook

Component documentation is available in Storybook:

```bash
npm run storybook
```

Visit `http://localhost:6006` to view the component library.

### API Documentation

API endpoints are documented in the backend service. Key endpoints:

- `GET /traffic/lights` - Get traffic light data
- `GET /api/cv/counts` - Get vehicle counts
- `GET /traffic/analytics` - Get traffic analytics
- `GET /config/status` - Get system status

## 🚀 Deployment

### Docker

Build and run with Docker:

```bash
# Build image
npm run docker:build

# Run container
npm run docker:run

# Or use Docker Compose
npm run docker:compose
```

### Kubernetes

Deploy to Kubernetes:

```bash
# Deploy
npm run k8s:deploy

# Check status
npm run k8s:status

# Delete deployment
npm run k8s:delete
```

### Production Build

```bash
npm run build
```

The built files will be in the `dist/` directory.

## 🔧 Configuration

### Environment Variables

Create a `.env.local` file:

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

### TypeScript Configuration

TypeScript is configured in `tsconfig.json` with strict mode enabled and path aliases:

```json
{
  "compilerOptions": {
    "baseUrl": "src",
    "paths": {
      "@/*": ["./*"],
      "@components/*": ["./components/*"],
      "@hooks/*": ["./hooks/*"],
      "@services/*": ["./services/*"]
    }
  }
}
```

### Vite Configuration

Vite is configured in `vite.config.ts` with:

- React plugin
- TypeScript checker
- Path aliases
- Proxy for API calls
- Build optimizations

## 🎨 Styling

### SCSS Modules

Components use SCSS modules for scoped styling:

```scss
// Component.module.scss
.container {
  display: flex;
  flex-direction: column;
}
```

### Global Styles

Global styles are in `src/styles/global.scss` and `src/index.css`.

### SAP UI5 WebComponents

The project uses SAP UI5 WebComponents for consistent UI:

```tsx
import { Card, CardHeader, Title, Text } from '@ui5/webcomponents-react';
```

## 🔍 Code Quality

### ESLint

ESLint is configured with:

- TypeScript support
- React hooks rules
- Storybook rules
- Custom rules for code quality

### Prettier

Prettier is configured for consistent code formatting.

### Pre-commit Hooks

Recommended pre-commit hooks:

```bash
# Install husky and lint-staged
npm install --save-dev husky lint-staged

# Add to package.json
{
  "lint-staged": {
    "src/**/*.{ts,tsx}": ["eslint --fix", "prettier --write"]
  }
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Kill process on port 3000
   npx kill-port 3000
   ```

2. **TypeScript errors**
   ```bash
   # Check TypeScript
   npx tsc --noEmit
   ```

3. **Build failures**
   ```bash
   # Clear cache and reinstall
   rm -rf node_modules package-lock.json
   npm install
   ```

4. **Test failures**
   ```bash
   # Clear Jest cache
   npm test -- --clearCache
   ```

### Debug Mode

Run in debug mode:

```bash
# Development with debug
DEBUG=* npm run dev

# Tests with debug
DEBUG=* npm run test
```

## 🤝 Contributing

### Development Workflow

1. Create a feature branch
2. Make changes
3. Run tests: `npm run test:all`
4. Run linting: `npm run lint`
5. Format code: `npm run format`
6. Create pull request

### Code Standards

- Use TypeScript for all new code
- Write tests for new components
- Follow the existing code style
- Update documentation as needed

## 📄 License

This project is part of the Smart India Hackathon 2025.

## 🆘 Support

For support and questions:

1. Check the troubleshooting section
2. Review the Storybook documentation
3. Check the test files for usage examples
4. Create an issue in the repository

---

**Built with ❤️ for Smart India Hackathon 2025**