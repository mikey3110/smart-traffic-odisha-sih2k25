import type { Meta, StoryObj } from '@storybook/react';
import { Dashboard } from './Dashboard';

const meta: Meta<typeof Dashboard> = {
  title: 'Pages/Dashboard',
  component: Dashboard,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: 'The main dashboard page that displays traffic management overview, real-time data, and system status.',
      },
    },
  },
  argTypes: {
    onIntersectionSelect: {
      action: 'intersection-selected',
      description: 'Callback when an intersection is selected',
    },
  },
};

export default meta;
type Story = StoryObj<typeof Dashboard>;

export const Default: Story = {
  args: {},
};

export const WithSelectedIntersection: Story = {
  args: {
    selectedIntersection: {
      id: 'intersection_1',
      name: 'Main Street & First Avenue',
      position: [20.2961, 85.8245],
      signalState: 'green',
      vehicleCount: 12,
      lastUpdate: new Date(),
      status: 'normal',
    },
  },
};
