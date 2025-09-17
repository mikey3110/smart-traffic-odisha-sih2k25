import type { Meta, StoryObj } from '@storybook/react';
import { TrafficMap } from './TrafficMap';

const meta: Meta<typeof TrafficMap> = {
  title: 'Components/Map/TrafficMap',
  component: TrafficMap,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: 'A real-time traffic map component that displays traffic intersections with live data from the backend API.',
      },
    },
  },
  argTypes: {
    height: {
      control: 'text',
      description: 'Height of the map container',
    },
    className: {
      control: 'text',
      description: 'Additional CSS classes',
    },
    showCameraFeeds: {
      control: 'boolean',
      description: 'Whether to show camera feed overlays',
    },
    onIntersectionClick: {
      action: 'intersection-clicked',
      description: 'Callback when an intersection is clicked',
    },
  },
};

export default meta;
type Story = StoryObj<typeof TrafficMap>;

export const Default: Story = {
  args: {
    height: '500px',
    showCameraFeeds: false,
  },
};

export const WithCameraFeeds: Story = {
  args: {
    height: '500px',
    showCameraFeeds: true,
  },
};

export const CustomHeight: Story = {
  args: {
    height: '700px',
    showCameraFeeds: false,
  },
};

export const WithCustomClass: Story = {
  args: {
    height: '500px',
    className: 'custom-traffic-map',
    showCameraFeeds: false,
  },
};
