import type { Meta, StoryObj } from '@storybook/react';
import { SystemStatus } from './SystemStatus';

const meta: Meta<typeof SystemStatus> = {
  title: 'Components/Layout/SystemStatus',
  component: SystemStatus,
  parameters: {
    layout: 'padded',
    docs: {
      description: {
        component: 'A system status component that displays the health and status of various system services.',
      },
    },
  },
  argTypes: {
    className: {
      control: 'text',
      description: 'Additional CSS classes',
    },
  },
};

export default meta;
type Story = StoryObj<typeof SystemStatus>;

export const Default: Story = {
  args: {},
};

export const WithCustomClass: Story = {
  args: {
    className: 'custom-system-status',
  },
};
