import React from 'react';
import { motion } from 'framer-motion';
import {
  NavigationList,
  NavigationListItem,
  Icon,
  Badge,
  Toolbar,
  ToolbarSpacer,
  Switch,
  Label
} from '@ui5/webcomponents-react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useApp } from '@/contexts/AppContext';
import { Permission } from '@/types';
import './Sidebar.scss';

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
}

interface MenuItem {
  id: string;
  label: string;
  icon: string;
  path: string;
  permission?: Permission;
  badge?: number;
  children?: MenuItem[];
}

const menuItems: MenuItem[] = [
  {
    id: 'dashboard',
    label: 'Dashboard',
    icon: 'home',
    path: '/dashboard',
    permission: Permission.VIEW_DASHBOARD
  },
  {
    id: 'traffic',
    label: 'Traffic Control',
    icon: 'traffic-light',
    path: '/traffic',
    permission: Permission.MANAGE_TRAFFIC,
    children: [
      {
        id: 'traffic-lights',
        label: 'Traffic Lights',
        icon: 'traffic-light',
        path: '/traffic/lights'
      },
      {
        id: 'intersections',
        label: 'Intersections',
        icon: 'intersection',
        path: '/traffic/intersections'
      },
      {
        id: 'vehicles',
        label: 'Vehicles',
        icon: 'car',
        path: '/traffic/vehicles'
      }
    ]
  },
  {
    id: 'analytics',
    label: 'Analytics',
    icon: 'analytics',
    path: '/analytics',
    permission: Permission.VIEW_ANALYTICS,
    children: [
      {
        id: 'performance',
        label: 'Performance',
        icon: 'performance',
        path: '/analytics/performance'
      },
      {
        id: 'reports',
        label: 'Reports',
        icon: 'document',
        path: '/analytics/reports'
      },
      {
        id: 'trends',
        label: 'Trends',
        icon: 'trending-up',
        path: '/analytics/trends'
      }
    ]
  },
  {
    id: 'simulation',
    label: 'Simulation',
    icon: 'simulation',
    path: '/simulation',
    permission: Permission.MANAGE_TRAFFIC
  },
  {
    id: 'configuration',
    label: 'Configuration',
    icon: 'settings',
    path: '/configuration',
    permission: Permission.CONFIGURE_SYSTEM,
    children: [
      {
        id: 'system-settings',
        label: 'System Settings',
        icon: 'settings',
        path: '/configuration/system'
      },
      {
        id: 'user-management',
        label: 'User Management',
        icon: 'group',
        path: '/configuration/users'
      },
      {
        id: 'alerts',
        label: 'Alerts & Notifications',
        icon: 'bell',
        path: '/configuration/alerts'
      }
    ]
  }
];

export function Sidebar({ collapsed, onToggle }: SidebarProps) {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, hasPermission, systemConfig, updateSystemConfig } = useApp();
  const [expandedItems, setExpandedItems] = React.useState<string[]>([]);

  const handleItemClick = (item: MenuItem) => {
    if (item.children) {
      // Toggle expanded state
      setExpandedItems(prev => 
        prev.includes(item.id) 
          ? prev.filter(id => id !== item.id)
          : [...prev, item.id]
      );
    } else {
      navigate(item.path);
    }
  };

  const handleSubItemClick = (item: MenuItem) => {
    navigate(item.path);
  };

  const isItemActive = (item: MenuItem) => {
    return location.pathname === item.path || 
           (item.children && item.children.some(child => location.pathname.startsWith(child.path)));
  };

  const isSubItemActive = (item: MenuItem) => {
    return location.pathname === item.path;
  };

  const canAccessItem = (item: MenuItem) => {
    if (!item.permission) return true;
    return hasPermission(item.permission);
  };

  const filteredMenuItems = menuItems.filter(canAccessItem);

  return (
    <div className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
      {/* Header */}
      <div className="sidebar-header">
        <motion.div
          className="logo-container"
          animate={{ scale: collapsed ? 0.8 : 1 }}
          transition={{ duration: 0.3 }}
        >
          <Icon name="traffic-light" />
          {!collapsed && (
            <motion.span
              className="logo-text"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3, delay: 0.1 }}
            >
              Traffic Control
            </motion.span>
          )}
        </motion.div>
        
        <Button
          icon="navigation-left-arrow"
          design="Transparent"
          onClick={onToggle}
          className="toggle-button"
        />
      </div>

      {/* Navigation */}
      <div className="sidebar-content">
        <NavigationList>
          {filteredMenuItems.map((item) => (
            <React.Fragment key={item.id}>
              <NavigationListItem
                icon={item.icon}
                text={collapsed ? '' : item.label}
                selected={isItemActive(item)}
                onClick={() => handleItemClick(item)}
                className={`nav-item ${isItemActive(item) ? 'active' : ''}`}
              >
                {!collapsed && item.badge && (
                  <Badge colorScheme="8" className="nav-badge">
                    {item.badge}
                  </Badge>
                )}
                {!collapsed && item.children && (
                  <Icon 
                    name={expandedItems.includes(item.id) ? 'slim-arrow-down' : 'slim-arrow-right'} 
                    className="expand-icon"
                  />
                )}
              </NavigationListItem>

              {/* Sub-items */}
              {!collapsed && item.children && expandedItems.includes(item.id) && (
                <motion.div
                  className="sub-items"
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  {item.children.map((subItem) => (
                    <NavigationListItem
                      key={subItem.id}
                      icon={subItem.icon}
                      text={subItem.label}
                      selected={isSubItemActive(subItem)}
                      onClick={() => handleSubItemClick(subItem)}
                      className={`sub-item ${isSubItemActive(subItem) ? 'active' : ''}`}
                    />
                  ))}
                </motion.div>
              )}
            </React.Fragment>
          ))}
        </NavigationList>
      </div>

      {/* Footer */}
      {!collapsed && (
        <motion.div
          className="sidebar-footer"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.2 }}
        >
          <div className="simulation-control">
            <Label>Simulation</Label>
            <Switch
              checked={systemConfig.simulation.enabled}
              onChange={(e) => updateSystemConfig({
                simulation: { ...systemConfig.simulation, enabled: e.target.checked }
              })}
            />
          </div>
          
          <div className="user-info">
            <Icon name="customer" />
            <span>{user?.name}</span>
          </div>
        </motion.div>
      )}
    </div>
  );
}
