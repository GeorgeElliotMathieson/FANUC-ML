# FANUC LR Mate 200iC Python Control via Ethernet

This repository contains Python scripts for implementing real-time control of a FANUC LR Mate 200iC robot with R-30iA Mate controller via Ethernet connection.

## Project Contents

- `fanuc_robot_controller.py` - Main controller class for robot communication
- `example_usage.py` - Example script demonstrating basic usage

## Hardware Setup

1. Connect an Ethernet cable directly from your Windows laptop to the R-30iA Mate controller's RJ-45 port.
2. Alternatively, connect both devices to the same network switch if you need other devices on the network.

## Network Configuration

### Configure the R-30iA Mate Controller

1. Access the controller's configuration screen via the teach pendant
2. Navigate to SETUP → Host Comm
3. Set a static IP address (e.g., 192.168.1.10)
4. Set subnet mask (e.g., 255.255.255.0)
5. Optional: Configure default gateway if needed

### Configure Your Windows Laptop

1. Open Network & Internet settings
2. Change adapter options
3. Right-click on your Ethernet connection and select Properties
4. Select Internet Protocol Version 4 (TCP/IPv4)
5. Set a static IP address (e.g., 192.168.1.11)
6. Set subnet mask (255.255.255.0)
7. Optional: Configure default gateway if needed

## Robot Controller Configuration

### Enable Socket Messaging

1. On the teach pendant, navigate to SETUP → Host Comm → TCP/IP
2. Enable the Socket Messaging option
3. Set the port number (common ports are 6000-9000)
4. Set timeout values (start with 60 seconds)

### Configure System Variables

1. Access the System Variables screen
2. Set $HOSTS_CFG.$PROTOCOL to 'SM' (Socket Messaging)
3. Set $HOSTS_CFG.$SERVER_PORT to your chosen port number
4. Set $HOSTS_CFG.$TIMEOUT to an appropriate value (seconds)
5. Ensure $HOSTC_CFG.$OPER is set to 'START'

### Set Access Controls

1. Navigate to SETUP → Security
2. Ensure socket connections are allowed
3. Set appropriate access levels for remote motion control

## Python Implementation

### Prerequisites

```bash
pip install socket threading time struct
```

### Basic Usage

```python
from fanuc_robot_controller import FanucRobotController

# Create controller instance
robot = FanucRobotController('192.168.1.10', 6000)

# Connect to the robot
robot.connect()

# Get current joint positions
positions = robot.get_joint_positions()
print(positions)

# Move to new position
robot.move_to_joint_positions([0, 10, 0, 0, 0, 0])

# Disconnect when done
robot.disconnect()
```

## Important Notes

### Command Syntax

The exact command syntax (`RDJPOS`, `MOVJ`, etc.) may vary based on your FANUC controller firmware version. Consult your FANUC documentation for precise command formats.

### Safety Considerations

- Implement emergency stop functionality
- Set appropriate speed limitations
- Consider collision detection
- Test in a safe environment first
- Never run untested code on a production robot

### Error Handling

- Implement robust error handling
- Add timeouts for all operations
- Have fallback procedures for communication failures

### Connection Management

- Keep track of connection state
- Implement reconnection strategies
- Properly close connections when done

### Performance Optimization

- Consider using binary protocols for higher performance
- Minimize unnecessary queries
- Implement buffering for command sequences

## Troubleshooting

1. **Connection Issues**
   - Verify IP addresses and port numbers
   - Check firewall settings on both devices
   - Test basic connectivity with ping
   - Use wireshark to debug communication issues

2. **Command Errors**
   - Verify correct command syntax for your controller model
   - Check permissions settings on the controller
   - Ensure robot is in the correct mode for remote operation

3. **Motion Problems**
   - Verify speed and acceleration settings
   - Check for joint limits and singularities
   - Ensure safe operating environment

## Disclaimer

This code is provided as a starting point and will need to be modified based on your specific FANUC controller model, firmware version, and application requirements. Always prioritize safety when working with industrial robots.