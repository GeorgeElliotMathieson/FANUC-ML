# FANUC Robot ML Platform - DirectML Conversion

This document outlines the conversion of the FANUC Robot ML Platform to a DirectML-only implementation, focusing exclusively on AMD GPU acceleration.

## Conversion Overview

The platform has been streamlined to focus entirely on DirectML operations for AMD GPUs, removing all non-DirectML code paths. This conversion ensures:

1. **Simplified Codebase**: Removed conditional code paths and duplicate implementations
2. **Focused Development**: All development efforts now target DirectML exclusively
3. **Optimized Performance**: Code specifically tuned for AMD GPUs with DirectML

## Key Changes

### Core Platform Changes

- Removed all conditional DirectML/non-DirectML code paths from `fanuc_platform.py`
- Set DirectML environment variables by default in all entry points
- Eliminated the `--directml` flag since all operations now use DirectML

### Entry Points

- `fanuc.bat` - Updated to always use DirectML
- `directml.bat` - Simplified to a forwarding script for backward compatibility

### Backward Compatibility

To maintain backward compatibility, all legacy scripts have been updated to use DirectML:

- `test_model.bat` - Now explicitly uses DirectML
- `evaluate_model.bat` - Now explicitly uses DirectML
- `test_directml.bat` - Maintained for backward compatibility
- `evaluate_directml.bat` - Maintained for backward compatibility
- `legacy.bat` - Updates to handle all legacy scripts with DirectML support

## Requirements

Since the platform now exclusively supports DirectML, the following requirements are mandatory:

1. Windows 10/11 with an AMD GPU
2. PyTorch 2.0.0+
3. torch-directml package

## Usage

All commands now operate with DirectML by default:

```bash
# Training with DirectML
fanuc.bat train --steps 1000000 --no-gui

# Evaluating a model with DirectML
fanuc.bat eval models/my-model --episodes 20

# Testing a model with DirectML
fanuc.bat test models/my-model
```

## Implementation Details

The DirectML-focused implementation leverages specific optimizations for AMD GPUs:

1. **Memory Management**: Optimized tensor operations for AMD GPU memory architecture
2. **DirectML-Specific Tensors**: Use of DirectML-specific tensor operations where beneficial
3. **Environment Variables**: Automatic configuration of DirectML environment variables
4. **Error Handling**: Improved error reporting for DirectML-specific issues

## Future Development

All future development will focus exclusively on enhancing DirectML support and performance. This includes:

1. Implementing additional DirectML-specific optimizations
2. Testing with various AMD GPU models to ensure broad compatibility
3. Updating to new versions of torch-directml as they become available
4. Expanding DirectML logging and debugging capabilities 