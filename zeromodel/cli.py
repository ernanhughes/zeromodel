"""
Zero-Model Intelligence Configuration CLI
"""

import argparse
import os

import yaml

from .config import DEFAULT_CONFIG, DEFAULT_CONFIG_PATH


def main():
    parser = argparse.ArgumentParser(description='Zero-Model Intelligence Configuration Manager')
    subparsers = parser.add_subparsers(dest='command', help='Configuration commands')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show current configuration')
    show_parser.add_argument('--path', type=str, default=DEFAULT_CONFIG_PATH,
                            help='Path to configuration file')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize configuration file')
    init_parser.add_argument('--path', type=str, default=DEFAULT_CONFIG_PATH,
                            help='Path to save configuration file')
    init_parser.add_argument('--force', action='store_true',
                            help='Overwrite existing configuration file')
    
    # Get command
    get_parser = subparsers.add_parser('get', help='Get a configuration value')
    get_parser.add_argument('key', type=str, help='Configuration key (e.g., zeromodel.precision)')
    get_parser.add_argument('--path', type=str, default=DEFAULT_CONFIG_PATH,
                           help='Path to configuration file')
    
    # Set command
    set_parser = subparsers.add_parser('set', help='Set a configuration value')
    set_parser.add_argument('key', type=str, help='Configuration key (e.g., zeromodel.precision)')
    set_parser.add_argument('value', type=str, help='Value to set')
    set_parser.add_argument('--path', type=str, default=DEFAULT_CONFIG_PATH,
                           help='Path to configuration file')
    
    args = parser.parse_args()
    
    if args.command == 'show':
        _show_config(args.path)
    elif args.command == 'init':
        _init_config(args.path, args.force)
    elif args.command == 'get':
        _get_config_value(args.key, args.path)
    elif args.command == 'set':
        _set_config_value(args.key, args.value, args.path)

def _show_config(config_path: str):
    """Show the current configuration"""
    if not os.path.exists(config_path):
        print(f"Configuration file not found at {config_path}")
        print("Use 'zeromodel config init' to create a default configuration")
        return
    
    with open(config_path, 'r') as f:
        print(f"Current configuration ({config_path}):")
        print(f.read())

def _init_config(config_path: str, force: bool):
    """Initialize a new configuration file"""
    if os.path.exists(config_path) and not force:
        print(f"Configuration file already exists at {config_path}")
        print("Use --force to overwrite")
        return
    
    # Create directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
    
    # Write default config
    with open(config_path, 'w') as f:
        yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created default configuration at {config_path}")

def _get_config_value(key: str, config_path: str):
    """Get a specific configuration value"""
    if not os.path.exists(config_path):
        print(f"Configuration file not found at {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Traverse the key path
    current = config
    for part in key.split('.'):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            print(f"Key '{key}' not found in configuration")
            return
    
    print(f"{key} = {current}")

def _set_config_value(key: str, value_str: str, config_path: str):
    """Set a specific configuration value"""
    if not os.path.exists(config_path):
        print(f"Configuration file not found at {config_path}")
        print("Use 'zeromodel config init' to create a default configuration")
        return
    
    # Parse the value
    try:
        # Try to convert to appropriate type
        if value_str.lower() in ['true', 'false']:
            value = value_str.lower() == 'true'
        elif value_str.lower() == 'null':
            value = None
        elif '.' in value_str:
            value = float(value_str)
        else:
            value = int(value_str)
    except ValueError:
        # Keep as string if conversion fails
        value = value_str
    
    # Load existing config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Traverse and set the key
    parts = key.split('.')
    current = config
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]
    
    # Set the value
    current[parts[-1]] = value
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Updated {key} = {value}")

if __name__ == "__main__":
    main()