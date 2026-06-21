# generate_grid.py
import yaml
from src import VolumeFactory, save_grid

def main():
    """Main function to generate and save the grid."""
    # Load the configuration file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Use the factory to create the master grid
    factory = VolumeFactory(config)
    master_grid = factory.create_volume()
    
    # Save the final grid
    output_conf = config['output_settings']
    save_grid(master_grid, output_conf['directory'], output_conf['filename'])

if __name__ == "__main__":
    main()