import pandas as pd
import os

def load_and_process_data(data_dir, weather_file, building_file):
    """
    Load and process weather and building data files from the data directory
    
    Args:
        data_dir (str): Path to the data directory
        weather_file (str): Name of the weather data file
        building_file (str): Name of the building data file
    """
    # Construct full file paths
    weather_path = os.path.join(data_dir, weather_file)
    building_path = os.path.join(data_dir, building_file)
    
    # Read the CSV files, ensuring mixed types don't cause issues
    weather_df = pd.read_csv(weather_path, low_memory=False)
    building_df = pd.read_csv(building_path, low_memory=False)
    
    # Inspect columns for merging
    common_column = 'timestamp'  # Update this based on the actual column name in your data
    if common_column not in weather_df.columns or common_column not in building_df.columns:
        raise ValueError(f"Column '{common_column}' not found in one of the datasets")
    
    # Merge weather and building data
    merged_df = pd.merge(weather_df, building_df, on=common_column, how='outer')
    
    return merged_df

def process_all_datasets(data_dir):
    """
    Process all test scenario datasets and combine them
    
    Args:
        data_dir (str): Path to the directory containing the data files
    """
    # Dictionary of file pairs for each scenario
    scenarios = {
        'heating_ff': ('Weather_FF_Heating.csv', 'Building_FF_Heating.csv'),
        'heating_base': ('Weather_Base_Heating.csv', 'Building_Base_Heating.csv'),
        'heating_sb': ('Weather_SB_Heating.csv', 'Building_SB_Heating.csv'),
        'heating_pre': ('Weather_Pre_Heating.csv', 'Building_Pre_Heating.csv'),
        'cooling_ff': ('Weather_FF_Cooling.csv', 'Building_FF_Cooling.csv'),
        'cooling_base': ('Weather_Base_Cooling.csv', 'Building_Base_Cooling.csv'),
        'cooling_sb': ('Weather_SB_Cooling.csv', 'Building_SB_Cooling.csv')
    }
    
    # List to store dataframes
    all_dfs = []
    
    # Process each scenario
    for scenario, (weather_file, building_file) in scenarios.items():
        try:
            df = load_and_process_data(data_dir, weather_file, building_file)
            df['scenario'] = scenario  # Add scenario identifier
            all_dfs.append(df)
        except FileNotFoundError as e:
            print(f"Warning: Could not find files for {scenario}: {e}")
        except Exception as e:
            print(f"Error processing {scenario}: {e}")
    
    # Combine all dataframes
    if not all_dfs:
        raise ValueError("No datasets were successfully loaded.")
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Select and rename required columns
    final_df = pd.DataFrame({
        'Day': pd.to_datetime(combined_df['timestamp']).dt.date,
        'Hours': pd.to_datetime(combined_df['timestamp']).dt.hour,
        'Minutes': pd.to_datetime(combined_df['timestamp']).dt.minute,
        'Voltage (V)': combined_df.get('voltage', None),
        'Current (A)': combined_df.get('current', None),
        'Power (W)': combined_df.get('power', None),
        'Power Factor': combined_df.get('power_factor', None),
        'Frequency (Hz)': combined_df.get('frequency', None),
        'Energy (kWh)': combined_df.get('energy', None),
        'Inside Temperature (°C)': combined_df.get('indoor_temperature', None),
        'Outside Temperature (°C)': combined_df.get('outdoor_temperature', None),
        'Inside Humidity (%)': combined_df.get('indoor_humidity', None),
        'Outside Humidity (%)': combined_df.get('outdoor_humidity', None)
    })
    
    return final_df

def main():
    # Specify the path to your data directory
    data_dir = "data"  # Change this to the path of your data folder
    
    try:
        # Process all datasets
        print("Starting data processing...")
        merged_data = process_all_datasets(data_dir)
        
        # Save to CSV
        output_file = os.path.join("output", 'merged_hvac_data.csv')
        os.makedirs("output", exist_ok=True)
        merged_data.to_csv(output_file, index=False)
        print(f"Successfully merged data and saved to {output_file}")
        
        # Display sample of the merged data
        print("\nFirst few rows of merged data:")
        print(merged_data.head())
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
