# Data Preprocessing
def load_and_preprocess_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Try different encodings
    encodings = ['latin1', 'utf-8', 'cp1252']
    data = None
    for encoding in encodings:
        try:
            data = pd.read_csv(file_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    
    if data is None:
        raise ValueError("Could not read file with any of the attempted encodings")

    data['Day'] = pd.to_datetime(data['Day'])
    
    # Print column names to debug
    logging.info(f"Original columns: {data.columns.tolist()}")
    
    def normalize_column_name(col):
        # Remove special characters and normalize spaces
        col = col.strip()
        col = col.replace('Â', '')
        col = col.replace('°', '')
        col = col.replace('º', '')
        col = col.replace('ÂºC', 'C')
        col = col.replace('Â°C', 'C')
        col = col.replace('°C', 'C')
        col = col.replace('(C)', '(°C)')
        return col

    # Normalize all column names
    data.columns = [normalize_column_name(col) for col in data.columns]

    # Define expected column names
    numerical_features = [
        'Voltage (V)', 'Current (A)', 'Power Factor', 'Frequency (Hz)', 'Energy (kWh)',
        'Inside Temperature (°C)', 'Outside Temperature (°C)', 'Inside Humidity (%)', 'Outside Humidity (%)'
    ]
    
    # Create a comprehensive mapping for temperature columns
    temp_variations = [
        ('Inside Temperature(C)', 'Inside Temperature (°C)'),
        ('Outside Temperature(C)', 'Outside Temperature (°C)'),
        ('Inside Temperature (C)', 'Inside Temperature (°C)'),
        ('Outside Temperature (C)', 'Outside Temperature (°C)'),
        ('Inside Temperature', 'Inside Temperature (°C)'),
        ('Outside Temperature', 'Outside Temperature (°C)'),
        ('InsideTemp(C)', 'Inside Temperature (°C)'),
        ('OutsideTemp(C)', 'Outside Temperature (°C)'),
        ('Inside_Temperature', 'Inside Temperature (°C)'),
        ('Outside_Temperature', 'Outside Temperature (°C)')
    ]
    
    # Try to find and map temperature columns
    for old_name, new_name in temp_variations:
        if old_name in data.columns:
            data = data.rename(columns={old_name: new_name})
            logging.info(f"Renamed column '{old_name}' to '{new_name}'")
    
    # Function to find closest matching column
    def find_closest_match(target, columns):
        import difflib
        matches = difflib.get_close_matches(normalize_column_name(target), 
                                          [normalize_column_name(col) for col in columns], 
                                          n=1, cutoff=0.7)
        if matches:
            orig_cols = [col for col in columns if normalize_column_name(col) == matches[0]]
            return orig_cols[0] if orig_cols else None
        return None

    # Try to map required columns
    for required_col in numerical_features:
        if required_col not in data.columns:
            match = find_closest_match(required_col, data.columns)
            if match:
                data = data.rename(columns={match: required_col})
                logging.info(f"Mapped '{match}' to required column '{required_col}'")
            else:
                logging.warning(f"Could not find match for required column: {required_col}")

    # Verify columns after mapping
    missing_cols = [col for col in numerical_features if col not in data.columns]
    if missing_cols:
        logging.error(f"Missing columns after mapping: {missing_cols}")
        logging.error(f"Available columns: {data.columns.tolist()}")
        # Create missing columns with NaN values instead of raising error
        for col in missing_cols:
            data[col] = np.nan
            logging.warning(f"Created missing column '{col}' with NaN values")

    # Use a separate scaler for the target variable
    target_scaler = StandardScaler()
    data['Power (W)'] = target_scaler.fit_transform(data[['Power (W)']])

    # Impute missing values
    imputer = KNNImputer(n_neighbors=5)
    data[numerical_features] = imputer.fit_transform(data[numerical_features])

    # Feature Engineering
    data['Temperature Difference'] = data['Outside Temperature (°C)'] - data['Inside Temperature (°C)']
    data['Humidity Difference'] = data['Outside Humidity (%)'] - data['Inside Humidity (%)']
    data['Power Factor * Current'] = data['Power Factor'] * data['Current (A)']
    data['Current Squared'] = data['Current (A)'] ** 2

    # Normalize input features
    feature_scaler = StandardScaler()
    feature_columns = numerical_features + ['Temperature Difference', 'Humidity Difference', 'Power Factor * Current', 'Current Squared']
    data[feature_columns] = feature_scaler.fit_transform(data[feature_columns])
    
    return data, feature_scaler, target_scaler
