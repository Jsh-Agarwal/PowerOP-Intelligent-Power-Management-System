
# PowerOP: Intelligent Power Management System

## Overview

**PowerOP** is a real-time power monitoring and management system designed to enhance energy efficiency in electronic devices. Leveraging IoT, AI, and machine learning, this project aims to optimize power consumption, predict usage patterns, and reduce energy waste. PowerOP enables adaptive power management by embedding AI-driven models into hardware, thus promoting eco-friendly, sustainable electronics with a reduced carbon footprint.

---

## Features

- **Real-Time Data Collection**: Continuously monitors power, current, and temperature metrics.
- **AI-Driven Power Optimization**: Predicts power usage patterns and optimizes energy consumption.
- **Sustainable Hardware Design**: Eco-friendly prototype developed with minimal energy waste.
- **Data Analytics**: Embedded data analysis to adjust operations dynamically for optimal power management.
- **Mobile App Integration**: Control and monitor the system using a mobile application.
- **Cloud Connectivity**: Seamless integration with cloud services for data storage and analysis.

---

## Project Structure

```plaintext
PowerOP/
├── azureIot/                   # Azure IoT related files
│   └── azureIot.ino            # Arduino code for Azure IoT integration
├── Dashboard/                  # Web dashboard for monitoring and control
│   ├── components/             # React components
│   ├── hooks/                  # Custom React hooks
│   ├── lib/                    # Utility libraries
│   ├── public/                 # Public assets
│   ├── styles/                 # Styling files
│   ├── next.config.mjs         # Next.js configuration
│   ├── package.json            # NPM package configuration
│   └── tailwind.config.ts      # Tailwind CSS configuration
├── IoT Codes/                  # Embedded C and Python codes for IoT devices
│   ├── final_code_with_supabase.py  # Final Python code with Supabase integration
│   ├── Finalcode.c             # Core implementation for sensor control and monitoring
│   ├── modbuscode.c            # Code for Modbus protocol communication
│   ├── PZEM-004TCODE.c         # Code specific to PZEM-004T sensor for power measurements
│   └── other sensor codes      # Additional sensor codes
├── ML_deployement/             # Machine learning deployment scripts
│   ├── __init__.py             # Initialization file for ML deployment
│   └── other ML scripts        # Additional ML scripts
├── ML_RnD/                     # Research and development for machine learning models
├── Mobile App/                 # Mobile application for system control
│   ├── android/                # Android-specific files
│   ├── app.json                # Expo configuration
│   └── other mobile app files  # Additional mobile app files
├── System Design/              # Architecture and design documentation
│   ├── hvac-architecture.mermaid  # HVAC system architecture diagram in mermaid format
│   └── output.png              # Example output of the system architecture
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
└── other project files         # Additional project files
```

---

## Getting Started

### Prerequisites

- **Python 3.8+**: Required for running Python scripts.
- **Embedded C Compiler**: For compiling embedded C files.
- **Mermaid (for VSCode)**: For visualizing system architecture diagrams.
- **Modbus Libraries**: Required for Modbus communication with compatible devices.
- **Node.js**: Required for running the web dashboard.
- **Expo CLI**: Required for running the mobile application.

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/PowerOP.git
   cd PowerOP
   ```
2. **Set up Virtual Environment**

   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```
3. **Install Dependencies**
   Install required Python packages with:

   ```bash
   pip install -r requirements.txt
   ```
4. **Install Node.js Dependencies**
   Navigate to the `Dashboard` directory and install dependencies:

   ```bash
   cd Dashboard
   npm install
   ```
5. **Install Expo CLI**

   ```bash
   npm install -g expo-cli
   ```

### Configuration

- **Modbus Setup**: Ensure Modbus-compatible devices are connected and configured as specified in `modbuscode.c`.
- **PZEM Sensor**: Connect PZEM-004T sensors according to `PZEM-004TCODE.c`.
- **HVAC Configuration**: Refer to `hvac-architecture.mermaid` for a structural overview of the HVAC system.
- **Azure IoT**: Configure Azure IoT settings in `azureIot.ino`.

---

## Usage

### Running the Model

Execute `model.py` to initiate the AI-driven power optimization model:

```bash
python Model Code\model.py
```

### Data Processing

The `data.py` and `data2.py` scripts handle preprocessing, transforming sensor data for analysis or model training.

### Embedded Code Execution

Compile and run `Finalcode.c` on the embedded hardware to implement real-time power monitoring and control.

### Running the Web Dashboard

Navigate to the `Dashboard` directory and start the development server:

```bash
cd Dashboard
npm run dev
```

### Running the Mobile App

Navigate to the `Mobile App` directory and start the Expo development server:

```bash
cd Mobile App
expo start
```

---

## Detailed File Explanations

- **Finalcode.c**: The main embedded C code for real-time sensor data collection, monitoring, and power adjustments.
- **model.py**: Implements the machine learning model for predicting power consumption. The model is trained on historical data and evaluated using `rmse.py`.
- **hvac-architecture.mermaid**: Diagram of the HVAC system architecture in mermaid syntax, outlining components involved in power management.
- **sensor_data.csv**: Sample dataset containing temperature, power, and current metrics, used for model validation.
- **azureIot.ino**: Arduino code for integrating with Azure IoT.
- **app.json**: Configuration file for the Expo mobile application.

---

## Results

![Figure_9](https://github.com/user-attachments/assets/5837a05b-3ad8-4bf9-9e65-72ba3d405710)

![Figure_10](https://github.com/user-attachments/assets/04f52cbe-20a8-4494-a03b-ccea89f210eb)

![Figure_3](https://github.com/user-attachments/assets/5fb2e042-7497-4d50-85e7-1b32d567014c)

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For any inquiries or suggestions, please reach out to the project maintainer:

- **Name**: Jsh Agarwal
- **Email**: [jshagarwal15@gmail.com](mailto:jshagarwal15@gmail.com)
