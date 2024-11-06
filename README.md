# PowerOP: Intelligent Power Management System

## Overview
**PowerOP** is a real-time power monitoring and management system designed to enhance energy efficiency in electronic devices. Leveraging IoT, AI, and machine learning, this project aims to optimize power consumption, predict usage patterns, and reduce energy waste. PowerOP enables adaptive power management by embedding AI-driven models into hardware, thus promoting eco-friendly, sustainable electronics with a reduced carbon footprint.

## Features
- **Real-Time Data Collection**: Continuously monitors power, current, and temperature metrics.
- **AI-Driven Power Optimization**: Predicts power usage patterns and optimizes energy consumption.
- **Sustainable Hardware Design**: Eco-friendly prototype developed with minimal energy waste.
- **Data Analytics**: Embedded data analysis to adjust operations dynamically for optimal power management.

## Project Structure

```plaintext
PoweOP/
│
├── Miniproject-1/
│   ├── data.py                 # Data handling and preprocessing
│   ├── data2.py                # Additional data processing code
│   ├── Finalcode.c             # Final embedded code implementation
│   ├── hvac-architecture.mermaid  # Architecture of HVAC system in mermaid diagram format
│   ├── meghacloudidecode.c     # Cloud IDE code for integration
│   ├── modbuscode.c            # Modbus protocol code for hardware communication
│   ├── model.py                # Machine learning model implementation
│   ├── mselect.py              # Model selection for best performance
│   ├── output.png              # Sample output graph/visualization
│   ├── pzem code/              # Contains PZEM sensor-related code
│   ├── PZEM-004TCODE.c         # Code for PZEM-004T sensor for power measurements
│   ├── rmse.py                 # RMSE calculation for model accuracy
│   ├── sensor_data.csv         # Sample sensor data for testing and validation
│   └── Temperaturewithlcddisplaynomodbus.c  # Temperature display code
```

## Getting Started

### Prerequisites
- **Python 3.8+**: Required for running Python scripts.
- **Embedded C Compiler**: For compiling embedded C files.
- **Mermaid**: For visualizing system architecture diagrams (e.g., via VSCode extension).
- **Modbus Libraries**: Required for communication with Modbus-enabled devices.

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
   Install the Python dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration
- **Modbus Setup**: Ensure Modbus-compatible devices are connected and configured as specified in `modbuscode.c`.
- **PZEM Sensor**: Connect the PZEM-004T sensors according to the instructions in `PZEM-004TCODE.c`.
- **HVAC Configuration**: Refer to `hvac-architecture.mermaid` for a visual overview of the HVAC system.

## Usage

1. **Running the Model**
   Execute `model.py` to initiate the AI-driven power optimization model.
   ```bash
   python model.py
   ```

2. **Data Processing**
   Use `data.py` and `data2.py` for data preprocessing steps, including sensor data transformation.

3. **Final Implementation**
   Compile and run `Finalcode.c` on the embedded hardware to deploy the optimized power management solution.

## Code Explanation

### [Finalcode.c](Miniproject-1/Finalcode.c)
The core embedded code responsible for real-time monitoring and control. It integrates data collection from various sensors and adjusts power output accordingly.

### [model.py](Miniproject-1/model.py)
Implements the machine learning model for predicting power consumption patterns. The model is trained using historical data and evaluates performance using `rmse.py`.

### [hvac-architecture.mermaid](Miniproject-1/hvac-architecture.mermaid)
This file contains the architecture diagram for an HVAC system. It provides a comprehensive view of the components involved in power management.

## Data
The file `sensor_data.csv` contains sample data collected from sensors. It includes metrics for temperature, power, and current, essential for testing and evaluating the AI model.

## Results
Sample results are visualized in `output.png`, demonstrating the model's performance in power optimization.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For any inquiries or suggestions, please reach out to the project maintainer:
- **Name**: Jsh Agarwal
- **Email**: [jshagarwal15@gmail.com](mailto:jshagarwal15@gmail.com)

