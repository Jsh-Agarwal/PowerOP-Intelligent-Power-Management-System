# PowerOP: Intelligent Power Management System

## Overview
**PowerOP** is a real-time power monitoring and management system designed to enhance energy efficiency in electronic devices. Leveraging IoT, AI, and machine learning, this project aims to optimize power consumption, predict usage patterns, and reduce energy waste. PowerOP enables adaptive power management by embedding AI-driven models into hardware, thus promoting eco-friendly, sustainable electronics with a reduced carbon footprint.

---

## Features

- **Real-Time Data Collection**: Continuously monitors power, current, and temperature metrics.
- **AI-Driven Power Optimization**: Predicts power usage patterns and optimizes energy consumption.
- **Sustainable Hardware Design**: Eco-friendly prototype developed with minimal energy waste.
- **Data Analytics**: Embedded data analysis to adjust operations dynamically for optimal power management.

---

## Project Structure

```plaintext
PowerOP/
├── data/                       # Data-related files
│   └── sensor_data.csv         # Sample sensor data for testing and validation
├── Model Code/                 # Code for model development and data processing
│   ├── data.py                 # Data handling and preprocessing
│   ├── data2.py                # Additional data processing functions
│   ├── model.py                # Machine learning model implementation
│   ├── mselect.py              # Model selection utility
│   └── rmse.py                 # RMSE calculation for model accuracy
├── output/                     # Visual outputs of the model
│   └── [all images]            # Output images showcasing model results
├── Sensor Codes/               # Embedded C code for various sensors
│   ├── Finalcode.c             # Core implementation for sensor control and monitoring
│   ├── meghacloudidecode.c     # Cloud IDE integration code
│   ├── modbuscode.c            # Code for Modbus protocol communication
│   ├── pzem code.c             # Code for PZEM sensor readings
│   ├── PZEM-004TCODE.c         # Code specific to PZEM-004T sensor for power measurements
│   └── Temperaturewithlcddisplaynomodbus.c  # LCD temperature display code without Modbus
└── System Design/              # Architecture and design documentation
    ├── hvac-architecture.mermaid  # HVAC system architecture diagram in mermaid format
    └── output.png              # Example output of the system architecture
```

---

## Getting Started

### Prerequisites

- **Python 3.8+**: Required for running Python scripts.
- **Embedded C Compiler**: For compiling embedded C files.
- **Mermaid (for VSCode)**: For visualizing system architecture diagrams.
- **Modbus Libraries**: Required for Modbus communication with compatible devices.

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

### Configuration

- **Modbus Setup**: Ensure Modbus-compatible devices are connected and configured as specified in `modbuscode.c`.
- **PZEM Sensor**: Connect PZEM-004T sensors according to `PZEM-004TCODE.c`.
- **HVAC Configuration**: Refer to `hvac-architecture.mermaid` for a structural overview of the HVAC system.

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

---

## Detailed File Explanations

- **Finalcode.c**: The main embedded C code for real-time sensor data collection, monitoring, and power adjustments.
- **model.py**: Implements the machine learning model for predicting power consumption. The model is trained on historical data and evaluated using `rmse.py`.
- **hvac-architecture.mermaid**: Diagram of the HVAC system architecture in mermaid syntax, outlining components involved in power management.
- **sensor_data.csv**: Sample dataset containing temperature, power, and current metrics, used for model validation.

---

## Results

Sample results are visualized in `output/output.png`, demonstrating the model's efficacy in power optimization.

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
