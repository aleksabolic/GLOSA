# Traffic GLosa Algorithm

This project implements a baseline GLosa (Graph-based Localized Spatial-Temporal) algorithm for traffic prediction. The GLosa algorithm leverages historical traffic data to predict future traffic conditions, providing valuable insights for traffic management and planning.

## Project Structure

```
traffic-glosa
├── src
│   ├── glosa.py        # Implementation of the GLosa algorithm
│   ├── visualize.py    # Visualization of traffic data and GLosa results
│   └── utils.py        # Utility functions for data handling
├── requirements.txt     # List of project dependencies
└── README.md            # Project documentation
```

## Installation

To set up the environment, you need to have Python installed on your machine. It is recommended to create a virtual environment for this project. You can do this using the following commands:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate

# On Ubuntu (fish)
source venv/bin/activate.fish
```

Once the virtual environment is activated, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

After setting up the environment, you can run the GLosa algorithm and visualize the results. Here are some example commands:

1. **Run the GLosa Algorithm**:
   ```bash
   python src/glosa.py
   ```

2. **Visualize the Results**:
   ```bash
   python src/visualize.py
   ```

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.