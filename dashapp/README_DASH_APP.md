# Crystalline Dash App

A web interface for training Augmented Neural ODEs on crystallization data using the Crystalline package.

## Features

- **Interactive Training Interface**: Set hyperparameters through a web UI
- **Real-time Progress**: See training progress in a terminal-style display
- **Interactive Visualizations**: View training results with Plotly charts
- **Model Summary**: Inspect trained model architecture and parameters
- **Local Deployment**: Runs entirely locally for data security

## Getting Started

### Prerequisites

- Python 3.12+
- uv package manager

### Installation

1. Navigate to the project directory:
   ```bash
   cd "/Users/danielepessina/Documents/Local Uni/GUI_Crystalline"
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

### Running the App

Start the Dash application:
```bash
uv run python app.py
```

The app will be available at: http://127.0.0.1:8050

## Usage

### Input Parameters

**Nucleation Parameters**
- A (nucleation rate): Controls the rate of crystal nucleation
- b (nucleation exponent): Exponent in the nucleation rate equation

**Growth Parameters**
- k (growth rate): Controls the rate of crystal growth
- g (growth exponent): Exponent in the growth rate equation

**Initial Concentrations**
- Base training: Comma-separated list of initial concentrations for training
- Test: Comma-separated list of initial concentrations for testing

**Data Parameters**
- Noise level: Amount of noise to add to simulated data
- Number of time steps: Number of time points in simulation
- Random seed: Seed for reproducible results

**Network Architecture**
- Width: Number of neurons per layer
- Depth: Number of hidden layers
- Activation function: Choice of activation function (Swish, ReLU, Tanh, etc.)

**Training Parameters**
- Learning rate (low/high): Range for learning rate scheduling
- Training steps (phase 1/2): Number of optimization steps in each phase
- Length strategy (low/high): Range for length scheduling

**Output Constraints**
- Negative constraint on output 0: Enforce non-positive values
- Positive constraint on output 1: Enforce non-negative values

### Output

**Training Progress**
- Real-time updates showing training progress
- Error messages if training fails
- Completion status

**Visualizations**
- Training and testing trajectory plots
- Model vs. data comparisons
- Metrics visualization

**Model Summary**
- Network architecture details
- Parameter counts per layer
- Total model parameters

## Technical Implementation

### Architecture

The app consists of several key components:

1. **`training_wrapper.py`**: Clean API wrapper around the Crystalline training functions
2. **`sim_runner.py`**: Input parsing and utility functions
3. **`postproc.py`**: Model summarization and data formatting
4. **`app.py`**: Main Dash application with UI and callbacks

### Data Flow

1. User inputs parameters through the web interface
2. Parameters are validated and converted to `TrainConfig` object
3. Training runs with progress callbacks updating the UI
4. Results are processed into plots and summary tables
5. All data is stored in memory for the session

### Key Features

- **Simplified Progress Tracking**: Uses in-memory progress aggregation instead of complex streaming
- **Local Storage**: Model and scaler objects stored in global memory for fast access
- **Error Handling**: Comprehensive error catching and user-friendly error messages
- **Input Validation**: Real-time validation of user inputs with visual feedback

## Example Usage

1. **Quick Test**: Use the default values and click "Run Training" for a quick demo
2. **Custom Experiment**:
   - Set base training concentrations: `13.0, 15.5, 16.5, 19.0`
   - Set test concentrations: `12.0, 14.3, 17.7, 20.0`
   - Adjust network size for faster/slower training
   - Click "Run Training" and watch the progress

## Development Notes

- The app runs entirely locally and doesn't require external services
- Training results are stored in memory and will be lost when the session ends
- For production use, consider adding database storage for persistent results
- The interface is optimized for research workflows with quick iteration

## Troubleshooting

**Import Errors**: Ensure all dependencies are installed with `uv sync`

**Training Failures**: Check the progress terminal for detailed error messages

**Performance Issues**: Reduce network size or number of training steps for faster training

**Browser Issues**: Try refreshing the page or using a different browser
