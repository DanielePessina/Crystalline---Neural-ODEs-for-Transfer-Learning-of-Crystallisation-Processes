"""Post-processing utilities for Dash app integration.

This module provides functions to summarize trained models into
DataFrame format suitable for display in Dash DataTables.
"""

import numpy as np
import pandas as pd


def summarize_model(model) -> pd.DataFrame:
    """Extract model summary as a DataFrame for Dash DataTable display.

    Parameters
    ----------
    model : AugmentedNeuralODE
        Trained model to summarize.

    Returns
    -------
    pd.DataFrame
        Summary table with model architecture and parameter information.
    """
    try:
        summary_data = []

        # Model architecture info
        if hasattr(model, 'func') and hasattr(model.func, 'mlp'):
            mlp = model.func.mlp

            # Count total parameters
            total_params = 0
            layer_count = 0

            if hasattr(mlp, 'layers'):
                for i, layer in enumerate(mlp.layers):
                    if hasattr(layer, 'weight') and hasattr(layer, 'bias'):
                        w_shape = layer.weight.shape if hasattr(layer.weight, 'shape') else 'Unknown'
                        b_shape = layer.bias.shape if hasattr(layer.bias, 'shape') else 'Unknown'

                        # Count parameters in this layer
                        w_params = np.prod(layer.weight.shape) if hasattr(layer.weight, 'shape') else 0
                        b_params = np.prod(layer.bias.shape) if hasattr(layer.bias, 'shape') else 0
                        layer_params = w_params + b_params
                        total_params += layer_params
                        layer_count += 1

                        summary_data.append({
                            'Component': f'Layer {i+1}',
                            'Type': 'Dense',
                            'Weight Shape': str(w_shape),
                            'Bias Shape': str(b_shape),
                            'Parameters': int(layer_params)
                        })

            # Add total summary
            summary_data.append({
                'Component': 'Total',
                'Type': 'Summary',
                'Weight Shape': f'{layer_count} layers',
                'Bias Shape': '-',
                'Parameters': int(total_params)
            })

        # Model hyperparameters if available
        if hasattr(model, 'func'):
            func = model.func

            # Augmentation dimension
            if hasattr(func, 'augment_dim'):
                summary_data.append({
                    'Component': 'Augment Dim',
                    'Type': 'Hyperparameter',
                    'Weight Shape': str(func.augment_dim),
                    'Bias Shape': '-',
                    'Parameters': 0
                })

        # Solver information
        if hasattr(model, 'solver'):
            solver_name = type(model.solver).__name__
            summary_data.append({
                'Component': 'ODE Solver',
                'Type': 'Configuration',
                'Weight Shape': solver_name,
                'Bias Shape': '-',
                'Parameters': 0
            })

        # If no model structure found, provide basic info
        if not summary_data:
            summary_data.append({
                'Component': 'Model',
                'Type': 'Unknown',
                'Weight Shape': 'No structure info',
                'Bias Shape': '-',
                'Parameters': 0
            })

    except Exception as e:
        # Fallback error handling
        summary_data = [{
            'Component': 'Error',
            'Type': 'Summary Failed',
            'Weight Shape': str(e)[:50],
            'Bias Shape': '-',
            'Parameters': 0
        }]

    return pd.DataFrame(summary_data)


def format_metrics_for_display(metrics: dict) -> pd.DataFrame:
    """Format metrics dictionary into a DataFrame for display.

    Parameters
    ----------
    metrics : dict
        Dictionary of metric names and values.

    Returns
    -------
    pd.DataFrame
        Formatted metrics table.
    """
    if not metrics:
        return pd.DataFrame({'Metric': ['No metrics'], 'Value': ['N/A']})

    data = []
    for key, value in metrics.items():
        # Format value based on type
        if isinstance(value, float):
            formatted_value = f"{value:.6f}"
        elif isinstance(value, int):
            formatted_value = str(value)
        elif hasattr(value, 'item'):  # JAX array
            formatted_value = f"{value.item():.6f}"
        else:
            formatted_value = str(value)

        data.append({
            'Metric': key,
            'Value': formatted_value
        })

    return pd.DataFrame(data)
