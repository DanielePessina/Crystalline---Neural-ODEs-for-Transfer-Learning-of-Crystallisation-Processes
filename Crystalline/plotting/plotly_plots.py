"""Interactive plotting utilities using Plotly for training and evaluation results."""

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def setup_plotly_output(mode="notebook"):
    """Setup Plotly output mode.

    Args:
        mode: Either "notebook" for VS Code interactive window or "browser" for standalone HTML
    """
    if mode == "notebook":
        # For VS Code/Jupyter notebooks
        import plotly.io as pio

        pio.renderers.default = "notebook"
    elif mode == "browser":
        # For browser display
        import plotly.io as pio

        pio.renderers.default = "browser"
    else:
        raise ValueError("Mode must be either 'notebook' or 'browser'")


def interactive_splitplot_model_vs_data(
    ts,
    ys_train,
    ys_test,
    model,
    scaler,
    length_strategy,
    extratitlestring,
    saveplot=False,
    filename_prefix="neural_ode_plot",
    output_mode="notebook",
):
    """Interactive Plotly version of splitplot_model_vs_data function.

    Creates an interactive 2x2 subplot layout showing model predictions vs data
    for both training and test sets, with separate plots for each output dimension.

    Args:
        ts: Time array for model evaluation
        ys_train: Training data array, shape (n_samples, n_timepoints, n_features)
        ys_test: Test data array, shape (n_samples, n_timepoints, n_features)
        model: Trained model with __call__ method for predictions
        scaler: Fitted scaler for inverse transformation
        length_strategy: List of training length fractions for vertical lines
        extratitlestring: Additional title string
        saveplot: Whether to save the plot (creates HTML file)
        filename_prefix: Prefix for saved filename
        output_mode: "notebook" for VS Code interactive window, "browser" for HTML file
    """
    # Setup Plotly output
    setup_plotly_output(output_mode)

    # Extract dimensions
    n_train = len(ys_train)
    n_test = len(ys_test)
    data_size = ys_train.shape[-1]
    length_size = ys_train.shape[1]

    # Create color palettes
    train_colors = pc.qualitative.Set2[:n_train] if n_train <= 8 else pc.qualitative.Set2 * (n_train // 8 + 1)
    test_colors = pc.qualitative.Dark2[:n_test] if n_test <= 8 else pc.qualitative.Dark2 * (n_test // 8 + 1)

    # Time array for plotting (matching original scale)
    ts_pred = np.linspace(0, 300, length_size)

    # Create subplot figure with 2x2 layout
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Training Set - Concentration",
            "Test Set - Concentration",
            "Training Set - D43",
            "Test Set - D43",
        ),
        shared_xaxes=True,
        shared_yaxes="rows",
        vertical_spacing=0.1,
        horizontal_spacing=0.08,
    )

    # Process training data and predictions
    for i, color in enumerate(train_colors[:n_train]):
        # Unscale the training data
        y_unscaled = scaler.inverse_transform(np.array(ys_train[i]).reshape(-1, data_size)).reshape(
            length_size, data_size
        )

        # Add scatter plots for real training data
        fig.add_trace(
            go.Scatter(
                x=ts_pred,
                y=y_unscaled[:, 0],
                mode="markers",
                marker={"color": color, "size": 6, "opacity": 0.7},
                name="Real Train Data" if i == 0 else f"Train Data {i + 1}",
                showlegend=(i == 0),  # Only show legend for first trace of each type
                legendgroup="train_data",
                hovertemplate=(
                    "<b>Train Data %{fullData.name}</b><br>"
                    "Time: %{x:.2f} min<br>"
                    "Concentration: %{y:.4f} mg/ml<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=ts_pred,
                y=y_unscaled[:, -1],
                mode="markers",
                marker={"color": color, "size": 6, "opacity": 0.7},
                name="Real Train Data" if i == 0 else f"Train Data {i + 1}",
                showlegend=False,
                legendgroup="train_data",
                hovertemplate=(
                    "<b>Train Data %{fullData.name}</b><br>Time: %{x:.2f} min<br>D43: %{y:.4f} µm<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

        # Generate and add model predictions for training data
        y_pred = np.array(model(ts, ys_train[i, 0])).reshape(-1, data_size)
        y_unscaled_pred = scaler.inverse_transform(y_pred).reshape(length_size, data_size)

        fig.add_trace(
            go.Scatter(
                x=ts_pred,
                y=y_unscaled_pred[:, 0],
                mode="lines",
                line={"color": color, "width": 2, "dash": "solid"},
                name="Model Prediction" if i == 0 else f"Train Pred {i + 1}",
                showlegend=(i == 0),
                legendgroup="train_pred",
                hovertemplate=(
                    "<b>Train Prediction %{fullData.name}</b><br>"
                    "Time: %{x:.2f} min<br>"
                    "Concentration: %{y:.4f} mg/ml<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=ts_pred,
                y=y_unscaled_pred[:, -1],
                mode="lines",
                line={"color": color, "width": 2, "dash": "solid"},
                name="Model Prediction" if i == 0 else f"Train Pred {i + 1}",
                showlegend=False,
                legendgroup="train_pred",
                hovertemplate=(
                    "<b>Train Prediction %{fullData.name}</b><br>Time: %{x:.2f} min<br>D43: %{y:.4f} µm<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    # Process test data and predictions
    for i, color in enumerate(test_colors[:n_test]):
        # Unscale the test data
        y_unscaled = scaler.inverse_transform(np.array(ys_test[i]).reshape(-1, data_size)).reshape(
            length_size, data_size
        )

        # Add scatter plots for real test data
        fig.add_trace(
            go.Scatter(
                x=ts_pred,
                y=y_unscaled[:, 0],
                mode="markers",
                marker={"color": color, "size": 7, "opacity": 0.7},
                name="Real Test Data" if i == 0 else f"Test Data {i + 1}",
                showlegend=(i == 0),
                legendgroup="test_data",
                hovertemplate=(
                    "<b>Test Data %{fullData.name}</b><br>"
                    "Time: %{x:.2f} min<br>"
                    "Concentration: %{y:.4f} mg/ml<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=ts_pred,
                y=y_unscaled[:, -1],
                mode="markers",
                marker={"color": color, "size": 7, "opacity": 0.7},
                name="Real Test Data" if i == 0 else f"Test Data {i + 1}",
                showlegend=False,
                legendgroup="test_data",
                hovertemplate=(
                    "<b>Test Data %{fullData.name}</b><br>Time: %{x:.2f} min<br>D43: %{y:.4f} µm<extra></extra>"
                ),
            ),
            row=2,
            col=2,
        )

        # Generate and add model predictions for test data
        y_pred = np.array(model(ts, ys_test[i, 0])).reshape(-1, data_size)
        y_unscaled_pred = scaler.inverse_transform(y_pred).reshape(length_size, data_size)

        fig.add_trace(
            go.Scatter(
                x=ts_pred,
                y=y_unscaled_pred[:, 0],
                mode="lines",
                line={"color": color, "width": 2, "dash": "solid"},
                name="Model Prediction" if i == 0 else f"Test Pred {i + 1}",
                showlegend=False,  # Don't show again since we already showed it for training
                legendgroup="test_pred",
                hovertemplate=(
                    "<b>Test Prediction %{fullData.name}</b><br>"
                    "Time: %{x:.2f} min<br>"
                    "Concentration: %{y:.4f} mg/ml<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=ts_pred,
                y=y_unscaled_pred[:, -1],
                mode="lines",
                line={"color": color, "width": 2, "dash": "solid"},
                name="Model Prediction" if i == 0 else f"Test Pred {i + 1}",
                showlegend=False,
                legendgroup="test_pred",
                hovertemplate=(
                    "<b>Test Prediction %{fullData.name}</b><br>Time: %{x:.2f} min<br>D43: %{y:.4f} µm<extra></extra>"
                ),
            ),
            row=2,
            col=2,
        )

    # Add vertical lines for training splits
    for length in length_strategy:
        split_time = ts_pred[int(length_size * length)] if int(length_size * length) < len(ts_pred) else ts_pred[-1]
        for row in [1, 2]:
            for col in [1, 2]:
                fig.add_vline(
                    x=split_time, line_dash="dash", line_color="red", opacity=0.7, line_width=2, row=row, col=col
                )

    # Update layout
    fig.update_layout(
        title={"text": f"Interactive Plot: {extratitlestring}", "x": 0.5, "font": {"size": 16}},
        width=1000,
        height=600,
        showlegend=True,
        legend={"orientation": "h", "yanchor": "top", "y": -0.15, "xanchor": "center", "x": 0.5, "tracegroupgap": 10},
        hovermode="closest",
    )

    # Update axes labels
    fig.update_xaxes(title_text="Time [min]", row=2, col=1)
    fig.update_xaxes(title_text="Time [min]", row=2, col=2)
    fig.update_yaxes(title_text="Concentration [mg/ml]", row=1, col=1)
    fig.update_yaxes(title_text="Concentration [mg/ml]", row=1, col=2)
    fig.update_yaxes(title_text="D43 [µm]", row=2, col=1)
    fig.update_yaxes(title_text="D43 [µm]", row=2, col=2)

    # Save if requested
    if saveplot:
        if output_mode == "browser":
            fig.write_html(f"{filename_prefix}.html")
        else:
            # For notebook mode, we can still save to HTML
            fig.write_html(f"{filename_prefix}.html")
            print(f"Plot saved to {filename_prefix}.html")

    # Display the plot
    fig.show()

    return fig


def interactive_splitplot_model_vs_data_1d(
    ts,
    ys_train,
    ys_test,
    model,
    scaler,
    length_strategy,
    extratitlestring,
    saveplot=False,
    filename_prefix="neural_ode_plot",
    output_mode="notebook",
):
    """Interactive Plotly version for 1D data (single output dimension).

    Similar to interactive_splitplot_model_vs_data but for 1D data,
    creating only train/test comparison plots.
    """
    # Setup Plotly output
    setup_plotly_output(output_mode)

    # Extract dimensions
    n_train = len(ys_train)
    n_test = len(ys_test)
    data_size = ys_train.shape[-1]
    length_size = ys_train.shape[1]

    if data_size != 1:
        # Call the 2D version for multi-dimensional data
        return interactive_splitplot_model_vs_data(
            ts,
            ys_train,
            ys_test,
            model,
            scaler,
            length_strategy,
            extratitlestring,
            saveplot,
            filename_prefix,
            output_mode,
        )

    # Create color palettes
    train_colors = pc.qualitative.Set2[:n_train] if n_train <= 8 else pc.qualitative.Set2 * (n_train // 8 + 1)
    test_colors = pc.qualitative.Dark2[:n_test] if n_test <= 8 else pc.qualitative.Dark2 * (n_test // 8 + 1)

    # Time array for plotting
    ts_pred = np.linspace(0, 300, length_size)

    # Create subplot figure with 1x2 layout
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Training Set", "Test Set"), shared_yaxes=True, horizontal_spacing=0.08
    )

    # Process training data and predictions
    for i, color in enumerate(train_colors[:n_train]):
        # Unscale the training data
        y_unscaled = scaler.inverse_transform(np.array(ys_train[i]).reshape(-1, data_size)).reshape(
            length_size, data_size
        )

        # Add scatter plots for real training data
        fig.add_trace(
            go.Scatter(
                x=ts_pred,
                y=y_unscaled[:, 0],
                mode="markers",
                marker={"color": color, "size": 6, "opacity": 0.7},
                name="Real Train Data" if i == 0 else f"Train Data {i + 1}",
                showlegend=(i == 0),
                legendgroup="train_data",
                hovertemplate=(
                    "<b>Train Data %{fullData.name}</b><br>Time: %{x:.2f} min<br>Value: %{y:.4f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

        # Generate and add model predictions for training data
        y_pred = np.array(model(ts, ys_train[i, 0])).reshape(-1, data_size)
        y_unscaled_pred = scaler.inverse_transform(y_pred).reshape(length_size, data_size)

        fig.add_trace(
            go.Scatter(
                x=ts_pred,
                y=y_unscaled_pred[:, 0],
                mode="lines",
                line={"color": color, "width": 2, "dash": "solid"},
                name="Model Prediction" if i == 0 else f"Train Pred {i + 1}",
                showlegend=(i == 0),
                legendgroup="train_pred",
                hovertemplate=(
                    "<b>Train Prediction %{fullData.name}</b><br>Time: %{x:.2f} min<br>Value: %{y:.4f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    # Process test data and predictions
    for i, color in enumerate(test_colors[:n_test]):
        # Unscale the test data
        y_unscaled = scaler.inverse_transform(np.array(ys_test[i]).reshape(-1, data_size)).reshape(
            length_size, data_size
        )

        # Add scatter plots for real test data
        fig.add_trace(
            go.Scatter(
                x=ts_pred,
                y=y_unscaled[:, 0],
                mode="markers",
                marker={"color": color, "size": 7, "opacity": 0.7},
                name="Real Test Data" if i == 0 else f"Test Data {i + 1}",
                showlegend=(i == 0),
                legendgroup="test_data",
                hovertemplate=(
                    "<b>Test Data %{fullData.name}</b><br>Time: %{x:.2f} min<br>Value: %{y:.4f}<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

        # Generate and add model predictions for test data
        y_pred = np.array(model(ts, ys_test[i, 0])).reshape(-1, data_size)
        y_unscaled_pred = scaler.inverse_transform(y_pred).reshape(length_size, data_size)

        fig.add_trace(
            go.Scatter(
                x=ts_pred,
                y=y_unscaled_pred[:, 0],
                mode="lines",
                line={"color": color, "width": 2, "dash": "solid"},
                name="Model Prediction" if i == 0 else f"Test Pred {i + 1}",
                showlegend=(i == 0),
                legendgroup="test_pred",
                hovertemplate=(
                    "<b>Test Prediction %{fullData.name}</b><br>Time: %{x:.2f} min<br>Value: %{y:.4f}<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

    # Add vertical lines for training splits
    for length in length_strategy:
        split_time = ts_pred[int(length_size * length)] if int(length_size * length) < len(ts_pred) else ts_pred[-1]
        for col in [1, 2]:
            fig.add_vline(x=split_time, line_dash="dash", line_color="red", opacity=0.7, line_width=2, row=1, col=col)

    # Update layout
    fig.update_layout(
        title={"text": f"Interactive 1D Plot: {extratitlestring}", "x": 0.5, "font": {"size": 16}},
        width=1000,
        height=400,
        showlegend=True,
        legend={"orientation": "h", "yanchor": "top", "y": -0.15, "xanchor": "center", "x": 0.5, "tracegroupgap": 10},
        hovermode="closest",
    )

    # Update axes labels
    fig.update_xaxes(title_text="Time [min]", row=1, col=1)
    fig.update_xaxes(title_text="Time [min]", row=1, col=2)
    fig.update_yaxes(title_text="Value", row=1, col=1)

    # Save if requested
    if saveplot:
        if output_mode == "browser":
            fig.write_html(f"{filename_prefix}.html")
        else:
            fig.write_html(f"{filename_prefix}.html")
            print(f"Plot saved to {filename_prefix}.html")

    # Display the plot
    fig.show()

    return fig


# Example usage function
def example_usage():
    """Example of how to use the interactive plotting functions."""
    print("Example usage:")
    print("1. For VS Code interactive window:")
    print("   interactive_splitplot_model_vs_data(ts, ys_train, ys_test, model, scaler,")
    print("                                       length_strategy, 'My Model', output_mode='notebook')")
    print()
    print("2. For browser window:")
    print("   interactive_splitplot_model_vs_data(ts, ys_train, ys_test, model, scaler,")
    print("                                       length_strategy, 'My Model', output_mode='browser')")
    print()
    print("3. The function will automatically open the appropriate output based on output_mode")
    print("4. Hover over points/lines to see detailed information including series name, x, and y values")
