"""Main Dash application for Crystalline Augmented NODE training.

This application provides a web interface for training Augmented Neural ODEs
on crystallization data with interactive result visualization.
"""

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate
from sim_runner import (
    build_graphs_and_table,
    get_default_tl_values,
    get_default_values,
    parse_inputs_to_cfg,
    parse_tl_inputs_to_cfg,
    validate_inputs,
)
from training_wrapper import train_AugNODE_dash, train_AugNODE_TL_dash

# Initialize Dash app with local Bootstrap theme and custom styles
app = Dash(
    __name__,
    external_stylesheets=["/assets/bootswatch-lux.min.css", "/assets/custom-styles.css"],
    serve_locally=True,
)
app.title = "Crystalline AugNODE Trainer"
server = app.server

# Global storage for results (since this runs locally)
GLOBAL_STORAGE = {
    "model": None,
    "scaler": None,
    "progress": "",
    "training_in_progress": False,
    "training_progress": 0,
    "training_status": "Ready",
    "training_results": None,
    "current_step": 0,
    "total_steps": 0,
    "current_phase": "",
    "current_loss": None,
    # TL specific state
    "tl_training_in_progress": False,
    "tl_training_status": "Ready",
    "tl_current_step": 0,
    "tl_total_steps": 0,
    "tl_current_phase": "",
    "tl_current_loss": None,
}

# Default values
defaults = get_default_values()
tl_defaults = get_default_tl_values()


# Create the Settings Panel with two side-by-side columns
def create_settings_panel():
    """Create a compact settings panel with simulation and neural ODE groups side by side.

    Replaces the previous tabbed layout with two columns so both groups are visible
    simultaneously. This change is purely presentational; component IDs and callbacks
    remain unchanged for compatibility.
    """

    # Simulation Settings Tab Content - More compact layout
    simulation_content = dbc.Container(
        [
            # Nucleation and Growth parameters in a 4-column grid for compactness
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Nucleation A", className="small fw-bold"),
                            dbc.Input(
                                id="nucl-A",
                                type="number",
                                placeholder="A",
                                value=defaults["nucl-A"],
                                step=0.01,
                                size="sm",
                            ),
                        ],
                        width=3,
                        className="mb-2",
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Nucleation b", className="small fw-bold"),
                            dbc.Input(
                                id="nucl-b",
                                type="number",
                                placeholder="b",
                                value=defaults["nucl-b"],
                                step=0.01,
                                size="sm",
                            ),
                        ],
                        width=3,
                        className="mb-2",
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Growth k", className="small fw-bold"),
                            dbc.Input(
                                id="growth-k",
                                type="number",
                                placeholder="k",
                                value=defaults["growth-k"],
                                step=0.01,
                                size="sm",
                            ),
                        ],
                        width=3,
                        className="mb-2",
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Growth g", className="small fw-bold"),
                            dbc.Input(
                                id="growth-g",
                                type="number",
                                placeholder="g",
                                value=defaults["growth-g"],
                                step=0.01,
                                size="sm",
                            ),
                        ],
                        width=3,
                        className="mb-2",
                    ),
                ],
                className="g-2",
            ),
            # Initial conditions in a row
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Base initial conditions", className="small fw-bold"),
                            dbc.Input(
                                id="base-inits",
                                type="text",
                                size="sm",
                                placeholder="e.g., 15.5, 16.5, 19",
                                value=defaults["base-inits"],
                            ),
                            dbc.FormText("Comma-separated floats", className="small"),
                        ],
                        width=6,
                        className="mb-2",
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Test initial conditions", className="small fw-bold"),
                            dbc.Input(
                                id="test-inits",
                                type="text",
                                size="sm",
                                placeholder="e.g., 12, 14.3, 17.7, 20",
                                value=defaults["test-inits"],
                            ),
                            dbc.FormText("Comma-separated floats", className="small"),
                        ],
                        width=6,
                        className="mb-2",
                    ),
                ],
                className="g-2",
            ),
            # Noise level with compact slider
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Noise level", className="small fw-bold mb-1"),
                            dcc.Slider(
                                id="noise-level",
                                min=0,
                                max=0.25,
                                step=0.01,
                                value=defaults["noise-level"],
                                marks={0: "0", 0.125: "0.125", 0.25: "0.25"},
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ],
                        width=12,
                        className="mb-1",
                    ),
                ],
            ),
        ],
        fluid=True,
        className="compact-form py-2",
    )

    # Neural ODE Hyperparameters Tab Content - More compact layout
    neural_ode_content = dbc.Container(
        [
            # Architecture parameters in a 4-column grid
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Width", className="small fw-bold"),
                            dbc.Input(
                                id="width-size", type="number", value=defaults["width-size"], step=1, min=1, size="sm"
                            ),
                        ],
                        width=3,
                        className="mb-2",
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Depth", className="small fw-bold"),
                            dbc.Input(id="depth", type="number", value=defaults["depth"], step=1, min=1, size="sm"),
                        ],
                        width=3,
                        className="mb-2",
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Timesteps", className="small fw-bold"),
                            dbc.Input(
                                id="ntimesteps", type="number", value=defaults["ntimesteps"], step=1, min=1, size="sm"
                            ),
                        ],
                        width=3,
                        className="mb-2",
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Seed", className="small fw-bold"),
                            dbc.Input(id="seed", type="number", value=defaults["seed"], step=1, size="sm"),
                        ],
                        width=3,
                        className="mb-2",
                    ),
                ],
                className="g-2",
            ),
            # Activation function in full width, learning rates in 2 columns
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Activation Function", className="small fw-bold"),
                            dcc.Dropdown(
                                id="activation",
                                options=[
                                    {"label": "Swish", "value": "swish"},
                                    {"label": "ReLU", "value": "relu"},
                                    {"label": "Tanh", "value": "tanh"},
                                    {"label": "Sigmoid", "value": "sigmoid"},
                                    {"label": "ELU", "value": "elu"},
                                    {"label": "GELU", "value": "gelu"},
                                ],
                                value=defaults["activation"],
                                clearable=False,
                                style={"fontSize": "0.875rem"},
                            ),
                        ],
                        width=4,
                        className="mb-2",
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Learning rate low (×10⁻⁴)", className="small fw-bold"),
                            dbc.Input(
                                id="lr-low",
                                type="number",
                                value=defaults["lr-low"] * 10000,
                                placeholder="Low",
                                step=0.1,
                                min=0.1,
                                size="sm",
                            ),
                        ],
                        width=4,
                        className="mb-2",
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Learning rate high (×10⁻⁴)", className="small fw-bold"),
                            dbc.Input(
                                id="lr-high",
                                type="number",
                                value=defaults["lr-high"] * 10000,
                                placeholder="High",
                                step=0.1,
                                min=0.1,
                                size="sm",
                            ),
                        ],
                        width=4,
                        className="mb-2",
                    ),
                ],
                className="g-2",
            ),
            # Training strategy parameters in 4 columns
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Steps Stage 1", className="small fw-bold"),
                            dbc.Input(id="steps1", type="number", value=defaults["steps1"], step=1, min=1, size="sm"),
                        ],
                        width=3,
                        className="mb-2",
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Steps Stage 2", className="small fw-bold"),
                            dbc.Input(id="steps2", type="number", value=defaults["steps2"], step=1, min=1, size="sm"),
                        ],
                        width=3,
                        className="mb-2",
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Length 1", className="small fw-bold"),
                            dbc.Input(id="len1", type="number", value=defaults["len1"], step=0.01, min=0.01, size="sm"),
                        ],
                        width=3,
                        className="mb-2",
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Length 2", className="small fw-bold"),
                            dbc.Input(id="len2", type="number", value=defaults["len2"], step=0.01, min=0.01, size="sm"),
                        ],
                        width=3,
                        className="mb-2",
                    ),
                ],
                className="g-2",
            ),
            # Output constraints and time toggle
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Physical Constraints", className="small fw-bold mb-2"),
                            dbc.Checklist(
                                id="constraints",
                                options=[
                                    {"label": "y₀ ≥ 0", "value": "neg0"},
                                    {"label": "y₁ ≥ 0", "value": "pos1"},
                                ],
                                value=[],
                                switch=True,
                                inline=True,
                                className="custom-switches",
                            ),
                        ],
                        width=6,
                        className="mb-1",
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Time Embedding", className="small fw-bold mb-2"),
                            dbc.Checklist(
                                id="include-time",
                                options=[
                                    {"label": "Include Time", "value": "true"},
                                ],
                                value=[],
                                switch=True,
                                inline=True,
                                className="custom-switches",
                            ),
                        ],
                        width=6,
                        className="mb-1",
                    ),
                ],
            ),
        ],
        fluid=True,
        className="compact-form py-2",
    )

    # Side-by-side columns for settings
    settings_columns = dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Simulation Settings"),
                        dbc.CardBody(simulation_content),
                    ],
                    className="mb-2",
                ),
                md=6,
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Neural ODE Hyperparameters"),
                        dbc.CardBody(neural_ode_content),
                    ],
                    className="mb-2",
                ),
                md=6,
            ),
        ],
        className="g-2",
    )

    return dbc.Card(
        [
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Button(
                                        "RUN TRAINING", id="run-btn", color="primary", size="lg", className="w-100 mb-2"
                                    ),
                                ],
                                width=12,
                            )
                        ]
                    ),
                    settings_columns,
                ],
                className="p-2",  # Reduced padding for compactness
            ),
        ],
        className="mb-3 shadow-sm settings-card",
        style={"minHeight": "auto"},  # Allow dynamic height
    )


# Create the Center Panel (Progress + Plots) - now full width
def create_results_panel():
    """Create the results panel with progress tracking and plots."""
    progress_card = dbc.Card(
        [
            dbc.CardHeader("Training Progress"),
            dbc.CardBody(
                [
                    html.Div(id="progress-status", className="mb-3"),
                    dbc.Progress(
                        id="training-progress",
                        value=0,
                        striped=True,
                        animated=True,
                        color="primary",
                        className="mb-3",
                        style={"height": "25px"},
                    ),
                    html.Div(id="progress-details", className="small text-muted"),
                    dcc.Interval(
                        id="progress-interval",
                        n_intervals=0,
                        interval=500,  # Update every 500ms
                        disabled=True,  # Start disabled
                    ),
                ]
            ),
        ],
        className="mb-3 shadow-sm",
    )

    plots_card = dbc.Card(
        [
            dbc.CardHeader("Results"),
            dbc.CardBody(dcc.Loading(html.Div(id="plots", className="plots-grid"), type="default")),
        ],
        className="shadow-sm",
    )

    return html.Div([progress_card, plots_card])


# Create the Header Section
def create_header():
    return dbc.Card(
        [
            dbc.CardBody(
                [
                    html.H1("Crystalline AugNODE Trainer", className="text-center mb-4"),
                    dcc.Markdown(
                        """
                        Welcome to the **Crystalline Augmented Neural ODE Training Interface**. This application allows you to:

                        - Configure simulation parameters for nucleation and growth processes
                        - Train Augmented Neural ODEs with real-time progress monitoring
                        - Visualize training results with interactive plots
                        - Apply constraints and customize network architectures

                        *Adjust the parameters in the left panel and click **RUN** to start training.*
                        """,
                        className="text-muted",
                    ),
                ]
            )
        ],
        className="mb-4 shadow-sm",
    )


def create_tl_panel():
    """Create the Transfer Learning panel: intro text, two columns of settings, progress and plots."""
    # TL intro text
    intro = dbc.Card(
        [
            dbc.CardBody(
                [
                    html.H4("Transfer Learning", className="mb-2"),
                    dcc.Markdown(
                        """
                        Fine-tune a pre-trained Augmented NODE on new conditions. This uses the model and scaler from the current session.
                        Choose a freezing strategy or enable weight deviation penalty.
                        """,
                        className="text-muted",
                    ),
                ]
            )
        ],
        className="mb-3 shadow-sm",
    )

    # Left: Simulation settings (TL)
    sim_col = dbc.Card(
        [
            dbc.CardHeader("Simulation Settings"),
            dbc.CardBody(
                dbc.Container(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Nucleation A", className="small fw-bold"),
                                        dbc.Input(
                                            id="tl-nucl-A",
                                            type="number",
                                            value=tl_defaults["nucl-A"],
                                            step=0.01,
                                            size="sm",
                                        ),
                                    ],
                                    width=3,
                                    className="mb-2",
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Nucleation b", className="small fw-bold"),
                                        dbc.Input(
                                            id="tl-nucl-b",
                                            type="number",
                                            value=tl_defaults["nucl-b"],
                                            step=0.01,
                                            size="sm",
                                        ),
                                    ],
                                    width=3,
                                    className="mb-2",
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Growth k", className="small fw-bold"),
                                        dbc.Input(
                                            id="tl-growth-k",
                                            type="number",
                                            value=tl_defaults["growth-k"],
                                            step=0.01,
                                            size="sm",
                                        ),
                                    ],
                                    width=3,
                                    className="mb-2",
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Growth g", className="small fw-bold"),
                                        dbc.Input(
                                            id="tl-growth-g",
                                            type="number",
                                            value=tl_defaults["growth-g"],
                                            step=0.01,
                                            size="sm",
                                        ),
                                    ],
                                    width=3,
                                    className="mb-2",
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Base initial conditions", className="small fw-bold"),
                                        dbc.Input(
                                            id="tl-base-inits", type="text", value=tl_defaults["base-inits"], size="sm"
                                        ),
                                        dbc.FormText("Comma-separated floats", className="small"),
                                    ],
                                    width=6,
                                    className="mb-2",
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Test initial conditions", className="small fw-bold"),
                                        dbc.Input(
                                            id="tl-test-inits", type="text", value=tl_defaults["test-inits"], size="sm"
                                        ),
                                        dbc.FormText("Comma-separated floats", className="small"),
                                    ],
                                    width=6,
                                    className="mb-2",
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Noise level", className="small fw-bold mb-1"),
                                        dcc.Slider(
                                            id="tl-noise-level",
                                            min=0,
                                            max=0.25,
                                            step=0.01,
                                            value=tl_defaults["noise-level"],
                                            marks={0: "0", 0.125: "0.125", 0.25: "0.25"},
                                            tooltip={"placement": "bottom", "always_visible": True},
                                        ),
                                    ],
                                    width=12,
                                    className="mb-1",
                                ),
                            ]
                        ),
                    ],
                    fluid=True,
                    className="compact-form py-2",
                )
            ),
        ],
        className="mb-2",
    )

    # Right: TL hyperparameters and toggles
    hp_col = dbc.Card(
        [
            dbc.CardHeader("Transfer Learning Hyperparameters"),
            dbc.CardBody(
                dbc.Container(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Timesteps", className="small fw-bold"),
                                        dbc.Input(
                                            id="tl-ntimesteps",
                                            type="number",
                                            value=tl_defaults["ntimesteps"],
                                            step=1,
                                            min=1,
                                            size="sm",
                                        ),
                                    ],
                                    width=3,
                                    className="mb-2",
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Seed", className="small fw-bold"),
                                        dbc.Input(
                                            id="tl-seed", type="number", value=tl_defaults["seed"], step=1, size="sm"
                                        ),
                                    ],
                                    width=3,
                                    className="mb-2",
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("LR low (×10⁻⁴)", className="small fw-bold"),
                                        dbc.Input(
                                            id="tl-lr-low",
                                            type="number",
                                            value=tl_defaults["lr-low"] * 10000,
                                            step=0.1,
                                            min=0.1,
                                            size="sm",
                                        ),
                                    ],
                                    width=3,
                                    className="mb-2",
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("LR high (×10⁻⁴)", className="small fw-bold"),
                                        dbc.Input(
                                            id="tl-lr-high",
                                            type="number",
                                            value=tl_defaults["lr-high"] * 10000,
                                            step=0.1,
                                            min=0.1,
                                            size="sm",
                                        ),
                                    ],
                                    width=3,
                                    className="mb-2",
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Steps 1", className="small fw-bold"),
                                        dbc.Input(
                                            id="tl-steps1",
                                            type="number",
                                            value=tl_defaults["steps1"],
                                            step=1,
                                            min=1,
                                            size="sm",
                                        ),
                                    ],
                                    width=3,
                                    className="mb-2",
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Steps 2", className="small fw-bold"),
                                        dbc.Input(
                                            id="tl-steps2",
                                            type="number",
                                            value=tl_defaults["steps2"],
                                            step=1,
                                            min=1,
                                            size="sm",
                                        ),
                                    ],
                                    width=3,
                                    className="mb-2",
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Length 1", className="small fw-bold"),
                                        dbc.Input(
                                            id="tl-len1",
                                            type="number",
                                            value=tl_defaults["len1"],
                                            step=0.01,
                                            min=0.01,
                                            size="sm",
                                        ),
                                    ],
                                    width=3,
                                    className="mb-2",
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Length 2", className="small fw-bold"),
                                        dbc.Input(
                                            id="tl-len2",
                                            type="number",
                                            value=tl_defaults["len2"],
                                            step=0.01,
                                            min=0.01,
                                            size="sm",
                                        ),
                                    ],
                                    width=3,
                                    className="mb-2",
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Scale Strategy", className="small fw-bold"),
                                        dcc.Dropdown(
                                            id="tl-scale-strategy",
                                            options=[
                                                {"label": "Keep scaler", "value": "keep_scaler"},
                                                {"label": "Refit scaler", "value": "refit_scaler"},
                                            ],
                                            value=tl_defaults["scale-strategy"],
                                            clearable=False,
                                            style={"fontSize": "0.875rem"},
                                        ),
                                    ],
                                    width=4,
                                    className="mb-2",
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Freeze Layers", className="small fw-bold"),
                                        dcc.Dropdown(
                                            id="tl-idx-frozen",
                                            options=[
                                                {"label": l, "value": v}
                                                for l, v in [
                                                    ("None", "none"),
                                                    ("First", "first"),
                                                    ("First Two", "first_two"),
                                                    ("First Three", "first_three"),
                                                    ("Last", "last"),
                                                    ("Last Two", "last_two"),
                                                    ("All", "all"),
                                                ]
                                            ],
                                            value=tl_defaults["idx-frozen"],
                                            clearable=False,
                                            style={"fontSize": "0.875rem"},
                                        ),
                                    ],
                                    width=4,
                                    className="mb-2",
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Freeze Mode", className="small fw-bold"),
                                        dcc.Dropdown(
                                            id="tl-freeze-mode",
                                            options=[
                                                {"label": "Weights", "value": "weights"},
                                                {"label": "Biases", "value": "biases"},
                                                {"label": "Both", "value": "both"},
                                            ],
                                            value=tl_defaults["freeze-mode"],
                                            clearable=False,
                                            style={"fontSize": "0.875rem"},
                                        ),
                                    ],
                                    width=4,
                                    className="mb-2",
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Checklist(
                                            id="tl-penalise",
                                            options=[{"label": "Penalise Deviations", "value": "true"}],
                                            value=["true"] if tl_defaults["penalise"] else [],
                                            switch=True,
                                            inline=True,
                                            className="custom-switches",
                                        ),
                                    ],
                                    width=4,
                                    className="mb-1",
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Penalty λ", className="small fw-bold"),
                                        dbc.Input(
                                            id="tl-penalty-lambda",
                                            type="number",
                                            value=tl_defaults["penalty-lambda"],
                                            min=0.0,
                                            step=0.1,
                                            size="sm",
                                        ),
                                    ],
                                    width=4,
                                    className="mb-1",
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Penalty Strategy", className="small fw-bold"),
                                        dcc.Dropdown(
                                            id="tl-penalty-strategy",
                                            options=[
                                                {"label": l, "value": v}
                                                for l, v in [
                                                    ("All", "all"),
                                                    ("None", "none"),
                                                    ("First", "first"),
                                                    ("First Two", "first_two"),
                                                    ("First Three", "first_three"),
                                                    ("Last", "last"),
                                                    ("Last Two", "last_two"),
                                                ]
                                            ],
                                            value=tl_defaults["penalty-strategy"],
                                            clearable=False,
                                            style={"fontSize": "0.875rem"},
                                        ),
                                    ],
                                    width=4,
                                    className="mb-1",
                                ),
                            ]
                        ),
                    ],
                    fluid=True,
                    className="compact-form py-2",
                )
            ),
        ],
        className="mb-2",
    )

    settings_columns = dbc.Row([dbc.Col(sim_col, md=6), dbc.Col(hp_col, md=6)], className="g-2")

    # Progress + plots (separate IDs from fresh)
    progress_card = dbc.Card(
        [
            dbc.CardHeader("TL Progress"),
            dbc.CardBody(
                [
                    html.Div(id="tl-progress-status", className="mb-3"),
                    dbc.Progress(
                        id="tl-training-progress",
                        value=0,
                        striped=True,
                        animated=True,
                        color="primary",
                        className="mb-3",
                        style={"height": "25px"},
                    ),
                    html.Div(id="tl-progress-details", className="small text-muted"),
                    dcc.Interval(id="tl-progress-interval", n_intervals=0, interval=500, disabled=True),
                ]
            ),
        ],
        className="mb-3 shadow-sm",
    )
    plots_card = dbc.Card(
        [
            dbc.CardHeader("Results"),
            dbc.CardBody(dcc.Loading(html.Div(id="tl-plots", className="plots-grid"), type="default")),
        ],
        className="shadow-sm",
    )

    return html.Div(
        [
            intro,
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "RUN TRANSFER LEARNING", id="tl-run-btn", color="primary", size="lg", className="w-100 mb-2"
                        ),
                        width=12,
                    ),
                ]
            ),
            settings_columns,
            progress_card,
            plots_card,
        ]
    )


# App layout with header and tabs for Fresh vs TL
app.layout = html.Div(
    [
        html.Div(
            [
                create_header(),
                dbc.Tabs(
                    [
                        dbc.Tab(
                            html.Div([create_settings_panel(), create_results_panel()]),
                            label="Fresh Training",
                            tab_id="tab-fresh",
                        ),
                        dbc.Tab(create_tl_panel(), label="Transfer Learning", tab_id="tab-tl"),
                    ],
                    id="mode-tabs",
                    active_tab="tab-fresh",
                    className="mb-2",
                ),
            ],
            className="container-fluid px-4",
        )
    ]
)


# Progress bar update callback
@app.callback(
    [
        Output("training-progress", "value"),
        Output("training-progress", "label"),
        Output("progress-status", "children"),
        Output("progress-details", "children"),
        Output("progress-interval", "disabled"),
    ],
    [Input("progress-interval", "n_intervals")],
    prevent_initial_call=True,
)
def update_progress_bar(n_intervals):
    """Update the progress bar based on global training state."""
    print(
        f"DEBUG: Progress callback triggered - interval #{n_intervals}, training_in_progress: {GLOBAL_STORAGE['training_in_progress']}"
    )

    if not GLOBAL_STORAGE["training_in_progress"]:
        # Training not running, disable interval
        return 0, "", "Ready to train", "", True

    current_step = GLOBAL_STORAGE["current_step"]
    total_steps = GLOBAL_STORAGE["total_steps"]

    if total_steps > 0:
        progress_value = min((current_step / total_steps) * 100, 100)
    else:
        progress_value = 0

    # Create progress label
    label = f"{progress_value:.1f}%" if progress_value >= 5 else ""

    # Status message
    status = GLOBAL_STORAGE["training_status"]

    # Detailed information
    details = []
    if GLOBAL_STORAGE["current_phase"]:
        details.append(f"Phase: {GLOBAL_STORAGE['current_phase']}")
    if current_step > 0 and total_steps > 0:
        details.append(f"Step: {current_step}/{total_steps}")
    if GLOBAL_STORAGE["current_loss"] is not None:
        details.append(f"Loss: {GLOBAL_STORAGE['current_loss']:.3e}")

    details_text = " | ".join(details)

    return progress_value, label, status, details_text, False


# Callback to enable progress interval when training starts
@app.callback(
    Output("progress-interval", "disabled", allow_duplicate=True),
    [Input("run-btn", "n_clicks")],
    prevent_initial_call=True,
)
def enable_progress_interval(n_clicks):
    """Enable the progress interval when training button is clicked."""
    print(f"DEBUG: Enable interval callback - n_clicks: {n_clicks}")
    return n_clicks == 0  # Disabled when n_clicks is 0, enabled otherwise


# TL Progress bar update
@app.callback(
    [
        Output("tl-training-progress", "value"),
        Output("tl-training-progress", "label"),
        Output("tl-progress-status", "children"),
        Output("tl-progress-details", "children"),
        Output("tl-progress-interval", "disabled"),
    ],
    [Input("tl-progress-interval", "n_intervals")],
    prevent_initial_call=True,
)
def update_tl_progress_bar(n_intervals):
    if not GLOBAL_STORAGE["tl_training_in_progress"]:
        return 0, "", "Ready to TL-train", "", True
    cs = GLOBAL_STORAGE["tl_current_step"]
    ts = GLOBAL_STORAGE["tl_total_steps"]
    pv = min((cs / ts) * 100, 100) if ts > 0 else 0
    label = f"{pv:.1f}%" if pv >= 5 else ""
    status = GLOBAL_STORAGE["tl_training_status"]
    details = []
    if GLOBAL_STORAGE["tl_current_phase"]:
        details.append(f"Phase: {GLOBAL_STORAGE['tl_current_phase']}")
    if cs > 0 and ts > 0:
        details.append(f"Step: {cs}/{ts}")
    if GLOBAL_STORAGE["tl_current_loss"] is not None:
        details.append(f"Loss: {GLOBAL_STORAGE['tl_current_loss']:.3e}")
    return pv, label, status, " | ".join(details), False


@app.callback(
    Output("tl-progress-interval", "disabled", allow_duplicate=True),
    [Input("tl-run-btn", "n_clicks")],
    prevent_initial_call=True,
)
def tl_enable_progress_interval(n_clicks):
    return n_clicks == 0


# Training callback with background processing and progress updates
@app.callback(
    [
        Output("plots", "children"),
        Output("run-btn", "disabled"),
    ],
    [Input("run-btn", "n_clicks")],
    [
        State("nucl-A", "value"),
        State("nucl-b", "value"),
        State("growth-k", "value"),
        State("growth-g", "value"),
        State("base-inits", "value"),
        State("test-inits", "value"),
        State("noise-level", "value"),
        State("width-size", "value"),
        State("depth", "value"),
        State("ntimesteps", "value"),
        State("seed", "value"),
        State("activation", "value"),
        State("lr-low", "value"),
        State("lr-high", "value"),
        State("steps1", "value"),
        State("steps2", "value"),
        State("len1", "value"),
        State("len2", "value"),
        State("constraints", "value"),
        State("include-time", "value"),
    ],
    prevent_initial_call=True,
)
def run_training(
    n_clicks,
    nucl_A,
    nucl_b,
    growth_k,
    growth_g,
    base_inits,
    test_inits,
    noise_level,
    width_size,
    depth,
    ntimesteps,
    seed,
    activation,
    lr_low,
    lr_high,
    steps1,
    steps2,
    len1,
    len2,
    constraints,
    include_time,
):
    """Execute training with real-time progress monitoring."""
    print(f"DEBUG: Training callback triggered - n_clicks: {n_clicks}")

    if n_clicks == 0 or n_clicks is None:
        print("DEBUG: Training callback prevented - no clicks")
        raise PreventUpdate

    try:
        # Reset global state
        print("DEBUG: Resetting global state...")
        GLOBAL_STORAGE["training_in_progress"] = True
        GLOBAL_STORAGE["current_step"] = 0
        GLOBAL_STORAGE["total_steps"] = 0
        GLOBAL_STORAGE["training_status"] = "Initializing..."
        GLOBAL_STORAGE["current_phase"] = ""
        GLOBAL_STORAGE["current_loss"] = None
        print(f"DEBUG: Global state reset, training_in_progress: {GLOBAL_STORAGE['training_in_progress']}")

        # Validate inputs first
        base_inits_str = base_inits or ""
        test_inits_str = test_inits or ""
        print(f"DEBUG: Validating inputs - base_inits: '{base_inits_str}', test_inits: '{test_inits_str}'")
        is_valid, error_msg = validate_inputs(base_inits_str, test_inits_str)

        if not is_valid:
            print(f"DEBUG: Input validation failed: {error_msg}")
            GLOBAL_STORAGE["training_in_progress"] = False
            GLOBAL_STORAGE["training_status"] = f"Error: {error_msg}"
            return [], False  # Return normal values

        print("DEBUG: Input validation passed, parsing configuration...")
        # Parse configuration with learning rate conversion
        cfg = parse_inputs_to_cfg(
            nucl_A,
            nucl_b,
            growth_k,
            growth_g,
            base_inits,
            test_inits,
            noise_level,
            width_size,
            depth,
            ntimesteps,
            seed,
            activation,
            lr_low,
            lr_high,
            steps1,
            steps2,
            len1,
            len2,
            constraints,
            include_time,
        )

        # Convert learning rates from the ×10⁻⁴ format back to actual values
        lr_low_display, lr_high_display = lr_low, lr_high  # lr-low and lr-high values
        actual_lr_low = (lr_low_display or 4) * 1e-4
        actual_lr_high = (lr_high_display or 10) * 1e-4
        print(f"DEBUG: Learning rates - low: {actual_lr_low}, high: {actual_lr_high}")

        # Override the learning rate strategy in the config using dataclasses.replace
        import dataclasses

        cfg = dataclasses.replace(cfg, lr_strategy=(actual_lr_low, actual_lr_high))
        GLOBAL_STORAGE["total_steps"] = sum(cfg.steps_strategy)
        GLOBAL_STORAGE["training_status"] = "Starting training..."
        print(f"DEBUG: Configuration ready, total steps: {GLOBAL_STORAGE['total_steps']}")

        # Create progress updater that updates global storage
        def update_global_progress(current_step, total_steps, phase="", loss=None, status="Training..."):
            print(f"DEBUG: Progress update - Step {current_step}/{total_steps}, Phase: {phase}, Status: {status}")
            GLOBAL_STORAGE["current_step"] = current_step
            GLOBAL_STORAGE["total_steps"] = total_steps
            GLOBAL_STORAGE["current_phase"] = phase
            GLOBAL_STORAGE["current_loss"] = loss
            GLOBAL_STORAGE["training_status"] = status

        print("DEBUG: Starting actual training...")
        # Run training with progress updates
        res = train_AugNODE_dash(cfg, progress_updater=update_global_progress)
        print("DEBUG: Training completed successfully")  # Store results globally
        GLOBAL_STORAGE["model"] = res.model
        GLOBAL_STORAGE["scaler"] = res.scaler
        GLOBAL_STORAGE["training_results"] = res

        # Build outputs with responsive plots
        graphs, columns, data = build_graphs_and_table(res)

        # Ensure plots are responsive and properly sized for 2x2 subplot layout
        for graph_div in graphs:
            if hasattr(graph_div, "children") and hasattr(graph_div.children, "figure"):
                fig = graph_div.children.figure
                fig.update_layout(
                    autosize=True,
                    width=None,  # Let CSS handle width
                    height=600,  # Fixed height for 2x2 subplot visibility
                    margin={"l": 40, "r": 40, "t": 60, "b": 40},
                    legend={"orientation": "h", "yanchor": "bottom", "y": -0.15, "xanchor": "center", "x": 0.5},
                )

        # Training completed
        GLOBAL_STORAGE["training_in_progress"] = False
        GLOBAL_STORAGE["training_status"] = "Training completed successfully!"
        GLOBAL_STORAGE["current_step"] = GLOBAL_STORAGE["total_steps"]

        return graphs, False  # Return normal values

    except Exception as e:
        print(f"DEBUG: Exception caught in training: {type(e).__name__}: {str(e)}")
        import traceback

        print(f"DEBUG: Full traceback: {traceback.format_exc()}")
        GLOBAL_STORAGE["training_in_progress"] = False
        GLOBAL_STORAGE["training_status"] = f"Training failed: {str(e)}"
        return [], False  # Return normal values

    finally:
        GLOBAL_STORAGE["training_in_progress"] = False


# TL Training callback
@app.callback(
    [Output("tl-plots", "children"), Output("tl-run-btn", "disabled")],
    [Input("tl-run-btn", "n_clicks")],
    [
        State("tl-nucl-A", "value"),
        State("tl-nucl-b", "value"),
        State("tl-growth-k", "value"),
        State("tl-growth-g", "value"),
        State("tl-base-inits", "value"),
        State("tl-test-inits", "value"),
        State("tl-noise-level", "value"),
        State("tl-ntimesteps", "value"),
        State("tl-seed", "value"),
        State("tl-lr-low", "value"),
        State("tl-lr-high", "value"),
        State("tl-steps1", "value"),
        State("tl-steps2", "value"),
        State("tl-len1", "value"),
        State("tl-len2", "value"),
        State("tl-scale-strategy", "value"),
        State("tl-idx-frozen", "value"),
        State("tl-freeze-mode", "value"),
        State("tl-penalise", "value"),
        State("tl-penalty-lambda", "value"),
        State("tl-penalty-strategy", "value"),
    ],
    prevent_initial_call=True,
)
def run_tl_training(
    n_clicks,
    nucl_A,
    nucl_b,
    growth_k,
    growth_g,
    base_inits,
    test_inits,
    noise_level,
    ntimesteps,
    seed,
    lr_low_display,
    lr_high_display,
    steps1,
    steps2,
    len1,
    len2,
    scale_strategy,
    idx_frozen,
    freeze_mode,
    penalise,
    penalty_lambda,
    penalty_strategy,
):
    if n_clicks == 0 or n_clicks is None:
        raise PreventUpdate

    if GLOBAL_STORAGE["model"] is None or GLOBAL_STORAGE["scaler"] is None:
        GLOBAL_STORAGE["tl_training_status"] = "Error: No base model found. Train a model first."
        GLOBAL_STORAGE["tl_training_in_progress"] = False
        return [], False

    try:
        GLOBAL_STORAGE["tl_training_in_progress"] = True
        GLOBAL_STORAGE["tl_current_step"] = 0
        GLOBAL_STORAGE["tl_total_steps"] = 0
        GLOBAL_STORAGE["tl_training_status"] = "Initializing TL…"
        GLOBAL_STORAGE["tl_current_phase"] = ""
        GLOBAL_STORAGE["tl_current_loss"] = None

        base_inits_str = base_inits or ""
        test_inits_str = test_inits or ""
        is_valid, error_msg = validate_inputs(base_inits_str, test_inits_str)
        if not is_valid:
            GLOBAL_STORAGE["tl_training_in_progress"] = False
            GLOBAL_STORAGE["tl_training_status"] = f"Error: {error_msg}"
            return [], False

        # Convert LR inputs from x1e-4 scale back
        actual_lr_low = (lr_low_display or 6) * 1e-4
        actual_lr_high = (lr_high_display or 6) * 1e-4

        cfg = parse_tl_inputs_to_cfg(
            nucl_A,
            nucl_b,
            growth_k,
            growth_g,
            base_inits,
            test_inits,
            noise_level,
            ntimesteps,
            seed,
            actual_lr_low,
            actual_lr_high,
            steps1,
            steps2,
            len1,
            len2,
            scale_strategy,
            idx_frozen,
            freeze_mode,
            penalise,
            penalty_lambda,
            penalty_strategy,
        )

        GLOBAL_STORAGE["tl_total_steps"] = sum(cfg.steps_strategy)
        GLOBAL_STORAGE["tl_training_status"] = "Starting TL…"

        def update_tl_progress(current_step, total_steps, phase="", loss=None, status="Training…"):
            GLOBAL_STORAGE["tl_current_step"] = current_step
            GLOBAL_STORAGE["tl_total_steps"] = total_steps
            GLOBAL_STORAGE["tl_current_phase"] = phase
            GLOBAL_STORAGE["tl_current_loss"] = loss
            GLOBAL_STORAGE["tl_training_status"] = status

        res = train_AugNODE_TL_dash(
            cfg,
            GLOBAL_STORAGE["model"],
            GLOBAL_STORAGE["scaler"],
            progress_updater=update_tl_progress,
        )

        graphs, columns, data = build_graphs_and_table(res)
        for graph_div in graphs:
            if hasattr(graph_div, "children") and hasattr(graph_div.children, "figure"):
                fig = graph_div.children.figure
                fig.update_layout(
                    autosize=True,
                    width=None,
                    height=600,
                    margin={"l": 40, "r": 40, "t": 60, "b": 40},
                    legend={"orientation": "h", "yanchor": "bottom", "y": -0.15, "xanchor": "center", "x": 0.5},
                )

        GLOBAL_STORAGE["tl_training_in_progress"] = False
        GLOBAL_STORAGE["tl_training_status"] = "TL completed successfully!"
        GLOBAL_STORAGE["tl_current_step"] = GLOBAL_STORAGE["tl_total_steps"]

        return graphs, False

    except Exception as e:
        import traceback

        print(f"DEBUG: TL Exception: {type(e).__name__}: {str(e)}")
        print(f"DEBUG: TL Traceback: {traceback.format_exc()}")
        GLOBAL_STORAGE["tl_training_in_progress"] = False
        GLOBAL_STORAGE["tl_training_status"] = f"TL failed: {str(e)}"
        return [], False
    finally:
        GLOBAL_STORAGE["tl_training_in_progress"] = False


@app.callback(
    [Output("tl-penalty-lambda", "disabled"), Output("tl-penalty-strategy", "disabled")],
    [Input("tl-penalise", "value")],
)
def toggle_penalty_inputs(values):
    enabled = bool(values and "true" in values)
    return (not enabled, not enabled)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=True)
