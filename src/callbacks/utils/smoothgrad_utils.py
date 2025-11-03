import numpy as np
import plotly.express as px


def add_smoothgrad_scatter(
    fig, shifted_filtered_raw_df, time_range, selected_channels, filter, all_channels
):
    """
    Adds a scatter plot with smooth gradient color to an existing figure based on the given selection.
    Also updates the layout and trace styles for smooth gradient visualizations.

    Parameters:
    - fig: The existing plotly figure to which the scatter plot will be added.
    - shifted_filtered_raw_df: The dataframe containing the time-series data for the plot.
    - time_range: The time range for the plot (start, end).
    - selected_channels: The list of selected channels to be plotted.
    - filter: The array or dataset containing the color gradient data.

    Returns:
    - fig: The updated plotly figure with the scatter plot and layout modifications.
    """

    # Calculate the time indices and channel indices for the given time range and selected channels
    time_range_indices = np.arange(
        round(time_range[0] * 150), round(time_range[1] * 150) + 1
    ).astype(int)
    channel_indices = np.where(np.isin(all_channels, selected_channels))[0]

    # Filter the data based on the calculated indices
    filtered_sa_array = filter[time_range_indices[:, None], channel_indices]

    # Melt the dataframe for plotting
    scatter_df = shifted_filtered_raw_df.melt(
        id_vars=["Time"], var_name="Channel", value_name="Value"
    )
    scatter_df["Color"] = filtered_sa_array.flatten("F")  # Flatten color array

    # Filter out values where the color value is less than or equal to 0
    scatter_df_filtered = scatter_df[scatter_df["Color"] > 0]

    # Create the scatter plot with the filtered data
    scatter_fig = px.scatter(
        scatter_df_filtered,
        x="Time",
        y="Value",
        color="Color",
        color_continuous_scale="Reds",
        labels={"value": "Value", "variable": "Channel", "Time": "Time (s)"},
        opacity=1,
    )

    # Add scatter traces to the main figure
    fig.add_traces(scatter_fig.data)

    # Update the layout for smoothGrad visualizations
    fig.update_layout(
        coloraxis_colorbar=dict(
            title=dict(text="SmoothGrad"),
            thicknessmode="pixels",
            thickness=10,
            lenmode="fraction",
            len=0.15,
            y=0,
            x=0.9,
            orientation="h",
            ticks="outside",
            dtick=1,
        ),
        coloraxis=dict(cmin=0, cmax=0.95),
    )

    fig.update_traces(line=dict(width=1), marker=dict(size=3))

    return fig
