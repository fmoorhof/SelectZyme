from base64 import b64encode
import io

from dash.dependencies import Input, Output, State

from src.visualizer import plot_2d


def register_callbacks(app, df, X_red, X_red_centroids):
    # Define callbacks
    @app.callback(
        [
            Output("data-table", "data"),  # output data to table displayed at page
            Output("shared-data", "data"),
        ],  # output data to update the dcc.Store in app.py
        [
            Input("plot", "clickData"),  # input users selection/click data
            Input("plot", "selectedData"),  # input from box/lasso selection
            Input("shared-data", "data"),
        ],  # input from dcc.Store in app.py (to load existing table at another page)
        State(
            "data-table", "data"
        ),  # !if user edits the table (delete rows, edit cells), changes are saved here
        # State('shared-data', 'data')
    )
    def update_table(
        clickData, boxSelect, shared_table, data_table
    ):  # (input 1, input 2, state)
        """Updates the existing table with new rows based on the clickData or boxSelect data.
        boxSelect (dict): Data from a box selection event, containing information about the selected points.
        shared_table (list): The current state of the table, represented as a list of dictionaries.
        data_table (list): The modified state of the table by the user, represented as a list of dictionaries.
        tuple: A tuple containing the updated shared_table twice.
        - If clickData is None, the function returns the shared_table without any changes.
        - If data_table is not None and differs from shared_table, shared_table is updated with data_table.
        - If shared_table is None, it initializes it as an empty list.
        - The function processes each selected point, extracts the accession, and looks up the corresponding row in the dataframe `df`.
        - The selected row is converted to a dictionary and appended to the shared_table.
        - If boxSelect contains points, each point is processed; otherwise, the first point from clickData is processed.
        """
        if clickData is None:
            return shared_table  # fix: not working, still table loads empty table

        # if user deletes entries or modifies cells
        if data_table is not None and data_table != shared_table:
            shared_table = (
                data_table  # set modified changes of user to shared_table (dcc.Store)
            )

        if shared_table is None or isinstance(shared_table, dict):
            shared_table = []

        if boxSelect and boxSelect["points"] != []:
            for point in boxSelect["points"]:
                _process_selection(df, shared_table, point)
        else:  # avoid adding previous clickData after box select
            _process_selection(df, shared_table, clickData["points"][0])

        return shared_table, shared_table

    @app.callback(
        [Output("plot", "figure"), Output("download-button", "href")],
        Input("legend-attribute", "value"),
    )
    def update_plot_and_download(legend_attribute):
        """
        Updates the plot and generates a download link for the updated figure.
        Args:
            legend_attribute (str): The attribute to be used for the legend in the plot.
        Returns:
            tuple: A tuple containing the updated figure and the updated download link.
                - updated_fig: The updated 2D plot figure.
                - updated_href: The HTML href link for downloading the updated figure.
        """
        updated_fig = plot_2d(
            df, X_red, X_red_centroids, legend_attribute
        )  # Update the figure
        updated_href = html_export_figure(
            updated_fig
        )  # Generate the updated download link
        return updated_fig, updated_href


def html_export_figure(fig):
    """
    Converts a Plotly figure to an HTML string and encodes it in base64 format.
    Args:
        fig (plotly.graph_objs._figure.Figure): The Plotly figure to be converted.
    Returns:
        str: A base64 encoded HTML string representing the figure.
    """
    buffer = io.StringIO()
    fig.write_html(buffer)
    html_bytes = buffer.getvalue().encode()
    encoded = b64encode(html_bytes).decode()
    return f"data:text/html;base64,{encoded}"


def _process_selection(df, shared_table, point):
    """
    Processes the selection of a data point and updates the shared table.
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        shared_table (list): A list to which the selected row's dictionary will be appended.
        point (dict): A dictionary containing information about the selected point,
                      including 'customdata' which holds the accession value.
    Returns:
        None
    """
    accession = point["customdata"]
    selected_row = df[df["accession"] == accession].iloc[0]
    selected_row[df.columns.get_loc("selected")] = True
    if (
        selected_row["xref_brenda"] != "unknown"
    ):  # todo: anyways not displayed!! add as row in template DataTable
        selected_row["BRENDA URL"] = (
            f"https://www.brenda-enzymes.org/enzyme.php?ecno={selected_row['xref_brenda'].split(';')[0]}&UniProtAcc={selected_row['accession']}&OrganismID={selected_row['organism_id']}"
        )
    shared_table.append(selected_row.to_dict())
