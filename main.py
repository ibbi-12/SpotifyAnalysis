import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from dash import Dash, html, dcc, Input, Output
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import dash_bootstrap_components as dbc

# Load the dataset
data = pd.read_csv('spotify-2023.csv', encoding='latin1')

# Data Cleaning
data['streams'] = pd.to_numeric(data['streams'].astype(str).str.replace(',', ''), errors='coerce')
data['in_deezer_playlists'] = pd.to_numeric(data['in_deezer_playlists'].astype(str).str.replace(',', ''), errors='coerce')
data['in_shazam_charts'] = pd.to_numeric(data['in_shazam_charts'].astype(str).str.replace(',', ''), errors='coerce')

# Fill missing values with 0
data.fillna(0, inplace=True)

# Calculate statistics
total_artists = len(data['artist(s)_name'].unique())
total_songs = len(data)
total_spotify_playlists = data['in_spotify_playlists'].sum()
total_apple_playlists = data['in_apple_playlists'].sum()
total_deezer_playlists = data['in_deezer_playlists'].sum()

# Remove non-numeric columns
numeric_df = data.select_dtypes(include=[float, int])

# Create a correlation matrix
correlation_matrix = numeric_df.corr()

# Convert the correlation matrix to a plotly figure
heatmap = ff.create_annotated_heatmap(
    z=correlation_matrix.values,
    x=list(correlation_matrix.columns),
    y=list(correlation_matrix.index),
    annotation_text=correlation_matrix.round(2).values,
    colorscale='Viridis'
)
heatmap.update_layout(
    title='Correlation Matrix Heatmap',
    template='plotly_dark',
    paper_bgcolor='#121212',
    plot_bgcolor='#121212',
    font=dict(color='white')
)

# Create visualizations
fig_years = px.histogram(data, x='released_year', nbins=20, title='Distribution of Track Releases Over Years', template='plotly_dark')

# Create the top artists figure
top_artists_df = data['artist(s)_name'].value_counts().nlargest(10).reset_index()
top_artists_df.columns = ['artist(s)_name', 'count']
fig_top_artists = px.bar(top_artists_df, x='artist(s)_name', y='count', title='Top 10 Artists by Number of Tracks', template='plotly_dark', labels={'artist(s)_name': 'Artist', 'count': 'Number of Tracks'})

fig_bpm = px.histogram(data, x='bpm', nbins=30, title='Distribution of BPM', template='plotly_dark')
fig_danceability = px.histogram(data, x='danceability_%', nbins=30, title='Distribution of Danceability', template='plotly_dark')
fig_valence = px.histogram(data, x='valence_%', nbins=30, title='Distribution of Valence', template='plotly_dark')
fig_energy = px.histogram(data, x='energy_%', nbins=30, title='Distribution of Energy', template='plotly_dark')
fig_acousticness = px.histogram(data, x='acousticness_%', nbins=30, title='Distribution of Acousticness', template='plotly_dark')
fig_instrumentalness = px.histogram(data, x='instrumentalness_%', nbins=30, title='Distribution of Instrumentalness', template='plotly_dark')
fig_liveness = px.histogram(data, x='liveness_%', nbins=30, title='Distribution of Liveness', template='plotly_dark')
fig_speechiness = px.histogram(data, x='speechiness_%', nbins=30, title='Distribution of Speechiness', template='plotly_dark')
fig_spotify_playlists = px.histogram(data, x='in_spotify_playlists', nbins=30, title='Popularity in Spotify Playlists', template='plotly_dark')
fig_apple_playlists = px.histogram(data, x='in_apple_playlists', nbins=30, title='Popularity in Apple Playlists', template='plotly_dark')
fig_deezer_playlists = px.histogram(data, x='in_deezer_playlists', nbins=30, title='Popularity in Deezer Playlists', template='plotly_dark')

# Prediction Model
# Select features and target
features = ['bpm', 'danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']
X = data[features]
y = data['streams']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Create a DataFrame for actual vs predicted values
pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
pred_df.reset_index(drop=True, inplace=True)
pred_df = pred_df.melt(var_name='Type', value_name='Streams')

fig_pred = px.scatter(pred_df, x=pred_df.index, y='Streams', color='Type', title='Actual vs Predicted Streams', template='plotly_dark')
fig_pred.update_layout(xaxis_title='Index', yaxis_title='Streams')

# Creating the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define search functionality
def search_data(query):
    return data[data['artist(s)_name'].str.contains(query, case=False)]

# Define the layout
app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#121212', 'color': 'white'}, children=[
    html.Div(style={'textAlign': 'center'}, children=[
        html.Img(src='https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg', style={'width': '100px'}),
        html.H1(children='Spotify', style={'textAlign': 'center', 'padding': '20px'})
    ]),

    # Search bar below Spotify logo
    html.Div(style={'textAlign': 'center', 'padding': '20px'}, children=[
        dcc.Input(id='search-input', type='text', placeholder='Search artists or tracks...', debounce=True, style={'width': '50%', 'margin': 'auto', 'padding': '10px'})
    ]),

    # Search results
    html.Div(id='search-results', style={'padding': '20px'}),

    # Statistics widgets
    dbc.Row([
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H5(f'Total Artists: {total_artists}', className='card-title'),
                    html.P('Number of unique artists in the dataset'),
                ]),
                className='mb-3',
                color='dark',
                inverse=True
            ),
            width=4
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H5(f'Total Songs: {total_songs}', className='card-title'),
                    html.P('Total number of songs in the dataset'),
                ]),
                className='mb-3',
                color='dark',
                inverse=True
            ),
            width=4
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H5(f'Total Spotify Playlists: {total_spotify_playlists}', className='card-title'),
                    html.P('Total number of playlists on Spotify'),
                ]),
                className='mb-3',
                color='dark',
                inverse=True
            ),
            width=4
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H5(f'Total Apple Playlists: {total_apple_playlists}', className='card-title'),
                    html.P('Total number of playlists on Apple Music'),
                ]),
                className='mb-3',
                color='dark',
                inverse=True
            ),
            width=4
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H5(f'Total Deezer Playlists: {total_deezer_playlists}', className='card-title'),
                    html.P('Total number of playlists on Deezer'),
                ]),
                className='mb-3',
                color='dark',
                inverse=True
            ),
            width=4
        ),
    ], className='mb-4'),

    # Tabs for different sections
    dcc.Tabs([
        dcc.Tab(label='Overview', children=[
            html.Div(id='overview-content', style={'padding': '20px'}, children=[
                html.P('This dashboard provides a comprehensive overview of the Spotify dataset, which includes information about various tracks, artists, and their popularity across different platforms.'),
                html.P('The dataset contains the following columns:'),
                html.Ul([
                    html.Li('track_name: The name of the track'),
                    html.Li('artist(s)_name: The name of the artist(s)'),
                    html.Li('artist_count: Number of artists for the track'),
                    html.Li('released_year: Year when the track was released'),
                    html.Li('released_month: Month when the track was released'),
                    html.Li('released_day: Day when the track was released'),
                    html.Li('in_spotify_playlists: The number of Spotify playlists the track is in'),
                    html.Li('in_spotify_charts: The position of the track in Spotify charts'),
                    html.Li('streams: The number of streams on Spotify'),
                    html.Li('in_apple_playlists: The number of Apple Music playlists the track is in'),
                    html.Li('in_apple_charts: The position of the track in Apple Music charts'),
                    html.Li('in_deezer_playlists: The number of Deezer playlists the track is in'),
                    html.Li('in_deezer_charts: The position of the track in Deezer charts'),
                    html.Li('in_shazam_charts: The position of the track in Shazam charts'),
                    html.Li('bpm: Beats per minute of the track'),
                    html.Li('key: The key the track is in'),
                    html.Li('mode: The mode (major or minor) of the track'),
                    html.Li('danceability_%: Danceability score of the track'),
                    html.Li('valence_%: Valence score of the track'),
                    html.Li('energy_%: Energy score of the track'),
                    html.Li('acousticness_%: Acousticness score of the track'),
                    html.Li('instrumentalness_%: Instrumentalness score of the track'),
                    html.Li('liveness_%: Liveness score of the track'),
                    html.Li('speechiness_%: Speechiness score of the track'),
                ]),
            ])
        ], style={'backgroundColor': '#1DB954', 'color': 'white', 'fontWeight': 'bold'}, selected_style={'backgroundColor': '#1DB954', 'color': 'black', 'fontWeight': 'bold'}),

        dcc.Tab(label='Visualizations', children=[
            html.Div(style={'padding': '20px'}, children=[
                dcc.Graph(id='heatmap', figure=heatmap),
                dcc.Graph(id='fig_years', figure=fig_years),
                dcc.Graph(id='fig_top_artists', figure=fig_top_artists),
                dcc.Graph(id='fig_bpm', figure=fig_bpm),
                dcc.Graph(id='fig_danceability', figure=fig_danceability),
                dcc.Graph(id='fig_valence', figure=fig_valence),
                dcc.Graph(id='fig_energy', figure=fig_energy),
                dcc.Graph(id='fig_acousticness', figure=fig_acousticness),
                dcc.Graph(id='fig_instrumentalness', figure=fig_instrumentalness),
                dcc.Graph(id='fig_liveness', figure=fig_liveness),
                dcc.Graph(id='fig_speechiness', figure=fig_speechiness),
                dcc.Graph(id='fig_spotify_playlists', figure=fig_spotify_playlists),
                dcc.Graph(id='fig_apple_playlists', figure=fig_apple_playlists),
                dcc.Graph(id='fig_deezer_playlists', figure=fig_deezer_playlists),
            ])
        ], style={'backgroundColor': '#1DB954', 'color': 'white', 'fontWeight': 'bold'}, selected_style={'backgroundColor': '#1DB954', 'color': 'black', 'fontWeight': 'bold'}),

        dcc.Tab(label='Prediction Model', children=[
            html.Div(style={'padding': '20px'}, children=[
                html.H2('Predicting Spotify Streams'),
                html.P('This section demonstrates a simple linear regression model used to predict the number of streams based on various features such as bpm, danceability, valence, energy, etc.'),
                dcc.Graph(id='fig_pred', figure=fig_pred)
            ])
        ], style={'backgroundColor': '#1DB954', 'color': 'white', 'fontWeight': 'bold'}, selected_style={'backgroundColor': '#1DB954', 'color': 'black', 'fontWeight': 'bold'}),

        dcc.Tab(label='About Us', children=[
            html.Div(style={'padding': '20px'}, children=[
                html.H2('About Us'),
                html.P('This dashboard was created by Mian Ibrahim Naveed as a project to explore and visualize music data from Spotify.'),
                html.P('Technologies used include Python, Plotly, Dash, Pandas, and Scikit-learn.'),
                html.P('For any inquiries, please contact us at example@email.com.')
            ])
        ], style={'backgroundColor': '#1DB954', 'color': 'white', 'fontWeight': 'bold'}, selected_style={'backgroundColor': '#1DB954', 'color': 'black', 'fontWeight': 'bold'}),
    ])
])

# Callback for search functionality
@app.callback(
    Output('search-results', 'children'),
    [Input('search-input', 'value')]
)
def update_search_results(search_query):
    if search_query:
        filtered_data = search_data(search_query)
        return [
            html.P(f'Showing results for: "{search_query}"'),
            html.P(f'Total results found: {len(filtered_data)}'),
            html.Table([
                html.Thead(html.Tr([html.Th(col) for col in filtered_data.columns])),
                html.Tbody([
                    html.Tr([html.Td(filtered_data.iloc[i][col]) for col in filtered_data.columns]) for i in range(min(len(filtered_data), 20))
                ])
            ])
        ]
    else:
        return None

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
