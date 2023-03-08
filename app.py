import base64
import datetime
import io
import plotly.graph_objs as go
import cufflinks as cf
import re
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import dash_table


import pandas as pd

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

dash_app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app = dash_app.server

colors = {"graphBackground": "#F5F5F5", "background": "#ffffff", "text": "#000000"}

dash_app.layout = html.Div(
    [

        html.H1(children='FRC Camel Heart Time Series'),

        html.Div(children='''
            Data Visualizatin for FRC camels.
        '''),
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            # Allow multiple files to be uploaded
            multiple=True,
        ),
        dcc.Dropdown(
        id='column-dropdown'
        ),
        dcc.Graph(id="Mygraph"),
        html.Div(id="stat-data-upload"),
        html.Div(id="output-data-upload"),
    ]
)


'''# Define callback
@dash_app.callback(
    dash.dependencies.Output('graph', 'figure'),
    [dash.dependencies.Input('column-dropdown', 'value')]
)
def update_graph(selected_column):
    fig = px.scatter(df, x='gdpPercap', y=selected_column, color='continent')
    fig.update_layout(title=f'{selected_column} vs GDP per capita')
    return fig'''


@dash_app.callback(Output('Mygraph', 'figure'), [
Input('upload-data', 'contents'),
Input('upload-data', 'filename'),
Input('column-dropdown', 'value'),
])



def update_graph(contents, filename,selected_column):
    
    x = []
    y = []
    
    df = pd.DataFrame(columns =['Time',"Parameter"])
    parameter = "Parameter"
    if contents and selected_column:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)
        #df = df.set_index(df.columns[0])


        for i in range(len(df.MS)):
            df['MS'].iloc[i] = int(df['MS'].iloc[i])/1000


        df['Time']= pd.to_datetime(df.MS, unit='s', errors='coerce')
        tf = df[['Time',selected_column]]
        df.Time = df.Time.dt.strftime('%H:%M:%S.%f')
        parameter = selected_column


    
    
        fig = px.line(df, x='Time', y=parameter)
        # plot only Google data for year 2018 with range slider
        fig = px.line(df, x='Time', y=parameter)
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        fig.update_layout(
            autosize=False,
            width=1700,
            height=800,)
        return fig
    fig = px.line()
    return fig




def parse_data(contents, filename):
    
    content_type, content_string = contents.split(",")

    decoded = base64.b64decode(content_string)
    try:
        if "csv" in filename:
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif "xls" in filename:
            df = pd.read_excel(io.BytesIO(decoded))
        elif "txt" or "tsv" in filename:

            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        
            #df.to_excel("ssiraj.xlsx")
            #print(df)
            lines = df[df.columns[-1]].tolist()
            lines.insert(0, df.columns[-1])
            
            #df.close()
            count = 0
            # remove /n at the end of each line
            for index, line in enumerate(lines):
                  lines[index] = line.strip()
                  if 'CHANNELS' in lines[index]:
                        count = index
                        name = lines[index]

            lines = lines[count+2:]


            i = 0  
            first_col = "" 
            second_col = ""
            third_col = ""
            four_col = ""

            if len(name.split('CHANNELS:')[-1].split(" ")) == 4:
                  a,b,c,d = name.split('CHANNELS:')[-1].split(" ")
                  col = [a,b,c,d]
                  df_result = pd.DataFrame(columns=col)

                  for line in lines:
                      #you can use "if" and "replace" in case you had some conditions to manipulate the txt data
                      if len(line.split(" "))==4:
                          first_col,second_col,third_col,four_col= line.split(" ")
                          #you have to kind of define what are the values in columns,for example second column includes:
                          #this is how you create next line data
                          df_result.loc[i] = [first_col, second_col,third_col,four_col]
                          i = i + 1
            elif len(name.split('CHANNELS:')[-1].split(" ")) == 2:
                  a,b = name.split('CHANNELS:')[-1].split(" ")
                  col = [a,b]
                  df_result = pd.DataFrame(columns=col)
                  for line in lines:
                      #you can use "if" and "replace" in case you had some conditions to manipulate the txt data
                      if len(line.split(" "))==2:
                          first_col,second_col= line.split(" ")
                          #you have to kind of define what are the values in columns,for example second column includes:
                          #this is how you create next line data
                          df_result.loc[i] = [first_col, second_col]
                          i = i + 1
            else:
                  a,b,c = name.split('CHANNELS:')[-1].split(" ")
                  col = [a,b,c]
                  df_result = pd.DataFrame(columns=col)
                  for line in lines:
                      #you can use "if" and "replace" in case you had some conditions to manipulate the txt data
                      if len(line.split(" "))==3:
                          first_col,second_col,third_col= line.split(" ")
                          #you have to kind of define what are the values in columns,for example second column includes:
                          #this is how you create next line data
                          df_result.loc[i] = [first_col, second_col,third_col]
                          i = i + 1

            df = df_result
            cols = df.columns
            df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
            #df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), delimiter=r"\s+")


    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."])

    return df

@dash_app.callback(
    Output("stat-data-upload", "children"),
    [Input("upload-data", "contents"), Input("upload-data", "filename")],
)


def update_stat(contents, filename):
    table = html.Div()

    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)
        df = df.reset_index(drop=True)
        
        af = df.describe()
        af['Parameters'] = af.index

        # shift column 'Name' to first position
        first_column = af.pop('Parameters')
          
        # insert column using insert(position,column_name,
        # first_column) function
        af.insert(0, 'Parameters', first_column)

        table = html.Div(
            [
                html.H5("Statistics"),
                dash_table.DataTable(
                    data=af.to_dict("rows"),
                    columns=[{"name": i, "id": i} for i in af.columns],
                ),
                html.Hr(),
            ]
        )

    return table


@dash_app.callback(
    Output("output-data-upload", "children"),
    [Input("upload-data", "contents"), Input("upload-data", "filename")],
)


def update_table(contents, filename):
    table = html.Div()

    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename)

        table = html.Div(
            [
                html.H5(filename),
                dash_table.DataTable(
                    data=df.to_dict("rows"),
                    columns=[{"name": i, "id": i} for i in df.columns],
                ),
                html.Hr(),
            ]
        )

    return table



@dash_app.callback(
    Output("column-dropdown", "options"),
    [Input("upload-data", "contents"), Input("upload-data", "filename")],
)

def parse_uploads(contents, filename):
    if contents:
        df = parse_data(contents[0], filename[0])
        return [{'label': i, 'value': i } for i in df.columns]
    return [{'label': i, 'value': i } for i in []]
if __name__ == "__main__":
    dash_app.run_server(debug=True)
