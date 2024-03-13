import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import math
from plotly.subplots import make_subplots
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
import streamlit as st

markdown_text0 = '''
# Interactive Visualization of the Mental Health in Tech Dataset

Visualization presented as a supplement to the Data Report for the application to the Data Manager role (Shashank Shekhar).

## Notes on Interactivity


Hover over the graphs to get exact values for the metrics/data points. 

The donut charts have a dropdown menu that can be used to select different features and visualize them.

Please note that the rendering of the visualizations might result in slight offsets depending on the browser.


'''
st.markdown(markdown_text0)

markdown_text_i = '''## Understanding the Meanings of the Dataset Features'''
st.markdown(markdown_text_i)

data = {
    "Features": [
        "Timestamp",
        "Age",
        "Gender",
        "Country",
        "state",
        "self_employed",
        "family_history",
        "treatment",
        "work_interfere",
        "no_employees",
        "remote_work",
        "tech_company",
        "benefits",
        "care_options",
        "wellness_program",
        "seek_help",
        "anonymity",
        "leave",
        "mental_health_consequence",
        "phys_health_consequence",
        "coworkers",
        "supervisor",
        "mental_health_interview",
        "phys_health_interview",
        "mental_vs_physical",
        "obs_consequence",
        "comments",
    ],
    "Description": [
        "Time the survey was submitted",
        "Respondent age",
        "Respondent gender",
        "Respondent country",
        "If you live in the United States, which state or territory do you live in?",
        "Are you self-employed?",
        "Do you have a family history of mental illness?",
        "Have you sought treatment for a mental health condition?",
        "If you have a mental health condition, do you feel that it interferes with your work?",
        "How many employees does your company or organization have?",
        "Do you work remotely (outside of an office) at least 50% of the time?",
        "Is your employer primarily a tech company/organization?",
        "Does your employer provide mental health benefits?",
        "Do you know the options for mental health care your employer provides?",
        "Has your employer ever discussed mental health as part of an employee wellness program?",
        "Does your employer provide resources to learn more about mental health issues and how to seek help?",
        "Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment",
        "How easy is it for you to take medical leave for a mental health condition?",
        "Do you think that discussing a mental health issue with your employer would have negative consequences?",
        "Do you think that discussing a physical health issue with your employer would have negative consequences?",
        "Would you be willing to discuss a mental health issue with your coworkers?",
        "Would you be willing to discuss a mental health issue with your direct supervisor(s)?",
        "Would you bring up a mental health issue with a potential employer in an interview?",
        "Would you bring up a physical health issue with a potential employer in an interview?",
        "Do you feel that your employer takes mental health as seriously as physical health?",
        "Have you heard of or observed negative consequences for coworkers with mental health conditions in your organization",
        "Any additional notes or comments",
    ],
}

df = pd.DataFrame(data)
#styled_df = df.style.applymap(lambda _: "background-color: LightBlue;", subset=pd.IndexSlice[:, :])
# Displaying the styled DataFrame with horizontal scrolling
st.dataframe(df, hide_index = True, use_container_width = True)


df = pd.read_csv('survey.csv')

df = df.drop(columns=['state', 'comments', 'Timestamp'])

prob_yes = df['self_employed'].value_counts(normalize=True)['Yes']
prob_no = df['self_employed'].value_counts(normalize=True)['No']

df['self_employed'] = df['self_employed']\
                      .fillna(pd.Series(np.random.choice(['Yes', 'No'], p=[prob_yes, prob_no], size=len(df))))


# fig = px.scatter(df, x="gdp per capita", y="life expectancy",
#                  size="population", color="continent", hover_name="country",
#                  log_x=True, size_max=60)

prob_sometimes = df['work_interfere'].value_counts(normalize=True)['Sometimes']
prob_never = df['work_interfere'].value_counts(normalize=True)['Never']
prob_rarely = df['work_interfere'].value_counts(normalize=True)['Rarely']
prob_often = df['work_interfere'].value_counts(normalize=True)['Often']

df['work_interfere'] = df['work_interfere']\
                      .fillna(pd.Series(np.random.choice(['Sometimes', 'Never', 'Rarely', 'Often']
                                                         , p=[prob_sometimes, prob_never, prob_rarely, prob_often], size=len(df))))

age = []
for i in df.Age:
    if (i<18) or (i>99):
        age.append(31)   # Median
    else:
        age.append(i)
df['Age'] = age

selfdescribe  = ['A little about you', 'p', 'Nah', 'Enby', 'Trans-female','something kinda male?','queer/she/they','non-binary','All','fluid', 'Genderqueer','Androgyne', 'Agender','Guy (-ish) ^_^', 'male leaning androgynous','Trans woman','Neuter', 'Female (trans)','queer','ostensibly male, unsure what that really means','trans']
male   = ['male', 'Male','M', 'm', 'Male-ish', 'maile','Cis Male','Mal', 'Male (CIS)','Make','Male ', 'Man', 'msle','cis male', 'Cis Man','Malr','Mail']
female = ['Female', 'female','Cis Female', 'F','f','Femake', 'woman','Female ','cis-female/femme','Female (cis)','femail','Woman','female']

df['Gender'].replace(to_replace = selfdescribe, value = 'Self-Describe', inplace=True)
df['Gender'].replace(to_replace = male, value = 'M', inplace=True)
df['Gender'].replace(to_replace = female, value = 'F', inplace=True)


markdown_text = '''
## Overall Distribution of the Data

'''

st.markdown(markdown_text)

# Univariate visualization of categorical variables

df_ = df.drop(['Age', 'Country'], axis=1)

buttons = []
i = 0
vis = [False] * 24

for col in df_.columns:
    vis[i] = True
    buttons.append({'label' : col,
             'method' : 'update',
             'args'   : [{'visible' : vis},
             {'title'  : col}] })
    i+=1
    vis = [False] * 24

fig = go.Figure()

for col in df_.columns:
    fig.add_trace(go.Pie(
             values = df_[col].value_counts(),
             labels = df_[col].value_counts().index,
             title = dict(text = 'Distribution of {}'.format(col),
                          font = dict(size=18, family = 'monospace'),
                          ),
             hole = 0.5,
             hoverinfo='label+percent',))

fig.update_traces(hoverinfo='label+percent',
                  textinfo='label+percent',
                  textfont_size=12,
                  opacity = 1,
                  showlegend = False,
                  marker = dict(colors = sns.color_palette('YlOrRd').as_hex()[1::2],
                              line=dict(color='#000000', width=1)))
              

fig.update_layout(margin=dict(t=50, b=50, l=50, r=50),  # Adjust the margins as needed
                  height=600,  # Set a fixed height for the plot
                  width=800,
                  updatemenus = [dict(
                        type = 'dropdown',
                        x = 1.15,
                        y = 0.85,
                        showactive = True,
                        active = 0,
                        buttons = buttons)],
                 annotations=[
                             dict(text = "<b>Choose Column</b> : ",
                             showarrow=False,
                             x = 1.06, y = 0.92, yref = "paper", align = "left")])

for i in range(1,22):
    fig.data[i].visible = False
st.plotly_chart(fig)


markdown_text1 = '''
## Visualization of the Distribution Across Countries and Age Groups


'''

st.markdown(markdown_text1)
# Univariate visualization of non-categorical variables
fig1 = make_subplots(rows = 2, cols=1)

fig1.append_trace(go.Bar(
                        y = df['Country'].value_counts(),
                        x = df['Country'].value_counts().index,
                        name = 'Observations from Countries (Log)',
                        text = df['Country'].value_counts(),
                        textfont = dict(size = 10,family = 'monospace'),
                        textposition = 'outside',
                        marker=dict(color="#FFD580")
                        ), row=1, col=1)

fig1.append_trace(go.Histogram(
                        x = df['Age'],
                        nbinsx = 8,
                        text = ['16', '500', '562', '149', '26', '5', '1'],
                        marker =  dict(color="#FFD580")),
                        row=2, col=1)


# For Subplot : 1

fig1.update_xaxes(
        row=1, col=1,
        tickfont = dict(size=10, family = 'monospace'),
        tickmode = 'array',
        ticktext = df['Country'].value_counts().index,
        tickangle = 60,
        ticklen = 6,
        showline = True,
        showgrid = True,
        ticks = 'outside')

fig1.update_yaxes(type = 'log',
        row=1, col=1,
        tickfont = dict(size=15, family = 'monospace'),
        tickmode = 'array',
        showline = True,
        showgrid = True,
        ticks = 'outside')

fig1.update_traces(
                  marker_line_color='black',
                  marker_line_width= 1.2,
                  opacity=1,
                  row = 1, col = 1)

fig1.update_xaxes(range=[-1,48], row = 1, col = 1)

# For Subplot : 2

fig1.update_xaxes(        
        title = dict(text = 'Age',
                     font = dict(size = 15,
                                 family = 'monospace')),
        row=2, col=1,
        tickfont = dict(size=15, family = 'monospace', color = 'black'),
        tickmode = 'array',
        ticktext = ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79'],
        ticklen = 6,
        showline = True,
        showgrid = True,
        ticks = 'outside')

fig1.update_yaxes(
        row=2, col=1,
        tickfont = dict(size=15, family = 'monospace'),
        tickmode = 'array',
        showline = True,
        showgrid = True,
        ticks = 'outside')

fig1.update_traces(
                  marker_line_color='black',
                  marker_line_width = 2,
                  opacity = 1,
                  row = 2, col = 1)


fig1.update_layout(height=1200, width=900,
                  title = dict(text = 'Visualization of Non-Categorical Variables<br>1. Observation from Countries(Log-Scaled)<br>2. Age Distribution',
                               x = 0.5,
                               font = dict(size = 16, color ='#27302a',
                               family = 'monospace')),
                #   plot_bgcolor='#edf2c7',
                #   paper_bgcolor = '#edf2c7',
                  showlegend = False)

st.plotly_chart(fig1)

country_counts = df['Country'].value_counts().reset_index()
country_counts.columns = ['Country', 'Number of Survey Respondents']

country_counts = pd.DataFrame(country_counts)
country_counts['Number of Survey Respondents'] = country_counts['Number of Survey Respondents']
fig5 = px.choropleth(country_counts, locations='Country', locationmode='country names', color='Number of Survey Respondents',
                    hover_name='Country', color_continuous_scale= 'YlOrRd',) 
fig5.update_layout(height=600, width=800,)

st.plotly_chart(fig5)

markdown_text2 = '''
## Country-Gender Distribution of the Data Points

'''
st.markdown(markdown_text2)
male_country   = df[df['Gender'] == 'M'][['Country', 'Gender']]
female_country = df[df['Gender'] == 'F'][['Country', 'Gender']]
male_country   = male_country.value_counts()
female_country = female_country.value_counts()

male_country   = pd.DataFrame(male_country).reset_index().rename(columns={0:'count'}).head(15)
female_country = pd.DataFrame(female_country).reset_index().rename(columns={0:'count'}).head(15)
male_country['count'] = male_country['count'] * -1

fig2 = make_subplots(rows=1, cols=2,
                    specs=[[{}, {}]],
                    shared_yaxes=True,
                    horizontal_spacing=0)

fig2.append_trace(go.Bar(
                 y = male_country.Country,
                 x = male_country['count'],
                 text = male_country['count'],
                 textfont = dict(size = 10, color = '#6aa87b'),
                 textposition = 'outside',
                 name = 'Male responses',
                 marker_color='#6aa87b',
                 orientation = 'h'),
                 row=1, col=1)

fig2.append_trace(go.Bar(
                 y = male_country.Country,
                 x = female_country['count'],
                 text = female_country['count'],
                 textfont = dict(size = 10, color = '#913f3f'),
                 textposition = 'outside',
                 name = 'Female responses',
                 marker_color='#913f3f',
                 orientation = 'h'),
                 row=1, col=2)


fig2.update_xaxes(
        tickfont = dict(size=15),
        tickmode = 'array',
        ticklen = 6,
        showline = False,
        showgrid = False,
        ticks = 'outside')

fig2.update_yaxes(showgrid=True,
                 categoryorder='total ascending',
                 ticksuffix=' ',
                 showline=False)

fig2.update_layout(height=600,  # Set a fixed height for the plot
                  width=900,
                  font_family   = 'monospace',
                  title         = dict(text = 'Gender of the survey respondents across Countries', x = 0.525),
                  margin        = dict(t=80, b=0, l=70, r=40),
                  hovermode     = "y unified",
                #   plot_bgcolor  = '#edf2c7',
                #   paper_bgcolor = '#edf2c7',
                  font          = dict(color='black'),
                  legend        = dict(orientation="h",
                                       yanchor="bottom", y=1,
                                       xanchor="center", x=0.5),
                  hoverlabel    = dict(bgcolor="#edf2c7", font_size=13, 
                                      font_family="Monospace"))
st.plotly_chart(fig2)

markdown_text3 = '''
## Past History of Treatment Across Gender and Age Groups
'''
st.markdown(markdown_text3)

fig3 = px.violin(df, y="Age", x="treatment", color="Gender", box=True, points="all")
st.plotly_chart(fig3)

markdown_text4 = '''
## Genderwise Distribution of Features
Please select an option from the dropdown menu
'''

st.markdown(markdown_text4)
male   = df[df.Gender == 'M'].drop(['Gender', 'Age', 'Country'], axis=1)
female = df[df.Gender == 'F'].drop(['Gender', 'Age', 'Country'], axis=1)
selfdescribe  = df[df.Gender == 'Self-Describe'].drop(['Gender', 'Age', 'Country'], axis=1)

buttons = []
i = 0
vis = [False] * 21

for col in male.columns:
    vis[i] = True
    buttons.append({'label' : col,
             'method' : 'update',
             'args'   : [{'visible' : vis},
             {'title'  : col}] })
    i+=1
    vis = [False] * 21

fig4 = make_subplots(rows=1, cols=3,  # Adjusted for three columns
                     specs=[[{'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}]])

# Add traces for Male, Female, and Self-Describe
for col in male.columns:
    fig4.add_trace(go.Pie(
        values=male[col].value_counts(),
        labels=male[col].value_counts().index,
        title=dict(text='Male distribution<br>of {}'.format(col),
                   font=dict(size=18, family='monospace'),
                   ),
        hole=0.5,
        hoverinfo='label+percent', ), 1, 1)

for col in female.columns:
    fig4.add_trace(go.Pie(
        values=female[col].value_counts(),
        labels=female[col].value_counts().index,
        title=dict(text='Female distribution<br>of {}'.format(col),
                   font=dict(size=18, family='monospace'),
                   ),
        hole=0.5,
        hoverinfo='label+percent', ), 1, 2)

for col in selfdescribe.columns:
    fig4.add_trace(go.Pie(
        values=selfdescribe[col].value_counts(),
        labels=selfdescribe[col].value_counts().index,
        title=dict(text='Self-Describe distribution<br>of {}'.format(col),
                   font=dict(size=18, family='monospace'),
                   ),
        hole=0.5,
        hoverinfo='label+percent', ), 1, 3)

# Update traces for styling
fig4.update_traces(hoverinfo='label+percent',
                   textinfo='label+percent',
                   textfont_size=12,
                   opacity=1,
                   showlegend=False,
                   marker=dict(colors=sns.color_palette('YlOrRd').as_hex()[1::2],
                               line=dict(color='#000000', width=1)))

fig4.update_layout(margin=dict(t=50, b=50, l=0, r=120),
                   height = 700, width = 1000,
                   font_family='monospace',
                   updatemenus=[dict(
                       type='dropdown',
                       x=0.62,
                       y=0.91,
                       showactive=True,
                       active=0,
                       buttons=buttons)],
                   annotations=[
                       dict(text="<b>Choose<br>Column<b> : ",
                            font=dict(size=14),
                            showarrow=False,
                            x=0.5, y=1, yref="paper", align="left")])

# Set visibility for the initial view
for trace in fig4.data:
    trace.visible = False
fig4.data[22].visible = True  # Adjusted for the new trace (22 instead of 21)
st.plotly_chart(fig4)
# Show the figure
# fig4.show()

# Choosing countries with more than 20 observations
markdown_text6 = '''
## Feature Distribution in Countries With More Than 30 Data Points
'''
st.markdown(markdown_text6)

us = df[df.Country == 'United States'].drop(['Age', 'Country'], axis=1)
uk = df[df.Country == 'United Kingdom'].drop(['Age', 'Country'], axis=1)
cd = df[df.Country == 'Canada'].drop(['Age', 'Country'], axis=1)
gr = df[df.Country == 'Germany'].drop(['Age', 'Country'], axis=1)
ir = df[df.Country == 'Ireland'].drop(['Age', 'Country'], axis=1)
nl = df[df.Country == 'Netherlands'].drop(['Age', 'Country'], axis=1)

buttons = []
i = 0
vis = [False] * 22

for col in us.columns:
    vis[i] = True
    buttons.append({'label' : col,
             'method' : 'update',
             'args'   : [{'visible' : vis},
             {'title'  : col}] })
    i+=1
    vis = [False] * 22

fig6 = make_subplots(rows=2, cols=2,
                    specs=[[{'type':'domain'}, {'type':'domain'}], [{'type':'domain'}, {'type':'domain'}]],
                    vertical_spacing = 0.2)

for col in us.columns:
    fig6.add_trace(go.Pie(
             values = us[col].value_counts(),
             labels = us[col].value_counts().index,
             title = dict(text = 'U.S. distribution<br>of {}'.format(col),
                          font = dict(size=18, family = 'monospace'),
                          ),
             hole = 0.5,
             hoverinfo='label+percent',),1,1)
    
for col in uk.columns:
    fig6.add_trace(go.Pie(
             values = uk[col].value_counts(),
             labels = uk[col].value_counts().index,
             title = dict(text = 'U.K. distribution<br>of {}'.format(col),
                          font = dict(size=18, family = 'monospace'),
                          ),
             hole = 0.5,
             hoverinfo='label+percent',),1,2)


for col in cd.columns:
    fig6.add_trace(go.Pie(
             values = cd[col].value_counts(),
             labels = cd[col].value_counts().index,
             title = dict(text = 'Canada distribution<br>of {}'.format(col),
                          font = dict(size=18, family = 'monospace'),
                          ),
             hole = 0.5,
             hoverinfo='label+percent',),2,1)
    
for col in gr.columns:
    fig6.add_trace(go.Pie(
             values = gr[col].value_counts(),
             labels = gr[col].value_counts().index,
             title = dict(text = 'Germany distribution<br>of {}'.format(col),
                          font = dict(size=18, family = 'monospace'),
                          ),
             hole = 0.5,
             hoverinfo='label+percent',),2,2)

fig6.update_traces(hoverinfo='label+percent',
                  textinfo='label+percent',
                  textfont_size=12,
                  opacity = 1,
                  showlegend = False,
                  marker = dict(colors = sns.color_palette('RdBu_r').as_hex(),
                              line=dict(color='#000000', width=1)))

fig6.update_traces(row=2, col=1, hoverinfo='label+percent',
                  textinfo='label+percent',
                  textfont_size=12,
                  opacity = 1,
                  showlegend = False,
                  marker = dict(colors = sns.color_palette('Reds').as_hex(),
                              line=dict(color='#000000', width=1)))

fig6.update_traces(row=2, col=2, hoverinfo='label+percent',
                  textinfo='label+percent',
                  textfont_size=12,
                  opacity = 1,
                  showlegend = False,
                  marker = dict(colors = sns.color_palette('Reds').as_hex(),
                              line=dict(color='#000000', width=1)))
              

fig6.update_layout(margin=dict(t=50, b=50, l=100, r=0),
                  height = 800,
                  font_family   = 'monospace',
                  updatemenus = [dict(
                        type = 'dropdown',
                        x = 0.60,
                        y = 0.55,
                        showactive = True,
                        active = 0,
                        buttons = buttons)],
                 annotations=[
                             dict(text = "<b>Choose Column<b> : ",
                                  font = dict(size = 14),
                             showarrow=False,
                             x = 0.50, y = 0.60, yref = "paper", align = "left")])

for i in range(0,88):
    fig6.data[i].visible = False

fig6.data[0].visible = True
fig6.data[22].visible = True
fig6.data[44].visible = True
fig6.data[66].visible = True

st.plotly_chart(fig6)

markdown_text7 = '''## Comparison of the Demographic Seeking Treatment vs Not Seeking Treatment'''
st.markdown(markdown_text7)

seek = df[df.treatment == 'Yes'].drop(['treatment', 'Country', 'Age'], axis=1)
dont = df[df.treatment == 'No'].drop(['treatment', 'Country', 'Age'], axis=1)

buttons = []
i = 0
vis = [False] * 21

for col in seek.columns:
    vis[i] = True
    buttons.append({'label' : col,
             'method' : 'update',
             'args'   : [{'visible' : vis},
             {'title'  : col}] })
    i+=1
    vis = [False] * 21

fig7 = make_subplots(rows=1, cols=2,
                    specs=[[{'type':'domain'}, {'type':'domain'}]])

for col in dont.columns:
    fig7.add_trace(go.Pie(
             values = dont[col].value_counts(),
             labels = dont[col].value_counts().index,
             title = dict(text = 'No Treatment: <br>Distribution<br>of {}'.format(col),
                          font = dict(size=18, family = 'monospace'),
                          ),
             hole = 0.5,
             hoverinfo='label+percent',),1,1)


for col in seek.columns:
    fig7.add_trace(go.Pie(
             values = seek[col].value_counts(),
             labels = seek[col].value_counts().index,
             title = dict(text = 'Seek Treatment: <br>Distribution<br>of {}'.format(col),
                          font = dict(size=18, family = 'monospace'),
                          ),
             hole = 0.5,
             hoverinfo='label+percent',),1,2)

fig7.update_traces(hoverinfo='label+percent',
                  textinfo='label+percent',
                  textfont_size=12,
                  opacity = 0.8,
                  showlegend = False,
                  marker = dict(colors = sns.color_palette('YlOrRd').as_hex()[1::2],
                              line=dict(color='#000000', width=1)))

fig7.update_traces(row=1, col=2, hoverinfo='label+percent',
                  textinfo='label+percent',
                  textfont_size=12,
                  opacity = 0.8,
                  showlegend = False,
                  marker = dict(colors = sns.color_palette('Reds').as_hex(),
                              line=dict(color='#000000', width=1)))
              

fig7.update_layout(margin=dict(t=50, b=50, l=60, r=50),
                  font_family   = 'monospace',
                  updatemenus = [dict(
                        type = 'dropdown',
                        x = 0.65,
                        y = 1.00,
                        showactive = True,
                        active = 0,
                        buttons = buttons)],
                 annotations=[
                             dict(text = "<b>Choose Column<b> : ",
                                  font = dict(size = 10),
                             showarrow=False,
                             x = 0.5, y = 1.05, yref = "paper", align = "left")])

for i in range(1,42):
    fig7.data[i].visible = False
fig7.data[21].visible = True
st.plotly_chart(fig7)

markdown_text8 = '''## Correlation Matrix to Identify Inter-Feature Relationships'''
st.markdown(markdown_text8)

st.image("./assets/correlation_matrix.png")

markdown_text9 = '''## Wordcloud to Visualize the Most Frequently Occuring Words in Comments'''
st.markdown(markdown_text9)
df1 = pd.read_csv("survey.csv")
df1 = df1.dropna(subset=['comments'])
comments_combined = " ".join(comment for comment in df1.comments)
word_cloud1 = WordCloud(collocations = False, background_color = 'white',
                        width = 2048, height = 1080).generate(comments_combined)
plt.imshow(word_cloud1, interpolation='bilinear')
plt.axis("off")
st.pyplot(plt)

markdown_text10 = '''## Visualizing Feature Importance Scores'''
st.markdown(markdown_text10)
feature_importance_dict = {'Age': 0.12649202262360504,
 'Gender': 0.024822932150002603,
 'Country': 0.06896437649260984,
 'self_employed': 0.014296077058401774,
 'family_history': 0.10822462938934918,
 'work_interfere': 0.17424085879142334,
 'no_employees': 0.0477775402987294,
 'remote_work': 0.02125296823359366,
 'tech_company': 0.0156113875078448,
 'benefits': 0.030874182891229493,
 'care_options': 0.061014546121503574,
 'wellness_program': 0.023771698822096083,
 'seek_help': 0.023297755618306507,
 'anonymity': 0.020609599295543218,
 'leave': 0.04207672948014938,
 'mental_health_consequence': 0.03055027070608146,
 'phys_health_consequence': 0.019544027750489704,
 'coworkers': 0.031062569469653176,
 'supervisor': 0.029017481924240817,
 'mental_health_interview': 0.016867936651136475,
 'phys_health_interview': 0.03087256832841105,
 'mental_vs_physical': 0.024174812715665638,
 'obs_consequence': 0.014583027679933878}

log_dict = {key: math.log(value)*-1 for key, value in feature_importance_dict.items()}

# Convert the dictionary to a DataFrame
df2 = pd.DataFrame(list(log_dict.items()), columns=['Feature', 'Score'])

# Create a radar chart
fig10 = px.line_polar(df2, r='Score', theta='Feature', line_close=True,
                     width=800, height=600,)

st.plotly_chart(fig10)