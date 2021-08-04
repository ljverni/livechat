import pandas as pd
from datetime import datetime
import calendar
import numpy as np
from datetime import timedelta
import json
import re
import cufflinks as cf
from matplotlib.gridspec import GridSpec
from matplotlib.legend import Legend
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from matplotlib import pyplot as plt
plt.style.use("seaborn")

from numpy import inf
import plotly.figure_factory as ff
import chart_studio.plotly as py
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
cf.go_offline()


#DF SOURCE#
df_source = pd.read_csv(r"C:\Users\l.verni\Documents\Local-Repo\analytics\Livechat\source.csv")

df_source.rename(columns={"conferenceId": "id", "chat start date Europe/London": "start_time", "chat start url": "url_source", "group name": "site", "last rate comment": "rate_comment", "last operator id": "agent_id", "operator 1 nick": "agent_1", "operator 2 nick": "agent_2", "operator 3 nick": "agent_3", "post chat: Would you use chat again?": "survey_chat", "post chat: How likely are you to use Techbuyer again?": "survey_tb"}, inplace=True)

df_source.drop(df_source[df_source.site == "DE Sales"].index, inplace=True) #droping germany site

df_source = df_source[["id", "start_time", "site", "rate_comment", "agent_id", "agent_1", "agent_2", "agent_3", "rate", "survey_chat", "survey_tb"]]
df_source.reset_index(drop=True, inplace=True)

df_source[["site", "rate", "agent_1", "agent_2"]] = df_source[["site", "rate", "agent_1", "agent_2"]].replace(" ", "_", regex=True)

df_source["date"] = df_source["start_time"].apply(lambda x: x[0:10]) #date column


#DF TRANSFERRED#
df_source["transferred"] = df_source["agent_2"].apply(lambda x: 1 if type(x) == str else 0)

df_transf = df_source.groupby(["date", "agent_1", "site"], as_index=False).agg({"transferred": "sum"}).rename(columns={"agent_1": "agent"})

#RATED COLUMNS#
df_source["rated_good"] = df_source["rate"].apply(lambda x: 1 if x == "rated_good" else 0)
df_source["rated_bad"] = df_source["rate"].apply(lambda x: 1 if x == "rated_bad" else 0)

#DF PERFORMANCE#
df_perf = pd.read_csv(r"C:\Users\l.verni\Documents\Local-Repo\analytics\Livechat\performance.csv").set_index("Agent")

#QA
import random
for i in range(0, len(df_source), 3):
    df_source.at[i, "qa_score"] = random.randint(0, 100)

#MAIN AGENT#
df_source["agent"] = ""
for i in range(len(df_source)):
    if str(df_source.iloc[i]["agent_2"]) == "nan":
          df_source.at[i, "agent"] = df_source.iloc[i]["agent_1"]
    elif str(df_source.iloc[i]["agent_2"]) != "nan" and str(df_source.iloc[i]["agent_3"]) == "nan":
        df_source.at[i, "agent"] = df_source.iloc[i]["agent_2"]
    else:
        df_source.at[i, "agent"] = df_source.iloc[i]["agent_3"]

#DF AGENT#
df_agent_daily = df_source.groupby(["date", "agent", "agent_id", "site"], as_index=False).agg({"rate": "count", "rated_good": "sum", "rated_bad": "sum"}).rename(columns={"rate": "chats_total"})


df_agent_weekly = df_source.groupby(["agent", "agent_id", "site"], as_index=False).agg({"rate": "count", "rated_good": "sum", "rated_bad": "sum"}).rename(columns={"rate": "chats_total"})

df_agent_weekly.insert(6, "availability", df_agent_weekly["agent_id"].apply(lambda x: df_perf.loc[x]["Accepting time"]))

df_t = df_transf.groupby(["agent", "site"], as_index=False).agg({"transferred": "sum"})
df_agent_weekly.insert(4, "transferred", df_t["transferred"]) #TRANSFERRED COLUMN

#DF SITE#
df_site = df_agent_weekly.groupby(["site"], as_index=False).agg({"chats_total": "sum", "transferred": "sum", "rated_good": "sum", "rated_bad": "sum", "availability": "sum"}).rename(columns={"rate": "chats_total"})

df_site_daily = df_source.groupby(["date", "site"], as_index=False).agg({"rate": "count", "rated_good": "sum", "rated_bad": "sum"}).rename(columns={"rate": "chats_total"})


#DF SURVEY#
df_survey = df_source[["site", "survey_chat", "survey_tb"]].drop(df_source[(df_source["survey_chat"].isnull())|(df_source["survey_tb"].isnull())].index).reset_index(drop=True)

survey_chat = "Would you use chat again?" 
df_survey_chat = df_survey[["site", "survey_chat"]] #DF SURVEY CHAT
df_survey_chat["Yes"] = df_survey_chat["survey_chat"].apply(lambda x: 1 if x == "Yes" else 0)
df_survey_chat["No"] = df_survey_chat["survey_chat"].apply(lambda x: 1 if x == "No" else 0)
df_survey_chat = df_survey_chat.drop(columns="survey_chat").groupby("site").agg({"Yes": "sum", "No": "sum"})

survey_tb = "How likely are you to use Techbuyer again?"
df_survey_tb = df_survey[["site", "survey_tb"]] #DF SURVEY TB
survey_tb_cols = df_survey_tb["survey_tb"].unique()
df_survey_tb[[col for col in survey_tb_cols]] = np.nan

for i in range(len(df_survey_tb)):
    col = df_survey_tb.iloc[i]["survey_tb"]
    df_survey_tb.at[i, col] = 1

df_survey_tb = df_survey_tb.fillna(0).drop(columns="survey_tb").groupby("site").agg({survey_tb_cols[0]: "sum", survey_tb_cols[1]: "sum", survey_tb_cols[2]: "sum", survey_tb_cols[3]: "sum"})

#COMMENTS OF THE WEEK#
comments = df_source[df_source["rate_comment"].notnull()][["rate_comment", "agent"]]
comments["rate_comment"] = comments["rate_comment"] + " (" +comments.agent + ")"
comments = comments.drop(columns="agent")



###############
#VISUALIZATION#
###############


for site in df_site["site"].unique():
    df = df_site[df_site["site"]==site].set_index(["site"])
    fig = plt.figure(figsize=(10, 1))
    ax = sns.heatmap(df, annot=True)


for site in df_agent_weekly["site"].unique():
    df = df_agent_weekly[df_agent_weekly["site"]==site].drop(columns=["agent_id", "site"]).set_index(["agent"])
    fig = plt.figure(figsize=(10, 2))
    ax = sns.heatmap(df, annot=True)



#QA SEPARATE DF
# for site in df_agent_weekly["site"].unique():
#     df = 
#     fig = plt.figure(figsize=(10, 2))
#     ax = sns.barplot(df.qa_score, df.agent, data=df, palette="Blues_d", orient="h")



#PLOT DAILY CHATS PER DAY PER SITE


#DONUT SURVEY  TB

#Mekko Chart SURVEY CHAT



