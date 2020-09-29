import re
from PIL import Image
from pathlib import Path
from typing import Union
import string

import io
import streamlit as st
import pandas as pd
import emoji
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import plotly.express as px

ProjectRoot = Path(__file__).resolve().parent.parent

st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown('<h1 align="center"> WhatsApp Group Chat Analyser </h1>',unsafe_allow_html=True)
st.markdown('<img src="https://i.imgur.com/USRuYJ8.png" align="center" />', unsafe_allow_html=True)
# GROUP_NAME = 'RandomGroup'
st.markdown('<h4 align="center"> Upload WhatsApp chat text file </h4>',  unsafe_allow_html=True)


class Chat_Analyser:
    def __init__(self,chat_data,user_name=None,group_name=None):
        self.user_name = user_name
        self.group_name = group_name
        # Reading WhatsApp data files
        self.group_data  = chat_data
        self.dataframe =  self.create_chat_data_frame()  
        self.dataframe.to_csv('df.csv',index=False)
        # match the given username with chat username using regex
        if self.user_name!=None and self.user_name!='':
            user_names = pd.Series(self.dataframe['Name_or_number'].unique())
            try:
                if '+' in self.user_name:
                    self.user_name = self.user_name.replace('+', "\+")
                self.user_name = user_names[user_names.str.contains(f"{self.user_name}",regex=True,flags=re.IGNORECASE).dropna()].values[0]
            except:
                self.user_name = None
        # replace media messages with media text
        self.dataframe['Messages'] = self.dataframe['Messages'].apply(lambda x : x if ': <Media omitted>\n' not in x else 'MEDIA')
        st.markdown('<h2 align="center"> Chat data overview </h2>', unsafe_allow_html=True)
        st.dataframe(self.dataframe.sample(n=10))  
        self.group_insights = {}
        self.user_insights = {}

    @staticmethod
    def handel_void_outputs(Data_extractor_output:str = ""):
        """this function returns a none value if list is empty"""
        if type(Data_extractor_output) == type(re.match('','')):
            # This is to handel name or mobile number in person name feild
            if Data_extractor_output.groups()[0] != None:
                return Data_extractor_output.groups()[0]
            else :
                return Data_extractor_output.groups()[1]
        elif Data_extractor_output is None:
            return None
            # This is to handel meessages,date and time
        elif  len(Data_extractor_output) == 0:
            return None
        else:
            return(Data_extractor_output[0])
        
    def data_extractor(self,group_data:str = "")-> dict: 
        """this function extracts date,time,name,messages"""
        # this is to extract dates from data
        date = self.handel_void_outputs(re.findall('\d{1}\/\d{2}\/\d{2}|\d{2}\/\d{2}\/\d{2}',group_data))
        # This is to extract time from data
        time = self.handel_void_outputs(re.findall('\d{1}\:\d{2}|\d{2}\:\d{2}',group_data))
        # This is to extract name or mobile number from the data
        name = self.handel_void_outputs(re.search('-\s([\w\s@\-_@\!\*\#]+)|-\s(\+\d{2}\s\d{5}\s\d{5})',group_data))
        # this is to extract messages from the data 
        messages = self.handel_void_outputs(re.findall(':\s[\w\s@\-_\$\@\!\*\#\.\(\)\[\]\>\<\+\&]*.*',group_data))
        return({'Date':date,'Time':time,'Name_or_number':name,'Messages':messages,})


        
    def create_chat_data_frame(self):
        """here we are mapping Data_extractor function with self.group_datas"""
        data_frame_whatsapp_chat = pd.DataFrame(map(self.data_extractor,self.group_data))
        # this is to remove te none values from the data frame
        data_frame_whatsapp_chat_without_null_values = data_frame_whatsapp_chat.dropna()
        return data_frame_whatsapp_chat_without_null_values

    def most_active_member_in_group(self):
        #  In this we are taking the count of Name_or_number to get most active member in group 
        user_chat_frequencey = self.dataframe['Name_or_number'].value_counts()
        self.user_insights['user_num_messages'] = dict(user_chat_frequencey).get(self.user_name,None)
        self.group_insights['num_messages_per_user'] = user_chat_frequencey
        st.markdown('<h2 align="center">Chat count per person </h2>',unsafe_allow_html=True)
        st.markdown('<h4 >Select user name </h4>',unsafe_allow_html=True)
        user_name = st.selectbox(' ',list(user_chat_frequencey.keys()))
        st.write(user_chat_frequencey[user_name])
        st.dataframe(self.dataframe[self.dataframe['Name_or_number']==user_name])
        st.markdown(f"<h5>Most active user in the group: {user_chat_frequencey.keys()[0]} </h5>",unsafe_allow_html=True)
        st.markdown(f"<h5>Most deactive users in the group: </h5>",unsafe_allow_html=True)
        st.dataframe(user_chat_frequencey[user_chat_frequencey<=5])
        if self.user_name != None:
            if self.user_insights['user_num_messages'] != None:
                
                st.markdown(f'<h2 align="center">{self.user_name}\'s messages in group </h2>',unsafe_allow_html=True)
                st.dataframe(self.dataframe[self.dataframe['Name_or_number']==self.user_name])
                
    def peak_chatting_time__of_the_day(self):
        # in this we are adding a new column date_time to dataframe
        self.dataframe['Date_Time'] = self.dataframe['Date'] + ' ' + self.dataframe['Time']
        # in this we are adding another column in dataframe to convert type and format of date_time 
        self.dataframe['datetimes'] = pd.to_datetime(self.dataframe['Date_Time'], format = '%d/%m/%y %H:%M')
        # in this we are assining a new column to the data frame of day sessions
        data_frame_with_seesions = self.dataframe.assign(session=pd.cut(self.dataframe.datetimes.dt.hour,[0,3,7,12,17,19,24],labels=['MidNight','EarlyMorning','Morning','Afternoon','evening','night']))
        # in this we will take count of the session column to get most active time in day
        peak_chatting_time =  data_frame_with_seesions.session.value_counts()
        
        # plot pie chat 
        labels = list(peak_chatting_time.keys())
        values = list(peak_chatting_time.values)

        # Use `hole` to create a donut-like pie chart
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.7)])
        st.markdown(f'<h2 align="center">Peak chatting time of the day</h2>',unsafe_allow_html=True)
        st.plotly_chart(fig)

    def average_messages_in_a_day(self):
        # In this we are taking the mean count of the date
        self.dataframe['Month'] =  self.dataframe["datetimes"].apply(lambda x: x.strftime("%b"))
        self.dataframe['year'] = pd.DatetimeIndex(self.dataframe['Date']).year
        moth_year = self.dataframe['Month'].astype(str) + "-" + self.dataframe['year'].astype(str)
        self.dataframe['Month_year'] = moth_year
        xAxis = (self.dataframe.groupby(["Month_year"])["Messages"].count()).keys()
        yAxis = (self.dataframe.groupby(["Month_year"])["Messages"].count()).values
        plt.plot(xAxis,yAxis)
        plt.title('Graph to show number of messages per month')
        plt.xlabel('Month')
        plt.ylabel('Number of messages')
        st.markdown('<h2 align="center" >  Number of messages per month  <h2>', unsafe_allow_html=True)
        st.pyplot(plt.show())
        



    def person_sends_more_emoji(self):
        messages =  list(self.dataframe['Messages'].values)
        def extract_emojis(s):
            return [c for c in s if c in emoji.UNICODE_EMOJI]
        emojies_in_chat = list(map(extract_emojis,messages)) 
        # here we are creating new column emojies 
        self.dataframe['emojies'] = emojies_in_chat
        # here we are using groupby function to find how many emojies sent by each person
        self.dataframe['number_of_emojies'] = self.dataframe['emojies'].apply(lambda x : len(x))
        sum_of_emojies = self.dataframe.groupby("Name_or_number")['number_of_emojies'].sum()
        emoji_data_frame = pd.DataFrame(sum_of_emojies)
        emoji_data_frame_sorted = emoji_data_frame.sort_values("number_of_emojies",ascending=False)
        st.markdown('<h2 align="center" >  Person who sends more emojies <h2>', unsafe_allow_html=True)
        st.dataframe(emoji_data_frame_sorted[0:3])

    def word_cloud(self):
        self.dataframe['messages_without'] = self.dataframe['Messages'].apply(lambda x : "\u200b" in x or 'MEDIA' in x or 'https' in x )
        text = " ".join(self.dataframe[self.dataframe['messages_without'] == False]['Messages'].apply(lambda x : x.strip(string.punctuation+string.whitespace)).tolist())
        # Define a function to plot word cloud
        def plot_cloud(wordcloud):
            # Set figure size
            plt.figure(figsize=(40, 30))
            # Display image
            plt.imshow(wordcloud) 
            # No axis details
            plt.axis("off")

        # Generate word cloud
        wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, background_color='black', colormap='Pastel1', collocations=False, stopwords = STOPWORDS).generate(text)
        # Plot
        #TODO pass matplot fig to st.plot instead of imshow
        st.markdown(f'<h2 align="center">Word cloud</h2>',unsafe_allow_html=True)
        st.pyplot(plot_cloud(wordcloud))
        
                    
    def Time_span(self):
        d =  self.dataframe['datetimes']
        starting_date = d.min()
        last_date = d.max()
        st.markdown(f"<h4>This file contains messages from : {starting_date} to {last_date} </h4>",unsafe_allow_html=True)
        

if __name__ == "__main__":
    st.markdown('<h4 > Person Name </h4>',unsafe_allow_html=True)
    user_name = st.text_input('',value='')
    st.markdown('<h4 > Group Name </h4>',unsafe_allow_html=True)
    group_name = st.text_input(' ',value='')
    file_buffer = st.file_uploader(label='')
    try:
        analyser = Chat_Analyser(file_buffer.readlines(), user_name=user_name, group_name=group_name)
        analyser.most_active_member_in_group()
        analyser.peak_chatting_time__of_the_day()
        analyser.average_messages_in_a_day()
        analyser.person_sends_more_emoji()
        analyser.Time_span()
    except Exception as e:
        st.markdown('<h3 align="center" > <font color="green"> !Please upload chat text file to begin analysis..... </font> <h3>', unsafe_allow_html=True)


