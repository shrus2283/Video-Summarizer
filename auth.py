import streamlit as st
# from streamlit import session_state
import yt_dlp 
import warnings
import re
warnings.filterwarnings("ignore")

from pathlib import Path

from ibm_watson import SpeechToTextV1
from ibm_watson.websocket import RecognizeCallback, AudioSource
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import pipeline
import gensim
from gensim.summarization import summarize
from gensim.summarization.textcleaner import split_sentences
from gensim.summarization import keywords

ydl_opts = {
    'format': 'm4a/bestaudio/best',
    # ℹ️ See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
    'postprocessors': [{  # Extract audio using ffmpeg
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '128',
    
    }],
    'ffmpeg-location' : ' ./C:/ffmpeg/ffmpeg-2022-12-08-git-9ca139b2aa-full_build/bin ' ,
    'outtmpl' : "D:/videosummarizer/edi.mp4" , 
}

st.set_page_config(page_title="Video Summarizer", page_icon="📽️")


# api key and url
apikey = 'AyZjk3cVLAeyN0q_sewRJQbx4a3Iho7IIpR7VR7sfKT8'
url = 'https://api.au-syd.speech-to-text.watson.cloud.ibm.com/instances/e0a13617-af58-4e03-a74a-61203f6b629c'

# link accepted only when submit button is pressed 
def get_link():
    link = st.text_input("Enter your YouTube video link below" , "")
    
    if (st.button("Submit")):
        return link
  
# show video only if the link is entered
def show_video(link):
    if link is None:
        return ""
    else:
        st.video(link)

def get_audio_file(link):
    if link is None:
        return ''
    
    _id=link.strip()
    
    def get_video(_id):
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            authautharva= ydl.download(_id)
    get_video(_id)
    
def speech_2_text(apikey , url , link):
    if link is None:
        return ''
    else:
        authenticator=IAMAuthenticator(apikey)
        stt=SpeechToTextV1(authenticator=authenticator)
        stt.set_service_url(url)
        
        with open('D:/videosummarizer/edi.mp3', 'rb') as f:
            res=stt.recognize(audio=f, content_type='audio/mp3',split_transcript_at_phrase_end="true",speech_detector_sensitivity=0.4,end_of_phrase_silence_time=0.6, model = 'en-WW_Medical_Telephony').get_result()
        
        #Data Preprocessing    
        text=[result['alternatives'][0]['transcript'].rstrip()+ '.\n' for result in res['results']]
        text=[para[0].title()+para[1:] for para in text]
        text=''.join(text)
        
        
        return text


 
def extractive(text , link,words):
    if link is None:
        return ''
    ext_summary=summarize(text,ratio=words)
    
    return (ext_summary)
     
def keywordss(ext_summary,link):
    if link is None:
        return ''
    keywordsss=keywords(ext_summary, words = 5,scores=True)
    return keywordsss
   
def abstractive(text , ext_summary , link):
    if link is None:
        return ''
    model_name="google/pegasus-cnn_dailymail"
    pegasus_tokenizer = PegasusTokenizer.from_pretrained(model_name)
    #pegasus_model=PegasusForConditionalGeneration.from_pretrained(model_name)
    
    
    if(size<=5700000):
        print("ifff")
        #tokens=pegasus_tokenizer(text,truncation=True,padding="max_length", return_tensors="pt")
        #encoded_summary=pegasus_model.generate(**tokens)
        #decoded_summary=pegasus_tokenizer.decode(encoded_summary[0], skip_special_tokens=True)
        
        summarizer=pipeline("summarization", model=model_name,tokenizer=pegasus_tokenizer,framework="pt")
        abs_summary=summarizer(text,min_length=50,max_length=200)
        abs_summ=abs_summary[0]["summary_text"]
    elif(size>5700000):
        print("elifff")
        #tokens=pegasus_tokenizer(ext_summary,truncation=True,padding="max_length", return_tensors="pt")
        #encoded_summary=pegasus_model.generate(**tokens)
        #decoded_summary=pegasus_tokenizer.decode(encoded_summary[0], skip_special_tokens=True)
        
        summarizer=pipeline("summarization", model=model_name,tokenizer=pegasus_tokenizer,framework="pt")
        abs_summary=summarizer(ext_summary,min_length=50,max_length=200)
        abs_summ=abs_summary[0]["summary_text"]
    abs_summ=re.sub("<n>","",abs_summ)
        
        
    return abs_summ

def generato(link, text):
    if link is None:
        return ''
    my_expander = st.expander(label="Generated Text")
    my_expander.write(text)
    
def extracto(link, ext_summary):
    if link is None:
        return ''
    my_expander = st.expander(label="Extractive Summary")
    my_expander.write(ext_summary)

def keywordo(link , keywordsss):
    if link is None:
        return ''
    my_expander = st.expander(label="Keywords")
    my_expander.write(keywordsss)

def abstracto(link, abs_summary):
    if link is None:
        return ''
    my_expander = st.expander(label="Abstractive Summary")
    my_expander.write(abs_summary)


st.title("Video Summarizer📽️")

link = get_link()
show_video(link)
words = st.slider('Enter Summary Ratio', 0.0, 1.0, 0.5)
with st.spinner('Audio fetching in progress'):
    get_audio_file(link)

file = Path() / 'D:/videosummarizer/edi.mp3'  # or Path('./doc.txt')
size = file.stat().st_size
#print(size)

with st.spinner('Speech-2-Text in progress'):
    text = speech_2_text(apikey , url , link)
    
text_len = len(text.split())

generato(link,text)


ext_summary=extractive(text , link,words)
ext_summary_len = len(ext_summary.split())
extracto(link, ext_summary)

keywordsss=keywordss(ext_summary , link)
keywordo(link, keywordsss)


abs_summary=abstractive(text,ext_summary ,link)
abs_summary_len = len(abs_summary.split())
abstracto(link, abs_summary)


st.write(" Original text length ->" , text_len)
st.write(" Extractive Summary length ->" , ext_summary_len)
st.write(" Abstractive Summary length ->" , abs_summary_len)



