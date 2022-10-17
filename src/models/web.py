import json
import streamlit as st
import matplotlib
import requests

#CONFIG
URL = "https://59e4tgb4xa.execute-api.eu-west-3.amazonaws.com/v1/predicted-word?words=" 

def fetch(text):
    """Function to post a request to SageMaker endpoint. The input needs to follow the next format:
       For the sentence Prev1 Prev2 Prev3 Word Next1 Next2 Next3, where the model needs to predict
       the Word word, text needs to be ["Prev1", "Prev2", "Prev3", "Next1", "Next2", "Next3"].
       The output of the request, will be the predicted central word, Word."""
    try:
        words = ",".join(text)
        #API Gateway request
        r = requests.get(url = URL + words)
        res = json.loads(r.text)
        return res['prediction'], r.status_code
    except Exception as err:
        return err, 0

try:
    st.set_page_config(page_title='Crystal gazers', page_icon=":crystal_ball:", layout="wide")

    font = {'weight': 'normal',
        'size': 22}
    matplotlib.rc('font', **font)

    st.title('Crystal gazers')
    st.header("Endpoint accessing")
    #session = requests.Session()
    st.write("Input 6 context word around the word you would like to predict.")
    with st.form("test"):
        col1, col2 = st.columns(2)
        with col1:
            text_left = st.text_input("Input previous words", key="text")
        with col2:
            text_right = st.text_input("Input following words", key="text")
        submitted = st.form_submit_button("Submit")

        if submitted:
            text = text_left.split() + text_right.split()
            if len(text) == 6:
                st.write("Results")
                res, status = fetch(text)
                if status == 200:
                    st.write(res)
                else:
                    st.error("Error")
            else:
                st.error("Input size needs to be 6 words, 3 on the left \
                    of the desired word and 3 on the right. Please try again.")
except NotImplementedError as nie:
    print(f'[ERROR] {nie}')
except Exception as e:
    raise e
