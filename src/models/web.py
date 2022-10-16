import json
import streamlit as st
import matplotlib
import boto3

#CONFIG
boto3.setup_default_session(profile_name='cgmiquel')
client = boto3.client('sagemaker-runtime')

def fetch(text):
    """Function to post a request to SageMaker endpoint. The input needs to follow the next format:
       For the sentence Prev1 Prev2 Prev3 Word Next1 Next2 Next3, where the model needs to predict
       the Word word, text needs to be ["Prev1", "Prev2", "Prev3", "Next1", "Next2", "Next3"].
       The output of the request, will be the predicted central word, Word."""
    try:
        #Sagemaker request
        data = {'input': text}
        body = str.encode(json.dumps(data))
        response = client.invoke_endpoint(
            EndpointName='pytorch-inference-2022-10-16-11-01-23-464',
            ContentType='application/json',
            Accept='application/json',
            Body=body)
        res_json = json.loads(response['Body'].read().decode("utf-8"))
        return res_json, 1
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
        text = st.text_input("Input text", key="text")
        submitted = st.form_submit_button("Submit")

        if submitted:
            text = text.split()
            if len(text) == 6:
                st.write("Results")
                res, status = fetch(text)
                if status:
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
