import streamlit as st
import json
import matplotlib
import boto3

#CONFIG
boto3.setup_default_session(profile_name='webapi')
client = boto3.client('sagemaker-runtime')
#URL = 'https://runtime.sagemaker.eu-west-3.amazonaws.com/endpoints/pytorch-inference-2022-10-15-19-35-33-706/invocations'


def fetch(text):
    try:
        #Sagemaker request
        data = {'input': text}
        body = str.encode(json.dumps(data))
        response = client.invoke_endpoint(
            EndpointName='pytorch-inference-2022-10-16-11-01-23-464', 
            ContentType='application/json',
            Accept='application/json',
            Body=body
            )
        res_json = json.loads(response['Body'].read().decode("utf-8"))
        #print(json.load(response))
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
            input = text.split()
            if len(input) == 6:
                st.write("Results")
                res, status = fetch(input)
                if status:
                    st.write(res)
                else:
                    st.error("Error")
            else:
                st.error("Input size needs to be 6 words, 3 on the left of the desired word and 3 on the right. Please try again.")
    

except NotImplementedError as nie:
    print(f'[ERROR] {nie}')
    pass
except Exception as e:
    raise e