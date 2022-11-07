import streamlit as st
import joblib
import torch as torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# Load tokenizer 
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")


# Load Model
# with open('model.pkl', 'rb') as file_1:
#   model = joblib.load(file_1) 
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")



def run():
#     # Membuat Form
    with st.form(key='form_parameters'):
            txt = st.text_area('Enter text: ')
            # Create tokens - number representation of our text
            tokens = tokenizer(txt, truncation=True, padding="longest", return_tensors="pt")
            


            submitted = st.form_submit_button('Summarize')


    if submitted:

       
        # Decode summary
        summary = model.generate(**tokens)
        # summary =torch.tensor.model(**tokens)
        pred=tokenizer.decode(summary[0])
        
    
        st.write('## summarize : ' + (pred))

if __name__ == '__main__':
    run()