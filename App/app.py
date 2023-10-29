import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

#Utils
import joblib

lr_pipe = joblib.load(open("models\emotion_classifier_pipeline_lr_2023.pkl","rb"))

def predict_emotions(docx):
    results = lr_pipe.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = lr_pipe.predict_proba([docx]) 
    return results


def main():
    st.title("Emotion Classifier App")
    menu = ["Home", "Statistics", "About"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        st.subheader("Home - How was your day?")
        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type here")
            submit_text = st.form_submit_button(label="Submit")

        if submit_text:
            col1,col2 = st.columns(2)

            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                st.write(prediction)
                st.write("Confidence:{}".format(np.max(probability)))

            with col2:
                st.success("Prediction Probability:")
                #st.write(probability)
                proba_df = pd.DataFrame(probability,columns=lr_pipe.classes_)
                #st.write(proba_df.T)

                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["Emotions","Probability"]
                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='Emotions',y='Probability',color='Emotions')
                st.altair_chart(fig,use_container_width=True)   


    elif choice == "Statistics":
        st.subheader("Statistics App")
    else:
        st.subheader("About")


if __name__ == '__main__':
    main()