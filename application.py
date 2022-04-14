import nltk
import streamlit as st
import pickle
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer


ps = PorterStemmer()
tf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


def clean_message(message):
    message = message.lower()
    message = nltk.word_tokenize(message)

    list1 = []
    for i in message:
        if i.isalnum():
            list1.append(i)

    message = list1[:]
    list1.clear()

    for i in message:
        if i not in stopwords.words('english') and i not in string.punctuation:
            list1.append(i)

    message = list1[:]
    list1.clear()

    for i in message:
        list1.append(ps.stem(i))

    return " ".join(list1)



st.title("Email Classifier")


user_input = st.selectbox("Message", ('-',"type your own message","Oh k...i'm watching here:)", 'Is that seriously how you spell his name?', 'Please call our customer service representative on 0800 169 6031 between 10am-9pm as you have WON a guaranteed £1000 cash or £5000 prize!','Here is your discount code RP176781. To stop further messages reply stop. www.regalportfolio.co.uk. Customer Services 08717205546'))

if user_input == "type your own message":
    user_input = st.text_input("Enter your message")
if st.button('Predict'):

    cleaned_message = clean_message(user_input)

    message_vector = tf.transform([cleaned_message])

    output = model.predict(message_vector)[0]

    if output == 1:
        st.header("Not Spam")
    else:
        st.header("Spam")

