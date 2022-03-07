from tkinter import *
from tkinter import ttk  # ttk is used for styling
from PIL import Image, ImageTk 
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import string
import nltk
# import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
# from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
ps=PorterStemmer()
class home:
    col=[]
    r=""
    def __init__(self, root):
        self.root = root
        self.root.geometry("1450x900+0+0")
        # self.resizeable(False,False)
        self.root.title("Email spam classifier")

        #variables

        self.var_location = StringVar()  # memberType
        self.var_type = StringVar()
        self.var_build = StringVar()
        self.var_furnish = StringVar()  # memberType
        self.var_bathroom = StringVar()
        self.var_parking = StringVar()
        self.var_carpet = StringVar()  # memberType

        #bg_img
        img1 = Image.open("image/imag111.jpg")
        img1 = img1.resize((1450, 900), Image.ANTIALIAS)
        self.photoimg1 = ImageTk.PhotoImage(img1)
        bg_img = Label(self.root, image=self.photoimg1)
        bg_img.place(x=0, y=0, width=1450, height=900)
        # self.title=Label(self,text="Email/SMS Spam Classifier",font="Bold 30")
        # self.title.place(x=200,y=10)
        login_frame = Frame(bg_img, bd=2, bg="orange", highlightthickness=5)
        login_frame.place(x=50, y=100, width=400, height=720)
        login_frame.config(highlightbackground="black", highlightcolor="black")

        account_label = Label(
           login_frame,
            text="ENTER MESSAGE",
            font=("fantasy", 23,"bold"),#17
            bg="#FFF8DC",
            fg="red",
        )
        account_label.grid(row=0, column=0, padx=0)
        account_label.place(x=50, y=20, anchor=NW)

        self.txt = Text(login_frame, bg="#F0F8FF" , fg="black" ,width=42, height=20)
        self.txt.grid(column=3, row=0)
        self.txt.place(x=20, y=90, anchor=NW)
        input = self.txt.get("5.0",END)
        print(input)
        submit_btn = Button(
            login_frame,
           command=self.submit_details,
            width=20,
            height=0,
            text="PREDICT",
            font=("times new roman", 18, "bold"),
            bg="#EEEEEE",
            fg="black",
        )
        submit_btn.grid(row=4, column=0)
        submit_btn.place(x=55, y=510, anchor=NW)

        result_text_label = Label(
            login_frame,
            textvariable=self.var_build,
            font=("Helvetica", 30,"bold"),
            fg="purple",
            bg="orange",
            width=10,
        )
        result_text_label.place(x=60, y=450, anchor=NW) # y=30
        df=pd.read_csv('spam.csv',encoding='Windows-1252')
        df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
        df.rename(columns={'v1':'Target','v2':'Text'},inplace=True)
        encoder=LabelEncoder()
        df['Target']=encoder.fit_transform(df['Target'])
        df=df.drop_duplicates(keep='first')
        df['Target'].value_counts()
        plt.pie(df['Target'].value_counts(),labels=['ham','spam'],autopct='%0.2f')
        plt.show()
        nltk.download('punkt')
        df['num_char']=df['Text'].apply(len)
        df['num_words']=df['Text'].apply(lambda x:len(nltk.word_tokenize(x)))
        df['Text'].apply(lambda x:nltk.sent_tokenize(x))
        df['num_sentences']=df['Text'].apply(lambda x:len(nltk.sent_tokenize(x)))
        tfidf=TfidfVectorizer(max_features=3000)
        df['transformed_text']=df['Text'].apply(self.transform)
        X=tfidf.fit_transform(df['transformed_text']).toarray()
        wc=WordCloud(width=1000,height=1000,min_font_size=10,background_color='white')
        spam_wc=wc.generate(df[df['Target']==1]['transformed_text'].str.cat(sep=" "))
        # X=np.hstack((X,df['num_char'].values.reshape(-1,1)))
        # from sklearn.preprocessing import MinMaxScaler
        # scaler=MinMaxScaler()
        plt.figure(figsize=(12,6))
        plt.imshow(spam_wc)
        spam_wc=wc.generate(df[df['Target']==0]['transformed_text'].str.cat(sep=" "))
        plt.figure(figsize=(12,6))
        spam_words=[]
        for msg in df[df['Target']==1]['transformed_text'].tolist():
            for word in msg.split():
                spam_words.append(word)
        plt.figure(figsize=(12,6))
        Counter(spam_words).most_common(40)
        plt.barplot(pd.DataFrame(Counter(spam_words).most_common(40))[0],pd.DataFrame(Counter(spam_words).most_common(40))[1])
        plt.xticks(rotation='vertical')
        plt.show()        
        plt.imshow(spam_wc)
        ham_words=[]
        for msg in df[df['Target']==0]['transformed_text'].tolist():
            for word in msg.split():
                ham_words.append(word)
        plt.figure(figsize=(12,6))
        plt.barplot(pd.DataFrame(Counter(ham_words).most_common(40))[0],pd.DataFrame(Counter(ham_words).most_common(40))[1])
        plt.xticks(rotation='vertical')
        plt.show()    
        # X=scaler.fit_transform(X)
        y=df['Target'].values
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)#20% data test
        gnb=GaussianNB()
        mnb=MultinomialNB()
        bnb=BernoulliNB()
        mnb.fit(X_train,y_train)
        y_pred1=mnb.predict(X_test)
        print(accuracy_score(y_test,y_pred1))
        print(confusion_matrix(y_test,y_pred1))
        print(precision_score(y_test,y_pred1))
        



    def transform(self,text):
        text=text.lower()
        text=nltk.word_tokenize(text)
        y=[]
        for i in text:
            if i.isalnum():
                y.append(i)
        text=y[:]
        y.clear()
        for i in text:
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(i)
        text=y[:]
        y.clear()
        for i in text:
            y.append(ps.stem(i))
        return " ".join(y)    
      
    def submit_details(self):
        input = self.txt.get("1.0",END)
        
        tfidf=pickle.load(open('vectorizer.pkl','rb'))
        modal=pickle.load(open('model.pkl','rb'))

        print(input)
        transformed_sms=self.transform(input)
        #vectorize
        vector_input=tfidf.transform([transformed_sms])
        #predict
        result=modal.predict(vector_input)[0]
        #display
        if result == 1:
            print("Spam")
            home.r="SPAM TEXT"
        else:
            print("Not Spam")
            home.r="HAM TEXT"
        self.var_build.set(home.r)
if __name__ == "__main__":
    root = Tk()
    obj = home(root)
    root.mainloop()
