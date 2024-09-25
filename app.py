import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
import pickle

def main():
    st.title("Water Potability Prediction")
    st.sidebar.title("Water Potability Prediction")
    st.markdown("Lets Find Quality of Water")
    st.sidebar.markdown("Lets Find Quality of Water")

#---- load data---
    #st.cache_data(persist=True)
    def load_data():
        data=pd.read_csv("ready.csv")
        return data

    #st.cache_data(persist=True)
    def split(df):
        #x=pd.get_dummies(data=df,drop_first=True,dtype="int64")
        x=df.drop(["Potability"],axis=1)
        y=df.Potability
        #_________#
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=12,shuffle=True)
        return x_train,x_test,y_train,y_test
    
    df=load_data()

    x_train,x_test,y_train,y_test = split(df)
    Regressor=st.sidebar.selectbox("Regressor",("LR","DCT","RF"))

    # DTC Hyper Parameters Set
    if Regressor=="DCT":
        st.sidebar.subheader("HyperParameters")
        max_depth =st.sidebar.number_input("max_depth",1,100,step=1,key="max_depth")
        min_samples_split = st.sidebar.number_input("min_samples_split",1,1000,step=1,key="min_samples_split")
        min_samples_leaf=st.sidebar.number_input("min_samples_leaf",1,100,step=1,key="min_samples_leaf")
    # Train Regressor
    

        if st.sidebar.button("Regressor",key="Regressor"):
            st.subheader("Decision Tree Regressor")
            dtr = DecisionTreeRegressor(max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf)
            dtr.fit(x_train,y_train)
            y_pred=dtr.predict(x_test)
            acc=sqrt(mean_squared_error(y_pred,y_test))
            result=dtr.predict(x_test)
            st.write("Rmse:",acc)
            st.write("RSquare:",r2_score(y_test,y_pred))
            st.write("Input Parameters",x_test)
            st.write("Price",result)
           
            
    # Random Forest 
    if Regressor=="RF":
        st.sidebar.subheader("HyperParameters")
        max_depth =st.sidebar.number_input("max_depth",1,100,step=1,key="max_depth")
        min_samples_split = st.sidebar.number_input("min_samples_split",1,1000,step=1,key="min_samples_split")
        min_samples_leaf=st.sidebar.number_input("min_samples_leaf",1,100,step=1,key="min_samples_leaf")
        n_estimators=st.sidebar.number_input("n_estimators",1,1000,key="n_estimators")
    # Train Regressor
       

        if st.sidebar.button("Regressor",key="Regressor"):
            st.subheader("Random Forest Regressor")
            rf= RandomForestRegressor(max_depth=max_depth,min_samples_split=min_samples_split,
                                       min_samples_leaf=min_samples_leaf,n_estimators=n_estimators)
            rf.fit(x_train,y_train)
            y_pred_rf=rf.predict(x_test)
            acc1=sqrt(mean_squared_error(y_pred_rf,y_test))
            result=rf.predict(x_test)
            st.write("Rmse:",acc1)
            st.write("RSquare:",r2_score(y_test,y_pred_rf))
            st.write("Input Parameters",x_test)
            st.write("Price",result)
            
#      # Linear Regression 
    if Regressor=="LR":
        st.sidebar.subheader("HyperParameters")
        st.sidebar.subheader("No Hyper Parameters are passed")
       
    # Train classifier
       

        if st.sidebar.button("Regressor",key="Regressor"):
            st.subheader("Linear Regression")
            lr = LinearRegression()
            lr.fit(x_train,y_train)
            y_pred_lr=lr.predict(x_test)
            acc_lr=sqrt(mean_squared_error(y_pred_lr,y_test))
            st.write("Rmse:",acc_lr)
            st.write("RSquare:",r2_score(y_test,y_pred_lr))
            st.write("Input Parameters",x_test)
            st.write("Price",y_pred_lr)
            
    if st.sidebar.checkbox("Show Training data",False):
        st.subheader("Training Dataset")
        st.write(x_train)
    if st.sidebar.checkbox("Show raw data",False):
        st.subheader("Original Dataset")
        st.write(df)

    

# #---Done --
#     st.markdown("Developed by External Guide Avinash Pawar and WBL intern : Sana Khan at NIELIT Daman")

if __name__ == '__main__':
    main()
