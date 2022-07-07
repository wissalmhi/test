import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

st.title('Customer segmentation using RFM and Kmeans')
st.sidebar.markdown('## Data Import')
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="xlsx")


if uploaded_file is not None:
    uploaded_file.seek(0)
    df = pd.read_excel(uploaded_file)
    df = df.astype(str)
    st.subheader('User Input file')
    st.write(df)
    #Preprocessing data to make it for model
    df=df.dropna()
    df['Quantity']=df['Quantity'].astype(float)
    df['UnitPrice']=df['UnitPrice'].astype(float)
    df=df[(df['Quantity']>0)]
    df['InvoiceDate']=pd.to_datetime(df['InvoiceDate'])
    now=df['InvoiceDate'].max()
    df['sales']=df['Quantity']*df['UnitPrice']
    RFM=df.groupby(['CustomerID']).agg({'InvoiceDate':lambda x :(now-x.max()).days,'InvoiceNo': lambda x : len(x.unique()),'sales': lambda x :x.sum()})
    RFM['InvoiceDate']=RFM['InvoiceDate'].astype(int)
    RFM.rename(columns={'InvoiceDate':'recency', 'InvoiceNo':'frequency','sales':'monetary_value'},inplace=True)
    RFM.reset_index(level='CustomerID',inplace=True)
    st.subheader('Transformed Dataset into RFM')
    st.write(RFM)
    #normalization
    scaler=StandardScaler()
    labels=list(RFM.columns[1:4])
    sd_rfm=RFM[labels]
    sd_rfm=scaler.fit_transform(sd_rfm)
    #model
    model_kmeans=pickle.load(open("Kmeans.pkl", "rb"))
    predict_cluster=model_kmeans.predict(sd_rfm)
    RFMK=RFM.copy()
    RFMK['cluster']=predict_cluster
    RFMK['cluster'].replace({0:'Average Customer',1:'VIP customer',2:'Lost Customer',3:'Best Customer'},inplace=True)
    Data_w_clust=pd.merge(df,RFMK[['CustomerID','cluster']],on='CustomerID', how='left')
    
    st.subheader('Initial dataset wiith clusters for each customer')
    st.write(Data_w_clust)
    st.markdown("""
                | Cluster | Customer type | RFM Characterictics | Action |
                | --- | --- | --- | --- |
                | 0 | Average Customer | Average spendings and recency, not very frequent though | Emphasizing customer relationship managemet to enhance shopp ing experience and hence strengthen the engagement. |
                | 1 | VIP customer | Frequent and recent shoppers. Heavy spendings. | Potential to be target customers for launch of new luxury products. |
                | 2 | Lost customer | Low frequency and spend ing amount adhas not placing an order recently. | Business might have lost them. Survey to be done on reason of being churned. Enhance the quality of products or services to avoid further losing. |
                | 3 | Best Customer | Moderate receny(past 2 weeks) and frequency shoppers, make heavy spendings | Potential to be target customers for launch of new  products. |
                """)
    filter = st.selectbox('Select customer', Data_w_clust['CustomerID'].unique())
    st.write(Data_w_clust[Data_w_clust['CustomerID']==filter])
else:
    st.sidebar.warning("You need to upload a csv or excel file")


