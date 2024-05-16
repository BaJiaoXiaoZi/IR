# 删减版
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
from joblib import load
import copy
from io import BytesIO
from xgboost.sklearn import XGBClassifier
from shap import Explainer
from shap.explainers._tree import TreeExplainer as Tree
import seaborn
from PIL import Image

def to_excel(df: pd.DataFrame):
    in_memory_fp = BytesIO()
    df.to_excel(in_memory_fp)
    # Write the file out to disk to demonstrate that it worked.
    in_memory_fp.seek(0, 0)
    return in_memory_fp.read()

# 导入数据集，划分特征和标签
# 或者不设置输入，设置一个demo表下载，让他们填完表再重新上传


df_demo = pd.read_csv('insulin_resistance_demo_s.csv')  # 后面一定要.to_csv,因为st.download_button只能传csv文件，不能传dataframe
file_name = "insulin_resistance_demo_s.csv"
st.sidebar.download_button(
        f"📥Click to download {file_name}",
        df_demo.to_csv(index=False),
        file_name,
        f"text/{file_name}",
        key=file_name
        )
select_file = st.sidebar.file_uploader('.csv')

if select_file:
    if st.sidebar.button(' :red[📥Click here to Analysis!]'):

        # 提取数据
        @st.cache_data
        def load_data(path):
            df_ = pd.read_csv(path)
            return df_
        df = load_data(select_file)
        df_ = pd.DataFrame()
        st.write('#### :black[Review the Data Entered: ]')
        st.write('##### :black[👇Please verify the accuracy of the information you provided below.]')
        st.write(df)
        st.write('___')
        # 读取模型（目前无用）
        model = load('x_s.ldz')
        df = df.iloc[[1]].astype('float64')
        a = (model.predict_proba(df)[0][1])*100
        b = round(a, 2)
        # 读取SHAP结果
        explainer = load('e_s.ldz')
        shap_value = explainer(df)

        # 画出个性化线
        a = shap.plots.waterfall(shap_value[0], show=True)
        row0_spacer1, rows_0, row0_1, rows_1 = st.columns((0.5, 0.10, 0.25, 0.15))

        with row0_spacer1:
            rowcc_spacer1, rowcc_0, rowcc_1 = st.columns((0.3, 0.5, 0.1))
            rowcc_spacer1.write('#### :red[Insulin Resistance Risk:]')
            with rowcc_0:
                st.write('####', b, '%')
            st.write('#### :green[ ]')
            st.pyplot(a)
            st.write('###### :green[ ]')
            st.write('#### :black[🎇Understanding the Color Coding: ]')
            st.write('##### :black[&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The red block indicates that this variable contributes to an increase in predicted insulin resistance, while the blue block indicates a reduction. Please prioritize managing the variables marked in red.]')


        with row0_1:
            st.write('#### :green[ ]')
            image1 = Image.open('analysis.png')
            st.image(image1, caption=None, width=600, use_column_width=True, clamp=False, channels='RGB')
            st.write('###### :green[ ]')
            st.write('#### :black[🎇Please Note:]')
            st.write('##### :black[&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For abbreviated codes displayed on the chart, refer to the corresponding variables in the table above.]')
    else:
        st.sidebar.write('### 👆👆👆 :red[Notice here!]👆👆👆')
        st.write('#### :green[ ]')
        st.write('#### :green[ ]')
        st.write('#### :green[ ]')
        st.write('#### :green[ ]')
        st.write('#### :green[ ]')
        st.write('#### :green[ ]')
        st.write('#### :green[ ]')
        st.write('#### :green[ ]')
        st.write('#### :green[ ]')
        st.write('#### Waiting for the button to click！')


else:
    # st.title('请上传文件')
    # st.write(':red[请注意：输入的变量请与论文一致，且有ID列(注意区分大小写)]')
    st.markdown(
        "<h1 style='text-align: center; color: black;'>Simplified Version of ICAIR</h1>",
        unsafe_allow_html=True)
    cols_1, cols_2, col2_0 = st.columns((0.8, 0.5, 0.1))
    cols_1.write('#### :green[ ]')
    cols_1.write('#### :green[ ]')
    cols_1.write('#### :green[ ]')
    cols_1.write('#### :green[ ]')
    cols_1.write('#### :green[ ]')
    cols_1.markdown("<h4 style='text-align: left; black: gray;'>😀Welcome! The simplified version of the ICAIR app should now be available to you! </h2>", unsafe_allow_html=True)
    cols_1.write('#### :green[ ]')
    cols_1.write('#### :green[ ]')

    cols_1.write('#### :green[ ]')
    cols_1.write('#### :green[ ]')
    cols_1.write('#### :green[ ]')
    cols_1.write('#### :green[ ]')
    cols_1.markdown("<h4 style='text-align: left; color: black;'>Analysis Procedure for Simplified Version of the App: </h2>", unsafe_allow_html=True)
    cols_1.write('#### ')
    cols_1.markdown("<h5 style='text-align: left; color: black;'>①&nbsp;&nbsp;&nbsp;	Download the Data Table: </h2>",
                unsafe_allow_html=True)
    cols_1.write('###### :gray[Click the button on the left to download the table named "insulin_resistance_demo_s.csv" ]')
    cols_1.markdown("<h5 style='text-align: left; color: black;'>②&nbsp;&nbsp;&nbsp;	Update the Table: </h2>",
                unsafe_allow_html=True)
    cols_1.write('###### :gray[Replace the contents of the table with your personal information.]')
    cols_1.markdown("<h5 style='text-align: left; color: black;'>③&nbsp;&nbsp;&nbsp;	Upload for Analysis: </h2>",
                unsafe_allow_html=True)
    cols_1.write('###### :gray[Once updated, please upload the file to proceed with the analysis.]')
    cols_1.write('#### :green[ ]')
    # st.markdown("<h5 style='text-align: left; color: red;'>Please ensure that the input variables are consistent with those specified in the Table , as case sensitivity matters.</h3>", unsafe_allow_html=True)
    # col5_0, col5_1 = st.columns((0.9,0.1))
    image1 = Image.open('input_variables.jpg')
    cols_2.write('#### :green[ ]')
    cols_2.image(image1, caption=None, width=600, use_column_width=True, clamp=False, channels='RGB')
    st.write('#### :green[ ]')
    st.write('#### :green[ ]')
    st.write('#### :green[ ]')
    st.write('#### :green[ ]')
    st.write('#### :green[ ]')
    st.write('#### :green[ ]')
    st.write('#### :green[ ]')
    st.write('#### :green[ ]')
    st.write('#### :green[ ]')
    st.write('#### :green[ ]')
    st.write('#### :green[ ]')
    st.write('#### :green[ ]')
    st.write('#### :green[ ]')
    st.write('#### :green[ ]')
    st.write('#### :green[ ]')
    st.write('#### :green[ ]')
    st.write('#### :green[ ]')
    st.write('#### :green[ ]')
    st.write('#### :green[ ]')
    st.markdown('---')