# 分为三个tab，第一个tab对app进行总的介绍，第二个tab介绍总的版本，和介绍输入值，第三个介绍精华版
import copy
import os
import streamlit as st
import plotly
import pandas as pd
import scipy as sp
import numpy as np
import scipy
from PIL import Image
import plotly.express as px
from streamlit_lottie import st_lottie
import requests
from io import BytesIO
from streamlit_extras.add_vertical_space import add_vertical_space
from xgboost.sklearn import XGBClassifier
import shap
from joblib import load


st.set_page_config(layout='wide')

def mahalanobis(x=None, data=None, cov=None):
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal


def md_calc(f):
    df_h1 = pd.read_csv('DRHD_health.csv')
    df_t = f

    df_h1 = df_h1.drop('id', axis=1)
    df_t = df_t.drop('ID', axis=1)
    df_h = (df_h1 - df_h1.mean())/df_h1.std()
    df_t = (df_t - df_h1.mean())/df_h1.std()

    a = mahalanobis(x=df_t, data=df_h)
    a_t = a**0.5
    MD = a_t.diagonal()
    MD_ori = MD/MD.std()
    MD_log = np.log(MD)/np.log(MD).std()
    MD_ori = np.around(MD_ori,2)
    MD_log = np.around(MD_log,2)
    return MD_ori, MD_log


# 定义一个返回该目录下所有文件的函数
def get_file_list(suffix, path):
    input_template_all = []
    input_template_all_path = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if os.path.splitext(name)[1] == suffix:
                input_template_all.append(name)
                input_template_all_path.append(os.path.join(root, name))
    return input_template_all, input_template_all_path


# 定义一个抓取图片的链接
def load_logourl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def to_excel(df: pd.DataFrame):
    in_memory_fp = BytesIO()
    df.to_excel(in_memory_fp)
    # Write the file out to disk to demonstrate that it worked.
    in_memory_fp.seek(0, 0)
    return in_memory_fp.read()




# logo = load_logourl("http://www-x-gxykzx-x-com.img.abc188.com/images/logo.gif")
# st_lottie(logo, speed=1, height=200, key="initial")

# row_t0, row_t1, row_t2, row_t3, row_t4 = st.columns((0.2, 1, 0.2, 1, 0.1))
# image1 = Image.open('C:/pycharm/PyCharm Community Edition 2022.2.3/DL/MD_streamlit/logo.png')
# row_t0.image(image1)
# image2 = Image.open('C:/pycharm/PyCharm Community Edition 2022.2.3/DL/MD_streamlit/ref.png')

st.markdown("<h1 style='text-align: center; color: black;'>Individualized Care Analysis of Insulin Resistance App</h1>", unsafe_allow_html=True)

row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
    (0.1, 1, 0.2, 1, 0.1)
)


with row0_2:
    add_vertical_space()

# st.markdown("# DRHD value app")  # 等价于st.title('DRHD value app')


# input_folder = st.sidebar.text_input('输入数据所在的目录:（输入后回车）', value=os.path.abspath('.'), key=None)
# path = input_folder
# _, file_list_csv = get_file_list('.csv', path)
# file_list = file_list_csv
# if file_list:
#     select_file = st.sidebar.selectbox('请选择你的数据(仅支持csv格式)', file_list)
#     st.write('当前目录:', select_file)
st.write("---")

tab0, tab1, tab2 = st.tabs(["🧾:green[Document of ICAIR APP]", "📠:green[Introduction of Complete Version]", "🖨:green[Introduction of Simplified Version]"])

# 提取数据


@st.cache_data
def load_data(path):
    df_ = pd.read_csv(path)
    return df_

# 对原始数据做一个备份


with tab0:
    col1_1,col1_0, col1_2 = st.columns((0.8, 0.1, 0.5))
    col1_1.markdown(
        "<h4 style='text-align: left; black: gray;'>📕The Individualized Care Analysis of Insulin Resistance (ICAIR)  app is an open-source web application designed for predicting insulin resistance in "
        "children based on obesity-related variables using the XGBoost algorithm. The complete version of ICAIR app requires eighteen inputs for  optimal performance. However, the application can also operate using a simplified algorithm if four key obesity-related variables (Waist-to-Height ratio, body mass index, triglycerides, and total bilirubin) are provided. The app returns individualized care plans that provide an explanation of the model’s decision-making process. </h2>",
        unsafe_allow_html=True)
    col1_1.markdown(
        "<h4 style='text-align: left; black: gray;'>📓The complete set of eighteen inputs includes: gender, age, alkaline phosphatase, aspartate aminotransferase, alanine aminotransferase, gamma glutamyltransferase, total bilirubin, total protein, systolic blood pressure, body mass index, Waist-to-Height ratio, glycated hemoglobin, and high-density lipoprotein.  </h2>",
        unsafe_allow_html=True)
    col1_1.markdown('---')
    col1_1.write('#### The analysis procedure of this app:')
    col1_1.write('##### :black[① Review Versions: ]')
    col1_1.write('###### :black[Begin by carefully reviewing both the complete and simplified versions of the ICAIR analysis.]')
    col1_1.write('##### :black[② Select the Appropriate Version:]')
    col1_1.write('###### :black[Choose the version that matches the physiological parameters you have available:]')
    col1_1.write('###### :black[•  Simplified Version: Opt for this if you have only the Waist-to-Height Ratio, Body Mass Index, Triglycerides, and Total Bilirubin.]')
    col1_1.write('###### :black[•  Complete Version: Choose this if you have the full set of 18 physiological parameters.]')
    col1_1.write('##### :black[③ Input Data and Analyze:]')
    col1_1.write('###### :black[Once you have selected the appropriate version, input your data to receive the insulin resistance analysis results.]')


    col1_1.markdown(
        "<h5 style='text-align: left; black: gray;'>🎆Please be assured that this app does not store any of your inputs, allowing you to use it with confidence.</h5>",
        unsafe_allow_html=True)
    image1 = Image.open('statement.jpg')
    col1_2.write('#### :green[ ]')
    # col1_2.write('#### :green[ ]')
    # col1_2.write('#### :green[ ]')
    # col1_2.write('#### :green[ ]')
    # col1_2.write('#### :green[ ]')
    # col1_2.write('#### :green[ ]')
    col1_2.image(image1, caption=None, width=600, use_column_width=True, clamp=False, channels='RGB')



# 渲染数据
with tab1:
    st.write('#### :green[The complete version was constructed by more comprehensive data.]')
    st.write('#### :green[ ]')

    st.write(
        '##### :black[👇Here is the matters of the variables need attention:]')
    st.write('***')
    col4_1, col4_2 = st.columns(2)
    # st.write('### 👇:green[This page shows the summary of your data]')
    # st.markdown('---')
    with col4_1:
        st.write('###### :black[Age Group: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1. Less than 14 years old; 2.14 to 16 years old; 3. Over 16 years old]')
        st.write('###### :black[Gender: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.male 2.female]')

        st.write('###### :black[Waist-to-height Ratio: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Waist circumference/height]')
        st.write('###### :black[BMI: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;body mass/height^2]')
        st.write('###### :black[Triglyceride: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the unit is mg/dL]')
        st.write('###### :black[Total Bilirubin: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the unit is mg/dL]')

        st.write('###### :black[AST: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the unit is U/L]')

        st.write('###### :black[Alkaline Phosphatase: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the unit is U/L]')
        st.write('###### :black[SBP: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the unit is mmHg]')
        st.write('###### :black[DBP: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the unit is mmHg]')

        st.write('###### :black[GGT: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the unit is U/L]')

    with col4_2:
        st.write('###### :black[ALT: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the unit is U/L]')
        st.write('###### :black[LDL-Cholesterol: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the unit is mg/dL]')
        st.write('###### :black[Glycohemoglobin: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the unit is %]')
        st.write('###### :black[Total Protein: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the unit is g/L]')
        st.write('###### :black[HDL-Cholesterol: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the unit is mg/dL]')
        st.write('###### :black[Albumin: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the unit is g/L]')

        st.write('###### :black[Non-Hispanic White: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.No 1.Yes]')
        st.write('###### :black[Non-Hispanic Black: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.No 1.Yes]')
        st.write('###### :black[Mexican American: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.No 1.Yes]')
        st.write('###### :black[Other Races: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.No 1.Yes]')


with tab2:
    # st.write('### 👇:green[This page shows the calculated DRHD value of your data]')
    # st.markdown('---')
    st.write(
        '#### :green[The simplified version was constructed by key variable, which were the four variables that contributed the most to the model!]')
    st.write('#### :green[ ]')

    st.write(
        '##### :black[👇Here is the details of the key variables:]')
    st.write('***')
    col5_1, col5_2 = st.columns((1, 1))

    with col5_1:
        st.write('###### :black[Waist-to-height Ratio: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;vWaist circumference/height]')
        st.write('###### :black[BMI: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;body mass/height^2]')


    with col5_2:
        st.write('###### :black[Triglyceride: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the unit is mg/dL]')
        st.write('###### :black[Total Bilirubin: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the unit is mg/dL]')
        # st.write('***')

    # col3_1, col3_2 = st.columns(2)


# st.title('请上传文件')
# st.write(':red[请注意：输入的变量请与论文一致，且有ID列(注意区分大小写)]')
