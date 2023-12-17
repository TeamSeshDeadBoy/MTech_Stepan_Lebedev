import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
import scipy.stats as stats
import streamlit as st



def calcMinMax(data, arr):
    """
    Функция, считающая минимальные и максимальные аргументы для выбранных колонок pd.DataFrame для слайдеров st.slider()
    Максимальным мы считаем Max-1, так как при проверке гипотезы мы ставим границу '>'
    :param data: данные pd.Dataframe
    :param arr: cписок названий колонок
    """
    minMaxes = []
    for i in arr:
        minMaxes.append(data[i].min())
        minMaxes.append(data[i].max() - 1)
    return minMaxes

def hist(data, nbins=20):
    """
    Функция строит гистограмму с заданным кол-вом bin
    :param data: данные pd.Dataframe
    :param nbins: кол-во бинов int
    """
    fig, ax = plt.subplots()
    ax.hist(data, bins=nbins)
    st.pyplot(fig)
    return fig, ax


def twoHistograms(x, y, arr, names=["Первая выборка", "Вторая выборка"]):
    """
    Функция, строящая две гистограммы плотности по двум выборкам.
    Пунктирные линии - средние значения выборок.
    :param x: выборка pd.Series
    :param y: выборка pd.Series
    """
    fig, ax = plt.subplots()
    ax.hist(x, alpha=0.5, weights=[1./len(x)]*len(x), bins = np.arange(arr[0]+1, arr[1]+ 2))
    ax.hist(y, alpha=0.5, weights=[1./len(y)]*len(y), bins = np.arange(arr[0]+1, arr[1]+ 2))
    ax.axvline(x.mean(), color='red', alpha=0.8, linestyle="dashed")
    ax.axvline(y.mean(), color='blue', alpha=0.8, linestyle="dashed")
    ax.legend(names)
    st.pyplot(fig)
    return fig, ax


def splitData(data, hypothesisNumber, work_days, age=0):
    """
    Функция, разделяющая данные на две подвыборки, в зависимости от критерия гипотезы и полученным переменным
    :param data: выборка pd.DataFrame
    :param hypothesisNumber: выборка pd.Series
    :param work_days: 
    :param age:
    """
    work_days_split = data[data["Количество больничных дней"] > work_days]
    if (hypothesisNumber == "1 Гипотеза"):
        first_samples = work_days_split[work_days_split["Пол"] == "М"]["Количество больничных дней"]
        second_samples = work_days_split[work_days_split["Пол"] == "Ж"]["Количество больничных дней"]
    else:
        first_samples = work_days_split[work_days_split["Возраст"] > age]["Количество больничных дней"]
        second_samples = work_days_split[work_days_split["Возраст"] <= age]["Количество больничных дней"]
    return first_samples, second_samples


def preprocessData(data):
    data.replace(r'(\")', r"", regex=True, inplace=True)
    data.rename(columns=lambda x: re.sub('(\")','',x), inplace=True)
    data["Количество больничных дней"] = df["Количество больничных дней"].astype("int64")
    data["Количество больничных дней"] = pd.to_numeric(df["Количество больничных дней"], errors='coerce')
    data["Возраст"] = pd.to_numeric(df["Возраст"], errors='coerce')
    return data
    
   
def loadData(file):
     df = pd.read_csv(file, encoding='cp1251', sep=',', quoting=3)
     return df
    

def _markdownPval(text, pval1, pval2=0, alpha=0.05):
    st.markdown(f"{text}   \np-value: {pval1}   \np-value: {pval2}")
    if (pval1 >= alpha and pval2 >= alpha):
        return False
    else:
        return True

    
def testShapiro(data1, data2):
    results_1 = stats.shapiro(data1)
    results_2 = stats.shapiro(data2)
    _markdownPval("Результаты теста шапиро для выборок:", results_1.pvalue, results_2.pvalue)
    
st.header('Тестовое задание на вакансию: Младший Исследователь данных (DS) / стажер')
st.sidebar.header("Загрузка данных")
file = st.sidebar.file_uploader("Убедитесь, что данные соответствуют предоставленному ранее формату")

df = pd.DataFrame()

if file:
    df = loadData(file)
    df = preprocessData(df)

if not df.empty:
    st.write("Загруженный датасет (С предопработкой из-за кодировки):", df)
    but = st.sidebar.checkbox("Показать информацию о ваших данных")
    if but:
        st.sidebar.write("Кол-во обьектов  : " , df.shape[0])
        st.sidebar.write("Кол-во признаков : " , df.shape[1])
        st.sidebar.write("\nПропущенные значения :  ", df.isnull().sum().values.sum())
        st.sidebar.write("\n      Уникальные значения :  \n", df.nunique())
        st.sidebar.write("\n       Типизация колонок :  \n", df.dtypes)
    st.header("Проверка гипотез")
    selected_hypothesis = st.selectbox('Выберите гипотезу для проверки', ["1 Гипотеза","2 Гипотеза"])
    
    min_age, max_age, min_days, max_days = calcMinMax(df, ["Возраст", "Количество больничных дней"])
    if selected_hypothesis == "1 Гипотеза":
        # age = st.slider('Возраст', min_value=min_age, max_value=max_age)
        work_days = st.slider('Кол-во пропущенных дней', min_value=min_days, max_value=max_days)
        st.write('Мужчины пропускают в течение года более ', work_days, 'рабочих дней по болезни значимо чаще женщин')
    else:
        age = st.slider('Возраст', min_value=min_age, max_value=max_age)
        work_days = st.slider('Кол-во пропущенных дней', min_value=min_days, max_value=max_days)
        st.write("Работники старше ",age," лет пропускают в течение года более ",work_days," рабочих дней по болезни значимо чаще своих более молодых коллег")
    
    start_button = st.button('Запустить проверку гипотезы')
    
    if start_button:
        if selected_hypothesis == "1 Гипотеза":
            names = ["Мужчины. Среднее", "Женщины. Среднее"]
            df_1, df_2 = splitData(df, selected_hypothesis, work_days)
        else:
            names = [f"{age} и младше. Среднее", f"Старше {age} лет. Среднее"]
            df_1, df_2 = splitData(df, selected_hypothesis, work_days, age)
        twoHistograms(df_1, df_2, [work_days, max_days], names)
        if (testShapiro(df_1, df_2)):
            st.write("Оба распределения близки к нормальному, для проверки гипотезы можем использовать t-test Сть")
        else:
            st.write("Оба (или одно) распределения далеки от нормального, нет возможности использовать t-test, используем тест Манна-уитни")
            
            