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
    :param work_days: гиперпараметр
    :param age: гиперпараметр
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
    """
    Функция, предобрабатывающая DataFrame с кодировкой cp1521
    Удаляет лишние кавычки
    :param data: обьект pd.DataFrame
    """
    data.replace(r'(\")', r"", regex=True, inplace=True)
    data.rename(columns=lambda x: re.sub('(\")','',x), inplace=True)
    data["Количество больничных дней"] = data["Количество больничных дней"].astype("int64")
    data["Количество больничных дней"] = pd.to_numeric(data["Количество больничных дней"], errors='coerce')
    data["Возраст"] = pd.to_numeric(data["Возраст"], errors='coerce')
    return data
    
   
def loadData(file):
    """
    Функция загружает из csv файла обьект pd.DataFrame
    :param file: путь к файлу
    """
    df = pd.read_csv(file, encoding='cp1251', sep=',', quoting=3)
    return df
    

def _markdownPval(text, pval1, pval2=0, alpha=0.05):
    """
    Функция оберки двух p-значений в отформатированный markdown
    """
    st.markdown(f"### {text}   \np-value: {pval1}   \np-value: {pval2}   \n$alpha$: {alpha}")
    if (pval1 >= alpha and pval2 >= alpha):
        st.markdown("**P-значения больше или равны доверительному интервалу**, не отрицаем нулевую гипотезу")
        return False
    else:
        st.markdown("**Одно или оба P-значений меньше доверительного интервала**, принимаем альтернативную гипотезу")
        return True

def _markdownSinglePval(text, pval1, alpha=0.05):
    """
    Функция оберки одного p-значения в отформатированный markdown
    """
    st.markdown(f"### {text}   \np-value: {pval1}   $alpha$: {alpha}")
    if (pval1 >= alpha):
        st.markdown("**P-значение больше или равно доверительному интервалу**, не отрицаем нулевую гипотезу")
        return False
    else:
        st.markdown("**P-значение меньше доверительного интервала**, принимаем альтернативную гипотезу")
        return True
    
def testShapiro(data1, data2, alpha=0.05):
    """
    Функция проводит для двух выборок тесты Шапиро и возвращает их p-значения
    :param data1: первая выборка pd.DataFrame
    :param data2: вторая выборка pd.DataFrame
    """
    st.markdown("### Гипотезы для теста Шапиро-Уилка:   \
        \n+ $H_0$: Выборка распределена по нормальному закону.   \
        \n+ $H_a$: Закон распределения выборки не является нормальным.")
    results_1 = stats.shapiro(data1)
    results_2 = stats.shapiro(data2)
    return _markdownPval("Результаты теста шапиро для выборок:", results_1.pvalue, results_2.pvalue, alpha)
    
    
    
def testMannWhitney(data1, data2, alpha=0.05):
    """
    Функция проводит для двух выборок тесты Шапиро и возвращает их p-значения
    :param data1: первая выборка pd.DataFrame
    :param data2: вторая выборка pd.DataFrame
    """
    results = stats.mannwhitneyu(data1, data2)
    return _markdownSinglePval("Результаты теста Манна-Уитни для выборок:", results.pvalue, alpha)
    
    
def testStudent(data1, data2, alpha=0.05):
    """
    :param data1: первая выборка pd.DataFrame
    :param data2: вторая выборка pd.DataFrame
    """
    results = stats.ttest_ind(data1, data2, equal_var=False)
    _markdownSinglePval("Результаты теста Стьюдента для выборок:", results.pvalue, alpha)    

    
def renderHypothesis(hypothesis, work_days, age=0, alpha=0.05):
    """
    Функция выводит формальный markdown для гипотез
    :param hypothesis: проверяемая гипотеза
    :param work_days: гиперпараметр кол-ва дней
    :param age=0: гиперпараметр возраст
    :param alpha=0.05: гиперпараметр уровень значимости
    """
    if hypothesis == "2 Гипотеза":
        st.markdown(f'### Формулировка гипотез:   \
            \n1. Null Hypothesis $H_0$ - Старшие сотрудники (> {age} лет) пропускают в течение года более {work_days} рабочих дней по болезни **реже, cопоставимо, или незначительно чаще** их более молодых коллег (<= {age} лет).   \
            \n2. Alternate Hypothesis $H_a$ - Старшие сотрудники (> {age} лет) пропускают в течение года более 2 рабочих дней по болезни значительно **чаще**, чем их более молодые коллеги (<= {age} лет).   \
            \n### Формальная формулировка гипотез:   \
            \n1. $H_0: \mu_1 <= \mu_2$   \
            \n1. $H_a: \mu_1 > \mu_2$,    \
            \nГде: $\mu_1$ - математическое ожидание выборки из старших сотрудников, $\mu_2$ - математическое ожидание выборки из молодых сотрудников   \
            \n### Определение уровня значимости:   \
            \nУровень значимости:    \
            \n$alpha$ = {alpha}')
    if hypothesis == "1 Гипотеза":
        st.markdown(f'### Формулировка гипотез:   \
            \n1. Null Hypothesis $H_0$ - Мужчины пропускают в течение года более {work_days} рабочих дней по болезни **реже, cопоставимо, или незначительно чаще** женщин.   \
            \n2. Alternate Hypothesis $H_a$ - Мужчины пропускают в течение года более {work_days} рабочих дней по болезни значительно **чаще** женщин.   \
            \n### Формальная формулировка гипотез:   \
            \n1. $H_0: \mu_1 <= \mu_2$   \
            \n1. $H_a: \mu_1 > \mu_2$,    \
            \nГде: $\mu_1$ - математическое ожидание выборки из старших сотрудников, $\mu_2$ - математическое ожидание выборки из молодых сотрудников   \
            \n### Определение уровня значимости:   \
            \nУровень значимости:    \
            \n$alpha$ = {alpha}')
        
        
def renderConclusion(hypothesis, test, testResult, work_days, age=0):
    """
    Функция выводит формальный markdown для гипотез
    :param hypothesis: проверяемая гипотеза
    :param test: статистический тест: MW | S
    :param testResult: реузльтат теста: True | False
    :param work_days: гиперпараметр кол-ва дней
    :param age=0: гиперпараметр возраст
    """
    if test == "MW":
        test_string = "Манна-Уитни"
    else:
        test_string = "Стьюдента"
    if hypothesis == "1 Гипотеза":
        if testResult:
            st.markdown(f'### Вывод:   \
                \n Тест {test_string} дал нам основания отвергнуть нулевую гипотезу, и мы имеем право сказать, что:   \
                \n### **Мужчины пропускают в течение года более {work_days} рабочих дней по болезни **значительно чаще** женщин.**')
        else:
            st.markdown(f'### Вывод:   \
                \n Тест {test_string} не дал нам оснований отвергнуть нулевую гипотезу, вывод:   \
                \n### **Мужчины пропускают в течение года более {work_days} рабочих дней по болезни **реже, cопоставимо, или незначительно чаще** женщин.**')
    if hypothesis == "2 Гипотеза":
        if testResult:
            st.markdown(f'### Вывод:   \
                \n Тест {test_string} дал нам основания отвергнуть нулевую гипотезу, и мы имеем право сказать, что:   \
                \n### **Старшие сотрудники (> {age} лет) пропускают в течение года более 2 рабочих дней по болезни **значительно чаще**, чем их более молодые коллеги (<= {age} лет).**')
        else:
            st.markdown(f'### Вывод:   \
                \n Тест {test_string} не дал нам оснований отвергнуть нулевую гипотезу, вывод:   \
                \n### **Старшие сотрудники (> {age} лет) пропускают в течение года более {work_days} рабочих дней по болезни **реже, cопоставимо, или незначительно чаще** их более молодых коллег (<= {age} лет).**')
    
def main():    
    st.header('Тестовое задание на вакансию: Младший Исследователь данных (DS) / стажер')
    st.sidebar.header("Загрузка данных")
    button_standart = st.checkbox("Использовать стандартные данные")
    if (not button_standart):
        file = st.sidebar.file_uploader("Убедитесь, что данные соответствуют предоставленному формату")
    if (button_standart):
        file = './data/М.Тех_Данные_к_ТЗ_DS.csv'

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
            age = min_age
            work_days = st.slider('Кол-во пропущенных дней', min_value=min_days, max_value=max_days)
            alpha = st.slider('Уровень значимости:', min_value=0.01, max_value=0.1)
            renderHypothesis("1 Гипотеза", work_days=work_days, alpha=alpha)
        else:
            age = st.slider('Возраст', min_value=min_age, max_value=max_age)
            work_days = st.slider('Кол-во пропущенных дней', min_value=min_days, max_value=max_days)
            alpha = st.slider('Уровень значимости:', min_value=0.01, max_value=0.2)
            renderHypothesis("2 Гипотеза", work_days=work_days, age=age, alpha=alpha)
        
        start_button = st.button('Запустить проверку гипотезы')
        
        if start_button:
            if selected_hypothesis == "1 Гипотеза":
                names = ["Мужчины. Среднее", "Женщины. Среднее"]
                df_1, df_2 = splitData(df, selected_hypothesis, work_days)
            else:
                names = [f"{age} и младше. Среднее", f"Старше {age} лет. Среднее"]
                df_1, df_2 = splitData(df, selected_hypothesis, work_days, age)
            twoHistograms(df_1, df_2, [work_days, max_days], names)
            if (testShapiro(df_1, df_2, alpha) == False):
                st.write("Оба распределения близки к нормальному, для проверки гипотезы можем использовать t-test Стьюдента со сформулированными выше гипотезами")
                test_result = testStudent(df_1, df_2, alpha)
                renderConclusion(selected_hypothesis, "S", test_result, work_days, age)
            else:
                st.write("Оба (или одно) распределения далеки от нормального, нет возможности использовать t-test, используем тест Манна-уитни со сформулированными выше гипотезами")
                test_result = testMannWhitney(df_1, df_2, alpha)
                renderConclusion(selected_hypothesis, "MW", test_result, work_days, age)
            
            
main()