## Тестовое задание на позицию Младший Исследователь Данных (стажер) M.Tech
Выполнил Лебедев Степан, 17.12.2023   
***
### Jupyter notebook:   
Jupyter Notebook с логикой выполнения задания находится в корневой папке репозитория, под именем <code>"Лебедев_Степан_Младший_Исследователь_Данных_(стажер).ipynb"</code>
***
### Dashboard Streamlit:
Дэшборд поднят на сервере streamlit, доступен по ссылке
#### <code>https://mtech-lebedev-ds.streamlit.app</code>
***
### Сборка и запуск Docker   
Собираем docker image:   
    <code>docker build -t streamlitapp:latest . </code>   
Запускаем docker container c собранным image:    
    <code>docker run -p 8501:8501 streamlitapp:latest</code>   
Чтобы открыть дэшборд локально, переходите по ссылке:
    <code>http://localhost:8501</code>