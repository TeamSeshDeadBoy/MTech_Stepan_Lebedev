### Сборка и запуск Docker   
Собираем docker image:   
    <code>docker build -t streamlitapp:latest . </code>   
Запускаем docker container c собранным image:    
    <code>docker run -p 8501:8501 streamlitapp:latest</code>   