## Centroid Tracker

Для запуска проекта на своём ПК необходимо:

* создать новое окружение;
* установить все необходимые зависимости;
* запустить tracker.

### Создаём окружение

Проект создавался с использованием conda в качестве пакетного менеджера и менеджера виртуальных окружений.
Список всех необходимых для работы проекта пакетов приведён в файле `requirements.txt`.
Для создания нового окружения с установкой всех зависимостей используйте команду:

```sh
conda create --name centroid_tracker --file requirements.txt
```

либо команду из Makefile:

```sh
make env
```

### Активируем окружение

После создания окружения и установки всех требуемых пакетов, необходимо активировать это окружение.
Используйте команду:

```sh
conda activate centroid_tracker
```

либо команду из Makefile:

```sh
make activate
```

### Запускаем трекер

Для запуска трекера на вашем видео используйте команду:

```sh
python app0.py --input <src> --output <dst>
```

где `src` - это путь к вашему видеофайлу, а `dst` - путь к результирующему файлу.

В папке `input` лежит видеофайл, который можно использовать в качестве примера. Для этого можно просто вызвать команду из Makefile:

```sh
make demo
```