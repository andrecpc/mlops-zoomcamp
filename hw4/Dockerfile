FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install poetry

COPY [ "pyproject.toml", "poetry.lock", "./" ]

RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-ansi

COPY [ "predict.py", "./" ]

ENTRYPOINT [ "python", "predict.py", "2021", "4" ]