FROM python:3.11-slim

ENV POETRY_VIRTUALENVS_CREATE=false

WORKDIR /onboarding

RUN pip install poetry

COPY pyproject.toml poetry.lock ./

RUN poetry install --without dev --no-root

COPY onboarding ./onboarding

COPY onboarding/challenge/README.md ./README.md

RUN poetry install --only-root

CMD ["poetry", "run", "uvicorn", "onboarding.challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]
#ENTRYPOINT ["poetry", "run", "uvicorn"]
#CMD ["onboarding.challenge.api:app", "--host", "0.0.0.0", "--port", "8000"]
