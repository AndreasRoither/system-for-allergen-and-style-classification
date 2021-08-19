FROM python:3.9.6-slim as base
LABEL maintainer="Andreas Roither <andi.roither@protonmail.com"

# First stage
FROM base as builder

RUN mkdir /install
WORKDIR /install

# setup python virtual environment
ENV VIRTUAL_ENV=/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# install build utilities
RUN apt-get update && \
	apt-get install -y gcc make apt-transport-https ca-certificates build-essential

# Installing python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Second stage
FROM python:3.9.6-slim as app

# set virtual environment as default so it works on CMD and RUN
ENV VIRTUAL_ENV=/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

# Copy all necessary files
COPY --from=builder /venv /venv
COPY /src ./src
COPY /data ./data
COPY /models ./models

EXPOSE 8000

# Running Python Application
WORKDIR ./src

# Other way to run the unit test
# RUN python unit_test.py
# RUN pytest

ENTRYPOINT ["uvicorn"]
CMD ["main:app", "--reload", "--host", "0.0.0.0"]