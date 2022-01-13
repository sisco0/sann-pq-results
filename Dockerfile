FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install -r requirements.txt --use-feature=2020-resolver

COPY . .

CMD [ "python", "./realvalidation.py" ]