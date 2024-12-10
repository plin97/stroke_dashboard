FROM python:3.12

WORKDIR /code

COPY ./range_visuals/requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

EXPOSE 7860

CMD ["shiny", "run", "range_visuals/app.py", "--host", "0.0.0.0", "--port", "7860"]