FROM python:3.7

COPY . .

RUN pip install opencv-python \
    && pip3 install redis \
    && pip3 install numpy

RUN apt update
RUN apt install libgl1-mesa-glx -y

CMD ["main.py"]
ENTRYPOINT ["python3"]
