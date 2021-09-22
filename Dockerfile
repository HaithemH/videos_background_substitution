FROM python:3.8
# LABEL about the custom image
LABEL maintainer="hermessi.haithem@gmail.com"
LABEL version="0.1"
LABEL description="This is custom Docker Image for backaground video substitution non-moving backgrounds" 

ENV ROOT=/editor
RUN mkdir -p $ROOT
ADD . $ROOT
WORKDIR $ROOT
RUN pip install -r requirements.txt
ENTRYPOINT ["/bin/echo", "Hi, this container is to test the backaground video substitution  !"]
