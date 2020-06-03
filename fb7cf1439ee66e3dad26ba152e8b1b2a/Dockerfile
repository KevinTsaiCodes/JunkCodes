FROM ubuntu
RUN apt-get -y update
RUN apt-get -y install python-dev
RUN apt-get -y install vim git
RUN git clone https://gist.github.com/fb7cf1439ee66e3dad26ba152e8b1b2a.git
ADD server.py .
ADD client.py .
