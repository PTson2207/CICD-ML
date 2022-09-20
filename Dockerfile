
FROM centos/python-38-centos7
 


RUN pip3 install --upgrade pip

RUN pip3 install tensorflow==2.10.0

RUN pip3 install keras==2.10.0

CMD ["python3" , "/code/code_file2.py"]