## ChatBot

This project is a simple project of a chatbot that answers any question related to given documents.

## Instruction to run the project

- clone the project from github `git clone`

- install docker on the required machine 
`curl -fsSL https://get.docker.com -o get-docker.sh`    
`sudo sh get-docker.sh`

ensure that the docker file is running using
`docker --version`


- make sure you have access to huggingface and that your machine login the website
`huggingface-cli login --token <huggingface_token>`

- build the project using the following command 
`sudo docker build -t chatbot /path/to/project/`

- run the project using the following command 
`sudo docker run -p 7860:7860 chatbot`

the project will be running 
`http://localhost:7860`




