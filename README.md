# Document Question Answering Application

A web application that allows user to upload txt documents and able to answer users' questions related to the uploaded documents based on fine-tuned ALBERT models.

The web application is built on Flask framework and ALBERT models are built on pytorch framework using transformer libary.

All required packages are listed on requirements.txt. Please follow below commands to create the virtual environment.

```bash
python -m venv .vevn

.venv\Scripts\activate

pip install requirements.txt

```

Lauch the web application

```bash
python application.py
```

For GPU environment, please follow the instruction (https://pytorch.org/get-started/locally/) to install pytorch-gpu compatible with your cuda.

<img src="./WebUI.png" alt="WebUI.png" style="width: 800px;"/>
