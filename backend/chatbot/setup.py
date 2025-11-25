from setuptools import find_packages, setup 

setup(
    name="hope_chatbot",
    version="0.0.1",
    author="Morris Mulandi",
    author_email="mulandimorris1@gmail.com",
    packages=find_packages(),
    install_requires=[
        "flask==3.1.1",
        "langchain==0.3.26",
        "sentence-transformers==4.1.0",
        "pypdf==5.6.1",
        "python-dotenv==1.1.0",
        "langchain-pinecone==0.2.8",
        "langchain-openai==0.3.24",
        "langchain-community==0.3.26",
    ],
)