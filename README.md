use python 3.10 for this project as the audio extraction library can work with this version only

create a virtual environment : .\venv\Scripts\activate 

install the required libraries using :  pip install -r requirements.txt                   

run the app using :  uvicorn app.main:app --reload  

The issue you are facing regarding while re uploading the same image or audio the path error shows up is die to mssing ffmpeg installation on your device the solution for the same :
- open cmd and type :  winget install ffmpeg
- after the installation has been sucessfully done add the bin path to the System environment variables
- if u fail to find the path just type : where ffmpeg , in cmd and you will get the path
