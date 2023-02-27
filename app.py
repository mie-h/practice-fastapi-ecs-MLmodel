from fastapi import FastAPI
from BankNotes import BankNote
import pickle


app = FastAPI()
pickle_in = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_in)


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.get("/detail")
def get_detail():
    return {"data": "Data were extracted from images that were taken from genuine and forged banknote-like specimens. For digitization, an industrial camera usually used for print inspection was used. The final images have 400x 400 pixels. Due to the object lens and distance to the investigated object gray-scale pictures with a resolution of about 660 dpi were gained. Wavelet Transform tool were used to extract features from images."}


@app.post("/predict")
def predict_banknote(data: BankNote):
    data_dict = data.dict()
    variance = data_dict['variance']
    skewness = data_dict['skewness']
    curtosis = data_dict['curtosis']
    entropy = data_dict['entropy']

    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    if(prediction[0]>0.5):
        prediction="Fake note"
    else:
        prediction="Its a Bank note"
    return {
        'prediction': prediction
    }
