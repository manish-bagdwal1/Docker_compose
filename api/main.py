from fastapi import FastAPI
import pickle
import numpy as np
import psycopg2
import uvicorn
import os

app = FastAPI()

# Database Connection
DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    #conn = psycopg2.connect(DATABASE_URL, sslmode="require")  # Secure connection
    conn = cnx = psycopg2.connect(user="Golu", password="{Veron1c@}", host="man.postgres.database.azure.com", port=5432, database="postgres",sslmode="require")
    return conn


# Load Model
model_path = "model/model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "ML API is running"}

@app.post("/predict")
def predict(data: dict):
    input_data = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(input_data)

    # Save request & prediction to DB
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO prediction (input_data, prediction) VALUES (%s, %s)", 
                   (str(input_data.tolist()), str(prediction.tolist())))
    conn.commit()
    cursor.close()
    conn.close()

    return {"prediction": prediction.tolist()}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)
