import torch
import torch.nn as nn
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler



class Cl_Model(nn.Module):
    def __init__(self):
        super(Cl_Model, self).__init__()
        self.fc1 = nn.Linear(93, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, 8)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x=self.fc1(x)
        x=self.activation(x)
        x=self.bn1(x)
        x=self.fc2(x)
        x=self.bn2(x)
        x=self.activation(x)
        x=self.fc3(x)
        return x
    


model = Cl_Model()
model = torch.load('best-model.pt', weights_only=False)
rs = RobustScaler()
rs = joblib.load('robust_scaler.pkl')
learn_data = pd.read_csv("learn_data.csv")

print(learn_data.head(1))

st.write("Введите характеристики вина")
st.number_input("Год урожая", key = "year", step=1)
st.number_input("Полнотелость (целое число от 2 до 5)", key = "body", step=1)
st.number_input("Кислотность (целое число от 1 до 3)", key = "acidity", step=1)
st.number_input("Число отзывов", key = "num_reviews", step=1)
st.text_input("Регион (Латиницей с большой буквы, пример: Aragon, Rioja)", key = "region")
st.text_input("Тип вина (Латиницей с большой буквы, пример: Sauvignon Blanc, Syrah)", key = "type")


year = st.session_state.year
body = st.session_state.body
acidity = st.session_state.acidity
num_reviews = st.session_state.num_reviews
region = st.session_state.region
type = st.session_state.type


learn_data['year'][0]  = year
learn_data['body'][0]  = body
learn_data['acidity'][0] = acidity
learn_data['num_reviews'][0]  = num_reviews
type = "type_" + type
region = "region_" + region

if type in learn_data.columns:
    learn_data[type][0] = 1
if region in learn_data.columns:
    learn_data[region][0] = 1


input = np.array(learn_data.iloc[0].tolist())
input = rs.transform(input.reshape(1, -1))
output = model(torch.from_numpy(input).float())
output = output.detach().numpy()
output_class = np.argmax(output[0])

rating_map = {
    0: 4.2,
    1: 4.3,
    2: 4.4,
    3: 4.5,
    4: 4.6,
    5: 4.7,
    6: 4.8,
    7: 4.9
}
st.write(output)

st.write(f"Предсказанная оценка вашего вина: {rating_map[int(output_class)]}")
