import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier

airline = pd.read_csv("Invistico_Airline.csv")

target = airline.satisfaction.values

airline.drop(['satisfaction', 'Customer Type', 'Type of Travel', 'Flight Distance'], axis=1, inplace=True)

airline.Class.replace(['Eco', 'Eco Plus', 'Business'], [1, 2, 3], inplace=True)

null_index = airline.isnull().sum()[airline.isnull().sum() > 0].index

for i in null_index:
    if airline[i].dtype == 'float64' or airline[i].dtype == 'int64':
        airline[i] = airline[i].fillna(airline[i].mean())
    elif airline[i].dtype == 'object':
        airline[i] = airline[i].fillna(airline[i].mode()[0])
    else:
        print("We missed this col=>", i)

airline_new = airline.rename({'Arrival Delay in Minutes': 'Arrival_Delay_in_Minutes'}, axis=1, inplace=True)

airline_new = pd.get_dummies(airline, drop_first=True)

rf = RandomForestClassifier()
airline_new_train, airline_new_test, target_train, target_test = train_test_split(airline_new, target, test_size=.2,
                                                                                  random_state=42)
airline_new_test.fillna(airline_new_train.mean(), inplace=True)
airline_new_test.fillna(airline_new_train.mode(), inplace=True)
np.any(np.isnan(airline_new))
np.all(np.isfinite(airline_new))
rf.fit(airline_new_train, target_train)
result = rf.predict(airline_new_test)
print(airline_new_test)
acc = accuracy_score(target_test, result)

# rf.predict(age=100, delay = 0)
# Output = Satisfied

# Streamlit
st.write("""

# Airline Customer Satisfaction
##### By: Candy Awuor, Ran Wei, Sumaya Alzuhairy, Sunmin Ku, Yashab Narang

Customer Satisfaction is interesting when it comes to Airlines. There are many factors from seat comfort to if the 
plane was late. This app predicts the probability of a customer's satisfaction after flight using some the factors as 
inputs.""")

with st.form("my_form"):
    st.write("Satisfaction Predictor")
    gender_str = st.selectbox(
        'Gender',
        ('Male', 'Female'))
    age_val = st.slider("Age")
    class_str = st.selectbox(
        'Flight Class',
        ('Eco', 'Eco Plus', 'Business'))
    seat_comfort = st.slider('Seat Comfort', 0, 5, 5)
    time_conv = st.slider('Departure/Arrival time convenient', 0, 5, 5)
    food_bev = st.slider('Food and Beverages', 0, 5, 5)
    gate_loc = st.slider('Gate location', 0, 5, 5)
    flight_wifi = st.slider('Inflight wifi service', 0, 5, 5)
    flight_tv = st.slider('Inflight entertainment', 0, 5, 5)
    online_support = st.slider('Online support', 0, 5, 5)
    booking_ease = st.slider('Ease of Online booking', 0, 5, 5)
    flight_service = st.slider('On-board service', 0, 5, 5)
    leg_room = st.slider('Leg room service', 0, 5, 5)
    baggage_handling = st.slider('Baggage handling', 0, 5, 5)
    checkin_conv = st.slider('Checkin service', 0, 5, 5)
    cleanliness = st.slider('Cleanliness', 0, 5, 5)
    online_boarding = st.slider('Online boarding', 0, 5, 5)
    dep_delay = st.slider('Departure Delay in Minutes', 0, 1600, 0)
    arr_delay = st.slider('Arrival_Delay_in_Minutes', 0, 1600, 0)


    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        if gender_str == "Female":
            gender_int = 0
        elif gender_str == "Male":
            gender_int = 1

        if class_str == "Eco":
            class_int = 1
        elif class_str == "Eco Plus":
            class_int = 2
        elif class_str == "Business":
            class_int = 3

        st.write("Customer is ",
                 rf.predict([[age_val, class_int, seat_comfort, time_conv, food_bev, gate_loc, flight_wifi, flight_tv,
                              online_support, booking_ease, flight_service, leg_room, baggage_handling, checkin_conv,
                              cleanliness, online_boarding, dep_delay, arr_delay, gender_int]])[0])
        st.write("Predicted using Random Forest with accuracy percentage of: ", round(acc, 2))
