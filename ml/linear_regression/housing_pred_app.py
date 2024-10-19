import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score,roc_curve
import streamlit as st

stdscaler=StandardScaler()
output_stdscaler=StandardScaler()

def get_model():
    df_blr = pd.read_csv("datasets/housing_price/Bangalore.csv")
    df_blr["city"] = "BLR"
    df_mum = pd.read_csv("datasets/housing_price/Mumbai.csv")
    df_mum["city"] = "MUM"
    df_del = pd.read_csv("datasets/housing_price/Delhi.csv")
    df_del["city"] = "DEL"
    df_kol = pd.read_csv("datasets/housing_price/Kolkata.csv")
    df_kol["city"] = "KOL"
    df_chn = pd.read_csv("datasets/housing_price/Chennai.csv")
    df_chn["city"] = "CHN"
    df_hyd = pd.read_csv("datasets/housing_price/Hyderabad.csv")
    df_hyd["city"] = "HYD"
    df = pd.concat([df_blr, df_chn, df_del, df_hyd, df_kol, df_mum], ignore_index=True)
    print("Merged dataframe")
    print(df)
    ohe_encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False).set_output(transform='pandas')
    df_encoded_city=ohe_encoder.fit_transform(df[['city']])
    df=pd.concat([df,df_encoded_city],axis=1).drop(columns=['city'])
    print("concatenating df")
    print(df)
    col_to_scale=['Area','No. of Bedrooms','MaintenanceStaff','Gymnasium','SwimmingPool','LandscapedGardens','JoggingTrack','RainWaterHarvesting','IndoorGames','ShoppingMall','Intercom','SportsFacility',	'ATM',	'ClubHouse','School','24X7Security','PowerBackup','CarParking','StaffQuarter','Cafeteria','MultipurposeRoom','Hospital','WashingMachine','Gasconnection','AC','Wifi','Children\'splayarea','LiftAvailable','BED','VaastuCompliant','Microwave', 'GolfCourse','TV','DiningTable','Sofa','Wardrobe','Refrigerator']
    features_to_drop=['MaintenanceStaff','Gymnasium','LandscapedGardens','JoggingTrack','RainWaterHarvesting','IndoorGames','ShoppingMall','Intercom','SportsFacility',	'ATM',	'ClubHouse','School','24X7Security','PowerBackup','CarParking','StaffQuarter','Cafeteria','MultipurposeRoom','Hospital','WashingMachine','Gasconnection','AC','Wifi','Children\'splayarea','LiftAvailable','BED','VaastuCompliant','Microwave', 'GolfCourse','TV','DiningTable','Sofa','Wardrobe','Refrigerator']
    df[col_to_scale]=stdscaler.fit_transform(df[col_to_scale])
    df=df.drop(columns=features_to_drop,axis=1)
    
    Y=output_stdscaler.fit_transform(df[['Price']])
    df_processed=df.drop(['Location','Price'],axis=1)
    X=df_processed
    # df_processed
    print("scaled inputs")
    print(X)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    # X_train
    print("training data size "+str(len(X_train)))
    model=RandomForestRegressor()
    model.fit(X_train,Y_train)
    y_pred=model.predict(X_test)


    mse = mean_squared_error(Y_test, y_pred)
    r2 = r2_score(Y_test, y_pred)

    # Output the performance metrics
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R²): {r2}")
    return model

def predict_price(model, features):
    
    """Predicts the house price using the trained model."""
    scaled_features = stdscaler.fit_transform([features])  # Scale the input features
    return model.predict(scaled_features)[0]
def get_city_param(city):
    city_=[0.0,0.0,0.0,0.0,0.0,0.0]
    if city=="Bangalore":
        city_[0]=1.0
    if city=="Chennai":
        city_[1]=1.0
    if city=="Delhi":
        city_[2]=1.0
    if city=="Hyderabad":
        city_[3]=1.0
    if city=="Kolkata":
        city_[4]=1.0
    if city=="Mumbai":
        city_[5]=1.0
    return city_

def app(model):
    col_to_scale=['Area','No. of Bedrooms','Price','MaintenanceStaff','Gymnasium','SwimmingPool','LandscapedGardens','JoggingTrack','RainWaterHarvesting','IndoorGames','ShoppingMall','Intercom','SportsFacility',	'ATM',	'ClubHouse','School','24X7Security','PowerBackup','CarParking','StaffQuarter','Cafeteria','MultipurposeRoom','Hospital','WashingMachine','Gasconnection','AC','Wifi','Children\'splayarea','LiftAvailable','BED','VaastuCompliant','Microwave', 'GolfCourse','TV','DiningTable','Sofa','Wardrobe','Refrigerator']

    st.title("Dream House Price prediction")
    with st.form("prediction"):
        st.write("select the parameter to predict the house price")
        city=st.selectbox("select city", ['Delhi','Mumbai','Kolkata','Chennai','Bangalore','Hyderabad'])
        city_=get_city_param(city=city)
        area=st.slider("super area(sftst): ",min_value=100,max_value=10000,step=50,value=1000)
        resale=int(st.checkbox("Resale: "))
        no_of_bedrooms=st.slider("No of BedRooms : ",min_value=0,max_value=10,value=1)
        no_of_sweeming_pools=st.slider("No of Sweeming Pools: ",min_value=0,max_value=10,value=1)
        submit=st.form_submit_button("predict")


        if submit:
            features = [area,no_of_bedrooms,resale,no_of_sweeming_pools]  # Replace with actual feature values
            features.extend(city_)
            print(features)
            # Assuming you have a trained model named 'model' and a StandardScaler named 'stdscaler'
            try:
                # Get the scaled predicted price from the model
                print(features)
                predicted_price_scaled = predict_price(model, features)  # Replace 'model' with your trained model
                
                print("Predicted price (scaled): " + str(predicted_price_scaled))
                print(type(predicted_price_scaled))

                # If the target variable (price) was scaled, you should have a separate scaler for that.
                # Assuming `price_scaler` is the StandardScaler for the target price
                predicted_price_scaled = np.array(predicted_price_scaled).reshape(1, -1)
                predicted_price_val = output_stdscaler.inverse_transform(predicted_price_scaled)  # Use the scaler for the target
                
                print(int(predicted_price_val))
                
                # Show the predicted price on the Streamlit app
                st.write(f"Predicted Price: {predicted_price_val[0][0]:.2f}")
            except NameError:
                st.write("Please upload a trained model and scaler first.")


if __name__ == "__main__":
    model=get_model()
    app(model=model)