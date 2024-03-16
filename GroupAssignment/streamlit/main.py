import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import optuna
import joblib
from PIL import Image
import os

# Interface
st.sidebar.title('Category')
selection = st.sidebar.radio('Go to', ['Home','Conclusions', 'Technical Report','Simulator'])
current_path = os.path.abspath(__file__)
model_path = os.path.join(os.path.dirname(current_path), "bike_model.pkl")
bike_model = joblib.load(model_path)
if selection == 'Home':
    st.title("Prediction App")
    st.header("Overview - Prediction of the total number of bicycle users on an hourly basis")
    st.markdown("- This project delves deep into the dynamics of bike rental services, aiming to unravel crucial insights for enhancing user experience and operational efficiency. Through meticulous univariate and bivariate analysis, the project scrutinizes pivotal factors such as working day, temperature, felt temperature, humidity, and wind speed. Notably, the dataset exhibits centered values, indicating a balanced distribution with occasional outliers.") 
    st.markdown("- The analysis further unveils distinct patterns in bike rental counts, showcasing pronounced peaks during rush hours on working days, contrasted by smoother trends during non-working days. Additionally, seasonal variations in rental counts reveal notable disparities, with autumn emerging as the peak season and spring registering the lowest activity. Armed with these insights, the project proposes strategic interventions, including the introduction of pre-booking services, real-time bike location features, and optimized weekend bike deployments. Furthermore, considerations for springtime maintenance and updates are suggested to ensure seamless operations and foster user engagement.")

# Conclusions
if selection == 'Conclusions':
    st.title("Conclusions")
    st.subheader("Strategies for Enhanced User Experience and Operational Efficiency")
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Pre-booking Service and Real-time Bike Location Feature:")
        st.image("https://cdn-icons-png.freepik.com/512/2649/2649019.png", width=128)
        st.markdown("- Introducing a pre-booking service 10 minutes in advance for registered users can enhance user experience and convenience, thereby encouraging users to utilize the bike service.")
        st.markdown("- Providing a real-time bike location feature for free users is an added value service, which can improve user satisfaction and loyalty.")
    with col2:
        st.markdown("#### Weekend Bike Deployment in Central Areas:")
        st.image("https://icons.iconarchive.com/icons/iconsmind/outline/512/Bicycle-icon.png", width=128)
        st.markdown("- Deploying bikes around commercial centers during weekends is a wise decision as it caters to users commuting from work to home.")
        st.markdown("- Additionally, adjusting deployment strategies based on data analysis and user feedback ensures the adequacy and adaptability of bike numbers and locations.")
    st.markdown("#### Springtime Bike Maintenance and Updates:")
    st.image("https://png.pngtree.com/png-clipart/20230417/original/pngtree-maintenance-line-icon-png-image_9063268.png", width=128)
    st.markdown("- Spring is an opportune time for bike maintenance and updates, avoiding disruptions during peak demand periods.")
    st.markdown("- Moreover, consider introducing promotions or discounts tailored for spring to incentivize user engagement and boost bike utilization rates.")
    st.markdown("- Additionally, during spring, as temperatures rise and humidity increases, it's crucial to conduct maintenance on bike components susceptible to weather-related damage. By reducing bike supply during this period, potential losses can be minimized.")

# Data Exploration
elif selection == 'Technical Report':
    st.title('Technical Report')
    st.write("This page serves as a comprehensive technical report covering various aspects of the project, including data quality checks, data analysis, feature engineering, and model development. It provides insights into how abnormal data is identified and handled, details about the hyperparameters used for model training, and an evaluation of the model's performance. Additionally, it offers a deep dive into the methodologies employed to optimize the model and ensure its efficacy in predicting bike rental demand accurately.")
    with st.expander("Data Quality"):
        st.markdown("#### Null values")
        st.markdown("There are not any null values in the dataset.")
        st.markdown("#### Duplicates")
        st.markdown("There are not any duplicates in the dataset.")
        st.markdown("#### Outliers")
        st.markdown("We used z-score calculation to identify outliers in temperature-related features, such as *temperature*, *felt temperature*, *humidity*, and *windspeed*, discarding values with z-scores greater than 3 to ensure data accuracy and reliability for subsequent analyses and modeling. This quality assurance step enhances the robustness of our findings and predictive models by mitigating the impact of extreme anomalies.")
        st.markdown("#### Data types")
        st.markdown("We excluded the year column due to its limited information content. The remaining features have been verified to possess appropriate data types and are prepared for further analysis.")
    with st.expander("Analysis"):
        tab1, tab2 = st.tabs(["Univariate Analysis", "Bivariate Analysis"])
        tab1.markdown("In the presented visualization, we opted to focus solely on the distribution analysis of key variables, namely *workingday, temperature, felt temperature, humidity*, and *wind speed*. This decision was informed by the observed correlation among date-related features, which led us to prioritize these specific factors for our analysis.")
        tab1.markdown("We observe that the dataset exhibits centered values, with occurrences of extreme weather being infrequent. This suggests a smaller dataset available for predicting outcomes under such extreme weather conditions. Consequently, it's likely that the model's performance may degrade when confronted with inputs corresponding to extreme weather conditions.")
        tab1.image("https://github.com/KJCarlosYu/Myproject/blob/main/GroupAssignment/streamlit/output.png?raw=true", use_column_width=True)
        tab2.markdown("Here, we analyze the bike rental counts from two distinct perspectives: seasonal variations and hourly patterns.")
        tab2.markdown("In this visualization, it's evident that on working days, there's a notable increase in bike rentals during rush hours, specifically between 7 to 9 in the morning and 17 to 19 in the evening. Conversely, on non-working days, the trend appears smoother, with bike rentals peaking between 10 to 20, indicating a more evenly distributed usage pattern throughout the day.")
        tab2.image("https://github.com/KJCarlosYu/Myproject/blob/main/GroupAssignment/streamlit/count_hour.png?raw=true", use_column_width=True)
        tab2.markdown("In this chart, we depicted the bike rental counts categorized by hours and seasons, revealing a significant disparity among the seasons. Notably, autumn (season 3) exhibits the highest rental counts, while spring (season 1) registers the lowest.")
        tab2.image("https://github.com/KJCarlosYu/Myproject/blob/main/GroupAssignment/streamlit/season.png?raw=true", use_column_width=True)
    with st.expander("Feature Engineering"):
        tab1, tab2 = st.tabs(["New variables", "One-hot Encoding"])
        tab1.markdown("**rush_hr**: We've introduced a new variable, *rush_hr*, specifically tailored for working days, encompassing the timeframes from 7 to 9 in the morning and from 17 to 19 in the evening.")
        tab2.markdown("**One-hot Encoding** : We conducted one-hot encoding for the variables *season* and *weathersit*.")
    with st.expander("Model development"):
        tab1, tab2 = st.tabs(["Model selection", "Model development"])
        tab1.markdown("We first ran a model with the default hyperparameters to see which model performs better. According to the results shown below, random forest and xgboost have similar performances, so we decided to use random forest for the model.")
        data = {
                    "Model": ["Linear Regression", "Decision Tree", "Random Forest", "Support Vector Machine", "XGBoost"],
                    "Mean Absolute Error (MAE)": [84.72, 53.28, 41.06, 91.39, 40.81],
                    "Root Mean Squared Error (RMSE)": [113.63, 87.38, 62.86, 142.16, 60.99],
                    "R-squared (R2)": [0.60, 0.76, 0.88, 0.37, 0.88]
                }
        df = pd.DataFrame(data)
        tab1.write("#### Model Evaluation Results")
        tab1.write(df)
        tab2.markdown("For model development, we used *optuna* to conduct the hyperparameter tuning to select the best model. The following is the result of the model.")
        data = {
            "Model": ["Random Forest"],
            "Mean Absolute Error (MAE)": [42.74],
            "Root Mean Squared Error (RMSE)": [64.83],
            "Mean Absolute Percentage Error (MAPE)": ["41.37%"],
            "R-squared (R2)": [0.87]
        }
        df = pd.DataFrame(data)
        tab2.write("#### Model Evaluation Results")
        tab2.write(df)
        image_path = "./model_plot.png"
        tab2.image("https://github.com/KJCarlosYu/Myproject/blob/main/GroupAssignment/streamlit/model_plot.png?raw=true", use_column_width=True)
        feature_names = ['mnth', 'hr', 'holiday', 'weekday', 'workingday', 'temp', 'atemp',
                 'hum', 'windspeed','rush_hr','season_1', 'season_2', 'season_3',
                 'season_4', 'weathersit_1', 'weathersit_2', 'weathersit_3',
                 'weathersit_4']
# Plot the feature importances
        feature_importances = bike_model.feature_importances_
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#EAE0D5')  # Set the background color for the figure
        ax.set_facecolor('#EAE0D5')  # Set the background color for the plot area
        ax.bar(feature_names, feature_importances)
        ax.set_xlabel('Feature')
        ax.set_ylabel('Feature Importance')
        ax.set_title('Feature Importance of the Best Model')
        plt.xticks(rotation=45)
        plt.show()
        tab2.write("Here, we've visualized the most influential features for our prediction. It's evident that *hour* stands out as the most significant feature among all.")
        tab2.pyplot(fig)

# Model Prediction Page
elif selection == 'Simulator':
    st.title('Bike rental counts Simulator')
    st.write('On this page, you can make predictions for bike rental counts by inputting data. Here are a few pointers:')
    st.markdown("- You can get predictions for a whole day by not selecting a specific hour, while inputting a specific hour will yield a numerical value.")
    st.markdown("- The model's accuracy may decrease during extreme weather conditions.")
    st.markdown("- The model can automatically extract data such as the day of the week, working day, holiday, season, etc., based on the date you input, eliminating the need for manual input of each variable.")
    with st.form(key='form'):
        weathersit_options = {
        1: 'Clear, Few clouds, Partly cloudy, Partly cloudy',
        2: 'Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist',
        3: 'Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds',
        4: 'Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog'
        }
        date = st.date_input(label='Date', value=datetime.now())
        hour = st.time_input(label='Time', step=3600, value=None)
        temp = st.slider(label='Temperature',value=20, min_value=0, max_value=41, step=1)
        felt_temp = st.slider(label='Felt temperature',value=25, min_value=0, max_value=50, step=1)
        humidity = st.slider(label='Humidity',value=50, min_value=0, max_value=100, step=1)
        wind_speed = st.slider(label='Wind_speed',value=33, min_value=0, max_value=67, step=1)
        selected_weather_value = st.selectbox('Select Weather Condition', options=list(weathersit_options.values()), index=0)
        selected_weather_key = None

        for key, value in weathersit_options.items():
            if value == selected_weather_value:
                selected_weather_key = key
            break
        submit_button = st.form_submit_button(label="Submit")

# Prediction
    if submit_button:
        # Extract date
        def parse_date(date):
            date_str = date.strftime("%m-%d")
            def get_season(month):
                season_mapping = {1: 4, 2: 4, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 4}
                return season_mapping[month]
            month = date.month
            season = get_season(month)
            holidays = ['01-01', '01-15', '02-19', '04-16', '05-27', '06-19', '07-04', '09-02', '10-14', '11-11', '11-28', '12-25']
            holiday = 1 if date_str in holidays else 0
            weekday = date.weekday()
            workingday = 1 if date.weekday() < 5 and holiday == 0 else 0
            return season, month, holiday, weekday, workingday

        season, month, holiday, weekday, workingday = parse_date(date)
        X = pd.DataFrame({'mnth': [month], 'hr': [hour.hour if hour is not None else None],'season':[season], 'holiday': [holiday], 'weekday': [weekday], 
                "workingday": [workingday], 'temp': [temp], 'atemp': [felt_temp], 'hum': [humidity], 
                'windspeed': [wind_speed], 'weathersit': [selected_weather_key]})

        X['rush_hr'] = 0
        if hour is not None and (7 <= hour.hour <= 9 or 17 <= hour.hour <= 19) and workingday == 1:
            X['rush_hr'] = 1

        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, categories=[[1, 2, 3, 4], [1, 2, 3, 4]])
        categorical_columns = ['season', 'weathersit']
        categorical_columns_encoded = encoder.fit_transform(X[categorical_columns])
        encoded_feature_names = encoder.get_feature_names_out(categorical_columns)
        X_encoded = pd.DataFrame(categorical_columns_encoded, columns=encoded_feature_names, index=X.index)
        X = pd.concat([X.drop(categorical_columns, axis=1), X_encoded], axis=1)

        temp_min, temp_max = 0, 41
        atemp_min, atemp_max = 0, 50
        hum_min, hum_max = 0, 100
        windspeed_min, windspeed_max = 0, 67    
        def linear_scale(value, min_val, max_val):
            return (value - min_val) / (max_val - min_val)

        X_scaled = X.copy() 
        X_scaled['temp'] = linear_scale(X_scaled['temp'], temp_min, temp_max)
        X_scaled['atemp'] = linear_scale(X_scaled['atemp'], atemp_min, atemp_max)
        X_scaled['hum'] = linear_scale(X_scaled['hum'], hum_min, hum_max)
        X_scaled['windspeed'] = linear_scale(X_scaled['windspeed'], windspeed_min, windspeed_max)

        if hour is None:
            hours = list(range(24))
            predictions = []

            for h in hours:
                X_scaled['hr'] = h
                y_pred = bike_model.predict(X_scaled)
                predictions.append(y_pred)

            plt.figure(figsize=(10, 5))  
            plt.plot(hours, predictions)
            plt.xlabel('Hour')
            plt.ylabel('Predicted Count')
            plt.title('Predicted Count vs. Hour')
            plt.xlim(0, 24)
            st.line_chart(data=predictions, color=None, width=400, height=300, use_container_width=True)

        else:
            y_pred = bike_model.predict(X_scaled)
            output = pd.DataFrame({"Number of bikes rented":[int(y_pred)]})
            output