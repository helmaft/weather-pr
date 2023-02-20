import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

import numpy as np


from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.metrics import classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import seaborn as sns








# Set the page title and header
st.set_page_config(page_title='Seattle Weather Analysis', page_icon=':sunny:')
st.title('Seattle Weather Analysis')


# Display the cover image
cover_image = "image.jpg"
st.image(cover_image, use_column_width=True)


        
# Set the title and link of the sidebar
st.sidebar.title("Helma Falahati ")
st.sidebar.markdown("[GitHub](https://github.com/helmaft)")


# Load the data
df = pd.read_csv('seattle-weather.csv')
df['date'] = df['date'].apply(pd.to_datetime)
df['year']= df['date'].dt.year
df['month']= df['date'].dt.month
df['day']= df['date'].dt.day
df = df.drop("date", axis='columns')
# Calculate the average temperature for each month
temp_by_month = df.groupby('month')['temp_max'].mean()






# Define the sidebar options
options = ['Data', 'Histogram', 'Horizontal Bar Chart', 'Pie Chart', 'Treemap', 'Average Temperature by Month', 'scatter plot of temperature vs. precipitation'
           ,'Histogram of wind speed','Box plot of temperature by month','Heatmap of average temperature by year and month','Correlation between Features',
           'Scatter Plot Matrix','Scatter plot matrix of temperature, precipitation, and wind','3D scatter plot of temperature, precipitation, and wind',
           'Parallel coordinates plot of temperature, precipitation, and wind','Density plot of wind speed by temperature','Violin plot of temperature by month',]

# Display the sidebar
selection = st.sidebar.selectbox('Select a chart', options)

# Display the selected chart
if selection == 'Data':
    st.write('## Seattle Weather Data')
    st.write(df.head())
elif selection == 'Histogram':
    fig = px.histogram(df, x='weather', nbins=60)
    fig.update_layout(title='Distribution of Weather Types')
    st.plotly_chart(fig)
elif selection == 'Horizontal Bar Chart':
    weather_counts = df['weather'].value_counts()
    fig = go.Figure(go.Bar(
        x=weather_counts.values,
        y=weather_counts.index,
        orientation='h'
    ))
    fig.update_layout(title='Distribution of Weather Types')
    st.plotly_chart(fig)
elif selection == 'Pie Chart':
    weather_counts = df['weather'].value_counts()
    fig = go.Figure(go.Pie(
        labels=weather_counts.index,
        values=weather_counts.values
    ))
    fig.update_layout(title='Distribution of Weather Types')
    st.plotly_chart(fig)
elif selection == 'Treemap':
    weather_counts = df.groupby('weather').size()
    weather_counts.index = weather_counts.index.map('_'.join)
    fig = px.treemap(
        names=weather_counts.index,
        parents=['Weather'] * len(weather_counts),
        values=weather_counts.values
    )
    fig.update_layout(title='Distribution of Weather Types')
    st.plotly_chart(fig)
elif selection == 'Average Temperature by Month':
    fig = px.bar(temp_by_month, x=temp_by_month.index, y=temp_by_month.values)
    fig.update_layout(
        title='Average Temperature by Month',
        xaxis_title='Month',
        yaxis_title='Average Maximum Temperature (Celsius)',
    )
    st.plotly_chart(fig)
    
elif selection == 'scatter plot of temperature vs. precipitation':  
    fig = px.scatter(df, x='precipitation', y='temp_max')
    # Set the chart title and axis labels
    fig.update_layout(
        title='Relationship Between Temperature and Precipitation',
        xaxis_title='Precipitation (mm)',
        yaxis_title='Maximum Temperature (Celsius)',
    )

    # Show the chart
    st.plotly_chart(fig)
    
elif selection == 'Histogram of wind speed':
    # Create a histogram of wind speed
    fig = px.histogram(df, x='wind')

    # Set the chart title and axis labels
    fig.update_layout(
        title='Distribution of Wind Speed',
        xaxis_title='Wind Speed (m/s)',
        yaxis_title='Frequency',
    )
    st.plotly_chart(fig)

elif selection == 'Box plot of temperature by month':
    # Create a box plot of temperature by month
    fig = px.box(df, x='month', y='temp_max')

    # Set the chart title and axis labels
    fig.update_layout(
        title='Distribution of Maximum Temperature by Month',
        xaxis_title='Month',
        yaxis_title='Maximum Temperature (Celsius)',
    )
    
    st.plotly_chart(fig)

elif selection == 'Heatmap of average temperature by year and month':
    # Calculate the average temperature by year and month
    df_avg_temp = df.groupby(['year', 'month'], as_index=False).agg({'temp_max': 'mean'})

    # Pivot the data to create a matrix of average temperatures
    temp_matrix = df_avg_temp.pivot(index='month', columns='year', values='temp_max')

    # Create a heatmap of average temperature by year and month
    fig = px.imshow(temp_matrix, 
                    x=temp_matrix.columns, 
                    y=temp_matrix.index, 
                    color_continuous_scale='RdBu_r', 
                    title='Average Temperature by Year and Month')

    # Set the axis labels and colorbar title
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Month',
        coloraxis_colorbar_title='Temperature (°C)'
    )
    st.plotly_chart(fig)

elif selection == 'Correlation between Features':
    corr_matrix = df.corr()

    # Create a heatmap of the correlation matrix
    fig = px.imshow(corr_matrix, 
                    x=corr_matrix.columns, 
                    y=corr_matrix.columns, 
                    color_continuous_scale='RdBu_r', 
                    title='Correlation between Features')

    # Set the axis labels and colorbar title
    fig.update_layout(
        xaxis_title='Features',
        yaxis_title='Features',
        coloraxis_colorbar_title='Correlation Coefficient'
    )
    st.plotly_chart(fig)
    
elif selection == 'Scatter Plot Matrix':
    fig = px.scatter_matrix(df, dimensions=["precipitation", "temp_max", "temp_min", "wind"], color="month")
    fig.show()
    st.plotly_chart(fig)
    
elif selection == 'Scatter plot matrix of temperature, precipitation, and wind':
    # Create a scatter plot matrix of temperature, precipitation, and wind
    fig = px.scatter_matrix(
        df, 
        dimensions=['temp_max', 'precipitation', 'wind'],
        title='Relationships Between Temperature, Precipitation, and Wind'
    )
    st.plotly_chart(fig)
    
elif selection == '3D scatter plot of temperature, precipitation, and wind':
# Create a 3D scatter plot of temperature, precipitation, and wind
    fig = go.Figure(data=[go.Scatter3d(
        x=df['temp_max'],
        y=df['precipitation'],
        z=df['wind'],
        mode='markers',
        marker=dict(
            size=5,
            color=df['wind'],
            colorscale='Viridis',
            opacity=0.8
        )
    )])

    # Set the axis labels and title
    fig.update_layout(
        scene=dict(
            xaxis_title='Maximum Temperature (Celsius)',
            yaxis_title='Precipitation (mm)',
            zaxis_title='Wind Speed (km/h)'
        ),
        title='Relationships Between Temperature, Precipitation, and Wind'
    )
    st.plotly_chart(fig)
    
    
elif selection == 'Parallel coordinates plot of temperature, precipitation, and wind' :
    # Create a parallel coordinates plot of temperature, precipitation, and wind
    fig = px.parallel_coordinates(
        df, 
        dimensions=['temp_max', 'precipitation', 'wind'],
        color='temp_max',
        color_continuous_scale=px.colors.sequential.Viridis,
        title='Relationships Between Temperature, Precipitation, and Wind'
    )
    st.plotly_chart(fig)


elif selection == 'Density plot of wind speed by temperature':
        # Create a density plot of wind speed by temperature
    fig = px.density_heatmap(
        df, 
        x='temp_max', 
        y='wind',
        title='Density Plot of Wind Speed by Temperature',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig)


elif selection == 'Violin plot of temperature by month':
    # Create a violin plot of maximum temperature by month
    fig = px.violin(df, x='month', y='temp_max', title='Distribution of Maximum Temperature by Month')
    st.plotly_chart(fig)



st.title('Machine Learning Models')


# Add a separation line with custom style
st.markdown(
    """
    <style>
    .horizontalLine {
        margin-top: 20px;
        margin-bottom: 20px;
        border: 5px solid #e6e6e6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Add the separation line
st.markdown('<p class="horizontalLine"></p>', unsafe_allow_html=True)








options_models =['DecisionTreeClassifier','Random Forest' ,'SVM','GradientBoostingClassifier','KNN','Linear Discriminant Analysis']
# Display the sidebar 2 
selection = st.sidebar.selectbox('Select a Model', options_models)


if selection == 'DecisionTreeClassifier':
    # Define the input features and output variable
    X = df[['precipitation', 'temp_max', 'temp_min', 'wind', 'year', 'month', 'day']]
    y = df['weather']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f'The accuracy of the Decision Tree Classifier is {accuracy:.2f}')

    def show_results(y_test, y_pred):
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
        st.write("\nClassification Report:")
        st.write(classification_report(y_test, y_pred))       
    
    
    # Create a button to run the model
    if st.button("Decision Tree Classifier: Run Model"):
        # Make predictions on the testing data
        if selection == 'DecisionTreeClassifier':
            y_pred = model.predict(X_test)
        else:
            st.write("Model not supported")
            st.stop()

        # Show the results
        show_results(y_test, y_pred)
    
    
elif selection == 'Random Forest':
    X = df[['precipitation', 'temp_max', 'temp_min', 'wind', 'year', 'month', 'day']]
    y = df['weather']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f'The accuracy of the Random Forest is {accuracy:.2f}')
    # Display the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
    
    
    st.set_option('deprecation.showPyplotGlobalUse', False) # disable deprecation warning

    importances = model.feature_importances_
    features = X_train.columns
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title('Feature Importance')
    ax.bar(range(len(features)), importances[indices], color='r', align='center')
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels([features[i] for i in indices], rotation=90)
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    st.pyplot(fig)
    
    def show_results(y_test, y_pred):
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
        st.write("\nClassification Report:")
        st.write(classification_report(y_test, y_pred))       
    
    
    # Create a button to run the model
    if st.button("Random Forest :Run Model"):
        # Make predictions on the testing data
        if selection == 'Random Forest':
            y_pred = model.predict(X_test)
        else:
            st.write("Model not supported")
            st.stop()

        # Show the results
        show_results(y_test, y_pred)
    
    
    
    
    
    
elif selection == 'SVM':
    
    X = df[['precipitation', 'temp_max', 'temp_min', 'wind', 'year', 'month', 'day']]
    y = df['weather']
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the SVM model and train it on the training data
    svm_model = SVC(kernel='linear', C=1, gamma='auto')
    svm_model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = svm_model.predict(X_test)   
    
    
    
    def show_results(y_test, y_pred):
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
        st.write("\nClassification Report:")
        st.write(classification_report(y_test, y_pred))       
    
    
    # Create a button to run the model
    if st.button("Support Vector Machine : Run Model"):
        # Make predictions on the testing data
        if selection == 'SVM':
            y_pred = svm_model.predict(X_test)
        else:
            st.write("Model not supported")
            st.stop()

        # Show the results
        show_results(y_test, y_pred)

elif selection == 'GradientBoostingClassifier':
    
    X = df[['precipitation', 'temp_max', 'temp_min', 'wind', 'year', 'month', 'day']]
    y = df['weather']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the GBM model and train it on the training data
    gbm_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gbm_model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = gbm_model.predict(X_test)


    def show_results(y_test, y_pred):
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
        st.write("\nClassification Report:")
        st.write(classification_report(y_test, y_pred))       
    
    
    # Create a button to run the model
    if st.button("Gradient Boosting Classifier: Run Model"):
        # Make predictions on the testing data
        if selection == 'GradientBoostingClassifier':
            y_pred = gbm_model.predict(X_test)
        else:
            st.write("Model not supported")
            st.stop()

        # Show the results
        show_results(y_test, y_pred)



elif selection == 'KNN':
    X = df[['precipitation', 'temp_max', 'temp_min', 'wind', 'year', 'month', 'day']]
    y = df['weather']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the KNN model and train it on the training data
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)

    # Define a function to display the results
    def show_results(y_test, y_pred):
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
        st.write("\nClassification Report:")
        st.write(classification_report(y_test, y_pred))


    # Create a button to run the model
    if st.button("K-Nearest Neighbors: Run Model"):
        # Make predictions on the testing data
        if selection == 'KNN':
            y_pred = knn_model.predict(X_test)
        else:
            st.write("Model not supported")
            st.stop()

        # Show the results
        show_results(y_test, y_pred)

elif selection == 'Linear Discriminant Analysis':
    # Define the input and target variables
    X = df[['precipitation', 'temp_max', 'temp_min', 'wind', 'year', 'month', 'day']]
    y = df['weather']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the LDA model and train it on the training data
    lda_model = LinearDiscriminantAnalysis()
    lda_model.fit(X_train, y_train)

    # Define a function to display the results
    def show_results(y_test, y_pred):
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
        st.write("\nClassification Report:")
        st.write(classification_report(y_test, y_pred))

    # Create a button to run the model
    if st.button("Linear Discriminant Analysis: Run Model"):
        # Make predictions on the testing data
        y_pred = lda_model.predict(X_test)

        # Show the results
        show_results(y_test, y_pred)
        
        
        
