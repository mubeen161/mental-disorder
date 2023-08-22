import streamlit as st
import pickle
from sklearn.preprocessing import LabelEncoder 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
from PIL import Image
# Define Prediction page content

st.set_page_config(
    page_title="Brainwave Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

custom_theme = """
        [theme]
        primaryColor="#19454c"
        backgroundColor="#316371"
        secondaryBackgroundColor="#131743"
        textColor="#ffffff"
        font="serif"
    """

    # Apply the custom theme
st.write(f"<style>{custom_theme}</style>", unsafe_allow_html=True)

hide_streamlit_style = """
        <style>
        #MainMenu, .stNotification, .stSystemWarning {
            display: none;
        }
        </style>
    """
# Load the trained model from the pickle file
model_filename11 = 'human_stress_model.pkl'  # Replace with the actual filename
with open(model_filename11, 'rb') as file:
    model1 = pickle.load(file)

st.markdown(hide_streamlit_style, unsafe_allow_html=True)
model = pickle.load(open("model.pkl", "rb"))
df = pd.read_csv('https://raw.githubusercontent.com/mubeen161/Datasets/main/EEG.machinelearing_data_BRMH.csv')
df = df.rename({'sex': 'gender', 'eeg.date': 'eeg date', 'main.disorder': 'main disorder',
                'specific.disorder': 'specific disorder'}, axis=1)
df['age'] = df['age'].round(decimals=0)
# df=df.drop('gender',axis=1)
df1=df.loc[:,'gender':'specific disorder']
df1=df1.drop('eeg date',axis=1)
def reformat_name(name):
    splitted = name.split(sep='.')
    if len(splitted) < 5:
        return name
    if splitted[0] != 'COH':
        result = f'{splitted[2]}.{splitted[4]}'
    else:
        result = f'{splitted[0]}.{splitted[2]}.{splitted[4]}.{splitted[6]}'
    return result
df.rename(reformat_name, axis=1, inplace=True)
# st.set_page_config(page_title='Streamlit Dashboard')


# Mean powers per main disorder
main_mean = df.groupby('main disorder').mean(numeric_only=True).reset_index()
# Mean powers per specific disorder
spec_mean = df.groupby('specific disorder').mean(numeric_only=True).reset_index()
# List of bands
msd=['Mood disorder','Addictive disorder','Trauma and stress related disorder','Schizophrenia','Anxiety disorder','Healthy control','Obsessive compulsive disorder']
bands = ['delta', 'theta', 'alpha', 'beta', 'highbeta', 'gamma']
ssd=['Acute stress disorder','Adjustment disorder','Alcohol use disorder','Behavioral addiction disorder','Bipolar disorder','Depressive disorder','Healthy Control','Obsessive compulsive disorder','Panic disorder','Posttraumatic stress disorder','Schizophrenia','Social anxiety disorder']
# Convert from wide to long
main_mean = pd.wide_to_long(main_mean, bands, ['main disorder'], 'channel', sep='.', suffix='\w+')
spec_mean = pd.wide_to_long(spec_mean, bands, ['specific disorder'], 'channel', sep='.', suffix='\w+')

# Define channels
chs = {'FP1': [-0.03, 0.08], 'FP2': [0.03, 0.08], 'F7': [-0.073, 0.047], 'F3': [-0.04, 0.041],
       'Fz': [0, 0.038], 'F4': [0.04, 0.041], 'F8': [0.073, 0.047], 'T3': [-0.085, 0], 'C3': [-0.045, 0],
       'Cz': [0, 0], 'C4': [0.045, 0], 'T4': [0.085, 0], 'T5': [-0.073, -0.047], 'P3': [-0.04, -0.041],
       'Pz': [0, -0.038], 'P4': [0.04, -0.041], 'T6': [0.07, -0.047], 'O1': [-0.03, -0.08], 'O2': [0.03, -0.08]}
channels = pd.DataFrame(chs).transpose()
def predict_stress_level(inputss):
    inputss = np.array(inputss).reshape(1, -1)
    prediction1 = model1.predict(inputss)
    return prediction1[0]

# Main function to create the Streamlit app
def stress_pred():
    st.title("Stress Level Prediction")
    
    image = Image.open('moderate stress.png')
    st.image(image)
    
    st.markdown("""
        <style>
        .big-font {
            font-size:50px !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="big-font">Enter the following inputs to predict stress level:</p>', unsafe_allow_html=True)
    
    # Input fields
    name = st.text_input("Name ")
    age = st.selectbox("Gender", ('Male', 'Female', 'Others'))
    sr = st.slider("Snoring Rate (Snoring frequency/total sleep time)", min_value=0.0, max_value=100.0, step=0.1)
    rr = st.slider("Respiration Rate (no. of breaths/min)", min_value=5.0, max_value=50.0, step=0.1)
    t = st.slider("Body Temperature (in F)", min_value=85.0, max_value=110.0, step=0.1)
    bo = st.slider("Blood Oxygen (in %)", min_value=70.0, max_value=100.0, step=0.1)
    sr_1 = st.slider("Sleeping Hours", min_value=0.0, max_value=24.0, step=0.5)
    hr = st.slider("Heart Rate", min_value=40.0, max_value=90.0, step=0.1)
    
    st.markdown("""
        <style>
        .op-font {
            font-size:10px !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Predict button
    if st.button("Predict Stress Level"):
        inputs = [sr, rr, t, bo, sr_1, hr]
        predicted_sl = predict_stress_level(inputs)
        if predicted_sl == 0:
            st.markdown('<p class="op-font">Minimal Stress:</p>', unsafe_allow_html=True)
            st.text("Occasional moments of mild tension or discomfort")
            image1 = Image.open('stress 1.0.png')
            st.image(image1)
        elif predicted_sl == 1:
            st.markdown('<p class="op-font">Mild Stress: </p>', unsafe_allow_html=True)
            st.text("Feeling a bit on edge, but you can still handle things")
            image2 = Image.open('stress 2.png')
            st.image(image2)
        elif predicted_sl == 2:
            st.markdown('<p class="op-font">Moderate Stress: </p>', unsafe_allow_html=True)
            st.text("Multiple deadlines & complex tasks require heightened concentration & adaptive problem solving ")
            image3 = Image.open('stress 1.png')
            st.image(image3)
        elif predicted_sl == 3:
            st.markdown('<p class="op-font">High Stress:</p>', unsafe_allow_html=True)
            st.text(" Pressure builds up, messing with your concentration and ability to unwind. ")
            image4 = Image.open('stress 4.png')
            st.image(image4)
        else:
            st.markdown('<p class="op-font">Severe Stress: </p>', unsafe_allow_html=True)
            st.text("Experiencing profound tension significantly impairs cognitive function & emotional well-being , severe health ramifications emphasizing the urgent need for help to reduce health effects ")
            image5 = Image.open('stress 5.png')
            st.image(image5)

def home_page():
    st.title('Brain-Wave Analytics')
    # st.write('This is the home page.')

    # Random text
    st.header('Muffakham Jah College of Engineering and Technology')
  
    
    # Display images
    image_paths = ['project/eeg.jpg', 'project/wave.png', 'project/process.jpg', 'project/density.png','project/pie.png','project/bar.png','project/chart.png','project/future.png']
    # List of captions for the images
    captions = ['EEG Recording', 'Types of Brain Waves', 'Process of EEG to ML', 'Density Plot of main disorder','Main Disorder Ratio','Specific Disorder Ratio','Male-Famale Ratio','Wearable Devices Available']
    # Define the number of columns in the grid
    num_columns = 2
    # Calculate the number of rows based on the number of images and columns
    num_rows = len(image_paths) // num_columns
    # Loop over the rows and columns to display the images in a grid
    for row in range(num_rows):
        col1, col2 = st.columns(num_columns)
        for col in [col1, col2]:
            if image_paths and captions:
                image_path = image_paths.pop(0)
                caption = captions.pop(0)
                image = Image.open(image_path)
                col.image(image, caption=caption, use_column_width=True)

def plot_eeg(levels, positions, axes, fig, ch_names=None, cmap='Spectral_r', cb_pos=(0.9, 0.1),cb_width=0.04, cb_height=0.9, marker=None, marker_style=None, vmin=None, vmax=None, **kwargs):
  if 'mask' not in kwargs:
    mask = np.ones(levels.shape[0], dtype='bool')
  else:
    mask = None
  im, cm = mne.viz.plot_topomap(levels, positions, axes=axes, names=ch_names,cmap=cmap, mask=mask, mask_params=marker_style, show=False, **kwargs)
def prediction_page():
    # st.title('Prediction')
    # Add code for prediction here
    st.title("Disorder Prediction Based on EEG data")
    st.text_input("Enter EEG Data in numericial format :")
    data=pd.read_csv('https://raw.githubusercontent.com/mubeen161/Datasets/main/EEG.machinelearing_data_BRMH.csv')
    data=data.rename(columns={"specific.disorder": "sd", "main.disorder": "md"})
    data = data.fillna(0)
    st.text("OR")
    # Preprocess the data
    md = LabelEncoder()
    data['md'] = md.fit_transform(data['md'])
    sex=LabelEncoder()
    data['sex'] = sex.fit_transform(data['sex'])
    sd = LabelEncoder()
    data['sd'] = sd.fit_transform(data['sd'])
    data = data.drop(['eeg.date', 'no.'], axis=1)
    data=data.round(4)
    X = data.drop('sd', axis=1)
    y = data['sd']
    X = X.round(3)
    X["age"] = X["age"].round(0)
    selected_data = st.selectbox("Select a person's Data from Dataset :", X.index)

    # Retrieve the selected row from X_test
    input_array = X.loc[selected_data].values

    # Make prediction on user input
    if st.button("Predict"):
        features = np.array(input_array).reshape(1, -1)
        prediction = model.predict(features)
        t = prediction[0]

        if t == 0:
            ls = "Acute Stress Disorder"
        elif t == 1:
            ls = "Adjustment Disorder"
        elif t == 2:
            ls = "Alchohol Use Disorder"
        elif t == 3:
            ls = "Bipolar Disorder"
        elif t == 4:
            ls = "Behavioral Addictive Disorder"
        elif t == 5:
            ls = "Depressive Disorder"
        elif t == 6:
            ls = "Healthy Control"
        elif t == 7:
            ls = "Obsessive Compulsive Disorder"
        elif t == 8:
            ls = "Panic Disorder"
        elif t == 9:
            ls = "Post Traumatic Stress Disorder"
        elif t == 10:
            ls = "Schizophrenia"
        elif t == 11:
            ls = "Social Anxiety Disorder"
        else:
            ls = "Unknown Disorder"

        st.success(f"This individual has the highest probability of having {ls}")
        st.text("Please note that this result is not entirely reliable, and further medical assessment maybe required for a conclusive diagnosis.")

def disorder_comparison():
    st.write("Disorder Comparison selected")
    # Add your code for Disorder Comparison functionality here
    test_schizo = main_mean.loc[st.selectbox("Disorder 1",msd), st.selectbox("bands 1",bands)]
    test_control = main_mean.loc[st.selectbox("Disorder 2",msd), st.selectbox("bands 2",bands)]
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
   
    # Plot the first subplot
    plot_eeg(test_schizo, channels.to_numpy(), ax1, fig, marker_style={'markersize': 4, 'markerfacecolor': 'black'})
    ax1.set_title('Disorder 1')
   
    # Plot the second subplot
    plot_eeg(test_control, channels.to_numpy(), ax2, fig, marker_style={'markersize': 4, 'markerfacecolor': 'black'})
    ax2.set_title('Disorder 2')
   
    # Display the plot
    st.pyplot(fig)

def stress_level_page():
    st.title('Stress Level')
    # Add code for stress level here
    column_name = st.selectbox("Select Band",bands)
    highest_value = main_mean[column_name].max()
    lowest_value = main_mean[column_name].min()
    

    min_value = round(lowest_value, 2)
    max_value = round(highest_value, 2)


    fig, ax = plt.subplots(figsize=(6, 1))
    ax.set_axis_off()


    cmap = plt.cm.YlOrRd
    norm = plt.Normalize(min_value, max_value)


    ax.imshow(np.arange(min_value, max_value).reshape(1, -1), cmap=cmap, norm=norm, aspect='auto')
    ax.text(0, 0, str(min_value), ha='left', va='center', color='black', weight='light')
    ax.text(max_value - min_value, 0, str(max_value), ha='right', va='center', color='black', weight='light')
    plt.title(f'Stress level using {column_name} brain waves')
    st.pyplot(fig)
    # st.write("Brain Simulation selected")
    img_path = "stresslevel.jpg"
    img=Image.open(img_path)
    # Display the GIF image
    st.image(img,caption='Reference Chart' ,use_column_width=True)



    # Display the plot using Streamlit
def main():
    # Dropdown menu for page selection
    page_options = {
        'Home': home_page,
        'Prediction': prediction_page,
        # 'Plots': plots_page,
        # 'Wave Compare': wave_compare_page,
        # 'Brain Compare': brain_compare_page,
        # 'Stress Level': stress_levels_page,
        # 'Topographic Brain Activity':topographic_brain_activity,
        'Disorder Comparison':disorder_comparison,
        'Stress Level': stress_level_page,
        # 'Brain Simulation':brain_simulation,
        # 'AI - Assistant ':chat
        'Stress Prediction':stress_pred
    }
    selected_page = st.sidebar.selectbox('Select a page', list(page_options.keys()))
    page = page_options[selected_page]

    # Display selected page content
    page()


if __name__ == '__main__':
    main()
    