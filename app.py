import tkinter as tk
from tkinter import messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder



# Function to check login credentials
def check_login():
    if username_entry.get() == "admin" and password_entry.get() == "admin":
        login_window.destroy()  # Close login window
        open_main_page()
    else:
        messagebox.showerror("Login Failed", "Invalid Username or Password")

# Function to animate title text
def animate_title(label, text, index=0):
    label.config(text=text[:index])
    if index < len(text):
        label.after(100, animate_title, label, text, index + 1)

# Function to animate button hover effects
def on_enter(event):
    event.widget.config(bg="#1abc9c", fg="white")

def on_leave(event):
    event.widget.config(bg="#3498db", fg="white")

# Function for Uploading Dataset
def upload():
    global filename, df
    filename = filedialog.askopenfilename(initialdir="dataset", title="Select a File", filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*")))
    if filename:  # Proceed only if a file is selected
        pathlabel.config(text=filename)
        try:
            df = pd.read_csv(filename)
            # Replace '?' with NaN
            df.replace('?', np.nan, inplace=True)
            # Fill missing values with mode for each column
            df.fillna(df.mode().iloc[0], inplace=True)

            # Update the text box
            text.delete('1.0', tk.END)
            text.insert(tk.END, 'Dataset loaded successfully.\n')
            text.insert(tk.END, f"Dataset Path: {filename}\n")
            text.insert(tk.END, f"Dataset Size: {df.shape[0]} rows and {df.shape[1]} columns\n")
            text.insert(tk.END, f"Columns: {', '.join(df.columns)}\n")
        except Exception as e:
            text.delete('1.0', tk.END)
            text.insert(tk.END, f"Error loading dataset: {str(e)}\n")

# Function for Preprocessing Data
def preprocess():
    global df, X, y  # Ensure global access
    
    if df is None:
        text.delete("1.0", tk.END)
        text.insert(tk.END, "Error: No dataset loaded. Please upload a dataset first.\n")
        return

    try:
        df.replace('?', np.nan, inplace=True)

        # Fill missing values with mode
        df['collision_type'] = df['collision_type'].fillna(df['collision_type'].mode()[0])
        df['property_damage'] = df['property_damage'].fillna(df['property_damage'].mode()[0])
        df['police_report_available'] = df['police_report_available'].fillna(df['police_report_available'].mode()[0])

        # Drop unnecessary columns
        to_drop = ['policy_number', 'policy_bind_date', 'policy_state', 'insured_zip', 'incident_location',
                   'incident_date', 'incident_state', 'incident_city', 'insured_hobbies', 'auto_make',
                   'auto_model', 'auto_year', '_c39']
        df.drop(to_drop, inplace=True, axis=1)
        df.drop(columns=['age', 'total_claim_amount'], inplace=True, axis=1)

        # Splitting data
        X = df.drop('fraud_reported', axis=1)
        y = df['fraud_reported']

        # Handling categorical features
        cat_df = X.select_dtypes(include=['object'])
        cat_df = pd.get_dummies(cat_df, drop_first=True)

        # Handling numerical features
        num_df = X.select_dtypes(include=['int64'])
        X = pd.concat([num_df, cat_df], axis=1)

        # Visualization
        plt.figure(figsize=(25, 20))
        plotnumber = 1
        for col in X.columns:
            if plotnumber <= 24:
                ax = plt.subplot(5, 5, plotnumber)
                sns.histplot(X[col], kde=True)  # Updated from sns.distplot (deprecated)
                plt.xlabel(col, fontsize=15)
            plotnumber += 1
        plt.tight_layout()
        plt.show()

        # Update text box
        text.delete("1.0", tk.END)
        text.insert(tk.END, "Data preprocessing completed successfully!\n")
        text.insert(tk.END, f"Processed Dataset Size: {X.shape[0]} rows, {X.shape[1]} columns\n")

    except Exception as e:
        text.delete("1.0", tk.END)
        text.insert(tk.END, f"Error during preprocessing: {str(e)}\n")

# Function for Train_Test_Split
def train_test_split_func():
    global X_train, X_test, y_train, y_test, X  # Ensure global access

    if df is None:
        text.delete("1.0", tk.END)
        text.insert(tk.END, "Error: No dataset loaded. Please upload a dataset first.\n")
        return

    try:
        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        num_df = X_train[['months_as_customer', 'policy_deductable', 'umbrella_limit',
                          'capital-gains', 'capital-loss', 'incident_hour_of_the_day',
                          'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'injury_claim', 'property_claim',
                          'vehicle_claim']]

        # Scaling the numeric values in the dataset
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(num_df)
        scaled_num_df = pd.DataFrame(data=scaled_data, columns=num_df.columns, index=X_train.index)
        scaled_num_df.head()

        # Dropping old columns and adding scaled data
        X_train.drop(columns=scaled_num_df.columns, inplace=True)
        X_train = pd.concat([scaled_num_df, X_train], axis=1)

        text.delete('1.0', tk.END)
        text.insert(tk.END, f"Train-Test Split and Scaling completed.\n")
        text.insert(tk.END, f"Training Data Shape: {X_train.shape}\n")
        text.insert(tk.END, f"Training Data Shape: {X_test.shape}\n")
        text.insert(tk.END, f"Testing Data Shape: {y_train.shape}\n")
        text.insert(tk.END, f"Testing Data Shape: {y_test.shape}\n")
        text.insert(tk.END, f"Top 5 Rows Of Training Data:\n {X_train.head()}\n")



    except Exception as e:
        text.delete('1.0', tk.END)
        text.insert(tk.END, f"Error during Train-Test Split: {str(e)}\n")



def Run_SVM():
    global X_train, X_test, y_train, y_test, X,svc_train_acc # Ensure global access

    svc = SVC()
    svc.fit(X_train, y_train)

    y_pred = svc.predict(X_test)
    # accuracy_score, confusion_matrix and classification_report

    svc_train_acc = accuracy_score(y_train, svc.predict(X_train))*100
    svc_test_acc = accuracy_score(y_test, y_pred)*100

    text.delete("1.0", tk.END)

    print(f"Training accuracy of Support Vector Classifier is : {svc_train_acc}")
    print(f"Test accuracy of Support Vector Classifier is : {svc_test_acc}")
    text.insert(tk.END, f"Training accuracy of Support Vector Classifier is : {svc_train_acc}\n")
    text.insert(tk.END, f"Test accuracy of Support Vector Classifier is : {svc_test_acc}\n")

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    # Convert confusion matrix to a string format
    cm_str = '\n'.join(['\t'.join(map(str, row)) for row in cm])
    text.insert(tk.END, f"Confusion Matrix:\n{cm_str}\n")

    cr = classification_report(y_test, y_pred)
    print(cr)
    text.insert(tk.END, f"Classification Report:\n{cr}\n")




# Function for Running Decision Tree Classifier
def Run_DecisionTree():
    global X_train, X_test, y_train, y_test, X,dtc_train_acc  # Ensure global access

    # Initialize the Decision Tree Classifier
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)

    # Predict on the test set
    y_pred = dtc.predict(X_test)

    # Calculate accuracy
    dtc_train_acc = accuracy_score(y_train, dtc.predict(X_train))*100
    dtc_test_acc = accuracy_score(y_test, y_pred)*100

    text.delete("1.0", tk.END)

    # Print and display the results
    print(f"Training accuracy of Decision Tree is : {dtc_train_acc}")
    print(f"Test accuracy of Decision Tree is : {dtc_test_acc}")
    text.insert(tk.END, f"Training accuracy of Decision Tree is : {dtc_train_acc}\n")
    text.insert(tk.END, f"Test accuracy of Decision Tree is : {dtc_test_acc}\n")

    # Print confusion matrix and classification report
    print(confusion_matrix(y_test, y_pred))
    text.insert(tk.END, f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")

    print(classification_report(y_test, y_pred))
    text.insert(tk.END, f"Classification Report:\n{classification_report(y_test, y_pred)}\n")



# Function for Running Random Forest Classifier
def Run_RandomForest():
    global X_train, X_test, y_train, y_test, X, rand_clf_train_acc  # Ensure global access

    # Initialize the Random Forest Classifier
    rand_clf = RandomForestClassifier(criterion='entropy', max_depth=10, max_features='sqrt', 
                                      min_samples_leaf=1, min_samples_split=3, n_estimators=140)
    rand_clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = rand_clf.predict(X_test)

    # Calculate accuracy
    rand_clf_train_acc = accuracy_score(y_train, rand_clf.predict(X_train))*100
    rand_clf_test_acc = accuracy_score(y_test, y_pred)*100

    text.delete("1.0", tk.END)

    # Print and display the results
    print(f"Training accuracy of Random Forest is : {rand_clf_train_acc}")
    print(f"Test accuracy of Random Forest is : {rand_clf_test_acc}")
    text.insert(tk.END, f"Training accuracy of Random Forest is : {rand_clf_train_acc}\n")
    text.insert(tk.END, f"Test accuracy of Random Forest is : {rand_clf_test_acc}\n")

    # Print confusion matrix and classification report
    print(confusion_matrix(y_test, y_pred))
    text.insert(tk.END, f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")

    print(classification_report(y_test, y_pred))
    text.insert(tk.END, f"Classification Report:\n{classification_report(y_test, y_pred)}\n")



# Function for Running Logistic Regression
def Run_LogisticRegression():
    global X_train, X_test, y_train, y_test, X, log_reg_train_acc  # Ensure global access

    # Initialize the Logistic Regression model
    log_reg = LogisticRegression(max_iter=1000)  # Set a higher max_iter if convergence is not reached
    log_reg.fit(X_train, y_train)

    # Predict on the test set
    y_pred = log_reg.predict(X_test)

    # Calculate accuracy
    log_reg_train_acc = accuracy_score(y_train, log_reg.predict(X_train))*100
    log_reg_test_acc = accuracy_score(y_test, y_pred)*100

    text.delete("1.0", tk.END)

    # Print and display the results
    print(f"Training accuracy of Logistic Regression is : {log_reg_train_acc}")
    print(f"Test accuracy of Logistic Regression is : {log_reg_test_acc}")
    text.insert(tk.END, f"Training accuracy of Logistic Regression is : {log_reg_train_acc}\n")
    text.insert(tk.END, f"Test accuracy of Logistic Regression is : {log_reg_test_acc}\n")

    # Print confusion matrix and classification report
    print(confusion_matrix(y_test, y_pred))
    text.insert(tk.END, f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")

    print(classification_report(y_test, y_pred))
    text.insert(tk.END, f"Classification Report:\n{classification_report(y_test, y_pred)}\n")



def display_graphs():
    global svc_train_acc, dtc_train_acc, rand_clf_train_acc, log_reg_train_acc
    # Data
    models = ['SVM', 'Decision Tree', 'Random Forest', 'Logistic Regression']
    accuracy = [svc_train_acc, dtc_train_acc, rand_clf_train_acc, log_reg_train_acc]

    # Create Bar Graph
    plt.figure(figsize=(10, 5))

    # Bar Graph
    plt.subplot(1, 2, 1)  # First subplot (1 row, 2 columns, 1st position)
    plt.bar(models, accuracy, color=['blue', 'green', 'red', 'purple'])
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Models')
    plt.ylabel('Accuracy (%)')
    plt.xticks( rotation=45 )  # Rotate x-axis labels for better readability
    plt.ylim(0, 120)  # Y-axis from 0 to 100
    for i, v in enumerate(accuracy):
        plt.text(i, v + 1, str(v) + '%', ha='center', color='black')

    # Create Pie Chart
    plt.subplot(1, 2, 2)  # Second subplot (1 row, 2 columns, 2nd position)
    plt.pie(accuracy, labels=models, autopct='%1.1f%%', colors=['blue', 'green', 'red', 'purple'], startangle=90, explode=(0, 0.1, 0, 0))
    plt.title('Model Accuracy Distribution')

    # Display the graphs
    plt.tight_layout()
    plt.show()


def prediction():
    global X_test, y_test, svc, dtc, df
    try:  # Add a try-except block for better error handling
        # Encode string columns to numerical values
        label_encoder = LabelEncoder()
        for column in df.columns:
            if df[column].dtype == 'object':
                df[column] = label_encoder.fit_transform(df[column])

        X = df[['months_as_customer', 'policy_deductable', 'umbrella_limit',
                'capital-gains', 'capital-loss', 'incident_hour_of_the_day',
                'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'injury_claim', 'property_claim',
                'vehicle_claim']]
        y = df['fraud_reported']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        dtc = DecisionTreeClassifier()
        dtc.fit(X_train, y_train)
        print("DONE")

        predict_file = filedialog.askopenfilename(initialdir="dataset", title="Select a File", filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*")))

        if predict_file:
            input_df = pd.read_csv(predict_file)

            # Data Preprocessing for input file (same as training data)
            input_df.replace('?', np.nan, inplace=True)
            input_df.fillna(input_df.mode().iloc[0], inplace=True)  # Use mode for filling NaN

            label_encoder = LabelEncoder()
            for column in input_df.columns:
                if input_df[column].dtype == 'object':
                    input_df[column] = label_encoder.fit_transform(input_df[column])

            X_predict = input_df[['months_as_customer', 'policy_deductable', 'umbrella_limit',
                                   'capital-gains', 'capital-loss', 'incident_hour_of_the_day',
                                   'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'injury_claim', 'property_claim',
                                   'vehicle_claim']]

            if X_predict is not None:
                y_pred = dtc.predict(X_predict)

                text.delete("1.0", tk.END)
                text.insert(tk.END, "Prediction completed successfully.\n")
                text.insert(tk.END, "Predictions:\n")

                for i, prediction_value in enumerate(y_pred):
                    result = "YES-FRAUD DETECTED" if prediction_value == 1 else "NO"  # Display YES/NO
                    text.insert(tk.END, f"Prediction {i+1}: {result}\n")

            else:
                text.delete('1.0', tk.END)
                text.insert(tk.END, "Error: Input data format mismatch or no data.\n")

        else:
            text.delete('1.0', tk.END)  # Clear any previous messages
            text.insert(tk.END, "No file selected.\n") # Inform the user file not selected.

    except Exception as e:  # Catch any exceptions during the process
        text.delete('1.0', tk.END)
        text.insert(tk.END, f"An error occurred: {e}\n")  # Display error message


    

# Function to open the main page
def open_main_page():
    global text, pathlabel
    main_window = tk.Tk()
    main_window.title("Fraud Detection & Analysis")
    main_window.geometry("1200x700")
    main_window.configure(bg="#2c3e50")

    # Main Page Title with Animation
    title_text = "FRAUD DETECTION AND ANALYSIS FOR INSURANCE CLAIM USING MACHINE LEARNING"
    title_label = tk.Label(
        main_window, text="", font=("Helvetica", 18, "bold"), fg="white", bg="#2c3e50"
    )
    title_label.pack(pady=20)
    animate_title(title_label, title_text)

    # Button Section
    button_frame = tk.Frame(main_window, bg="#2c3e50")
    button_frame.pack(pady=20)

    button_texts = [
        "Upload Dataset", "Preprocess Data", "Train_Test_Model", "Run_SVM",
        "Run_DecisionTree", "Run_RandomForest", "Run_LogisticRegression", "display_graphs",
        "prediction"]

    buttons = []
    for i, text in enumerate(button_texts):
        btn = tk.Button(
            button_frame, text=text, font=("Arial", 12, "bold"), width=20, height=2,
            bg="#3498db", fg="white", relief="raised", cursor="hand2"
        )
        btn.grid(row=i // 5, column=i % 5, padx=10, pady=10, sticky="ew")
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        buttons.append(btn)

    # Assign Upload Dataset Function to Button
    buttons[0].config(command=upload)
    buttons[1].config(command=preprocess)  
    buttons[2].config(command=train_test_split_func)
    buttons[3].config(command=Run_SVM) 
    buttons[4].config(command=Run_DecisionTree)  
    buttons[5].config(command=Run_RandomForest)  
    buttons[6].config(command=Run_LogisticRegression)
    buttons[7].config(command=display_graphs)
    buttons[8].config(command=prediction)









    # Label to Show File Path
    pathlabel = tk.Label(main_window, text="", font=("Arial", 10), bg="#2c3e50", fg="white")
    pathlabel.pack()

    # Large Text Box
    text = tk.Text(main_window, height=18, width=150, font=("Arial", 12), bg="#ecf0f1")
    text.pack(pady=20)

    main_window.mainloop()

# Login Window
login_window = tk.Tk()
login_window.title("Admin Login")
login_window.geometry("500x400")
login_window.configure(bg="#34495e")

# Animated Title in Login Page
login_title_text = "FRAUD DETECTION SYSTEM - ADMIN LOGIN"
login_title_label = tk.Label(
    login_window, text="", font=("Helvetica", 16, "bold"), fg="white", bg="#34495e"
)
login_title_label.pack(pady=15)
animate_title(login_title_label, login_title_text)

# Login Form
tk.Label(login_window, text="Username:", font=("Arial", 12), bg="#34495e", fg="white").pack()
username_entry = tk.Entry(login_window, font=("Arial", 12))
username_entry.pack(pady=5)

tk.Label(login_window, text="Password:", font=("Arial", 12), bg="#34495e", fg="white").pack()
password_entry = tk.Entry(login_window, font=("Arial", 12), show="*")
password_entry.pack(pady=5)

# Login Button with Animation
login_button = tk.Button(
    login_window, text="Login", font=("Arial", 12, "bold"),
    bg="#27ae60", fg="white", width=15, command=check_login
)
login_button.pack(pady=15)
login_button.bind("<Enter>", on_enter)
login_button.bind("<Leave>", on_leave)

login_window.mainloop()
