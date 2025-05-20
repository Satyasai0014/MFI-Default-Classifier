import streamlit as st
import math
import pandas as pd
import datetime
import sqlite3
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(layout="wide")

# Database setup
def init_db():
    conn = sqlite3.connect("emi_schedule.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS payment_schedule (
            customer_id TEXT,
            period INTEGER,
            dpd TEXT,
            amount_outstanding REAL,
            interest REAL,
            principal_paid REAL,
            principal_outstanding REAL,
            cumulative_interest REAL,
            interest_income_outstanding REAL,
            emi_to_be_paid REAL,
            date_of_payment TEXT
            
        )
    """)
    conn.commit()
    conn.close()

# Save payment schedule to the database
def save_schedule_to_db(customer_id, payment_schedule):
    conn = sqlite3.connect("emi_schedule.db")
    cursor = conn.cursor()

    # Delete existing schedule for the customer
    cursor.execute("DELETE FROM payment_schedule WHERE customer_id = ?", (customer_id,))

    # Insert new data
    for _, row in payment_schedule.iterrows():
        cursor.execute("""
            INSERT INTO payment_schedule (
                customer_id, period, amount_outstanding, interest, principal_paid,
                principal_outstanding, cumulative_interest, interest_income_outstanding,
                emi_to_be_paid, date_of_payment ,dpd
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            customer_id, row['Period'], row['Amount Outstanding'], row['Interest'], row['Principal Paid'],
            row['Principal Outstanding'], row['Cumulative Interest'], row['Interest Income Outstanding'],
            row['EMI to be Paid'], row['Date of Payment'], row['DPD']
        ))
    conn.commit()
    conn.close()

# Load payment schedule from the database
def load_schedule_from_db(customer_id):
    conn = sqlite3.connect("emi_schedule.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM payment_schedule WHERE customer_id = ?", (customer_id,))
    rows = cursor.fetchall()
    conn.close()

    if rows:
        columns = [
            'customer_id', 'Period', 'Amount Outstanding', 'Interest', 'Principal Paid',
            'Principal Outstanding', 'Cumulative Interest', 'Interest Income Outstanding',
            'EMI to be Paid', 'Date of Payment','DPD'
        ]
        return pd.DataFrame(rows, columns=columns)
    return None

# Function to calculate EMI
def calculate_emi(principal, rate, tenure, payment_frequency):
    if payment_frequency == "Daily":
        rate = rate / (365 * 100)
    elif payment_frequency == "Biweekly":
        rate = rate / (26 * 100)
    elif payment_frequency == "Weekly":
        rate = rate / (52 * 100)
    elif payment_frequency == "Monthly":
        rate = rate / (12 * 100)

    emi = (principal * rate * math.pow(1 + rate, tenure)) / (math.pow(1 + rate, tenure) - 1)
    return emi

# Function to generate payment schedule
def generate_payment_schedule(principal, rate, tenure, payment_frequency, start_date):
    emi = calculate_emi(principal, rate, tenure, payment_frequency)
    schedule = []

    principal_outstanding = principal
    cumulative_interest = 0

    for i in range(tenure):
        if payment_frequency == "Daily":
            interval_days = 1
            interest = principal_outstanding * (rate / (365 * 100))
        elif payment_frequency == "Biweekly":
            interval_days = 14
            interest = principal_outstanding * (rate / (26 * 100))
        elif payment_frequency == "Weekly":
            interval_days = 7
            interest = principal_outstanding * (rate / (52 * 100))
        elif payment_frequency == "Monthly":
            interval_days = 30
            interest = principal_outstanding * (rate / (12 * 100))

        principal_paid = emi - interest
        principal_outstanding -= principal_paid
        cumulative_interest += interest

        schedule.append({
            'Period': i + 1,
            'DPD': '0 DPD',  # Default value
            'Amount Outstanding': principal_outstanding + principal_paid,
            'Interest': round(interest, 3),
            'Principal Paid': round(principal_paid, 3),
            'Principal Outstanding': round(principal_outstanding, 3),
            'Cumulative Interest': round(cumulative_interest, 3),
            'Interest Income Outstanding': round(interest * (tenure - i), 3),
            'EMI to be Paid': round(emi, 3),
            'Date of Payment': (start_date + datetime.timedelta(days=interval_days * i)).strftime('%Y-%m-%d'),
            
        })
    return pd.DataFrame(schedule)
# Fetch all distinct customer IDs
def fetch_all_customers():
    conn = sqlite3.connect("emi_schedule.db")
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT customer_id FROM payment_schedule")
    customers = [row[0] for row in cursor.fetchall()]
    conn.close()
    return customers

# Fetch DPD summary
def fetch_dpd_summary():
    conn = sqlite3.connect("emi_schedule.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT dpd, COUNT(DISTINCT customer_id) AS count
        FROM payment_schedule
        WHERE dpd IS NOT NULL
        GROUP BY dpd
    """)
    dpd_summary = cursor.fetchall()
    conn.close()
    return dpd_summary

# Fetch details for customers in a specific DPD category
def fetch_customers_by_dpd(dpd_status):
    conn = sqlite3.connect("emi_schedule.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT customer_id
        FROM payment_schedule
        WHERE dpd = ?
    """, (dpd_status,))
    customers = [row[0] for row in cursor.fetchall()]
    conn.close()
    return customers

# Prepare DPD summary based on the latest DPD status of each customer
def prepare_latest_dpd_summary():
    conn = sqlite3.connect("emi_schedule.db")
    cursor = conn.cursor()

    # Fetch the latest DPD status for each customer
    query = """
        SELECT customer_id, dpd 
        FROM (
            SELECT customer_id, dpd, MAX(period) AS latest_period
            FROM payment_schedule
            GROUP BY customer_id
        )
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    # Summarize DPD counts
    dpd_summary = {}
    for customer_id, dpd_status in rows:
        dpd_status = dpd_status if dpd_status else "No DPD"
        if dpd_status not in dpd_summary:
            dpd_summary[dpd_status] = []
        dpd_summary[dpd_status].append(customer_id)

    return dpd_summary

# Function to display all customers in a popover
def show_all_customers():
    conn = sqlite3.connect("emi_schedule.db")
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT customer_id FROM payment_schedule")
    all_customers = [row[0] for row in cursor.fetchall()]
    conn.close()
    return all_customers

def generate_customer_id():
    # Get the current year and extract the last two digits
    year_suffix = str(datetime.date.today().year)[-2:]

    # Load the last used customer ID from the database
    last_customer_id = get_last_customer_id_from_db()

    # Generate the next customer ID
    if last_customer_id is None or not last_customer_id.startswith(year_suffix):
        # Start a new sequence for the current year
        new_customer_id = int(year_suffix + "0001")
    else:
        # Increment the sequence for the current year
        new_customer_id = int(last_customer_id) + 1

    return new_customer_id

# Main application
def main():
    init_db()
    # logo = "logo.png"
    st.subheader("", divider="rainbow")
    # st.image(logo, use_column_width=100)
    col1, col2 = st.columns([1, 4])  # Adjust ratio as needed
    
    with col1:
        st.image("logo.png", width=120)  # Replace "logo.png" with your logo path

    with col2:
        
        # st.title("MFI Credit Risk Management Tool - Default Classifier")
        st.markdown(
        """
        <div style="text-align: left; font-size: 32px; font-weight: bold;">
            MFI Credit Risk Management Tool  - <br>  A  Default Classifier
        </div>
        """,
        unsafe_allow_html=True
        )
        
        # st.title("  MFI  Credit Risk Management Tool - Default Classifier")
    # st.subheader('',divider="rainbow")

    tab1, tab2, tab3 = st.tabs(["About","Default Classifier","Exposure Calculator"])
    # tab1, tab2, tab3,tab4,tab5 = st.tabs(["Home","New User Creation", "Exposure Calculator", "Portfolio View","Classifier"])
    # import streamlit as st  

    with tab1:

        sub_tabs = st.tabs(["Home", "About MFI","User Manual","Research Framework","Our Team","Research Publications"])
       
        
        with sub_tabs[0]:
            st.image("MFI web app photo.png")
            # Expanders for More Details

        with sub_tabs[1]:
            st.subheader("FAQ")
            with st.expander(" What is Microfinance?"):
                st.write("""
                    `Microfinance` is a financial service offered by several financial institutions to individuals or groups who are not included in conventional banking services
                """)
    
            with st.expander(" Who are the players in MFI ?"):
                st.write("""
                               The microfinance sector in India is served by a diverse set of institutions that cater to the financial needs of low-income individuals. These players differ in their regulatory structure, outreach, and business models. Here's a breakdown of the key participants:
            
            - **`NBFC-MFIs (Non-Banking Financial Companies - Microfinance Institutions)`**: 39%  
              These are specialized financial institutions focused primarily on microfinance. They have the largest share in the sector.
            
            - **`Banks`**: 33%  
              Commercial banks play a major role by directly offering microloans or lending to MFIs for onward lending.
            
            - **`SFBs (Small Finance Banks)`**: 16%  
              SFBs are newer entities aimed at promoting financial inclusion, and they have a growing presence in microfinance.
            
            - **`NBFCs (Non-Banking Financial Companies)`**: 11%  
              Though not exclusively focused on microfinance, these entities contribute significantly through partnerships and lending.
            
            - **`Others`**: 1%  
              This includes cooperatives, NGOs, and other informal players involved in micro-lending.
            
            These players collectively form the backbone of India’s financial inclusion efforts by expanding access to credit and empowering underserved communities.
        
                """)
    
            with st.expander("What are the Products of MFI ? "):
                st.write("""
                    Core Products Offered by Microfinance Institutions (MFIs)

                    Microfinance Institutions play a vital role in promoting financial inclusion by offering a suite of tailored financial products designed specifically for underserved and low-income segments of the population. The core offerings include:
                    
                    - **`Micro Credit`**: Small, collateral-free loans provided to individuals or groups to support income-generating activities, business expansion, or emergency needs.
                    
                    - **`Micro Savings`**: Secure and accessible savings options that encourage financial discipline and help customers build a financial cushion for the future.
                    
                    - **`Micro Insurance`**: Affordable insurance solutions that protect low-income households against risks such as health issues, accidents, and natural disasters.
                    
                    - **`Remittance Services`**: Reliable and cost-effective money transfer services that enable individuals to send and receive funds, especially important for migrant workers and rural families.
                    
                    These products collectively empower communities by fostering self-reliance, enabling asset creation, and reducing financial vulnerability.

                """)
            st.markdown("------")
            with st.expander(" 🔍  Quick Links "):
                st.write(""" 
                    ### 🔗 Useful Links

                    - Reserve Bank of India (RBI) : [(https://www.rbi.org.in/)] 
                    - Reserve Bank of India (RBI) NEW website : [https://website.rbi.org.in/en/web/rbi]
                    - SA-DHAN : [(https://www.sa-dhan.net/)]
                    - Microfinance Institutions Network (MFIN) : [(https://mfinindia.org/)]
                    - Ministry of Corporate Affairs (MCA) : [(https://www.mca.gov.in/)]
                    - Insurance Regulatory and Development Authority of India (IRDAI) : [(https://irdai.gov.in/)]
                    - Pradhan Mantri Jeevan Jyoti Bima Yojana (PMJJBY) : [(https://www.myscheme.gov.in/schemes/pmjjby)]
                    - Department of Financial Services, India : [(https://financialservices.gov.in/beta/en)]
                """)

        # Tab 2: Create Payment Schedule
        # Tab 2: Create Payment Schedule
        # Tab 2: Create Payment Schedule
        with sub_tabs[2]:
            # st.write("User Manual come here..!")
            st.title("User Manual")

            st.write("This manual contains a step-by-step guide to use this tool.The tool has three main tabs")

            # ABOUT in big letters
            st.markdown("<h2 style='text-align: left; color: teal;'>ABOUT</h2>", unsafe_allow_html=True)
            
            st.markdown("""
                               
            
            This section contains three sub-tabs:
            
            - **Home**: Provides detailed information on the objectives, innovation, users and features of this tool
            - **About MFI**: Gives background on Microfinance Institutions (MFIs) and includes useful resources and quick links for further reading.
            - **Research Framework**: Gives the overall framework used for this tool.
            - **Our Team**: Introduces the team behind this tool.
            - **Research Publications**: Provides all the information about the research publications.
            ---
            """)
            
            
            # Default Classifier in big letters
            st.markdown("<h2 style='text-align: left; color: teal;'>Default Classifier</h2>", unsafe_allow_html=True)
            
            st.markdown("""
                               
            
            This tab contains the detailed description about the model which is running behind the servers for making prediction.This section contains five sub-tabs:
            
            - **About Default Classifier**: Contains information about the tab ***“Default Classifier”***. Overview of the model and its purpose.
            - **Dataset Description**: Preview of the data and explanation of each feature.
            - **Mehtodology**: Step-by-step approach used to build the classifier.
            - **Data Visualization**: Interactive plots showing feature distributions and class counts.
            - **Run Model**: Input interface for predictions using top 8 selected features. Applies pre-trained scaler, encoder, and model to return output with probability.
            ---
            """)
            # Exposure Calculator in big letters
            st.markdown("<h2 style='text-align: left; color: teal;'>Exposure Calculator</h2>", unsafe_allow_html=True)
            
            st.markdown("""
                               
            
            This section contains four sub-tabs:
            
            - **About Exposure Calculator**: Provides an overview of the ***“Exposure Calculator”***, explaining its objective and relevance.
            - **New User Creation**: Allows users to create a new customer profile, generate their payment schedule, and store the information in the database.
            - **Expoure Calculation**:  Enables retrieval of a customer's payment schedule from the database. Loan Officers can update the Days Past Due (DPD), which will automatically compute the total exposure.
            - **Portfolio View**: Displays a comprehensive view of all customers in the database, including their respective DPD status and a summary of customer distribution.

            
            ---
            """)
        with sub_tabs[3]:
            st.write("Research Framework will come here..!")
        with sub_tabs[4]:
            # st.write("Team Details will come here..!")
            with st.container():
                st.markdown("""
                    <div style="margin-bottom: 40px;">
                        <h2 style="color: #0e76a8; margin-bottom: 5px;">Aparna V</h2>
                        <p style="font-size: 16px; line-height: 1.6;">
                            Mrs. Aparna V is an Assistant Professor in Department of Management and Commerce at Sri Satya Sai Institute of Higher Learning. From the past eight years, she is teaching papers in the area of financial management and Accounting related papers. She is currently perusing a Ph.D. in Credit Risk Management in Microfinance Institutions.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                    <div style="margin-bottom: 40px;">
                        <h2 style="color: #0e76a8; margin-bottom: 5px;">Dr. C. Jayashree</h2>
                        <p style="font-size: 16px; line-height: 1.6;">
                            Dr. Jayashree specializes in the area of marketing and has a strong background in accounting and finance, with nearly a decade of teaching experience. She has also worked as an accountant and administrative secretary at a private company in Chennai. Dr. Jayashree holds a Ph.D. degree from the University of Madras and has numerous national and international publications to her credit. Her research interests focus on fintech solutions and their impact in the Indian context.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                    <div style="margin-bottom: 40px;">
                        <h2 style="color: #0e76a8; margin-bottom: 5px;">Satya Sai Mudigonda, AIAI</h2>
                        <p style="font-size: 16px; line-height: 1.6;">
                            Satya Sai is a Professor of Practice and Coordinator at the Center of Excellence in Actuarial Data Science, Sri Sathya Sai Institute of Higher Learning (SSSIHL). He is an Associate of the Institute of Actuaries of India, with over 30 years of experience as a Senior Tech Actuarial Consultant, he has managed multi-million-dollar international assignments for major insurers. He has guided three PhD scholars and has numerous international journal publications in Actuarial Data Science. His research focuses on Fraud Detection, Crop Insurance, and Group Health Insurance.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
    with tab2:
        sub_tabs = st.tabs(["About the Classifier","Data Description","Methodology", "Data Visualization", "Run Model"])
        with sub_tabs[0]:
            st.markdown("## Default Classifier")
    
            st.markdown("""
                Our **Default Classifier** is a smart, data-driven prediction model designed to assess the risk of loan default.  
                It helps financial institutions identify potential defaulters early and take proactive risk mitigation measures.
                """)
            
            st.markdown("---")
                
            st.markdown("###  What is a Default?")
            st.markdown("""
                A **default** occurs when a borrower fails to make required loan repayments within the agreed time frame.  
                Understanding and predicting defaults is crucial to:
                - Maintain portfolio health  
                - Ensure repayment discipline  
                - Improve financial outreach strategies
                """)
            
            st.markdown("---")
            
            st.markdown("###  How the Model Works")
            st.markdown("""
                The classifier uses a **Decision Tree algorithm** trained on historical borrower data to classify whether a loan applicant is likely to:
                - ✅ **Repay** the loan on time (*Not Default*), or
                - ❌ **Fail** to repay the loan (*Default*)
            
                It processes borrower and loan-related information through:
                - A **MinMaxScaler** for numerical inputs  
                - **LabelEncoders** for categorical values  
                - A trained **machine learning model** that outputs:
                  - A **prediction label**: Default or Not Default  
                  - A **confidence score** as a probability  
                """)
            
            st.markdown("---")
            
            st.markdown("### 🧾 Features Used in Prediction")
            
            st.markdown("""
                <style>
                .feature-table {
                    width: 100%;
                    border-collapse: collapse;
                }
                .feature-table th, .feature-table td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }
                .feature-table th {
                    background-color: #f2f2f2;
                    color: #333;
                }
                </style>
            
                <table class="feature-table">
                    <tr>
                        <th>Feature</th>
                        <th>Description</th>
                    </tr>
                    <tr><td>Age</td><td>Age of the applicant</td></tr>
                    <tr><td>Occupation</td><td>Main source of livelihood (e.g., Agriculture, Labour, Business)</td></tr>
                    <tr><td>Education Level</td><td>Highest level of formal education completed</td></tr>
                    <tr><td>Household Size</td><td>Number of people in the applicant's household</td></tr>
                    <tr><td>Income Level (INR/month)</td><td>Monthly income of the applicant</td></tr>
                    <tr><td>Loan Amount (INR)</td><td>Amount of loan applied for</td></tr>
                    <tr><td>Loan Tenure (Months)</td><td>Duration of the loan repayment period</td></tr>
                    <tr><td>Interest Rate (%)</td><td>Annual rate of interest charged on the loan</td></tr>
                </table>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            st.markdown("###  Prediction Output")
            st.markdown("""
                After running the model, users receive:
                - A **visual prediction**:  
                  - 🟥 **Default**  
                  - 🟩 **Not Default**
                - A **personalized summary** of the applicant’s profile
                - A **confidence score** displayed as a percentage
            
                #### 📋 Example Output:
                > The applicant aged 30 with an occupation of *Agriculture*, education level *Primary*, household size of 3,  
                > monthly income of ₹5,000, applied for a loan of ₹5,00,000 for a tenure of 12 months at an interest rate of 25.0%.  
                > Based on the model’s prediction, the applicant is likely to **Not Default** with a probability of **91.34%**.
                """)
            
            st.markdown("---")
            
            st.markdown("###  Why It Matters")
            st.markdown("""
                - Helps identify **high-risk borrowers** before disbursing a loan  
                - Supports **credit decisioning** for loan officers and field agents  
                - Enhances **portfolio monitoring** by flagging possible defaulters  
                - Provides **data-backed justification** for approvals or rejections  
                """)
        with sub_tabs[1]:
            st.header("Dataset Description")
            df = pd.read_excel("Data_MFI.xlsx")
        
                # Display the first few rows
            st.write("### Data Preview")
            st.dataframe(df.head())
            if st.checkbox("View Full Data"):
                st.write("## Full Dataset")
                st.dataframe(df)
            data_desc = {
            "Attribute": [
                "Age", "Gender", "Marital Status", "Education Level", "Household Size", "Occupation (Income Source)",
                "Income Level", "Savings Behaviour", "Loan Repayment History", "Days Past Due (DPD)", "Previous Delinquencies",
                "Multiple Borrowing", "Rural or Semi-Urban Classification", "Employment Stability", "Membership in SHGs/JLGs",
                "Community Participation", "Loan Amount", "Loan Tenure", "Interest Rate", "Repayment Frequency", "Type of Loan Product",
                "Loan Cycle", "Restructured Loans"
            ],
            "Description": [
                "Age of the loan borrower.", "Gender of the loan borrower.", "Marital status of the loan borrower.",
                "Highest level of education attained by the loan borrower.", "Number of members in the borrower's household.",
                "Primary source of income for the borrower.", "Monthly income of the borrower in INR.",
                "Pattern of savings maintained by the borrower.", "Borrower’s past repayment behavior in previous loans.",
                "Number of days the borrower has delayed in repaying the loan.", "Number of times the borrower has defaulted or delayed payments in the past.",
                "Indicates whether the borrower has taken multiple loans simultaneously.",
                "Classification of the borrower’s place of residence as rural or semi-urban.",
                "Stability of the borrower’s employment or source of income.",
                "Indicates whether the borrower is a member of Self-Help Groups (SHG) or Joint Liability Groups (JLG).",
                "Level of the borrower’s involvement in community activities.", "The amount borrowed by the loan applicant.",
                "Duration for which the loan is granted, in months.", "The percentage of interest charged on the loan.",
                "The schedule at which the borrower is required to repay the loan.", "The specific type of loan availed by the borrower.",
                "Indicates whether the borrower is taking a loan for the first time or is a repeat borrower.",
                "Indicates whether the loan has been restructured due to financial difficulties."
            ]
        }

            # Create a DataFrame
            df_desc = pd.DataFrame(data_desc)  
            st.write("### About the Data")
            st.write("This table provides an overview of the attributes and their descriptions in the dataset.")
            st.dataframe(df_desc, use_container_width =True)
        with sub_tabs[2]:
            st.header("Methodolody")
            st.image("methodology_updated.jpg")

            with st.expander(" What is Data Preprocessing ?"):
                st.write("""
                        `Data Preprocessing` is the process of cleaning, transforming, and organizing raw data to make it suitable for analysis or machine learning models. It involves handling missing values, removing duplicates, converting data into a usable format, and scaling numerical values to improve the model's performance and accuracy.

                """)

            with st.expander(" What is Feature Engineering ?"):
                st.write("""
                        `Feature Engineering` is the process of creating new features or modifying existing ones to improve the performance of a machine learning model. It involves selecting, transforming, or generating relevant information from raw data to help the model make better predictions.

                """)

            with st.expander(" What is Label Encoding and Min Max Scaling?"):
                st.write("""
                        `Label encoding` is a technique used to convert categorical (text-based) data into numerical values so that machine learning models can process it. Each unique category is assigned a unique integer.
                        
                       `Min-Max Scaling` (also called Normalization) is a technique used to transform numerical data into a fixed range, usually [0, 1]. It ensures that all values are on the same scale, making machine learning models work better.

                """)
                st.subheader("Label Encoding")
                st.image("labelencoding.jpg")
                st.subheader("MinMax Scaling")
                st.image("minmaxscalar.png")
    
            with st.expander(" What is SMOTE ?"):
                st.write(""" 
                        `SMOTE (Synthetic Minority Over-sampling Technique)` is a technique used in machine learning to handle imbalanced datasets by generating synthetic samples for the minority class. Instead of simply duplicating existing samples, SMOTE creates new synthetic data points to balance the dataset, making models perform better.                 
                    
                """)
            with st.expander("What is Feature Selection?"):
                st.write("""
                    `Feature Selection` is the process of identifying and selecting the most relevant input variables (features) from a dataset that contribute the most to the prediction or analysis task.  
                    
                    It helps in:
                    - Reducing overfitting by eliminating noisy or irrelevant data
                    - Improving model performance and accuracy
                    - Reducing training time and computational cost
                    - Enhancing model interpretability
            
                    Common techniques include filter methods (e.g., correlation), wrapper methods (e.g., recursive feature elimination), and embedded methods (e.g., Lasso regression).
                """)

        
        with sub_tabs[3]:
            st.header("Data Visualization")
            
            try:
                # exceL_file_path = (r"C:\Users\Satya\Desktop\CADS Working\MFI\Data_MFI.xlsx")
                # df = pd.read_excel("Data_MFI.xlsx")
        
                # # Display the first few rows
                # st.write("### 📝 Data Preview")
                # st.dataframe(df.head())
        
                # Select a feature to visualize
                selected_feature = st.selectbox("Select a Feature to Visualize", df.columns)
        
                # Check feature type (Categorical or Numerical)
                if df[selected_feature].dtype == "object" or df[selected_feature].nunique() < 10:
                    # Categorical Feature: Show count plot
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.countplot(x=selected_feature, data=df, palette="viridis", ax=ax)
                    ax.set_title(f"Class Distribution of {selected_feature}")
                    ax.set_xlabel(selected_feature)
                    ax.set_ylabel("Count")
                    plt.xticks(rotation=30)
                    st.pyplot(fig)
        
                    # Display value counts as a table
                    st.write("### 🔢 Class Counts")
                    st.write(df[selected_feature].value_counts())
        
                else:
                    # Numerical Feature: Show histogram
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.histplot(df[selected_feature], bins=20, kde=True, color="blue", ax=ax)
                    ax.set_title(f"Distribution of {selected_feature}")
                    ax.set_xlabel(selected_feature)
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig)
        
                    # Display basic statistics
                    st.write("### 📊 Summary Statistics")
                    st.write(df[selected_feature].describe())
        
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
                
        
        user_data = []
        with sub_tabs[4]:
            
            
            # Load the saved MinMaxScaler, LabelEncoders, and Models
            # Load the saved MinMaxScaler, LabelEncoders, and Models
            scaler = joblib.load("latest-scaler.pkl")
            label_encoders = joblib.load("latest-label_encoder.pkl")
            model1 = joblib.load("latest_DT_model.joblib")  # Decision Tree Model
            
            st.header("Default Prediction")
            
            # Ensure session state contains user data
            if "user_data" not in st.session_state:
                st.session_state.user_data = None
            
            # Define input fields
            age = st.slider("Age", min_value=18, max_value=60, value=30, step=1)
            
            occupation_options = [
                "Labour", "Business", "Agriculture", "Dairy", "House-wife", "Salaried-Others",
                "Shop Owner", "Others", "Performing Arts", "Salaried-Govt", "Professional",
                "Unemployed", "Retired/Pensioner", "Goat Rearing", "Student", "Driver",
                "Migrant Labour", "Fishing", "Small Industry", "Rental Income",
                "Working Abroad", "Agri Trading"
            ]
            occupation = st.selectbox("Occupation", occupation_options)
            
            education_level_options = ["Illiterate", "Primary", "Secondary", "Graduate"]
            education_level = st.selectbox("Education Level", education_level_options)
            
            household_size = st.slider("Household Size", min_value=2, max_value=8, value=3, step=1)
            income_level = st.number_input("Income Level (INR/month)", min_value=1000, max_value=20000, value=5000, step=500)
            loan_amount = st.number_input("Loan Amount (INR)", min_value=10000, max_value=200000, value=50000, step=5000)
            loan_tenure = st.slider("Loan Tenure (Months)", min_value=3, max_value=36, value=12, step=3)
            interest_rate = st.number_input("Interest Rate (%)", min_value=23.0, max_value=27.0, value=25.0, step=0.5)
            
            # Submit Button
            if st.button("Save"):
                st.session_state.user_data = {
                    "Age": age,
                    "Occupation": occupation,
                    "Education Level": education_level,
                    "Household Size": household_size,
                    "Income Level (INR/m)": income_level,
                    "Loan Amount (INR)": loan_amount,
                    "Loan Tenure (Months)": loan_tenure,
                    "Interest Rate (%)": interest_rate
                }
            
            if "user_data" in st.session_state and st.session_state.user_data:
                df_raw = pd.DataFrame([st.session_state.user_data])
                st.write("### 📝 Raw User Data")
                st.dataframe(df_raw)
            
                # Copy raw data for transformation
                df_transformed = df_raw.copy()
            
                # Apply MinMax Scaling to numerical features
                numerical_features = ["Age", "Household Size", "Income Level (INR/m)", "Loan Amount (INR)", "Loan Tenure (Months)", "Interest Rate (%)"]
                df_transformed[numerical_features] = scaler.transform(df_transformed[numerical_features])
            
                # Encode categorical features using LabelEncoder
                df_transformed["Occupation"] = label_encoders["Occupation"].transform(df_transformed["Occupation"])
                df_transformed["Education Level"] = label_encoders["Education Level"].transform(df_transformed["Education Level"])
            
                # Ensure final feature order
                feature_order = ["Age", "Education Level", "Household Size", "Occupation", "Income Level (INR/m)", "Loan Amount (INR)", "Loan Tenure (Months)", "Interest Rate (%)"]
                df_final = df_transformed[feature_order]
                
                if st.button("Run Model & Predict"):
                    try:
                        prediction = model1.predict(df_final)
                        print(prediction)
                        prediction_probs = model1.predict_proba(df_final)[:, 1]
                        print(prediction_probs)
            
                        prediction_labels = ["Default" if p == "Default" else "Not Default" for p in prediction]
            
                        # ✅ Display Prediction Result
                        st.markdown(
                            f"""
                            <div style="text-align: center; padding: 20px; background-color: {'#ffcccc' if prediction_labels[0] == 'Default' else '#ccffcc'};">
                                <h2 style="font-size: 36px; font-weight: bold;">
                                    {prediction_labels[0]}
                                </h2>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            
                        # ✅ Detailed Description
                        result_text = (
                            f"The applicant aged {df_raw.iloc[0]['Age']} with an occupation of {df_raw.iloc[0]['Occupation']}, "
                            f"education level {df_raw.iloc[0]['Education Level']}, household size of {df_raw.iloc[0]['Household Size']}, "
                            f"monthly income of INR {df_raw.iloc[0]['Income Level (INR/m)']}, applied for a loan of INR {df_raw.iloc[0]['Loan Amount (INR)']} "
                            f"for a tenure of {df_raw.iloc[0]['Loan Tenure (Months)']} months at an interest rate of {df_raw.iloc[0]['Interest Rate (%)']}%. "
                            f"Based on the model's prediction, the applicant is likely to **{prediction_labels[0]}** "
                            f"with a probability of **{prediction_probs[0]:.2%}**."
                        )
                        
                        if prediction_labels[0] == "Default":
                            st.error(result_text)
                        else:
                            st.success(result_text)
                    
                    except ValueError as e:
                        st.error(f"Model Prediction Error: {e}")

    
    with tab3:
        sub_tabs = st.tabs(["About Exposure Calculator","New User Creation", "Exposure Calculation", "Porfolio View"])

        with sub_tabs[0]:
            # st.write("Here the content about the exposure calculator will come")
           
            
            st.markdown(" ## Exposure Calculator")
            st.markdown("""
            <div style="font-family: 'Segoe UI', sans-serif; padding: 10px; line-height: 1.6;"> 
            <p>
            The <strong>Exposure Calculator</strong> is a core feature of our loan management system designed to provide a transparent view of the customer’s outstanding obligations at any point in time.
            </p>
            
            <h3>What is Exposure?</h3>
            <p>
            In lending , <strong>exposure</strong> refers to the potential risk a lender faces from a borrower's failure to repay a loan, representing the total amount the lender could lose if the borrower defaults.
            </p> 
          
            
            <h3> How Does It Work?</h3>
            <p>Our exposure calculation engine dynamically computes exposure based on the payment schedule and actual payments made by the borrower. Here's what it considers:</p>
            
            <ul>
              <li><strong>Principal Due:</strong> The portion of the loan amount that was scheduled to be paid.</li>
              <li><strong>Interest Due:</strong> Interest applicable for the period based on the outstanding balance.</li>
              <li><strong>Outstanding Amount:</strong> Any unpaid principal or interest carried forward from previous periods.</li>
              <li><strong>DPD (Days Past Due):</strong> Delayed payments increase the overdue amount and exposure.</li>
            </ul>
            
            <p>
            Each time the schedule is updated or a payment is made, the calculator recalculates exposure accordingly — helping stakeholders track risk in <strong>real-time</strong>.
            </p>
            
            <h3> Why It Matters</h3>
            <ul>
              <li>Assists in <strong>risk assessment</strong> and <strong>portfolio monitoring</strong>.</li>
              <li>Useful for <strong>collections</strong>, <strong>NPA tracking</strong>, and <strong>credit decisioning</strong>.</li>
              <li>Provides a clear view of <strong>who owes how much and why</strong>.</li>
            </ul>
            
            </div>
            """, unsafe_allow_html=True)
        with sub_tabs[1]:
            st.header("Payment Schedule Creation")
            
            if "valid_customer_id" not in st.session_state:
                st.session_state.valid_customer_id = None
    
    
            # Custom CSS for centering buttons
            st.markdown(
                """
                <style>
                .center-button {
                    display: flex;
                    justify-content: center;
                    margin-top: 10px;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
        
            # Customer ID Input
            if not st.session_state.valid_customer_id:
                customer_id = st.text_input("Enter Customer ID (6 Alphanumeric Characters)")
                st.caption("Examples: A1B2C3, X9Y8Z7, 1A2B3C")  # Examples below the text box
                
                # Validate Customer ID
                if st.button("Validate Customer ID"):
                    if len(customer_id) != 6 or not customer_id.isalnum():
                        st.error("Customer ID must be exactly 6 alphanumeric characters.")
                    else:
                        st.session_state.valid_customer_id = customer_id
                        st.success(f"A user is created with Customer ID: **{customer_id}**")
            
            # If a valid customer ID is entered, show the form to fill remaining details
            if st.session_state.valid_customer_id:
                st.markdown(f"### Customer ID: **{st.session_state.valid_customer_id}**")
                
                # Principal Input
                principal = st.number_input("Enter Principal Amount", min_value=1000, max_value=500000, value=1000, step = 1000)
                
                # Interest Rate Input
                interest_rate = st.number_input("Enter Interest Rate (%)", min_value=12.0, max_value=30.0, value=12.0, step = 0.5)
                
                # Payment Frequency Input (single-line layout)
                payment_frequency = st.radio(
                    "Choose Payment Frequency",
                    ("Daily", "Weekly", "Biweekly", "Monthly"),
                    horizontal=True
                )
                
                # Dynamic Tenure Input
                if payment_frequency == "Daily":
                    tenure_label = "Enter Tenure (in Days)"
                    max_tenure = 365
                elif payment_frequency == "Weekly":
                    tenure_label = "Enter Tenure (in Weeks)"
                    max_tenure = 156
                elif payment_frequency == "Biweekly":
                    tenure_label = "Enter Tenure (in Biweekly count)"
                    max_tenure = 78
                elif payment_frequency == "Monthly":
                    tenure_label = "Enter Tenure (in Months)"
                    max_tenure = 36
                
                tenure = st.number_input(
                    tenure_label, min_value=1, max_value=max_tenure, value=max_tenure // 2
                )
                
                if tenure < 1 or tenure > max_tenure:
                    st.error("Check the tenure. The entered value is out of range.")
                
                # Start Date Input
                start_date = st.date_input("Select Start Date", value=datetime.date.today())
                
                # Generate and Save Payment Schedule Button
                if st.button("Generate and Save Payment Schedule"):
                    schedule = generate_payment_schedule(
                        principal, interest_rate, tenure, payment_frequency, start_date
                    )
                    save_schedule_to_db(st.session_state.valid_customer_id, schedule)
                    st.success(f"Payment schedule for Customer ID **{st.session_state.valid_customer_id}** has been saved.")
                    st.subheader("Generated Payment Schedule")
                    st.dataframe(schedule)

        with sub_tabs[2]:
            
            st.header("Exposure Calculator")
    
            customer_id = st.text_input("Enter Customer ID to View/Edit Schedule")
        
            if st.button("Load Payment Schedule"):
                if customer_id:
                    schedule = load_schedule_from_db(customer_id)
        
                    if schedule is not None:
                        if 'Original Principal Paid' not in schedule.columns:
                            schedule['Original Principal Paid'] = 0.0  # Initialize with zeros or any default value
                        st.session_state.schedule = schedule
                    else:
                        st.error("No schedule found for the given Customer ID.")
        
            if "schedule" in st.session_state and st.session_state.schedule is not None:
                schedule = st.session_state.schedule

                schedule["DPD"] = schedule["DPD"].fillna("Select")
                gb = GridOptionsBuilder.from_dataframe(schedule)
                gb.configure_default_column(editable=True)
                gb.configure_column("DPD", 
                                    cellEditor="agSelectCellEditor", 
                                    cellEditorParams={"values": ["Select", "0 DPD", "1< 30 DPD"]}, 
                                    cellRenderer=JsCode("""
                                        function(params) {
                                            if (!params.value || params.value === 'Select') {
                                                return '';
                                            } else if (params.value === '0 DPD') {
                                                return '🟢';
                                            } else if (params.value === '1< 30 DPD') {
                                                return '🟡';
                                            }
                                        }
                                    """))
                gb.configure_column("Original Principal Paid", editable=True)  # Add Editable Column
        
                grid_options = gb.build()
        
                grid_response = AgGrid(
                    schedule,
                    gridOptions=grid_options,
                    update_mode=GridUpdateMode.MANUAL,  # Use manual mode to handle updates explicitly
                    height=600,
                    fit_columns_on_grid_load=True,
                    allow_unsafe_jscode=True,
                )
        
                # Update the session state with the modified grid data
                st.session_state.schedule = pd.DataFrame(grid_response["data"])
        
                # Adjust Outstanding Amounts based on Original Principal Paid
                if 'Original Principal Paid' in st.session_state.schedule.columns:
                    for index, row in st.session_state.schedule.iterrows():
                        original_paid = row['Original Principal Paid']
                        try:
                            original_paid = float(original_paid)  # Convert to float
                        except ValueError:
                            st.error(f"Invalid value for Original Principal Paid at index {index}: {row['Original Principal Paid']}")
                            continue  # Skip this iteration if conversion fails
                        # Assuming 'Principal Outstanding' is the column for outstanding principal
                        if original_paid > 0:
                            st.session_state.schedule.at[index, 'Principal Outstanding'] -= original_paid
                            # Apply interest logic here if needed, e.g., adjusting interest based on new principal outstanding
        
                if st.button("Save Changes"):
                    save_schedule_to_db(customer_id, st.session_state.schedule)  # Save to database
                    st.success(f"Payment schedule for Customer ID {customer_id} saved successfully!")
        
                # Calculate severity if '1< 30 DPD' is selected after adjustments
                exposure_data = st.session_state.schedule[st.session_state.schedule['DPD'] == '1< 30 DPD']
                if not exposure_data.empty:
                    principal_exposure = exposure_data['Principal Outstanding'].sum()
                    interest_exposure = exposure_data['Interest Income Outstanding'].sum()
                    exposure = principal_exposure + interest_exposure
                    
                    st.markdown(
                        f"<div style='color:red; font-size:18px; font-weight:bold;'>"
                        f" 🚨 Exposure Detected: </div>"
                        f"<div style='font-size:16px;'>Principal Outstanding: <b>{principal_exposure:.4f}</b></div>"
                        f"<div style='font-size:16px;'>Interest Income Outstanding: <b>{interest_exposure:.4f}</b></div>"
                        f"<div style='color:red; font-size:18px;'>Total Exposure: <b> {exposure:.4f}</b></div>",
                        unsafe_allow_html=True,
                    )

        with sub_tabs[3]:
            st.header("Portfolio View")

            # Fetch data for monitoring
            all_customers = fetch_all_customers()
            dpd_summary = fetch_dpd_summary()
    
            st.subheader("Customer Summary")
            total_customers = len(all_customers)
            st.write(f"**Total Customers:** {total_customers}")
    
            # Prepare DPD summary with customer details
            dpd_data = {"DPD Status": [], "Customer Count": [], "Customer IDs": []}
            
            for dpd_status, count in dpd_summary:
                dpd_status_label = dpd_status if dpd_status else "No DPD"
                customers_in_dpd = fetch_customers_by_dpd(dpd_status_label)  # Fetch customers for this DPD status
                dpd_data["DPD Status"].append(dpd_status_label)
                dpd_data["Customer Count"].append(count)
                dpd_data["Customer IDs"].append(", ".join(customers_in_dpd) if customers_in_dpd else "None")
            
            # Convert the data into a DataFrame
            dpd_df = pd.DataFrame(dpd_data)
            
            # Filter out rows where DPD status is "Select"
            dpd_df = dpd_df[dpd_df["DPD Status"] != "Select"]
            
            # Display counts and customers by DPD status
            # st.subheader("DPD Breakdown with Customer Details")
            # st.dataframe(dpd_df)
            
            # Optional: Highlight rows based on DPD Status dynamically for better visibility
            st.markdown("<h4>Dynamic Highlights</h4>", unsafe_allow_html=True)
            st.table(dpd_df.style.applymap(
                lambda x: "background-color: lightpink;" if "1< 30 DPD" in x else "background-color: lightgreen;", 
                subset=["DPD Status"]
            ))
    
            # Show dynamic chart
            st.subheader("DPD Status Visualization")
            fig = px.pie(dpd_df, names="DPD Status", values="Customer Count", title="DPD Distribution")
            st.plotly_chart(fig)
    
            # Display details of customers in each DPD category as a table
            st.subheader("Customers in Each DPD Category")
            
            # Prepare data for the table
            dpd_customer_data = []
            for dpd_status in dpd_data["DPD Status"]:
                display_status = "Total" if dpd_status == 'Select' else dpd_status
                customers_in_dpd = fetch_customers_by_dpd(dpd_status)
                dpd_customer_data.append({
                    "DPD Status": display_status,
                    "Customer Count": len(customers_in_dpd),
                    "Customer IDs": ", ".join(customers_in_dpd) if customers_in_dpd else "None"
                })
    
            # Create a DataFrame from the data
            dpd_customer_df = pd.DataFrame(dpd_customer_data)
            
            # Display the DataFrame as a table
            st.dataframe(dpd_customer_df)

    # Tab 3: View/Edit Payment Schedule
    # with tab3:
        

    # Tab 4: Administration Monitoring
    # with tab4:
        
    
    # Create an empty list to store user inputs
    

    # with tab5:
        
        




st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #f8f9fa;
            color: #6c757d;
            text-align: center;
            padding: 8px;
            font-size: 14px;
            box-shadow: 0px -1px 5px rgba(0, 0, 0, 0.1);
        }
    </style>
    <script>
        function placeFooter() {
            var footer = document.querySelector(".footer");
            var bodyHeight = document.body.scrollHeight;
            var windowHeight = window.innerHeight;
            if (bodyHeight < windowHeight) {
                footer.style.position = "fixed";
            } else {
                footer.style.position = "relative";
            }
        }
        window.onload = placeFooter;
        window.onresize = placeFooter;
    </script>


    <div class="footer">
        © 2025 An SSSIHL Product. All Rights Reserved.
    </div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
