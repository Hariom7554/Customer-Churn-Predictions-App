import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📊 Customer Churn Prediction System")
st.markdown("### ⚙️ Fast, Cached, and Interactive")

# ------------------------------------------------------------
# STEP 1: LOAD DATA
# ------------------------------------------------------------
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)

uploaded_file = st.sidebar.file_uploader("📂 Upload Dataset", type=["csv", "xlsx", "xls"])

if uploaded_file:
    df = load_data(uploaded_file)
    st.success("✅ Data Loaded Successfully!")
    st.dataframe(df.head())

    df.replace(" ", np.nan, inplace=True)
    df.dropna(inplace=True)

    # ------------------------------------------------------------
    # STEP 2: AUTO ENCODE + MODEL TRAINING (CACHED)
    # ------------------------------------------------------------
    @st.cache_resource
    def train_model(df, target_col):
        le_dict = {}
        for col in df.columns:
            if df[col].dtype == "object":
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                le_dict[col] = le

        X = df.drop(columns=[target_col])
        y = df[target_col]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        model = RandomForestClassifier(
            n_estimators=150, random_state=42, n_jobs=-1, max_depth=12
        )
        model.fit(X_train, y_train)
        return model, scaler, le_dict, X, y, X_train, X_test, y_train, y_test

    target_col = "Churn"
    if target_col not in df.columns:
        st.error("❌ 'Churn' column not found in dataset.")
        st.stop()

    if st.sidebar.button("🚀 Train Model"):
        with st.spinner("Training model..."):
            model, scaler, le_dict, X, y, X_train, X_test, y_train, y_test = train_model(df.copy(), target_col)
        st.session_state.update({
            "trained_model": model,
            "scaler": scaler,
            "le_dict": le_dict,
            "X": X,
            "y": y
        })
        st.success("✅ Model trained successfully!")

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.metric("Model Accuracy", f"{acc*100:.2f}%")

        st.write("### Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.write("### Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

    # ------------------------------------------------------------
    # STEP 3: VISUALIZATION SECTION (IMPROVED)
    # ------------------------------------------------------------
    st.sidebar.header("📊 Visualization")

    if len(df) > 10000:
        st.info("Large dataset detected — using 10,000 rows for faster plotting.")
        df_viz = df.sample(10000, random_state=42)
    else:
        df_viz = df.copy()

    df_viz_encoded = df_viz.copy()
    for col in df_viz_encoded.columns:
        if df_viz_encoded[col].dtype == "object":
            le = LabelEncoder()
            df_viz_encoded[col] = le.fit_transform(df_viz_encoded[col].astype(str))

    graph_type = st.sidebar.selectbox(
        "Select Graph Type",
        [
            "Correlation Heatmap",
            "Bar Plot",
            "Line Plot",
            "Pair Plot",
            "Box Plot",
            "Histogram",
            "Scatter Plot",
            "Count Plot",
            "Violin Plot",
            "Distribution Plot (KDE)"
        ]
    )

    st.subheader("📈 Data Visualization")

    # ------------------ CORRELATION HEATMAP -------------------
    if graph_type == "Correlation Heatmap":
        numeric_df = df_viz_encoded.select_dtypes(include=[np.number])
        if numeric_df.shape[1] > 1:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=False)
            st.pyplot(fig)

    # ------------------ BAR PLOT -------------------
    elif graph_type == "Bar Plot":
        x_col = st.selectbox("X-axis", df.columns)
        y_col = st.selectbox("Y-axis", df.columns)
        st.bar_chart(df.groupby(x_col)[y_col].count())

    # ------------------ LINE PLOT -------------------
    elif graph_type == "Line Plot":
        x_col = st.selectbox("X-axis", df.columns)
        y_col = st.selectbox("Y-axis", df.columns)
        st.line_chart(df[[x_col, y_col]].set_index(x_col))

    # ------------------ PAIR PLOT -------------------
    elif graph_type == "Pair Plot":
        cols = st.multiselect("Select features", df_viz_encoded.columns, default=df_viz_encoded.columns[:4])
        if len(cols) > 1:
            sns.pairplot(df_viz_encoded[cols])
            st.pyplot(plt)

    # ------------------ BOX PLOT -------------------
    elif graph_type == "Box Plot":
        col = st.selectbox("Select Column", df_viz_encoded.columns)
        fig, ax = plt.subplots()
        sns.boxplot(x=df_viz_encoded[col], ax=ax)
        st.pyplot(fig)

    # ------------------ HISTOGRAM -------------------
    elif graph_type == "Histogram":
        col = st.selectbox("Select Column", df_viz_encoded.columns)
        fig, ax = plt.subplots()
        sns.histplot(df_viz_encoded[col], kde=True, ax=ax)
        st.pyplot(fig)

    # ------------------ SCATTER PLOT -------------------
    elif graph_type == "Scatter Plot":
        x_col = st.selectbox("X-axis", df_viz_encoded.columns)
        y_col = st.selectbox("Y-axis", df_viz_encoded.columns)
        fig, ax = plt.subplots()
        sns.scatterplot(x=df_viz_encoded[x_col], y=df_viz_encoded[y_col], ax=ax)
        st.pyplot(fig)

    # ------------------ COUNT PLOT -------------------
    elif graph_type == "Count Plot":
        col = st.selectbox("Select Column", df.columns)
        fig, ax = plt.subplots()
        sns.countplot(x=df[col], ax=ax)
        st.pyplot(fig)

    # ------------------ VIOLIN PLOT -------------------
    elif graph_type == "Violin Plot":
        col = st.selectbox("Select Column", df_viz_encoded.columns)
        fig, ax = plt.subplots()
        sns.violinplot(y=df_viz_encoded[col], ax=ax)
        st.pyplot(fig)

    # ------------------ DISTRIBUTION PLOT -------------------
    elif graph_type == "Distribution Plot (KDE)":
        col = st.selectbox("Select Column", df_viz_encoded.columns)
        fig, ax = plt.subplots()
        sns.kdeplot(df_viz_encoded[col], fill=True, ax=ax)
        st.pyplot(fig)

    # ------------------------------------------------------------
    # STEP 4: MANUAL PREDICTION
    # ------------------------------------------------------------
    st.sidebar.header("🤖 Prediction Mode")
    mode = st.sidebar.radio("Choose Mode", ["Manual Entry", "Batch Prediction"])

    if mode == "Manual Entry":
        st.subheader("🧮 Predict for New Customer")

        model = st.session_state.get("trained_model")
        scaler = st.session_state.get("scaler")
        le_dict = st.session_state.get("le_dict")

        if not model:
            st.warning("⚠️ Please train the model first.")
        else:
            input_data = {}
            for col in st.session_state["X"].columns:
                if col in le_dict:
                    le = le_dict[col]
                    options = list(le.classes_)
                    if len(options) == 2:
                        val = st.radio(f"{col}", options, horizontal=True)
                    else:
                        val = st.selectbox(f"{col}", options)
                    encoded_val = le.transform([val])[0]
                    input_data[col] = encoded_val
                else:
                    input_data[col] = st.number_input(
                        f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean())
                    )

            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)

            if st.button("🔍 Predict Churn"):
                pred = model.predict(input_scaled)[0]
                prob = model.predict_proba(input_scaled)[0][1]

                if prob < 0.3:
                    risk = "🟢 Low Risk"
                elif prob < 0.6:
                    risk = "🟡 Medium Risk"
                else:
                    risk = "🔴 High Risk"

                readable_input = {
                    col: le_dict[col].inverse_transform([int(val)])[0]
                    if col in le_dict else val
                    for col, val in input_data.items()
                }

                st.write("### Customer Input Summary")
                st.table(pd.DataFrame(readable_input, index=[0]))

                if pred == 1:
                    st.error(f"⚠️ Customer likely to **CHURN** (Probability: {prob:.2f}) — {risk}")
                else:
                    st.success(f"✅ Customer NOT likely to churn (Probability: {prob:.2f}) — {risk}")

    elif mode == "Batch Prediction":
        st.subheader("📤 Upload File for Batch Prediction")
        batch_file = st.file_uploader("Upload new data", type=["csv", "xlsx", "xls"], key="batch")
        if batch_file:
            if batch_file.name.endswith(".csv"):
                new_df = pd.read_csv(batch_file)
            else:
                new_df = pd.read_excel(batch_file)

            le_dict = st.session_state.get("le_dict")
            scaler = st.session_state.get("scaler")
            model = st.session_state.get("trained_model")

            for col in new_df.columns:
                if col in le_dict:
                    new_df[col] = le_dict[col].transform(new_df[col].astype(str))
            new_scaled = scaler.transform(new_df[st.session_state["X"].columns])

            preds = model.predict(new_scaled)
            probs = model.predict_proba(new_scaled)[:, 1]

            new_df["Predicted_Churn"] = preds
            new_df["Churn_Probability"] = probs
            st.dataframe(new_df.head())

            csv = new_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Predictions (CSV)", csv, "churn_predictions.csv")

else:
    st.info("👆 Upload your dataset to begin.")
