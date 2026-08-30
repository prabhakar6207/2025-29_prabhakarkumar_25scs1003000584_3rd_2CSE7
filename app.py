import streamlit as st
import pickle


# Page settings
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧"
)


# Saved model को load करना
try:

    with open("spam_model.pkl", "rb") as file:
        model = pickle.load(file)

except FileNotFoundError:

    st.error(
        "Model नहीं मिला। "
        "पहले train_model.py run करें।"
    )

    st.stop()


# Project heading
st.title("📧 Spam Email Classifier")

st.write(
    " ENTER YOUR EMAIL MESSAGE  BELOW.  "
    "Machine Learning MODEL WILL CLASSIFY . "
    "Email Spam OR NOT SPAM.."
)


st.divider()


# Email input
email = st.text_area(
    "📩 Email meassage",
    height=200,
    placeholder="Example: Congratulations! You won a prize..."
)


# Check button
if st.button(
    "🔍 Check Email",
    use_container_width=True
):

    if email.strip() == "":

        st.warning(
            "please enter an email message first"
        )

    else:

        # Prediction
        prediction = model.predict([email])[0]

        probability = model.predict_proba([email])[0]

        spam_probability = probability[1] * 100
        ham_probability = probability[0] * 100


        st.divider()


        # Result
        if prediction == 1:

            st.error("🚨 SPAM EMAIL")

            st.write(
                f"Spam Probability: "
                f"**{spam_probability:.2f}%**"
            )

        else:

            st.success("✅ NOT SPAM / HAM")

            st.write(
                f"Spam Probability: "
                f"**{spam_probability:.2f}%**"
            )


        # Probability
        st.subheader("📊 Prediction Confidence")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Not Spam",
                f"{ham_probability:.2f}%"
            )

        with col2:

            st.metric(
                "Spam",
                f"{spam_probability:.2f}%"
            )


st.divider()

st.caption(
    "Spam Email Classifier | "
    "Machine Learning Internship Project"
)