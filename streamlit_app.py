import streamlit as st
from hate.pipeline.prediction_pipeline import PredictionPipeline


@st.cache_resource(show_spinner=False)
def load_predictor():
    return PredictionPipeline()


def main():
    st.set_page_config(page_title="Hate Speech Classifier", page_icon="🛡️", layout="wide")
    st.title("🛡️ Hate Speech Classifier")
    st.markdown("Enter a sentence to detect hate or abusive content.")

    predictor = load_predictor()

    text = st.text_area("Text", height=200, placeholder="Type a tweet or sentence...")
    if st.button("Predict", type="primary"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Running inference..."):
                try:
                    label = predictor.predict(text)
                    st.success(f"Prediction: **{label}**")
                except Exception as exc:  # noqa: BLE001
                    st.error(
                        "Model artifacts not found. Run training first (python app.py) or check saved_model folder."
                    )
                    st.exception(exc)

    st.caption("Model trained locally using the provided labeled_data.csv dataset.")


if __name__ == "__main__":
    main()
