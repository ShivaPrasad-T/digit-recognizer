import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="Digit Recognizer", page_icon="🧠")
st.title("🧠 Handwritten Digit Recognition")
st.write("Draw a digit (0–9) below 👇")

# -------------------------------
# Load Model
# -------------------------------
try:
    model = load_model("digit_model.h5")
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error("❌ Could not load model. Make sure 'digit_model.h5' is in this folder.")
    st.error(str(e))
    st.stop()

# -------------------------------
# Clear Canvas Button
# -------------------------------
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = "canvas_0"

def clear_canvas():
    # Change the key to force canvas to reset
    st.session_state.canvas_key = "canvas_" + str(np.random.randint(10000))

st.button("🧹 Clear Canvas", on_click=clear_canvas)

# -------------------------------
# Drawing Canvas
# -------------------------------
canvas_result = st_canvas(
    fill_color="black",               # brush fill color
    stroke_width=18,                  # brush width
    stroke_color="white",             # brush color
    background_color="black",         # canvas background
    height=280,
    width=280,
    drawing_mode="freedraw",
    key=st.session_state.canvas_key   # unique key to reset canvas
)

# -------------------------------
# Process Drawing
# -------------------------------
if canvas_result.image_data is not None:
    # Convert RGBA → Grayscale
    img = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)

    # Show processed grayscale image
    st.image(img, caption="Processed Grayscale Image", use_column_width=True)

    # Resize & blur to match MNIST format
    img = cv2.resize(img, (28,28))
    img = cv2.GaussianBlur(img, (5,5), 0)

    # Normalize & reshape for model
    img = img.astype("float32") / 255.0
    img = img.reshape(1,28,28,1)

    # Predict digit
    try:
        prediction = model.predict(img)
        predicted_digit = np.argmax(prediction)

        st.subheader(f"🎯 Predicted Digit: {predicted_digit}")

        st.write("### 🔢 Confidence Levels")
        for i, prob in enumerate(prediction[0]):
            st.write(f"Digit {i}: {prob*100:.2f}%")
    except Exception as e:
        st.error("❌ Prediction failed. Check model input and compatibility.")
        st.error(str(e))



