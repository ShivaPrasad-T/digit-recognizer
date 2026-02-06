import streamlit as st
import numpy as np
from PIL import Image, ImageOps
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
    st.session_state.canvas_key = "canvas_" + str(np.random.randint(10000))

st.button("🧹 Clear Canvas", on_click=clear_canvas)

# -------------------------------
# Drawing Canvas
# -------------------------------
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=18,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key=st.session_state.canvas_key
)

# -------------------------------
# Process Drawing
# -------------------------------
if canvas_result.image_data is not None:
    # Convert numpy array to PIL Image
    img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')

    # Convert to grayscale
    img = ImageOps.grayscale(img)

    # Resize to 28x28 using Pillow resampling
    img = img.resize((28,28), Image.Resampling.LANCZOS)

    # Show processed image (sharp, no blur)
    st.image(img, caption="Processed Grayscale Image", use_column_width=True)

    # Normalize and reshape for model
    img_array = np.array(img).astype("float32") / 255.0
    img_array = img_array.reshape(1,28,28,1)

    # Predict digit
    try:
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)

        st.subheader(f"🎯 Predicted Digit: {predicted_digit}")

        st.write("### 🔢 Confidence Levels")
        for i, prob in enumerate(prediction[0]):
            st.write(f"Digit {i}: {prob*100:.2f}%")
    except Exception as e:
        st.error("❌ Prediction failed.")
        st.error(str(e))




