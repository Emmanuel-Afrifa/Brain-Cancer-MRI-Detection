from io import BytesIO
from PIL import Image
from src.inference.inference_loader import InferenceDataLoader
from src.inference.predict import predict
from src.inference.uploaded_image_dataset import get_uploaded_image_data
from src.interpret.interpret import BrainGradCAM
from src.models.base_model import BrainScanCNN
from src.utils.file_io import load_config, load_objects
import pandas as  pd
import streamlit as st
import torch

CHECKPOINT_PATH = "artifacts/models/model_brain_cnn.pth"
configs = load_config("configs/config.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = load_objects("artifacts/preprocessing/class_names.pth")

st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide"
)
st.title("Brain Cancer MRI Scan Classification")

@st.cache_resource
def load_model():
    checkpoint = torch.load(CHECKPOINT_PATH)
    model = BrainScanCNN(num_classes=3)
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model

@st.cache_resource
def get_predictions_n_dataset(uploaded_file):
    dataset = get_uploaded_image_data(uploaded_file)
    dataloader = InferenceDataLoader(dataset=dataset, config=configs).get_inference_loaders()
    pred, pred_probs = predict(data_loader=dataloader, model=model, device=device)
    return pred, pred_probs, dataset
    
uploaded_file = st.file_uploader("Upload a Brain MRI Scan image for prediction", type=['.png', '.jpg', '.jpeg', '.bmp', '.tiff'])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image=image, caption="Uploaded Brain scan image")
    st.session_state["uploaded_image"] = uploaded_file

mode = st.selectbox("Select Prediction Mode", options=["Predict Only", "Interpret (GRAD-CAM)"])
    
if st.button("Run"):
    model = load_model()
    pred, pred_probs, dataset = get_predictions_n_dataset(uploaded_file=uploaded_file)
    
    st.session_state["pred"] = pred[0]
    st.session_state["pred_probs"] = pred_probs[0]   
    st.session_state["mode"] = mode

    # Optionally generate Grad-CAM visualizations
    if mode.lower() == "interpret (grad-cam)":
        img, _ = dataset[0]
        grad_cam = BrainGradCAM(model, target_layer_name="conv5")
        cam, target_class = grad_cam.compute_grad_cam(img)
        overlay = grad_cam.apply_colormap_on_img(img, cam=cam)
        fig = grad_cam.plot_overlays(img, cam, predicted_label=target_class, overlay=overlay, 
                                    class_list=class_names, save_name="")

        st.session_state["gradcam_fig"] = fig
        st.session_state["pred_label"] = class_names[st.session_state["pred"]]        
    
if "pred" in st.session_state:
    st.markdown(f"### Predicted Class: **{class_names[st.session_state['pred']]}**")
    st.bar_chart(st.session_state["pred_probs"])
    
    st.markdown("### Predicted Probabilities")
    st.dataframe(pd.DataFrame([st.session_state["pred_probs"]], columns=class_names))
    
if "gradcam_fig" in st.session_state and st.session_state.get("mode", "").lower() == "interpret (grad-cam)":
    st.pyplot(st.session_state["gradcam_fig"], width="stretch")

    buffer = BytesIO()
    st.session_state["gradcam_fig"].savefig(buffer, format="PNG", bbox_inches="tight")
    buffer.seek(0)

    st.download_button(
        label="Download Grad-CAM",
        data=buffer,
        file_name=f"{class_names[st.session_state['pred']]}_gradcam.png",
        mime="image/png"
    )