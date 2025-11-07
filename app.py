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

st.title("Brain Cancer MRI Scan Classification")


uploaded_file = st.file_uploader("Upload a Brain MRI Scan image for prediction", type=['.png', '.jpg', '.jpeg', '.bmp', '.tiff'])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image=image, caption="Uploaded Brain scan image")
    

mode = st.selectbox("Select Prediction Mode", options=["Predict Only", "Interpret (GRAD-CAM)"])

# n_limit = None
# if mode == "Interpret (GRAD-CAM)":
#     n_limit = st.slider("Select max number of images to interpret", min_value=1, max_value=20, value=1)
    
if st.button("Run"):
    checkpoint = torch.load(CHECKPOINT_PATH)
    model = BrainScanCNN(num_classes=3)
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    dataset = get_uploaded_image_data(uploaded_file)
    dataloader = InferenceDataLoader(dataset=dataset, config=configs).get_inference_loaders()
    pred, pred_probs = predict(data_loader=dataloader, model=model, device=device)
    print(pred, class_names)
    st.markdown(f"### Predicted Class: **{class_names[pred[0]]}**")
    st.bar_chart(pred_probs[0])
    
    st.markdown("### Predicted Probabilities")
    st.dataframe(pd.DataFrame([pred_probs[0]], columns=class_names))

    # Optionally generate Grad-CAM
    if mode.lower() == "interpret (grad-cam)":
        grad_cam = BrainGradCAM(model, target_layer_name="conv5")
        img, _ = dataset[0]
        cam, target_class = grad_cam.compute_grad_cam(img)
        overlay = grad_cam.apply_colormap_on_img(img, cam=cam)
        fig = grad_cam.plot_overlays(img, cam, predicted_label=target_class, overlay=overlay, 
                                     class_list=class_names, save_name="")
        

        st.pyplot(fig, width='stretch', use_container_width=True)

        buffer = BytesIO()
        fig.savefig(buffer, format="PNG", bbox_inches="tight")
        buffer.seek(0)

        st.download_button(
            label="Download Grad-CAM",
            data=buffer,
            file_name=f"{class_names[pred[0]]}_gradcam.png",
            mime="image/png"
        )

        # # Allow user to download
        # save_path = f"outputs/{pred_label}_gradcam.png"
        # overlay.save(save_path)
        # with open(save_path, "rb") as file:
        #     st.download_button("Download Grad-CAM", file, file_name=f"{pred_label}_gradcam.png")