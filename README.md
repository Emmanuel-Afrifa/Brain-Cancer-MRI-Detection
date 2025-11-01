# Brain-Cancer-MRI-Detection
This project uses neural networks to classify the type of brain cancer.

### Usage
---
- For training the model, use
    ```
    python -m src.main --mode train --config configs/config.yaml 
    ```
- For evaluation of the model performance, use
    ```
    python -m src.main --mode predict --config configs/config.yaml --input trial_predict_imgs
    ```

- For use of the trained model to make predictions, use
    ```
    python -m src.main --mode predict --config configs/config.yaml --input trial_predict_imgs
    ```

***NB: Here, `trial_predict_imgs` denotes the path of the directory that contains images to be predicted.***