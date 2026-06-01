# DressCheck Presentation Script

Good day everyone. We are Group 11, and today we will present our machine learning web application called **DressCheck**.

DressCheck is a government employee attire compliance classifier. It is designed to identify whether an uploaded upper-garment image is classified as **Compliant** or **Non-Compliant** based on our trained dataset.

The goal of this project is to demonstrate how image processing and machine learning can be applied to a practical attire classification problem. Instead of checking each image manually, the system allows users to upload a garment image and receive an automated prediction.

For our dataset, we prepared two image categories: **Compliant** and **Non-Compliant**. The compliant category contains upper-garment examples that match the expected government employee attire, while the non-compliant category contains examples that do not match the expected attire.

For preprocessing, each image is converted to RGB, resized to 128 by 128 pixels, and transformed into numerical features. We used **HOG**, or Histogram of Oriented Gradients, to extract edge and shape patterns from each image.

After feature extraction, we trained a **Support Vector Machine**, or SVM, classifier. The SVM learns the visual differences between compliant and non-compliant government employee attire. We also used a scaler so that the extracted feature values are standardized before prediction.

Now, let us move to the web application.

This is the DressCheck interface. The user can click the upload area or drag and drop an image. Once an image is selected, the website shows a preview so the user can confirm the uploaded file.

After that, the user clicks **Analyze Attire**. The image is sent to the prediction API, where it goes through the same machine learning pipeline: resizing, HOG feature extraction, scaling, and SVM classification.

The result is then displayed on the screen. If the image is classified as compliant, the system shows **Compliant** with an allowed verdict. If it is classified as non-compliant, it shows **Non-Compliant** with a violation verdict. The system also displays the confidence score of the model.

For our project requirements, we have the dataset, the Jupyter notebook, the website, and the user manual. The Jupyter notebook shows the full training workflow, including dataset loading, feature extraction, model training, evaluation, and saving the model files. The user manual explains how to use the app, how to interpret the results, and how to troubleshoot common issues.

Overall, DressCheck shows how a traditional machine learning model can be integrated into a simple web application. It provides a fast and easy way to classify government employee attire compliance using image input.

However, we also recognize the limitations of the system. Since the model is trained on a limited dataset, its accuracy may depend on image quality, lighting, clothing angle, and how similar the uploaded image is to the training examples. Because of this, DressCheck should be treated as a prototype and learning project, not as a final authority for government employee dress code enforcement.

That concludes our presentation. Thank you.
