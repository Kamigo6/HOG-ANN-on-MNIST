# Handwritten digit recognition
A capstone project of Introduction to Machine Learning, about handwritten digit recognition using ANN


# For local computer


# Prerequisites:
- Python 3.x
- TensorFlow
- matplotlib
- seaborn
- numpy
- pandas
- skimage
- scikit-learn
- PIL (pillow)
- tkinter
- cv2 (opencv- python)


# Installation
1. Clone the repository:
```bash
git clone https://github.com/Kamigo6/HOG-ANN-on-MNIST/tree/master
```
2. Navigation to the project directory:
```bash
cd your-repository
```


3. Install the required dependencies:
```bash
pip install -r requirements.txt
```
# Structure of file/folder
Files:
UI_final.py for running the UI
best_model.h5 is the best model we have trained (use to run the UI)
tmp.png is the image produced when you run the UI 
Folders:
“Code” for the main code of our project which includes 2 subfolders: “HOG” (contains codes for training the HOG model ) and “Pixel” (contains codes for training the Pixel model)
“Data” for the code to extract data, visualize data and test noise 
“Result” for the result we have found for both HOG and Pixel model which contains the network architecture, learning rate, dropout, batch normalization, final result and some other experiment
“Model” for the models we have trained
# Usage
If you want to train the model, please run the “HOG_ANN.ipynb”/“HOG_ANN_noise.ipynb” or “ANN.ipynb”/“ANN_noise.ipynb”
(We highly recommend the google colab)
	(We have already included our trained model “best_model.h5” if you don’t want to train it again)
- To use our app, please run the “UI_final.py”


NOTE: Please click “Proceed” only after you already clicked “Save” or else the system will run on the latest saved image. This system only works with numbers.






- If you want to train the model, this is the explanation of our code:


Our file code will train the model using the MNIST dataset and display the training loss, validation loss, and accuracy




2. The code will load the MNIST dataset, normalize the pixel values, define and compile the model architecture, and train the model using the fit() function.


3. After training, the code will save the trained model in the specified location.


4. The training loss and validation loss will be plotted using Matplotlib.


5. The code will make predictions on the test set, create a confusion matrix, and display a heatmap of the confusion matrix using Seaborn and Matplotlib.


6. The classification report, including precision, recall, and F1-score for each class, will be printed in the console.


7. The accuracy score of the model will also be printed.


# Troubleshooting
If you encounter any issues or errors while running the code, please refer to the following:


- Make sure all the dependencies are correctly installed.
- Check if the dataset is present in the correct directory.
- Verify that the model is saved as best_model.h5 after training.


# Google colab execution
To run this project on Google Colab, follow these steps:


1. Open Google Colab in your web browser.


2. Create a new notebook or open an existing notebook.


3. Copy and paste the code into a code cell in the notebook.


4. Click on the "Runtime" menu and select "Change runtime type".


5. In the "Runtime type" section, select "Python 3" as the runtime type and "GPU" as the hardware accelerator. This will enable GPU acceleration for faster training.


6. Click on the "Runtime" menu again and select "Run all". This will execute all the code cells in the notebook.


7. You can modify the code and experiment with different hyperparameters or network architectures to improve the performance of the model.


# Contributions
Contributions to this project are welcome. If you would like to contribute, please follow these steps:


1. Fork the repository.
2. Create a new branch for your feature/bug fix.
3. Make your changes and commit them.
4. Push the changes to your forked repository.
5. Submit a pull request.


# Contact Information
For any inquiries or feedback, please contact minh.nq205163@sis.hust.edu.vn or linh.pk205186@sis.hust.edu.vn


# Acknowledgements
The MNIST dataset is obtained from TensorFlow.
Thank you to  Dr.Nguyen Nhat Quang for guidance and support us throughout the project.

