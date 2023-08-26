1. Install all the libraries required.
    - Use pip install -r requirements.txt
2. Run flask server.
   - python server.py
3. Navigate to the application URL.
   - Open browser and naviagate to http://127.0.0.1:1234
4. Upload multiple csv files. ["Sensor", "Sensor_high_freq", "Percent_reference"] and click on upload button.
5. Once uploaded wait for a few seconds. 
   - You will see Test - Best Model and Test - Ensemble Model.
   - When you click on any of the button, you would see Confusion Matrix, Accuracy, Recall, Precision and F1 scores with bar plots on the screen.
6. The application is also hosted on Azure Cloud and can be accessed on https://morsehandrehab.eastus.cloudapp.azure.com/
7. Please go throught the EagleMLDatasetAnalysis.ipynb, this python notebook has exploratory analysis of dataset with my findings.
8. I also attached a three paged Report on the steps and findings of mine, open EagleML - Analysis Report.pdf.

Please note that the application only supports csv file formats only.