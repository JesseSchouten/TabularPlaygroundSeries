# TabularPlaygroundSeries
My participation on the Tabular Playground Series on kaggle: https://www.kaggle.com/competitions/tabular-playground-series-jul-2022.

The following setup was used during the project:
- WSL ubuntu 22.04
- Python 3.10

Steps:
- pip install -r requirements.txt
- download the data
    - Use the kaggle API: 
        - pip install kaggle
        - add a .kaggle folder with a kaggle.json file containing your API key to the root directory (https://www.kaggle.com/docs/api).
        - kaggle competitions download -c tabular-playground-series-jul-2022
        - unzip tabular-playground-series-jul-2022.zip
        - mv data.csv /data/raw/data.csv &&  mv sample_submission.csv /data/raw/sample_submission.csv
    - Download manually
    - Make sure to add it to the /data/raw folder!
- To train a baseline model and get a submission.csv file, run on the command line:
    - make trained_model
    - make predictions FILE_NAME=submission.csv
