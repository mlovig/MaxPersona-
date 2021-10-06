# Python 3.6 Enviroment
source "venv/bin/activate"

#Function used for following python files
python BERTPOS_functions.py

#This creates all 5 BERT models based on the 5 train test spilts (Multi Day Run-time)
#Saves all 5 models from our training labeled bert_taggeri.h5
python BERT_run.py

#Calculates the calibration and p-values, give results or metrics and generates plots.
#Final Results in BPS section of the table is called RESULTS.CSV
python BERT_out.py

