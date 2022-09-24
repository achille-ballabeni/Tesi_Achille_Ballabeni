Per eseguire correttamente i codici di inferenza occorre prima estrarre le immagini e i valori ground truth dai file tfrecord usando il notebook *TFRecord_to_NPY_inference.ipynb*.
Le 5 cartelle contenenti i file npy devono essere raggruppate in un unica cartella. È necessario adattare i percorsi che si trovano all'interno dei due script Python
in modo che coincidano a quelli utilizzati.

Per la conversione dei modelli TensorFlow in formato tflite si può utilizzare il notebook *TF_lite_conversion.ipynb*, sempre assicurandosi che i percorsi siano quelli corretti.
In alternativa si può ricorrere ai modelli già convertiti.