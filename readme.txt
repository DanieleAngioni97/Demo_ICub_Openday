```
conda create -n openday python=3.9
conda activate openday
pip install -r requirements.txt
```


lanciare demo_icub.py da un env python 3 con:
	- secml v 0.10
	- cv2 (OpenCV)
	- torch e torchvision

parametri demo (da impostare alla fine dello script):
	- clf_rate: numero (approssimativo) di frame al secondo usati per la
		    classificazione
	- n_acquisition: numero di frame acquisiti ogni volta che si mostra un
			 nuovo oggetto

comandi demo:
	- train <nome_classe>: acquisisce un certo numero di frame del nuovo
			       oggetto e addestra il clf
	- forget <nome_classe>: rimuove la classe specificata dal dataset e
				riaddestra il clf
	- reset: ricarica il clf addestrato sui 7 oggetti base iCub7
