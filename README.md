# SRLProject

To set-up the DSRL library, first create a virtual environment using python 3.8.

```
python3.8 -m venv .venv
```

Then activate it.
```
source .venv/bin/activate # Linux
.venv\Scripts\activate # Windows
```

Then we need to install the DSRL package from source, as there is a compatibility issue with the dataset paths on the pip version.

```
git clone https://github.com/liuzuxin/DSRL.git
cd DSRL
python3.8 -m pip install -e .
cd ..
```

And install the remaining requirements.
```
python3.8 -m pip install -r requirements.txt
```


You can then download the environment datasets by running:
```
python3.8 load_data.py
```