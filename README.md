MuDPaW v6.1

Multimodal Data Processor and Writer
Advanced Rheological Testing Science Initiative (artSI) @ MAX IV

1. Overview

MuDPaW is a multimodal data processing framework designed for synchronized analysis of:
	• Rheology
	• Polarized Light Imaging (PLI)
	• Polarimetry (PI)
	• SAXS / SWAXS
	• Tribology

Version 6.x is an updated version from TkInter to PyQt6 with a unified architecture for ease of use. The procedures outlined are specific to the coSAXS and ForMAX beamlines at MAX IV.

Important:  MuDPaW has been designed and tested on MAC OS X using Python v.3.1.1
	    MuDPaW was tested on Python 3.9


2. Installation (Terminal)

2.1 — Clone or unzip the repository

unzip MuDPaW_v6.1.zip
cd MuDPaW_v6.1\

2.2 - Locate and run installer.

python3 install.py
python3 Main.py

If you want to check the version:

Python 3 --versino


3. Python Requirements
	•	Python ≥ 3.10 (3.11 recommended)
	•	PyQt6
	•	numpy
	•	pandas
	•	matplotlib
	•	scipy
	•	h5py (for beamline data)
	•	paramiko (for SSH connector)

If install.py fails, you could try to manually install:

pip install -r requirements.txt



4. SSH Setup (Required for Beamline Connector)

If you plan to use:
	•	CoSAXS connector
	•	ForMAX connector
	•	Remote beamline data access

you must configure SSH key authentication.

4.1 Create an SSH Key (Mac / Linux)

Open Terminal and run from the local machine
ssh -o BatchMode=yes user@offline-fe1.maxiv.lu.se 'echo OK'
	•	If you get OK → done.
	•	If it asks for password → continue.

Determine which key model applies (remote policy test)

Still from your local machine:
ssh user@offline-fe1.maxiv.lu.se 'echo HOME=$HOME; ls -ld "$HOME" /home/$USER 2>&1 | head -n 5'

	•	If it shows a real directory → use authorized_keys method.
	•	If it says “No such file” / “cannot access” → central key registration, stop doing .ssh.


4.4 Test SSH Connection
ssh username@remote.server

Then run

ssh-keygen -R remote.server

If you see:
Host key verification failed

Then run 
ssh-keygen -R remote.server

5. Common Issues

GUI opens but buttons do nothing
	•	Ensure no Python errors appear in the terminal.
	•	Verify PyQt6 is correctly installed.
	•	Try running in clean environment:

python -m venv mudpaw_env
source mudpaw_env/bin/activate
pip install -r requirements.txt
python Main.py

