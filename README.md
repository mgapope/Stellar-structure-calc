Welcome, I am glad you took the time to look over my stellar structure code. This is a report done for AS.171.611 – Stellar Structure and Evolution. Some notes on running the code
There's a few packages that must be installed for the code to work, all of them can be installed via pip

pip install scipy
pip install numpy
pip install matplotlib
pip install mesa_reader
pip install pandas

The only other thing to note is the different opacity tables. For this repository, I've uploaded a few sample tables, and they're all stored under the OPACITIES folder.
With some tinkering of the initial mass, composition, and initial guesses, you should appear at a converged result for the internal strcture of a zero age main sequence star. Currently the only tested combinations have been with a 1 solar mass and 1.2 solar mass case, both with composition X = 0.7 Y = 0.28 and Z = 0.02. The current known issues is that the luminosity will be underestimated by ~30% and the radius will be overestimated by ~30%.
