# Reinforcement Learning Gust Control — Project Overview

Project summary
---------------
This student research project develops online estimation and reinforcement-learning-based control methods for unsteady aerodynamic systems exposed to large-amplitude gusts. The aim is to detect aerodynamic disturbances from surface pressure sensors on a 3D delta wing model, estimate the unsteady forces, and synthesise controllers that maintain desired aerodynamic performance (for example, zero pitching moment) during gust encounters.

Modelling Approach
----------------
This project will aim to encorperate both Physics Augmented Auto Encoders (PA-AE), LSTM networks and RL to control the wing in extreme aerodynamic conditions. 

Basic Code Structure
---------------
This project includes a rust backend which simply compiles into a python module to allow us to right performative backend code whilst maintining the advantage of flexible pythoin libraries. 

In this project the following libraries will be used
- Polars (not Pandas)
- qtpy (for any custom UI)
- PyTorch (Faster, less overhead, iterable)
- ruststuff (Custom rust backend)
- Matplotlib (One off plotting)
- Pydantic (Serialisable and safe typing)

All code must simply be fully type hinted. 

