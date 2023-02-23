# Tangetial Wasserstein Projections

Gunsilius, Hsieh & Lee (2022)


Overview
--------

The code in this repository constructs the results, plots, and tables found in the accompanying paper, available at https://arxiv.org/abs/2207.14727.


Data Availability and Provenance Statements
----------------------------

### Summary of Availability

- [ ] All data used herein are publicly available.

### Details on Each Data Source

- For Lego image replication, the data can be downloaded from Kaggle: https://www.kaggle.com/datasets/joosthazelzet/lego-brick-images. In our repository, select `Lego_bricks` in the folder `Data`, and the images used are contained therein. We kept the same file names for the files downloaded from the link above.

Datafiles: `Data/Lego_bricks`

- For Medicaid expansion application, the data can be downloaded from IPUMS: https://usa.ipums.org/usa/. We downloaded the variables stated in the main text: HINSCAID, EMPSTAT, UHRSWORK, INCWAGE. We selected additional technical person-level ID variables to allow us to select household head and spouse (if any). We applied further sample selection criteria mentioned in Appendix B.2 of the accompanying paper.

Datafiles available here: https://www.dropbox.com/sh/y43s568l44ny8pz/AADmFeUdanq9PzKHYhESaPBaa?dl=0


Dataset list
------------

| Data file | Source | Notes    |Provided |
|-----------|--------|----------|---------|
| `Data/Lego_bricks/0001.png` | Listed above | Target image in Section 4.2 of main text | Yes |
| `Data/Lego_bricks/0040.png` | Listed above | Control image in Section 4.2 of main text | Yes |
| `Data/Lego_bricks/0080.png` | Listed above | Control image in Section 4.2 of main text | Yes |
| `Data/Lego_bricks/0120.png` | Listed above | Control image in Section 4.2 of main text | Yes |
| `Data/Lego_bricks/0160.png` | Listed above | Control image in Section 4.2 of main text | Yes |
| `Data/Lego_bricks/0200.png` | Listed above | Control image in Section 4.2 of main text | Yes |
| `Data/Lego_bricks/0240.png` | Listed above | Control image in Section 4.2 of main text | Yes |
| `Data/Lego_bricks/0280.png` | Listed above | Control image in Section 4.2 of main text | Yes |
| `Data/Lego_bricks/0320.png` | Listed above | Control image in Section 4.2 of main text | Yes |
| `Data/Lego_bricks/0360.png` | Listed above | Control image in Section 4.2 of main text | Yes |
| `Data/Lego_bricks/0400.png` | Listed above | Control image in Section 4.2 of main text | Yes |
| `Data/Medicaid_Data/` | Listed above | ACS data used to obtain optimal weights lambda star | Yes; in Dropbox folder above |
| `Data/Medicaid_Data/Counterfactual` | Listed above | ACS data used to obtain counterfactual distributions in Section 4.3 | Yes; in Dropbox folder above |


Computational requirements
---------------------------

### Software Requirements

The simulations and applications were ran using Python/3.8.1.
  - ot (pip install pot) 
  - cvxpy
  - numpy
  - matplotlib
  - pandas
  - seaborn

### Memory and Runtime Requirements

#### Summary

Approximate time needed to reproduce the analyses on a standard (CURRENT YEAR) desktop machine:

- [ ] 1-3 days

#### Details

The code was last run on a **8-core Apple-based laptop with MacOS version 12.4**. 

Portions of the code were last run on a **36-core Intel server with 180 GB of RAM**. Computation took 2 hours. 

  
Description of programs/code
----------------------------

- Programs in `Python Code` will perform all simulations and estimations described in the main text. These programs will also generate all tables and plots found therein.


Instructions to Replicators
---------------------------
- Download the repository to your local computer.
- Download Medicare data from the `Medicaid Data/Pre-Intervention Data` subfolder on Dropbox (linked above) to the directory: `Data/Medicaid_Data`.
- Download Medicare data from the `Medicaid Data/Counterfactual Data` subfolder on Dropbox (linked above) to the directory: `Data/Medicaid_Data/Counterfactual`.
- Move the `Data` folder to the `Python Code` subdirectory.
- Run `Python Code/Mixed Gaussian Simulation.ipynb` for the Gaussian simulations.
- Run `Python Code/Lego Block Simulation.py` for the Lego Brick image replication.
- Run `Python Code/Medicaid.ipynb` for the Medicaid expansion application.

The .png files will be stored in `Python Code` subdirectory by default. The tables are generated within each .ipynb program. If you wish to save the .png files in alternative directories, please change the corresponding `plt.savefig()` commands in the programs.


### Details

- `Python Code/twp_utils.py`: defines barycentric projection and projection method described in the main text.
- `Python Code/Mixed Gaussian Simulation.ipynb`: contains the Gaussian simulations described in the main text.
- `Python Code/Lego Block Simulation.py`: contains the Lego Brick image replication described in the main text.
- `Python Code/Medicaid.ipynb`: contains the Medicaid expansion application described in the main text.

Apart from `Python Code/Lego Block Simulation.py`, it takes less than 5 minutes to finish running all the programs on a laptop with Apple M1 processor. Running `Python Code/Lego Block Simulation.py` takes 2 hours from start to finish on a cluster computer with 36 cores.

List of tables and programs
---------------------------

The provided code reproduces:

- [ ] All tables in the paper
- [ ] Figures 2 through 7 in the paper

| Figure/Table #     | Program                               |          Line (Block) Number |
|------------------- |---------------------------------------|------------------------------|
| Table 1            | Python Code/Mixed Gaussian Simulation.ipynb             | 8          | 
| Table 2            | Python Code/Medicaid.ipynb                              | 9          | 
| Table 3            | Python Code/Mixed Gaussian Simulation.ipynb             | 12         | 
| Figure 2 (Right)   | Python Code/Lego Block Simulation.py                    | 152        | 
| Figure 3 (Weights) | Python Code/Lego Block Simulation.py                    | 144        | 
| Figure 4           | Python Code/Medicaid.ipynb                              | 19, 20     | 
| Figure 5           | Python Code/Medicaid.ipynb                              | 21, 22     | 
| Figure 6           | Python Code/Medicaid.ipynb                              | 10, 12     |
| Figure 7           | Python Code/Medicaid.ipynb                              | 14, 15     |

## References

Steven Ruggles, Sarah Flood, Ronald Goeken, Megan Schouweiler and Matthew Sobek. IPUMS USA: Version 12.0 [dataset]. Minneapolis, MN: IPUMS, 2022. 
https://doi.org/10.18128/D010.V12.0
