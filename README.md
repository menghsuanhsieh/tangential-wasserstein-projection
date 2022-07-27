# Tangetial Wasserstein Projection

Gunsilius, Hsieh, & Lee (2022) - simulations and application code

Preprint at:


Overview
--------

The code in this repository constructs the results and plots found in the accompanying paper, available at: .

Data Availability and Provenance Statements
----------------------------

### Summary of Availability

- [ ] All data used herein are publicly available.

### Details on each Data Source

- For Lego image replication, the data can be downloaded from Kaggle: https://www.kaggle.com/datasets/joosthazelzet/lego-brick-images. In our repository, select `Lego` in the folder `data`, and the images used are contained therein. We kept the same file names for the files downloaded from the link above.

Datafiles: `data/Lego`

- For Medicaid expansion application, the data can be downloaded from IPUMS: https://usa.ipums.org/usa/. We downloaded the variables stated in the main text: HINSCAID, EMPSTAT, UHRSWORK, INCWAGE. We selected additional technical person-level ID variables to allow us to select household head and spouse (if any). We applied further sample selection criteria mentioned in Appendix B.2 of the accompanying paper.

Datafiles available here: https://www.dropbox.com/sh/y43s568l44ny8pz/AADmFeUdanq9PzKHYhESaPBaa?dl=0


Dataset list
------------

| Data file | Source | Notes    |Provided |
|-----------|--------|----------|---------|
| `data/Lego/.png` | Listed above | Control image | Yes |

Computational requirements
---------------------------

### Software Requirements

The simulations and applications were ran using Python/3.8.1.
  - ot (pip install pot) 
  - cvxpy
  - numpy
  - matplotlib
  - pandas

### Memory and Runtime Requirements

#### Summary

Approximate time needed to reproduce the analyses on a standard (CURRENT YEAR) desktop machine:

- [ ] 3-14 days

#### Details

The code was last run on a **4-core Apple-based laptop with MacOS version 12.4**. 

Portions of the code were last run on a **36-core Intel server with 100 GB of RAM**.  Computation took 8.5 hours. 

  
Description of programs/code
----------------------------

- Programs in `code/01_simulation` will perform simulation studys on all datasets referenced above and output the results in `data/analysis`. The `Dockerfile` in root dictionary will run them all. These programs will generate all tables and plots in the main text.


Instructions to Replicators
---------------------------
- Download the repository to your local computer.
- Install docker desktop using link: https://docs.docker.com/get-docker/
- Open terminal, if you use docker for the first time, type: `docker run -d -p 80:80 docker/getting-started`
- Get the container of R 4.1.0 using pull command: `docker pull rocker/r-ver:4.1.0`
- Go to the root dictionary (your local path of the root dictionary) and type: 
  - `docker build -t analysis .`
  - `docker run -v (your local path of the root dictionary’)/data/analysis:/home/results analysis`

The Rda and txt files will be stored in `data/analysis`

- Go to the dictionary `code/02_analysis` and change the local path in `table_gen.R`. Run the code and get results for tables.

### Details

- `python/01_simulation/archive_functions.R`:

Apart from `.ipynb`, it takes less than 5 minutes to finish running all the programs with 36 cores. `.ipynb` takes 8.5 hours to finish running with 36 cores.

- `code/02_analysis/table_gen.R`: generating the results of all tables

List of tables and programs
---------------------------

The provided code reproduces:

- [ ] All tables in the paper

| Figure/Table #    | Program                               | Line Number |
|-------------------|---------------------------------------|-------------|
| Table 1           | python/table_gen.R                    | 137, 146          | 
| Table 2           | python/table_gen.R                    | 137, 146          | 
| Table 3           | python/table_gen.R                    | 421, 422        | 
| Figure 2           | python/table_gen.R                    | 421, 422        | 
| Figure 3           | python/table_gen.R                    | 421, 422        | 
| Figure 4           | python/table_gen.R                    | 421, 422        | 
| Figure 5           | python/table_gen.R                    | 421, 422        | 

## References
