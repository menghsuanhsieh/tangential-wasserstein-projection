rm(list = ls())
set.seed(100)

pacman::p_load(haven, ramify, fastDummies, doParallel, foreach, data.table, sandwich, lmtest, maps)

registerDoParallel(cores = 36)
getDoParWorkers()
getDoParName()

# import usa Data
data <- fread("usa_00003.csv") # test: nrows = 1000)

# states data
data("state.fips")
state.fips <- as.data.table(state.fips)

# getting data for states
new <- state.fips[abb %in% c("MT",
                             "AL", "FL", "GA", "KS", "MS", "NC", "SC", "SD", "TN", "TX", "WI", "WY")]

states_interest <- unique(new$fips)
states_names_of_interest <- unique(new$abb)

# limiting data to states of interest
data_f <- data[STATEFIP %in% states_interest]

# age restrictions
data_f <- data_f[AGE >= 18 & AGE <= 65]

# drop missing sdata
data_f <- data_f[is.na(HINSCAID)==FALSE
                 & is.na(EMPSTAT)==FALSE & EMPSTAT!=0 & EMPSTAT!=3
                 & !(INCTOT %in% c(9999999, 0))
                 & !(UHRSWORK %in% c(0, 99))]

# export states as separate csv
for (k in 1:length(states_interest)) {
  
  fwrite(data_f[STATEFIP==states_interest[k]], file = paste0("TWP/states application data/", states_names_of_interest[k], ".csv"))
  
}

fwrite(data_f, file = paste0("states application data/full data.csv"))
