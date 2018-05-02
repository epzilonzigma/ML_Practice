### AO Tennis Point-Ending Prediction R-Script ###

### Model Usage and Rationalle ###
### We will use a nested logit model for each mens and womens' training set and use the results to predict the outcomes of mens and womens' test set respectively.
### Rationale why nested logit is used over multinomial logit is that the IIA assumption does not hold in the categorization of outcomes.
### A point either ends in a "Winner" (outcome included) or an "Error" (not included). Unforced Errors and Forced Errors are types of Errors.
### We will be using the 2 nest of the following categories: Winners - "W" and Errors - "UE" and "FE".

### Estimation will be done with "mlogit" package

library(mlogit)


### Train nested logit model for mens

#load data and pre-specified models

library(gdata)
setwd("C:/Users/Tony Cai/Documents")
mens.train <- read.csv("mens_train_file.csv", header = TRUE)
womens.train <- read.csv("womens_train_file.csv", header = TRUE)
model.spec <- read.csv("models.csv", header = FALSE)

#modify training data for mlogit package estimation

mt.data <- mlogit.data(mens.train, choice = "outcome", shape = "wide", chid.var = "id")
wt.data <- mlogit.data(womens.train, choice = "outcome", shape = "wide", chid.var = "id")
head(mt.data)
head(wt.data)

#estimation of models based on specifications provided
 
log.likelihoodm <- c()
log.likelihoodw <- c()

for (i in c(1:3)){
  model <- mlogit(formula(model.spec[i]),mt.data,nests=list(winnner = "W", error = c("UE","FE")), un.nest.el = TRUE)
  log.likelihoodm[i] <- model$logLike
  model <- mlogit(formula(model.spec[i]),wt.data,nests=list(winnner = "W", error = c("UE","FE")), un.nest.el = TRUE)
  log.likelihoodw[i] <- model$logLike
}

