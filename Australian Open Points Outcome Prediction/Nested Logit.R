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
amodel.spec <- as.matrix(model.spec)

#modify training data for mlogit package estimation

mt.data <- mlogit.data(mens.train, choice = "outcome", shape = "wide", chid.var = "id")
wt.data <- mlogit.data(womens.train, choice = "outcome", shape = "wide", chid.var = "id")
head(mt.data)
head(wt.data)

#estimation of models based on specifications provided
 
log.likelihoodm <- c()
log.likelihoodw <- c()

for (i in 1258675:nrow(amodel.spec)){
  model <- mlogit(formula(amodel.spec[i]),mt.data,nests=list(winnner = "W", error = c("UE","FE")), un.nest.el = TRUE)
  log.likelihoodm[i] <- model$logLik
  model <- mlogit(formula(amodel.spec[i]),wt.data,nests=list(winnner = "W", error = c("UE","FE")), un.nest.el = TRUE)
  log.likelihoodw[i] <- model$logLik
}

#exporting log.likelihood data into csv for model selection via computation and comparison of BIC

log.like <- as.matrix(cbind(amodel.spec,log.likelihoodm,log.likelihoodw))
write.csv(log.like, file = "loglikes.csv")

#predicting values of test set

### importing test dataset

mens.test <- read.csv("mens_test_file.csv", header = TRUE)
womens.test <- read.csv("womens_test_file.csv", header = TRUE)

mens.tt <- mlogit.data(mens.test, choice = "outcome", shape = "wide")
womens.tt <- mlogit.data(womens.test, choice = "outcome", shape = "wide")

#estimate selected optimal model by BIC

mens.p.model <- mlogit(outcome ~ 0 | serve+hitpoint+speed+net.clearance+distance.from.sideline+depth+outside.sideline+outside.baseline+player.distance.travelled+player.impact.depth+player.impact.distance.from.center+player.depth+opponent.depth+opponent.distance.from.center+previous.speed+previous.net.clearance+previous.distance.from.sideline+previous.depth+previous.hitpoint+server.is.impact.player, mt.data, nests =list(winnner = "W", error = c("UE","FE")), un.nest.el = TRUE)
womens.p.model <- mlogit(outcome ~ 0 | rally+hitpoint+speed+net.clearance+distance.from.sideline+depth+outside.sideline+outside.baseline+player.distance.travelled+player.impact.depth+player.impact.distance.from.center+player.depth+player.distance.from.center+opponent.depth+opponent.distance.from.center+previous.speed+previous.net.clearance+previous.distance.from.sideline+previous.depth+previous.hitpoint+server.is.impact.player, wt.data, nests =list(winnner = "W", error = c("UE","FE")), un.nest.el = TRUE)

#generate preditions for test sets and export as CSV

mens.predictions <- predict(mens.p.model, newdata = mens.tt, returnData = TRUE)
womens.predictions <- predict(womens.p.model, newdata = womens.tt, returnData = TRUE)

write.csv(mens.predictions, "mens_predictions.csv")
write.csv(womens.predictions, "womens_predictions.csv")
