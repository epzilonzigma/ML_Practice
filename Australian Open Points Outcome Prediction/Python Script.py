# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 19:50:14 2018

@author: Tony Cai
"""
##Creates model specifications to be estimated and selected

import itertools
import csv

## constructs a tuple of independent variables - copied and pasted from concatenation in Excel
variables = ('rally+','serve+','hitpoint+','speed+','net.clearance+','distance.from.sideline+','depth+','outside.sideline+','outside.baseline+','player.distance.travelled+','player.impact.depth+','player.impact.distance.from.center+','player.depth+','player.distance.from.center+','opponent.depth+','opponent.distance.from.center+','same.side+','previous.speed+','previous.net.clearance+','previous.distance.from.sideline+','previous.depth+','previous.hitpoint+','previous.time.to.net+','server.is.impact.player+')

## constructs list which houses results
modelcombo = []

## creates an array of all possible combinations of independent variables

for i in range(16,25):
  stuff = list(itertools.combinations(variables,i))
  for words in stuff:
    modelcombo.append(''.join(words))

modelcombonum = len(modelcombo)

## adjusts each entry to formula specification strings in R

for i in range(0,modelcombonum):
  modelcombo[i] = "outcome ~ 0 | " + modelcombo[i][0:len(modelcombo[i])-1]
  
## export data to R to be uploaded into R script for estimation

with open('models.csv', 'w', newline = "") as fp:
   a = csv.writer(fp, dialect='excel')
   for i in range(0,len(modelcombo)):
       a.writerow([modelcombo[i]])