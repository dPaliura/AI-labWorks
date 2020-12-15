# Set path to the root of local repo instead of [directory]
dir <- "[directory]/lab3/reporting"
setwd(dir)

##
#   Result for not supplemented KDD
##
validated <- read.csv("../data/output/KDD_validation.csv")
tested <- read.csv("../data/output/KDD_testing.csv")

boxplot(validated[,c("error_1", "error_50", "error_100")], main="Validated")

boxplot(tested[,c("error_1", "error_50", "error_100")], main="Tested")

sum(tested$expected != tested$encoded_1)
sum(tested$expected != tested$encoded_50)
sum(tested$expected != tested$encoded_100)


##
#   Result for supplemented KDD with supplement rate 10 attacks per 100 normals
##
validated <- read.csv("../data/output/KDD_suppl_10_percnt_validation.csv")
tested <- read.csv("../data/output/KDD_suppl_10_percnt_testing.csv")

boxplot(validated[,c("error_1", "error_50", "error_100")], main="Validated")

boxplot(tested[,c("error_1", "error_50", "error_100")], main="Tested")

sum(tested$expected != tested$encoded_1)
sum(tested$expected != tested$encoded_50)
sum(tested$expected != tested$encoded_100)


##
#   Result for supplemented KDD with supplement rate 10 attacks per 100 normals
##
validated <- read.csv("../data/output/KDD_suppl_10_percnt_3K_epoch_validation.csv")
tested <- read.csv("../data/output/KDD_suppl_10_percnt_3K_epoch_testing.csv")

boxplot(validated[,c("error_1", "error_50", "error_100", "error_3000")], main="Validated")

boxplot(tested[,c("error_1", "error_50", "error_100", "error_3000")], main="Tested")

sum(tested$expected != tested$encoded_1)
sum(tested$expected != tested$encoded_50)
sum(tested$expected != tested$encoded_100)


##
#   Result for supplemented KDD with supplement rate 10 attacks per 100 normals
##
validated <- read.csv("../data/output/KDD_validation_pnn.csv")
tested <- read.csv("../data/output/KDD_testing_pnn.csv")

boxplot(validated$expected - validated$recognized, main="Validated")

boxplot(tested$expected - tested$recognized, main="Tested")

validated.size = nrow(validated)
tested.size = nrow(tested)

validated.err.num = sum(validated$expected != validated$recognized)
tested.err.num = sum(tested$expected != tested$recognized)

cat(paste0("validated error percentage: ", 
          round(100*validated.err.num/validated.size, 2), "%\n",
          "tested error percentage: ", 
          round(100*tested.err.num/tested.size, 2), "%\n"))

