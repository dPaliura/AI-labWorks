# Set path to current local repo insted [directory]
dir <- "[directory]/lab3/data/dataProcessing"
setwd(dir)

# Read training data-set
train.set <- read.csv("KDDTrain_procsd.csv")

# Supplement rate is amount of classes "attack" to "normal" rate
suppl.rate = 1/10

# Get indeces of attack and normal records
attack.indcs <- which(train.set$class==0)
normal.indcs <- which(train.set$class==1)

# Get attack records
attack.recs <- train.set[attack.indcs,]

# Recount supplement rate taking into consideration number of attack records
suppl.rate <- suppl.rate/length(attack.indcs)
# Count splitting size
split.size <- round(length(normal.indcs)*suppl.rate)


# Split indeces of normal records into sets of lengths about split.size
splited.normal.indcs <- split(normal.indcs, 1:split.size)

# Supplementing training set
train.set.supplemented <- NULL
for (indcs in splited.normal.indcs){
    train.set.supplemented <- rbind(train.set.supplemented,
                                    train.set[indcs,],
                                    attack.recs)
}

rm(attack.recs, splited.normal.indcs, train.set, 
   attack.indcs, indcs,  normal.indcs,
   split.size, suppl.rate)

# Check if everything OK
summary(train.set.supplemented)
# There are excess variable 'X' in first row and we have to delete it
train.set.supplemented <- train.set.supplemented[-1]

# Sample data due to close lying supplemented rows
set.seed(42)
shuffle.times = 5
data.size = nrow(train.set.supplemented)
for (i in 1:shuffle.times){
    train.set.supplemented <- train.set.supplemented[sample(1:data.size),]
}

# Write got df into .csv file
write.csv(train.set.supplemented, "KDDTrain_supplmntd_10perc.csv")

rm(train.set.supplemented, i, shuffle.times, data.size)

