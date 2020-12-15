# Set path to local repo instead of [directory]
dir <- "[directory]/lab3/data/dataProcessing"
setwd(dir)

classes = c("perl", "normal")

# Read data
fields <- read.csv("dataSet/FieldNames.csv", head=FALSE)

train.set <- read.csv("dataSet/KDDTrain+.csv", head=FALSE)[,-43]
train.20.subset <- read.csv("dataSet/KDDTrain+_20PERCENT.csv", head=FALSE)[,-43]
test.set <- read.csv("dataSet/KDDTest+.csv", head=FALSE)[,-43]

# Filter data
train.set <- na.exclude(train.set[train.set[,42] %in% classes,])
train.20.subset <- na.exclude(train.20.subset[train.20.subset[,42] %in% classes,])
test.set <- na.exclude(test.set[test.set[,42] %in% classes,])

# Paste all data into one set
data <- rbind(train.set, train.20.subset, test.set)

# Add column names as given in FieldNames.csv file
names(data) <- c(as.character(fields[,1]), "class")

# Take a look at data
summary(data)

# And save indeces of sets in pasted data
train.size <- nrow(train.set)
train.20.size <- nrow(train.20.subset)
train.indcs <- 1:train.size
train.20.indcs <- 1:train.20.size + train.size
test.indcs <- 1:nrow(test.set) + train.size + train.20.size


# Remove non-actual variables
rm(train.size, train.20.size, 
   train.set, train.20.subset,
   test.set, classes)

# Encode class variable
data$class <- +(data$class == "normal")

# Encoding
for (i in 1:nrow(fields)){
    if (fields[i,2] == "symbolic"){
        field.indx <- which(names(data)==fields[i,1])
        classes <- unique(data[,field.indx])
        data[,field.indx] = sapply(data[,field.indx], function(x) which(x==classes))
    }
}

# Normalize data
for (i in 1:ncol(data)){
    rng <- range(data[,i])
    if (rng[1]!=0 | rng[2]!=1){
        if (rng[1] != rng[2]){
            data[,i] <- (data[,i] - rng[1])/(rng[2] - rng[1])
        }
    }
}

# Take a look at final data format and check if it contains any NA
summary(data)
any(is.na(data))

# Split data into subsets
train.set <- data[train.indcs,]
train.20.subset <- data[train.20.indcs,]
test.set <- data[test.indcs,]
rm(data, train.indcs, train.20.indcs, test.indcs, 
   classes, field.indx, rng, i)

# Check if each sample has row with class "perl"
sum(!train.set$class)
sum(!train.20.subset$class)
# No "perl" in 20% subset
sum(!test.set$class)

# Make replacement to add "perl into 20% subset"
train.20.subset[1,] <- train.set[which(!train.set$class)[1],]

# Save sets into files
write.csv(train.set, "../input/KDDTrain_procsd.csv")
write.csv(train.20.subset, "../input/KDDTrain_20PERCENT_procsd.csv")
write.csv(test.set, "../input/KDDTest_procsd.csv")

rm(train.set, train.20.subset, test.set, fields, dir)
