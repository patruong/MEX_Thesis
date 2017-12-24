rm(list = ls())
par(mfrow=c(1,1))
library(diffusionMap)
library(randomForest)
library(ggplot2)
library(reshape)
library(reshape2)  ## here is where we do a 'pivot table'
library(plyr)
library(zoo) # time series object
library(rgl) # plot3d

t <- seq(0,10, len=100)  # the parametric index
# Then convert ( sqrt(t), 2*pi*t ) to rectilinear coordinates
x = sqrt(t)* cos(2*pi*t) 
y = sqrt(t)* sin(2*pi*t)
#png("plot1.png");

# Polar Coordinate
plot(x,y);

# Regular Coordinate
plot(t)

# Add lines to connect adjacent points
plot(x,y, type = "b")

#################################################################
setwd("C:/Users/extra23/Desktop/Filer/MEX/Local laptop backup/Other Financial Data/TDA")

omx = read.zoo("OMXS30_perc_diff.csv", sep = ',', header = TRUE, format = "%Y-%m-%d") 
omx = read.zoo("OMXS30_perc_diff_NaN0.csv", sep = ',', header = TRUE, format = "%Y-%m-%d") 

omx = read.zoo("OMXS30_raw.csv", sep = ',', header =TRUE, format = "%Y-%m-%d")

windowed <- window(omx, start = as.Date("2016-02-01"),end = as.Date("2016-06-01"))
data <- as.data.frame(windowed)

# Convert to df
omx_df <- as.data.frame(omx)
class(omx_df)

# Test plotting
plot(omx[,1], main = colnames(omx)[1])
plot(omx[,2], main = colnames(omx)[2])
#plot(omx) #Plot all return curves - plot takes time

plot(windowed[,1], main = colnames(windowed)[1])
plot(windowed[,2], main = colnames(windowed)[2])

plot(data[,1], main = colnames(data)[1])
plot(data[,2], main = colnames(data)[2])

#################     nonlinearTseries           ##############################
library(nonlinearTseries)


ts = windowed[,1]
  plot(ts)

d = buildTakens(ts, 2, 10)

plot(d[,1], d[,2],
     xlab = "x",
     ylab = "y",
     main = colnames(ts))

# 3D
ts = windowed[,1]
d = buildTakens(ts, 3, 10)

plot3d(d[,1], d[,2], d[,3],
       xlab = "x",
       ylab = "y",
       zlab = "z",
       main = colnames(ts))
