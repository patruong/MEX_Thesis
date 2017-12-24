"
This scripts contains single window test cases
"


rm(list = ls())
par(mfrow=c(1,1))


library(zoo) # time series object
library(data.table) #fast df readin 
library(nanotime) # nanosecond time option
library(bit64) #for nanosecond data.table
library(TDA)
library(nonlinearTseries)
library(rgl) # plot3d
#library(plotly)
library(scatterplot3d) #3d scatter plot


#Change default time stamp to sub-seconds
Sys.time()
options("digits.secs"=6)
options("nanotimeFormat" = "%Y-%m-%d %H:%M:%E*S")
Sys.time()

# READIN DATA
setwd("C:/Users/Patrick/Documents/MEX/MEX/Local laptop backup/Other Financial Data/TDA/Paris_Intraday")

#df = fread("EURUSD_Paris.csv")
df = fread("standardized_EURUSD.csv")
df = fread("EURUSD_ma1000_standardized.csv")
#df_QN = fread("QN_sample_0-2000.csv")
#df_QN = fread("QN_a.csv")
#df_QN = fread("Standardized_Laplace_QuantumNoise_short_discrete.csv")
df_QN = fread("Standardized_Laplace_QuantumNoise_short_discrete77.csv")
df_QN = fread("LaplaceQN_ma1000_standardized_disc77.csv")


#############################################
### TDA SINGLE WINDOW TEST CASES ############
#############################################

# Alpha complex for low dimension, slow with high dim
# Rips complex for high dimension, slow with many dp

data <- df[1:2001]$logR_ask 

takens = buildTakens(data, embedding.dim = 35, time.lag = 1)
takens[is.na(takens)] <- 0
pca_takens <- prcomp(takens, scale = TRUE)

#library(rgl)
plot3d(takens[,1], takens[,2], takens[,3],
       xlab = "Takens Coordinate 1",
       ylab = "Takens Coordinate 2",
       zlab = "Takens Coordinate 3",
       main = "Takens delay embedding",
       xlim = c(-8,8), ylim = c(-8, 8), zlim = c(-8,8))
#text3d(dmap$X[,1], dmap$X[,2], dmap$X[,3], texts = colnames(omx))
#library(rgl)
plot3d(pca_takens$x[,1], pca_takens$x[,2], pca_takens$x[,3],
       xlab = "PCA Takens Coordinate 1",
       ylab = "PCA Takens Coordinate 2",
       zlab = "PCA Takens Coordinate 3",
       main = "PCA Takens delay embedding",
       xlim = c(-8,8), ylim = c(-8, 8), zlim = c(-8,8))
#text3d(dmap$X[,1], dmap$X[,2], dmap$X[,3], texts = colnames(omx))

scatterplot3d(x = takens[,1], y = takens[,2], z = takens[,3])

# Takens loop
for (i in seq(0,20000,2000)){
  takens = buildTakens(df[i:(i+2000)]$logR_ask, embedding.dim = 3, time.lag = 1)
  takens[is.na(takens)] <- 0
  pca_takens <- prcomp(takens, scale = TRUE)
  #scatterplot3d(x = takens[,1], y = takens[,2], z = takens[,3])
  scatterplot3d(x = pca_takens$x[,1], y = pca_takens$x[,2], z = pca_takens$x[,3])
  Sys.sleep(2)
}


DiagLim <- 0.1
maxdimension <- 1

# Diag data Euclidean - extreme slow memory error
#par(mfrow=c(1,2))
DiagOMX <- ripsDiag(takens, maxdimension = maxdimension, maxscale = DiagLim, printProgress = TRUE)
plot(DiagOMX[["diagram"]])

ptm <- proc.time()
DiagAlphaOMX <- alphaComplexDiag(pca_takens$x[,1:3], printProgress = TRUE)
plot(DiagAlphaOMX[["diagram"]])
proc.time() - ptm

tseq <- seq(0, 1, length = 1000) # domain
Land <- landscape(DiagAlphaOMX[["diagram"]], dimension = 1, KK = 1, tseq)
plot(tseq, Land, type = 'l')


# from dionysus - www.mrzv.org
# alpha complex when R2 or R3 high datapoint
# Rips complex when Rn but low datapoint


##################
## More Tests ####
##################

data <- df[1:2000]$logR_ask 
takens = buildTakens(data, embedding.dim = 35, time.lag = 1)
takens[is.na(takens)] <- 0
pca_takens <- prcomp(takens, scale = TRUE)

ptm <- proc.time()
DiagAlphaOMX <- alphaComplexDiag(pca_takens$x[,1:3], printProgress = TRUE)
plot(DiagAlphaOMX[["diagram"]])
proc.time() - ptm

ptm <- proc.time()
tseq <- seq(0, 1, length = 1000) # domain
Land <- landscape(DiagAlphaOMX[["diagram"]], dimension = 1, KK = 1, tseq)
plot(tseq, Land, type = 'l')
proc.time() - ptm

ptm <- proc.time()
DiagAlphaShape <- alphaShapeDiag(pca_takens$x[,1:3], printProgress = TRUE)
plot(DiagAlphaShape[["diagram"]])
proc.time() - ptm

ptm <- proc.time()
tseq <- seq(0, 1, length = 1000) # domain
LandA <- landscape(DiagAlphaShape[["diagram"]], dimension = 1, KK = 1, tseq)
plot(tseq, LandA, type = 'l')
proc.time() - ptm


