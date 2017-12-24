"""
Nanosecond configuration for R:
https://github.com/eddelbuettel/nanotime
"""

rm(list = ls())
par(mfrow=c(1,1))
library(zoo) # time series object
library(data.table) #fast df readin 
library(nanotime) # nanosecond time option
library(bit64) #for nanosecond data.table
library(TDA)
library(nonlinearTseries)
library(rgl) # plot3d


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
#df_sw = fread("EURUSD_ask_roll_sw100_g10.csv")
#df_QN = fread("QN_sample_0-2000.csv")
#df_QN = fread("QN_a.csv")
#df_QN = fread("Standardized_Laplace_QuantumNoise_short_discrete.csv")
df_QN = fread("Standardized_Laplace_QuantumNoise_short_discrete77.csv")
df_QN = fread("LaplaceQN_ma1000_standardized_disc77.csv")
#df_ts = read.zoo(df) 
df = rollmean(df, 1000)

# Creating test Sliding window
DF <- data.frame(a = 1:10, b = 21:30, c = letters[1:10])
replace(DF, 1:2, rollapply(DF[1:2], 3, sum, fill = NA))
replace(DF, 1:2, rollapply(DF[1:2], 3, sum, by = 3, fill = NA))
rollapply(DF[1:2], 3, sum, by = 3, fill = NA)

rollapply(df[1:2000]$ask, 200, by = 200, sum, fill = NA)

####################
### TDA ############
####################

# Alpha complex for low dimension, slow with high dim
# Rips complex for high dimension, slow with many dp

#windowed <- window(omx, start = as.Date("2015-04-09"),end = as.Date("2017-01-19"))
data <- df[1:2001]$logR_ask 
data <- df[5000:7000]$logR_ask 
data <- df[50000:52000]$logR_ask 
data <- df[200000:202000]$logR_ask 
D = dist(scale(data)) # use Euclidean distance on data
#D = as.dist((1- cor(data))/2) # Correlation distance

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

library(plotly)
library(scatterplot3d)
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


###########
## TDA ####
###########

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



###
data2 <- df[300000:302000]$logR_ask 
takens2 = buildTakens(data2, embedding.dim = 35, time.lag = 1)
takens2[is.na(takens2)] <- 0
pca_takens2 <- prcomp(takens2, scale = TRUE)

ptm <- proc.time()
DiagAlphaOMX2 <- alphaComplexDiag(pca_takens2$x[,1:3], printProgress = TRUE)
plot(DiagAlphaOMX2[["diagram"]])
proc.time() - ptm

ptm <- proc.time()
tseq <- seq(0, 1, length = 1000) # domain
Land2 <- landscape(DiagAlphaOMX2[["diagram"]], dimension = 1, KK = 1, tseq)
plot(tseq, Land2, type = 'l')
proc.time() - ptm


array <- array(list(), c(floor(length(df$logR_ask)/2000), 1))

###########
### LOOP ##
###########


ptm <- proc.time()
emb_dim = 35 #FNN
tau = 1 #AC / Martinegale
#AlphaShape_array <- array(list(), c(floor(length(df$logR_ask)/2000), 1))
#AlphaShape_array <- vector("list", floor(length(df$logR_ask)/2000))
#AlphaShape_array <- matrix(list(), floor(length(df$logR_ask)/2000), 1)
#AlphaShape_array <- list(1:3, "a", c(TRUE, FALSE, TRUE), c(2.3, 5.9))
AlphaShape_array <- list()

#Takens_array <- array(matrix(), c(floor(length(df$logR_ask)/2000), 1))
#PCATakens_array <- array(list(), c(floor(length(df$logR_ask)/2000), 1))
iter_i = 1
window_size = 1000
gap_size = window_size
days = 5
for (i in seq(0, (length(df$logR_ask)*(days/5))-window_size, gap_size)){
  #for (i in seq(0, 6000, 2000)){
  data <- df[i:(i+window_size)]$logR_ask 
  #print("step1")
  takens = buildTakens(data, embedding.dim = emb_dim, time.lag = tau)
  #print("step2")
  if (all(takens == takens[1])){ #(all(takens == 0)){
    # if all(takens == 0) or any other digit, it does not contain any information, therefore we can take taken[1] comparison
    "case when FX dont move in whole window -> no info for takens"
    DiagAlphaShape <- 0 # add empty birth death diagram (0, 0, 0), because Takens reconstruction failed
    #print("hi")
    plot(0)
  }
  else{
    takens[is.na(takens)] <- 0
    #print("step3")
    pca_takens <- prcomp(takens, scale = TRUE)
    #print("step4")
    DiagAlphaShape <- alphaShapeDiag(pca_takens$x[,1:3], printProgress = TRUE)
    plot(DiagAlphaShape[["diagram"]])
  }
  #print("step5")
  #DiagAlphaShape <- alphaComplexDiag(pca_takens$x[,1:3], printProgress = TRUE)
  
  #AlphaShape_array[iter_i] <- DiagAlphaShape
  AlphaShape_array <- c(AlphaShape_array, DiagAlphaShape)
  #plot(AlphaShape_array[3][["diagram"]])
  #plot(AlphaShape_array[4][["diagram"]], xlim = c(0,1), ylim = c(0,1))
  print("step6")
  print(proc.time() - ptm)
  #print("step7")
  print((iter_i)/length(seq(0, (length(df$logR_ask)*(days/5)), gap_size)))
  #print("step8")
  iter_i = iter_i + 1
}

tt <- AlphaShape_array[2000][["diagram"]]
index_1D <- which(tt[,1] %in% c(1))
index_2D <- which(tt[,1] %in% c(2))
tseq <- seq(0, 3, length = 2000)
# t[index_xD, columns] to choose

#plot all
plot(tt)
plot(tt[,2], tt[,3])

plot(tt[index_1D,2], tt[index_1D, 3])
plot(tt[index_2D,2], tt[index_2D, 3])

test <- data.frame(tt[index_1D,2], tt[index_1D,3])
pca_test <- prcomp(test)

eps = 0.1
plot(tseq, tseq+eps, 'l', xlim = c(0,3), ylim=c(0,3))

ptm <- proc.time()
tseq <- seq(0, 3, length = 2000) # domain
LandA <- silhouette(tt,  p = 1, dimension = 1, tseq)
plot(tseq, LandA, type = 'l')
proc.time() - ptm

ptm <- proc.time()
tseq <- seq(0, 3, length = 1000) # domain
Land2 <- landscape(tt, dimension = 1, KK = 1, tseq)
plot(tseq, Land2, type = 'l')
proc.time() - ptm

#####
##LOOP##

####

#emb_dim = 35 #FNN
#tau = 1 #AC / Martinegale
#AlphaShape_array <- array(list(), c(floor(length(df$logR_ask)/2000), 1))
#AlphaShape_array <- vector("list", floor(length(df$logR_ask)/2000))
#AlphaShape_array <- matrix(list(), floor(length(df$logR_ask)/2000), 1)
#AlphaShape_array <- list(1:3, "a", c(TRUE, FALSE, TRUE), c(2.3, 5.9))
#Landscape_array <- list()
#Landscape_array <- rbind(Landscape_array, data.frame( bla = c(LandA)))
#Landscape_array <- rbind(Landscape_array, data.frame( bla2 = c(Land2)))
#Landscape_array <- matrix(0, ncol = 4130, nrow = 1000)

#Takens_array <- array(matrix(), c(floor(length(df$logR_ask)/2000), 1))
#PCATakens_array <- array(list(), c(floor(length(df$logR_ask)/2000), 1))
#iter_i = 1


ptm <- proc.time()

tseq <- seq(0, 10, length = 300) # domain
AlphaShape_seq <- seq(1, length(AlphaShape_array),1)
Landscape_array <- data.frame(matrix(0, ncol = length(AlphaShape_seq), nrow = length(tseq)))
#Landscape_array[1:length(tseq),1] <- LandA
#Landscape_array[1:length(tseq),2] <- Land2

for (i in AlphaShape_seq){
  
  if (is.null(AlphaShape_array[i][["diagram"]])){
    Landscape <- integer(length(tseq))
    plot(tseq, Landscape, type = 'l')
    Landscape_array[1:length(tseq), i] <- Landscape
  }
  else{
    Landscape <- landscape(AlphaShape_array[i][["diagram"]], dimension = 1, KK = 1, tseq)
    plot(tseq, Landscape, type = 'l')
    Landscape_array[1:length(tseq), i] <- Landscape
  }
  print(proc.time() - ptm)
  print((i)/length(AlphaShape_seq))
  
  #iter_i = iter_i + 1
}

# DiffusionMap instead of PCA -> DiffusionMap perserves geometric strucutre and is robust


library(diffusionMap)

D = dist(scale(takens)) # use Euclidean distance on data
D = as.dist((1- cor(takens))/2) # Correlation distance

eps = epsilonCompute(D, p = 0.01)

dmap = diffuse(D, eps.val=eps, t=1, neigen=2) ## just run with the standard default settings
plot(dmap$X[,1],dmap$X[,2],
     xlab="Diffusion Map Coordinate 1", 
     ylab="Diffusion Map Coordinate 2",
     main="Diffusion Map of OMX30 stocks")
text(dmap$X[,1], dmap$X[,2], labels = colnames(omx), cex = 0.7, pos = 3)

dmap = diffuse(D, eps.val=eps, t=1, neigen=3) ## just run with the standard default settings
plot(dmap)

#library(rgl)
plot3d(dmap$X[,1], dmap$X[,2], dmap$X[,3],
       xlab = "Diffusion Map Coordinate 1",
       ylab = "Diffusion Map Coordinate 2",
       zlab = "Diffusion Map Coordinate 3",
       main = "Diffusion Map of OMXS30 stocks")
text3d(dmap$X[,1], dmap$X[,2], dmap$X[,3], texts = colnames(omx))


####################################
## Quantum Noise ###################
####################################

data <- df_QN[1:2001]$V1
emb_dim = 35 #FNN
tau = 1 #AC / Martinegale
takens = buildTakens(data, embedding.dim = emb_dim, time.lag = tau)
takens[is.na(takens)] <- 0
pca_takens <- prcomp(takens, scale = TRUE)

#DiagAlphaShape <- alphaShapeDiag(pca_takens$x[,1:3], printProgress = TRUE)
QN_complex <- alphaComplexDiag(pca_takens$x[,1:3], printProgress = TRUE)
tseq = seq(0, 10, length = 300)
QN_land <- landscape(QN_complex[["diagram"]], dimension = 1, KK = 1, tseq)
plot(QN_complex[["diagram"]])
plot(tseq, QN_land, type = 'l')

#print(bottleneck(QN_complex[["diagram"]], AlphaShape_array[1][[]]))

ptm <- proc.time()
emb_dim = 35 #FNN
tau = 1 #AC / Martinegale
#AlphaShape_array <- array(list(), c(floor(length(df$logR_ask)/2000), 1))
#AlphaShape_array <- vector("list", floor(length(df$logR_ask)/2000))
#AlphaShape_array <- matrix(list(), floor(length(df$logR_ask)/2000), 1)
#AlphaShape_array <- list(1:3, "a", c(TRUE, FALSE, TRUE), c(2.3, 5.9))
QN_complex_array <- list()

#Takens_array <- array(matrix(), c(floor(length(df$logR_ask)/2000), 1))
#PCATakens_array <- array(list(), c(floor(length(df$logR_ask)/2000), 1))
iter_i = 1
for (i in seq(0, length(df_QN$V1)-1000, 1000)){
  #for (i in seq(0, 6000, 2000)){
  data <- df_QN$V1[i:(i+1000)] 
  takens = buildTakens(data, embedding.dim = emb_dim, time.lag = tau)
  takens[is.na(takens)] <- 0
  pca_takens <- prcomp(takens, scale = TRUE)
  
  DiagAlphaShape <- alphaShapeDiag(pca_takens$x[,1:3], printProgress = TRUE)
  #DiagAlphaShape <- alphaComplexDiag(pca_takens$x[,1:3], printProgress = TRUE)
  plot(DiagAlphaShape[["diagram"]])
  #AlphaShape_array[iter_i] <- DiagAlphaShape
  QN_complex_array <- c(QN_complex_array, DiagAlphaShape)
  #plot(AlphaShape_array[3][["diagram"]])
  #plot(AlphaShape_array[4][["diagram"]], xlim = c(0,1), ylim = c(0,1))
  print(proc.time() - ptm)
  print((iter_i)/length(seq(0, length(df_QN$V1)-2000, 2000)))
  
  iter_i = iter_i + 1
}

############# LANDSCAPE OF QN

ptm <- proc.time()

tseq <- seq(0, 10, length = 300) # domain
QN_seq <- seq(1, length(QN_complex_array),1)
Landscape_QN_array <- data.frame(matrix(0, ncol = length(QN_seq), nrow = length(tseq)))
#Landscape_array[1:length(tseq),1] <- LandA
#Landscape_array[1:length(tseq),2] <- Land2

for (i in QN_seq){
  
  
  Landscape <- landscape(QN_complex_array[i][["diagram"]], dimension = 1, KK = 1, tseq)
  plot(tseq, Landscape, type = 'l')
  
  Landscape_QN_array[1:length(tseq), i] <- Landscape
  
  print(proc.time() - ptm)
  print((i)/length(QN_seq))
  
  #iter_i = iter_i + 1
}

Landscape_QN_array[1:length(tseq), 1]
typeof(Landscape_QN_array)

#write.csv(Landscape_QN_array, file = "W2000GW_Landscape_QN_AlphaComplex_dim1_KK1_tseq10l300.csv", row.names = FALSE)
#write.csv(Landscape_array, file = "W2000GW_Landscape_AlphaComplex10_embdim35_tau1.csv", row.names = FALSE)


### TDA mean landscape test

N <- 4000
XX1 <- circleUnif(N / 2)
XX2 <- circleUnif(N / 2, r = 2) + 3
X <- rbind (XX1, XX2)

maxscale <- 10
m <- 80 # subsample
n <- 10 # we will compute n landscapes using subsamples of size m
tseq <- seq(0, maxscale, length = 300)
#store rips diags
Diags <- list()
#store n landscapes
Lands <- matrix(0, nrow = n, ncol = length(tseq))

for (i in seq_len(n)){
  subX <- X[sample(seq_len(N), m), ]
  Diags[[i]] <- ripsDiag(subX, maxdimension = 1, maxscale = 5)
  Lands[i, ] <- landscape(Diags[[i]][["diagram"]], dimension = 1, KK = 1, tseq)
}

bootLand <- multipBootstrap(Lands, B = 100, alpha = 0.05, parallel = FALSE)

plot(tseq, bootLand[["mean"]], main = "Mean Landscape with 95 % band")
polygon(c(tseq, rev(tseq)), c(bootLand[["band"]][, 1], rev(bootLand[["band"]][, 2])), col = "pink")
lines(tseq, bootLand[["mean"]], lwd = 2, col = 2)

tseq <- seq(0, 10, length = 300) # domain
bootLand <- multipBootstrap(t(Landscape_QN_array), B = 4130, alpha = 0.05, parallel = FALSE)
plot(tseq, bootLand[["mean"]], main = "Mean Landscape with 95 % band")
polygon(c(tseq, rev(tseq)), c(bootLand[["band"]][, 1], rev(bootLand[["band"]][, 2])), col = "pink")
lines(tseq, bootLand[["mean"]], lwd = 2, col = 2)


## Remove noise from single complex
eps = 0.15
booleans = ((DiagAlphaShape[["diagram"]][,2]+eps) < DiagAlphaShape[["diagram"]][,3])
x <- DiagAlphaShape[["diagram"]][,2][booleans]
y <- DiagAlphaShape[["diagram"]][,3][booleans]
dim <- DiagAlphaShape[["diagram"]][,1][booleans]


## Remove noise features and plot persistence diagram
eps = 0.15
booleans = ((AlphaShape_array[1][["diagram"]][,2]+eps) < AlphaShape_array[1][["diagram"]][,3])
x <- AlphaShape_array[1][["diagram"]][,2][booleans]
y <- AlphaShape_array[1][["diagram"]][,3][booleans]
dim <- AlphaShape_array[1][["diagram"]][,1][booleans]

#booleans = ((QN_complex_array[1][["diagram"]][,2]+eps) < QN_complex_array[1][["diagram"]][,3])
#x <- QN_complex_array[1][["diagram"]][,2][booleans]
#y <- QN_complex_array[1][["diagram"]][,3][booleans]
#dim <- QN_complex_array[1][["diagram"]][,1][booleans]

#plot(x,y)  
#plot(x = c(0,5), y = c(0,5), type = 'l')
#lines(x = c(0,5), y = c(0,5))
points(x,y)

complex_noiseRed <- list()
complex_noiseRed <- c(dimension = list(dim), Birth = list(x), Death = list(y))
complex_noiseRed <- data.frame(complex_noiseRed)
#complex_noiseRed <- c(Birth = list(x))
#complex_noiseRed <- c(complex_noiseRed, dim)
max_y = max(complex_noiseRed[,3][2:length(complex_noiseRed[,3])])
max_x = max(complex_noiseRed[,2][1:length(complex_noiseRed[,2])])
max_xy = max(max_x, max_y)
#plot(complex_noiseRed[,2], complex_noiseRed[,3])
#plot(complex_noiseRed[,3] ~ complex_noiseRed[,2], 
#     xlim = c(0, max_x), ylim=c(0,max_y))
plot(complex_noiseRed[,3] ~ complex_noiseRed[,2], 
     col =  ifelse(complex_noiseRed[,1] == 0, 'black', ifelse(complex_noiseRed[,1] == 1, 'red', 'blue')),
     pch = ifelse(complex_noiseRed[,1] == 0, 16, ifelse(complex_noiseRed[,1] == 1, 2, 5)),
     lwd = 2,
     xlim = c(0, max_x), ylim=c(0,max_y),
     xlab = "Birth",
     ylab = "Death")
lines(x = c(0, max_xy), y = c(0, max_xy))

t_land = landscape(complex_noiseRed, dimension = 1, KK = 1, tseq) #works
plot(tseq, t_land, type = 'l', ylab = "epsilon")


################
### Noise reduced LANDSCAPE
################



ptm <- proc.time()

eps = 0.3
tseq <- seq(0, 20, length = 300) # domain
AlphaShape_seq <- seq(1, length(AlphaShape_array),1)
Landscape_array <- data.frame(matrix(0, ncol = length(AlphaShape_seq), nrow = length(tseq)))
#Landscape_array[1:length(tseq),1] <- LandA
#Landscape_array[1:length(tseq),2] <- Land2

for (i in AlphaShape_seq){
  booleans = ((AlphaShape_array[i][["diagram"]][,2]+eps) < AlphaShape_array[1][["diagram"]][,3])
  x <- AlphaShape_array[i][["diagram"]][,2][booleans]
  y <- AlphaShape_array[i][["diagram"]][,3][booleans]
  dim <- AlphaShape_array[i][["diagram"]][,1][booleans]
  
  complex_noiseRed <- list()
  complex_noiseRed <- c(dimension = list(dim), Birth = list(x), Death = list(y))
  complex_noiseRed <- data.frame(complex_noiseRed)
  
  Landscape <- landscape(complex_noiseRed, dimension = 1, KK = 1, tseq)
  plot(tseq, Landscape, type = 'l')
  
  Landscape_array[1:length(tseq), i] <- Landscape
  
  print(proc.time() - ptm)
  print((i)/length(AlphaShape_seq))
  
  #iter_i = iter_i + 1
}


############################
### TDA Toy-examples #######
############################

x <- seq(0, 16*pi, length = 1000)
x2 <- seq(0, pi, length = 10000)
x2 <- seq(0, pi, length = 1000)
x3 <- seq(0, 10, length = 500)

# cirlce equation

x_c <- seq(-7, 7, length = 500)
y_c <- abs(sqrt(49-x_c^2))/7

y = sin(x) #circle
y = 10*sin(10*x)+x2 #void
y = sin(4*x)+x2 #void
y = 4*sin(x2)*sin(32*x2)+ 4*x2
y = 10*y_c * sin(4*x) + 4*x2 #ball
y = 10*y_c * sin(4*x) + 4*x2 #ball
y = (4*sin(x2)*sin(32*x2) + 4*x2) *sin(4*x2)
y = sin(2*x2)

y <- jitter(y, factor = 10.0, amount = 0) # adding noise

# quantize
s <- 0.5
y <- round(y*s)
y <- y/s

y <- rollmean(y, 200) #rolling mean
plot(y, xlab = "")#, main = "f = 10")#, type = 'o')
emb_dim = 3
tau = 10#1000
takens = buildTakens(y, embedding.dim = emb_dim, time.lag = tau)
plot(takens[,1], takens[,2], xlab = "Takens Coordinate 1", ylab = "Takens Coordinate 2", main = expression(tau == 100))
plot3d(takens[,1], takens[,2], takens[,3],
       xlab = "Takens Coordinate 1",
       ylab = "Takens Coordinate 2",
       zlab = "Takens Coordinate 3",
       main = "Takens delay embedding")
pca_takens <- prcomp(takens, scale = TRUE)
plot(pca_takens$x[,1], pca_takens$x[,2], xlab = "PCA Takens Coordinate 1", ylab = "PCA Takens Coordinate 2", main = expression(tau == 100))
plot3d(pca_takens$x[,1], pca_takens$x[,2], pca_takens$x[,3],
       xlab = "PCA Takens Coordinate 1",
       ylab = "PCA Takens Coordinate 2",
       zlab = "PCA Takens Coordinate 3",
       main = "PCA Takens delay embedding")
#text3d(dmap$X[,1], dmap$X[,2], dmap$X[,3], texts = colnames(omx))

DiagAlphaShape <- alphaShapeDiag(takens[,1:3], printProgress = TRUE)
DiagAlphaShape <- alphaShapeDiag(pca_takens$x[,1:3], printProgress = TRUE)

#DiagAlphaShape <- alphaShapeDiag()
#DiagAlphaShape <- ripsDiag(takens[,1:2], printProgress = TRUE, maxdimension = 1, maxscale = 1)
#DiagAlphaShape <- alphaComplexDiag(takens[,1:2], printProgress = TRUE)

plot(DiagAlphaShape[["diagram"]], main = expression(tau == 1))

## Remove noise features and plot persistence diagram
eps = 0.000
booleans = ((DiagAlphaShape[["diagram"]][,2]+eps) < DiagAlphaShape[["diagram"]][,3])
x <- DiagAlphaShape[["diagram"]][,2][booleans]
y <- DiagAlphaShape[["diagram"]][,3][booleans]
dim <- DiagAlphaShape[["diagram"]][,1][booleans]

complex_noiseRed <- list()
complex_noiseRed <- c(dimension = list(dim), Birth = list(x), Death = list(y))
complex_noiseRed <- data.frame(complex_noiseRed)
#complex_noiseRed <- c(Birth = list(x))
#complex_noiseRed <- c(complex_noiseRed, dim)
max_y = max(complex_noiseRed[,3][2:length(complex_noiseRed[,3])])
max_x = max(complex_noiseRed[,2][1:length(complex_noiseRed[,2])])
max_xy = max(max_x, max_y)
max_y = max_x
#plot(complex_noiseRed[,2], complex_noiseRed[,3])
#plot(complex_noiseRed[,3] ~ complex_noiseRed[,2], 
#     xlim = c(0, max_x), ylim=c(0,max_y))
plot(complex_noiseRed[,3] ~ complex_noiseRed[,2], 
     col =  ifelse(complex_noiseRed[,1] == 0, 'black', ifelse(complex_noiseRed[,1] == 1, 'red', 'blue')),
     pch = ifelse(complex_noiseRed[,1] == 0, 16, ifelse(complex_noiseRed[,1] == 1, 2, 5)),
     lwd = 2,
     xlim = c(0, 3), ylim = c(0, 3),#c(0, max_x), ylim=c(0,max_y),
     xlab = "Birth",
     ylab = "Death")#,
#main = expression(tau == 20))
lines(x = c(0, max_xy), y = c(0, max_xy))
lines(x = c(0, max_x), y = c(0, max_x))
lines(x = c(0, 3), y = c(0, 3))

tseq <- seq(0, 10, length = 300)
Landscape <- landscape(DiagAlphaShape[["diagram"]], dimension = 1, KK = 1, tseq)
plot(tseq, Landscape, type = 'l', ylim = c(0, 0.4))#, main = expression(tau == 20))#, ylim=c(0,1.8))#,
#ylim = c(0, 0.4))

######### PLOT EXAMPLE

eps = 0.000
x <- c(0, 1, 2)
y <- c(3, 2, 2.2)
dim <- c(0,1,2)

complex_noiseRed <- list()
complex_noiseRed <- c(dimension = list(dim), Birth = list(x), Death = list(y))
complex_noiseRed <- data.frame(complex_noiseRed)

plot(complex_noiseRed[,3] ~ complex_noiseRed[,2], 
     col =  ifelse(complex_noiseRed[,1] == 0, 'black', ifelse(complex_noiseRed[,1] == 1, 'red', 'blue')),
     pch = ifelse(complex_noiseRed[,1] == 0, 16, ifelse(complex_noiseRed[,1] == 1, 2, 5)),
     lwd = 2,
     xlim = c(0, 3), ylim = c(0, 3),#c(0, max_x), ylim=c(0,max_y),
     xlab = "Birth",
     ylab = "Death")#,
#main = expression(tau == 20))
lines(x = c(0, max_xy), y = c(0, max_xy))
lines(x = c(0, max_x), y = c(0, max_x))
lines(x = c(0, 3), y = c(0, 3))
legend(2.5,1, legend = c(expression('H'[0]), expression('H'[1]), expression('H'[2])), 
       col = c("black", "red", "blue"), pch=c(16, 2, 5), 
       cex = 1.2, lty = c(NA, NA, NA), lwd = 2)


# PCA SCREE PLOT - PCA diagnostics using variance
barplot(pca_takens$sdev^2, names.arg = seq(1,length(pca_takens$sdev^2)), main = "", xlab = "PC component", ylab = "Variation")

pcaCharts <- function(x) {
  x.var <- x$sdev ^ 2
  x.pvar <- x.var/sum(x.var)
  print("proportions of variance:")
  print(x.pvar)
  
  par(mfrow=c(2,2))
  plot(x.pvar,xlab="Principal component", ylab="Proportion of variance explained", ylim=c(0,1), type='b')
  plot(cumsum(x.pvar),xlab="Principal component", ylab="Cumulative Proportion of variance explained", ylim=c(0,1), type='b')
  screeplot(x)
  screeplot(x,type="l")
  par(mfrow=c(1,1))
}
