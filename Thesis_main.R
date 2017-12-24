
"
Main code for the thesis

- Generates persistence diagrams for all windows
- EURUSD and QN data code seperation for clarity
- Function for removing noise in persistence diagram
- Loop for removing noise removed persistence landscapes


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


####################
### ALPHA COMPLEX ##
####################


ptm <- proc.time()

# Parameter selection
emb_dim = 35 #FNN
tau = 1 #AC / Martinegale
iter_i = 1
window_size = 1000
gap_size = window_size
days = 5


AlphaShape_array <- list() # Empty list to append complex

for (i in seq(0, (length(df$logR_ask)*(days/5))-window_size, gap_size)){
  data <- df[i:(i+window_size)]$logR_ask 
  takens = buildTakens(data, embedding.dim = emb_dim, time.lag = tau)
  if (all(takens == takens[1])){ #(all(takens == 0)){
    # if all(takens == 0) or any other digit, it does not contain any information, 
    # therefore we can take taken[1] comparison
    
    "case when FX dont move in whole window -> no info for takens"
    DiagAlphaShape <- 0 # add empty birth death diagram (0, 0, 0), because Takens reconstruction failed
    #print("hi")
    plot(0)
  }
  else{ #Normal case
    takens[is.na(takens)] <- 0
    pca_takens <- prcomp(takens, scale = TRUE)
    DiagAlphaShape <- alphaShapeDiag(pca_takens$x[,1:3], printProgress = TRUE)
    plot(DiagAlphaShape[["diagram"]])
  }
  
  AlphaShape_array <- c(AlphaShape_array, DiagAlphaShape) #Add complex to array
  
  # Print out progression
  print(proc.time() - ptm)
  print((iter_i)/length(seq(0, (length(df$logR_ask)*(days/5)), gap_size)))
  iter_i = iter_i + 1
}

#####################
## LANDSCAPE ARRAY ##
#####################

ptm <- proc.time()

tseq <- seq(0, 10, length = 300) # domain
AlphaShape_seq <- seq(1, length(AlphaShape_array),1) #lenth of AlphaShape_array
Landscape_array <- data.frame(matrix(0, ncol = length(AlphaShape_seq), nrow = length(tseq))) #allocate memory for Landscape memory

for (i in AlphaShape_seq){
  
  # if AlphaShape_array[i] is null then create landscape of 0
  # This is for example when all log-returns are 0 --> no topological properties
  if (is.null(AlphaShape_array[i][["diagram"]])){
    Landscape <- integer(length(tseq))
    plot(tseq, Landscape, type = 'l')
    Landscape_array[1:length(tseq), i] <- Landscape
  }
  else{ # Normal case
    Landscape <- landscape(AlphaShape_array[i][["diagram"]], dimension = 1, KK = 1, tseq)
    plot(tseq, Landscape, type = 'l')
    Landscape_array[1:length(tseq), i] <- Landscape
  }
  
  # Print Progress
  print(proc.time() - ptm)
  print((i)/length(AlphaShape_seq))
  
}


###########################################
## Quantum Noise Complex###################
###########################################

ptm <- proc.time()

emb_dim = 35 #FNN
tau = 1 #AC / Martinegale
window_size = 1000
gap_size = window_size
iter_i = 1

QN_complex_array <- list()

for (i in seq(0, length(df_QN$V1)-window_size, gap_size)){
  data <- df_QN$V1[i:(i+window_size)] 
  takens = buildTakens(data, embedding.dim = emb_dim, time.lag = tau)
  takens[is.na(takens)] <- 0
  pca_takens <- prcomp(takens, scale = TRUE)
  DiagAlphaShape <- alphaShapeDiag(pca_takens$x[,1:3], printProgress = TRUE)
  plot(DiagAlphaShape[["diagram"]])
  QN_complex_array <- c(QN_complex_array, DiagAlphaShape)
  
  # Print Progress
  print(proc.time() - ptm)
  print((iter_i)/length(seq(0, length(df_QN$V1)-window_size, gap_size)))
  
  iter_i = iter_i + 1
}


###########################################
## LANDSCAPE OF QN ########################
###########################################

ptm <- proc.time()

tseq <- seq(0, 10, length = 300) # domain
QN_seq <- seq(1, length(QN_complex_array),1)
Landscape_QN_array <- data.frame(matrix(0, ncol = length(QN_seq), nrow = length(tseq)))

for (i in QN_seq){
  
  Landscape <- landscape(QN_complex_array[i][["diagram"]], dimension = 1, KK = 1, tseq)
  plot(tseq, Landscape, type = 'l')
  
  Landscape_QN_array[1:length(tseq), i] <- Landscape
  
  # Print Progress
  print(proc.time() - ptm)
  print((i)/length(QN_seq))
  
}


#write.csv(Landscape_QN_array, file = "W2000GW_Landscape_QN_AlphaComplex_dim1_KK1_tseq10l300.csv", row.names = FALSE)
#write.csv(Landscape_array, file = "W2000GW_Landscape_AlphaComplex10_embdim35_tau1.csv", row.names = FALSE)


###########################################
## MEAN LANDSCAPES ########################
###########################################

array_item <- Landscape_QN_array
b_iter <- length(array_item)
tseq <- seq(0, 10, length = 300) # domain
bootLand <- multipBootstrap(t(array_item), B = b_iter, alpha = 0.05, parallel = FALSE)
plot(tseq, bootLand[["mean"]], main = "Mean Landscape with 95 % band")
polygon(c(tseq, rev(tseq)), c(bootLand[["band"]][, 1], rev(bootLand[["band"]][, 2])), col = "pink")
lines(tseq, bootLand[["mean"]], lwd = 2, col = 2)


##################################################################
## REMOVE NOISE FROM PERSISTENCE DIAGRAMS ########################
##################################################################

plotAlpha <- function(DiagAlphaShape, eps = 0){
  
  ## Remove noise features and plot persistence diagram
  
  # Create boolean table of relevant homology groups
  booleans = ((DiagAlphaShape[["diagram"]][,2]+eps) < DiagAlphaShape[["diagram"]][,3])
  
  # Use only relevant homology groups
  x <- DiagAlphaShape[["diagram"]][,2][booleans]
  y <- DiagAlphaShape[["diagram"]][,3][booleans]
  dim <- DiagAlphaShape[["diagram"]][,1][booleans]
  
  # Create new list with only noise reduced homology groups
  complex_noiseRed <- list()
  complex_noiseRed <- c(dimension = list(dim), Birth = list(x), Death = list(y))
  complex_noiseRed <- data.frame(complex_noiseRed)
  
  # Set parameters for plotting
  max_y = max(complex_noiseRed[,3][2:length(complex_noiseRed[,3])])
  max_x = max(complex_noiseRed[,2][1:length(complex_noiseRed[,2])])
  max_xy = max(max_x, max_y)
  
  # Plot noise reduced Birth-Death Diagram
  plot(complex_noiseRed[,3] ~ complex_noiseRed[,2], 
       col =  ifelse(complex_noiseRed[,1] == 0, 'black', ifelse(complex_noiseRed[,1] == 1, 'red', 'blue')),
       pch = ifelse(complex_noiseRed[,1] == 0, 16, ifelse(complex_noiseRed[,1] == 1, 2, 5)),
       lwd = 2,
       xlim = c(0, max_x), ylim = c(0, max_y),#c(0, max_x), ylim=c(0,max_y),
       xlab = "Birth",
       ylab = "Death")#,
  lines(x = c(0, max_xy), y = c(0, max_xy))
  lines(x = c(0, max_x), y = c(0, max_x))
  
  return(complex_noiseRed)
}

################################
### Noise reduced LANDSCAPE ####
################################

ptm <- proc.time()

eps = 0.3
tseq <- seq(0, 20, length = 300) # domain
AlphaShape_seq <- seq(1, length(AlphaShape_array),1)
Landscape_array <- data.frame(matrix(0, ncol = length(AlphaShape_seq), nrow = length(tseq)))


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
  
}


##############################
### PCA SCREE DIAGNOSTICS ####
##############################

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
