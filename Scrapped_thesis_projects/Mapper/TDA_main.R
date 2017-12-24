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
library(dimRed)


# Define plotting function for readin file
DimRed2d <- function(d, d_str, k, add_str = FALSE, labels = ""){
  x_str = paste(d_str, " Coordinate 1", sep = "")
  y_str = paste(d_str, " Coordinate 2", sep = "")
  if (add_str == FALSE){
    main_str = paste(d_str)
  }else {
    main_str = paste(d_str, " of ", add_str, sep = "")
  }
  
  
  results = tryCatch({
    # Cluster on distances
    cluster = hclust(dist(d))
    #plot(cluster, labels = colnames(omx))
    #abline(h=0.00005, col = 'blue', lwd=1)
    
    # Cluster on Correlation Distance
    #cluster_corr = hclust(D, method = "complete")
    #plot(cluster_corr)
    
    # DiffusionMap and Cluster viz
    clusterings = cutree(cluster, k = k)
    #clusterings
    
    #plot(dmap$X[,1], dmap$X[,2], col = clusterings)
    #text(dmap$X[,1], dmap$X[,2], labels = colnames(omx), cex = 0.7, pos = 3)
    
    plot(d$X,d$Y, col = clusterings,
         xlab=x_str, 
         ylab=y_str,
         main=main_str)
    text(d$X, d$Y, labels = labels, cex = 0.7, pos = 3)
    #return(cluster)
    }, error = function(error_handler){
          #message("YEST")
          cluster = hclust(dist(d$X[,1:2]))
          clusterings = cutree(cluster, k = k)
          plot(d$X[,1],d$X[,2], col = clusterings, 
               xlab=x_str, 
               ylab=y_str,
               main=main_str)
          text(d$X[,1], d$X[,2], labels = labels, cex = 0.7, pos = 3)
          #return(cluster)
  })
}
colnames(omx)
DimRed3d <- function(d, d_str, k, p_size = 10, add_str = FALSE, labels = ""){
  x_str = paste(d_str, " Coordinate 1", sep = "")
  y_str = paste(d_str, " Coordinate 2", sep = "")
  z_str = paste(d_str, " Coordinate 3", sep = "")
  if (add_str == FALSE){
    main_str = paste(d_str)
  }else {
    main_str = paste(d_str, " of ", add_str, sep = "")
  }
  
  results = tryCatch({
    # Cluster on distances
    cluster = hclust(dist(d))
    clusterings = cutree(cluster, k = k)
    
    plot3d(d$X, d$Y, d$Z, col = clusterings, size = p_size,
           xlab = x_str,
           ylab = y_str,
           zlab = z_str,
           main = main_str)
    text3d(d$X, d$Y, d$Z, texts = labels)
    return(cluster)
    
  }, error = function(error_handler){
    #message("YEST")
    cluster = hclust(dist(d$X[,1:3]))
    clusterings = cutree(cluster, k = k)
    plot3d(d$X[,1],d$X[,2], d$X[,3], col = clusterings, 
         xlab=x_str, 
         ylab=y_str,
         zlab=z_str,
         main=main_str)
    text3d(d$X[,1], d$X[,2], d$X[,3], texts = labels, cex = 0.7, pos = 3)
    return(cluster)
  })
}


TransMat <- function(d){
  # Transition Probability Matrix as constructed
  # by Phoa W. - Portfolio Concentration and the Geometry of Co-movement
  # WARNING: might already be incorporated in diffusion map algo
  K = 1 + d
  K = as.matrix(K)
  rowsum = rowSums(K)
  P = K/rowsum
  return(P)
}


# READIN DATA
setwd("C:/Users/extra23/Desktop/Filer/MEX/Local laptop backup/Other Financial Data/TDA")

#raw-data
omx = read.zoo("omxs30_raw.csv", sep = ',', header = TRUE, format = "%Y-%m-%d") 
omx = read.zoo("omxs30_raw_ffill.csv", sep = ',', header = TRUE, format = "%Y-%m-%d")
#log-return
omx = read.zoo("OMXS30_perc_diff.csv", sep = ',', header = TRUE, format = "%Y-%m-%d") 
omx = read.zoo("OMXS30_perc_diff_NaN0.csv", sep = ',', header = TRUE, format = "%Y-%m-%d") 

#DROP X.OMX
omx$X.OMX <- NULL
#FX
FX = read.zoo("FX_raw.csv", sep = ',', header = TRUE, format = "%Y-%m-%d") 
FX = read.zoo("FX_log.csv", sep = ',', header = TRUE, format = "%Y-%m-%d") 

# Convert to df
omx_df <- as.data.frame(omx)
class(omx_df)
FX_df <- as.data.frame(FX)
class(FX)
plot(omx[,30], main = colnames(omx)[30])


# Data indexing
#data <- omx[4000:4350]
#data <- omx[(length(omx[,1])-252):length(omx[,1])] #approx 1-yr
data <- omx[3900:4350] #ok data set for tda
data <- omx[3000:4350] # crash
data <- omx[3050:3100]
data <- omx[3916:4422]
#data <- omx

data <- FX[3000:4350]
data <- FX[1000:1150]

# Data indexing with dates instead - CHECK THIS
# Same as omx[3900:4350]
windowed <- window(omx, start = as.Date("2015-04-09"),end = as.Date("2017-01-19"))
data <- as.data.frame(windowed)

D = dist(scale(data)) # use Euclidean distance on data
D = as.dist((1- cor(data))/2) # Correlation distance
#P = TransMat(D)

eps = epsilonCompute(D, p = 0.01)

# Visually inspect to find epsilon in linear part of plot
eps_list = c()
for (i in (seq(0,1000))){
  eps_list[i] <- exp(-sum(((D^2)/((i/10)^2))))
}
plot(eps_list)
plot(eps_list, log ="y")
eps_list[150]


dmap = diffuse(D, eps.val=0.7, t=1, neigen=2) ## just run with the standard default settings
DimRed2d(dmap, "Diffusion Map", 5, labels = colnames(omx))
DimRed2d(dmap, "Diffusion Map", 5, labels = colnames(FX))


for (i in (seq(0,4700, len=(47+1)))){
  data <- FX[(i):(252+i)]
  #data <- omx[i:(252+i)]
  idx <- index(data)[1]
  #data <- as.data.frame(windowed)
  #D = dist(scale(data)) # use Euclidean distance on data
  D = as.dist((1- cor(data))/2) # Correlation distance
  #D = cor(data)
  #P = TransMat(D)
  dmap = diffuse(D, eps.val=400, t=1, neigen=2) ## just run with the standard default settings
  #DimRed2d(dmap, "Diffusion Map", 5, idx, labels = colnames(omx))
  DimRed2d(dmap, "Diffusion Map", 5, idx, labels = colnames(FX))
  #plot.dmap(x = dmap)
  Sys.sleep(1)
}


####TDA####################
library("TDA")
# Diag data Euclidean
par(mfrow=c(1,2))
DiagLim <- 0.5
maxdimension <- 1

DiagOMX <- ripsDiag(data, maxdimension, DiagLim, printProgress = TRUE)
plot(DiagOMX[["diagram"]])
#DiagAlphaOMX <- alphaComplexDiag(data, printProgress = TRUE)
#plot(DiagAlphaOMX[["diagram"]])

# Diag data Corr Distance - No alphaComplex?
par(mfrow=c(1,2))
DiagOMXcorr <- ripsDiag(D, maxdimension = 1, maxscale = 5, dist = "arbitrary", printProgress = TRUE)
plot(DiagOMXcorr[["diagram"]])
#DiagAlphaOMXcorr <- alphaComplexDiag(D, printProgress = TRUE)
#plot(DiagAlphaOMXcorr[["diagram"]])

tseq <- seq(0, maxscale, length = 1000) #domain
Land <- landscape(DiagOMXcorr[["diagram"]], dimension = 1, KK = 1, tseq)
Sil <- silhouette(DiagOMXcorr[["diagram"]], p = 1, dimension = 1, tseq)

plot(tseq, Land, type = "l")
plot(tseq, Sil, type = "l")




# Diag dmap data
Diag <- ripsDiag(X = dmap$X, maxdimension = 1, maxscale = 5,
                 library = "GUDHI", printProgress = FALSE)
par(mfrow=c(1,2))
plot(dmap$X, xlab="", ylab="")
plot(Diag[["diagram"]])
###########################
i <- 47
maxdimension <- 2
DiagLim <- 0.6
for (i in (seq(0,9400, len=(940+1)))){
  data <- omx[(i):(50+i)]
  #data <- omx[(1800+i):(1850+i)]
  idx <- index(data)[1]
  #data <- as.data.frame(windowed)
  #D = data
  D = dist(scale(data)) # use Euclidean distance on data
  D = as.dist((1- cor(data, method = "kendall"))/2) # Correlation distance
  #P = TransMat(D)
  #dmap = diffuse(D, eps.val=400, t=1, neigen=2) ## just run with the standard default settings
  #DimRed2d(dmap, "Diffusion Map", 5, idx, labels = colnames(FX))
  #plot.dmap(x = dmap)
  DiagOMX <- ripsDiag(D, maxdimension, DiagLim, "arbitrary",printProgress = TRUE)
  
  #tseq <- seq(0, 0.8, length = 1000) #domain
  #Land <- landscape(DiagOMX[["diagram"]], dimension = 1, KK = 1, tseq)
  #Sil <- silhouette(DiagOMX[["diagram"]], p = 1, dimension = 1, tseq)
  #plot(tseq, Land, type = "l", main = idx, ylim = c(0, 0.05))
  #plot(tseq, Sil, type = "l", main = idx) 
  plot(DiagOMX[["diagram"]], main = idx)
  #plot(DiagOMX[["diagram"]], rotated = TRUE, main = idx)
  #plot(DiagOMX[["diagram"]], barcode = TRUE, main = idx)
  Sys.sleep(1)
}

#2000-02-29 -> 2002-09-30
#2007-04-30 -> 2009-01-30
#2011-04-29 -> 2009-11-30
#2015-02-27 -> 2016-06-30

i <- 2924
maxdimension <- 30
DiagLim <- 0.6
for (i in c(47,385, 722,
            1959,2131,2303,
            2924,2969,3014,
            3912,4058,4204)){
  data <- omx[(i):(50+i)]
  #data <- omx[722:1959]
  #data <- omx[(1800+i):(1850+i)]
  idx <- index(data)[1]
  #data <- as.data.frame(windowed)
  #D = data
  D = dist(scale(data)) # use Euclidean distance on data
  D = as.dist((1- cor(data, method = "pearson"))/2) # Correlation distance
  #D = cor(data, method = "kendall")
  #D = cor(data)
  #P = TransMat(D)
  #dmap = diffuse(D, eps.val=400, t=1, neigen=2) ## just run with the standard default settings
  #DimRed2d(dmap, "Diffusion Map", 5, idx, labels = colnames(FX))
  #plot.dmap(x = dmap)
  DiagOMX <- ripsDiag(D, maxdimension, DiagLim, "arbitrary",printProgress = TRUE)
  plot_name <- paste("PH","_OMXS30_",idx, "_50d",".png", sep = "")
  png(plot_name)
  #tseq <- seq(0, 0.8, length = 1000) #domain
  #Land <- landscape(DiagOMX[["diagram"]], dimension = 1, KK = 1, tseq)
  #Sil <- silhouette(DiagOMX[["diagram"]], p = 1, dimension = 1, tseq)
  #plot(tseq, Land, type = "l", main = idx, ylim = c(0, 0.05))
  #plot(tseq, Sil, type = "l", main = idx) 
  idx <- "Dotcom Crash"
  plot(DiagOMX[["diagram"]], main = idx)
  #plot(DiagOMX[["diagram"]], rotated = TRUE, main = idx)
  #plot(DiagOMX[["diagram"]], barcode = TRUE, main = idx)
  dev.off()
  Sys.sleep(1)
}




##############PHASE SPACE RECONSTRUCTION
library(nonlinearTseries)

# Build the Takens vector for the Henon map using the x-coordinate time series
h = henon(n.sample=  3000,n.transient= 100, a = 1.4, b = 0.3,
          start = c(0.73954883, 0.04772637), do.plot = FALSE)
takens = buildTakens(h$x,embedding.dim=2,time.lag=1)
# using the x-coordinate time series we are able to reconstruct
# the state space of the Henon map
plot(takens)
plot(h$y)

h = henon(n.sample=  3000,n.transient= 100, a = 1.4, b = 0.3,
          start = c(0.73954883, 0.04772637), do.plot = FALSE)
takens = buildTakens(h$x,embedding.dim=3,time.lag=1)
plot3d(takens)

takens = buildTakens(omx[,30][2000:2252], embedding.dim = 2, time.lag=1)
plot(takens)


for (i in (seq(0,9400, len=(940+1)))){
  data <- omx[(i):(50+i)]
  idx <- index(data)[1]
  #data <- as.data.frame(windowed)
  #D = dist(scale(data)) # use Euclidean distance on data
  #D = as.dist((1- cor(data))/2) # Correlation distance
  #P = TransMat(D)
  #dmap = diffuse(D, eps.val=400, t=1, neigen=2) ## just run with the standard default settings
  #DimRed2d(dmap, "Diffusion Map", 5, idx, labels = colnames(FX))
  #plot.dmap(x = dmap)
  #DiagOMX <- ripsDiag(data, maxdimension, DiagLim, printProgress = TRUE)
  #plot(DiagOMX[["diagram"]], main = idx)
  takens = buildTakens(omx[,30][i:(50+i)], embedding.dim = 3, time.lag=10)
  #plot(takens)
  DiagOMX <- ripsDiag(takens, maxdimension, DiagLim, "arbitrary",printProgress = TRUE)
  plot(DiagOMX[["diagram"]], main = idx)
  Sys.sleep(1)
  plot3d(takens)
  
}

i <- 47
maxdimension <- 2
DiagLim <- 50
for (i in c(47,385, 722,
            1959,2131,2303,
            2924,2969,3014,
            3912,4058,4204)){
  data <- omx[(i):(50+i)]
  #data <- omx[(1800+i):(1850+i)]
  idx <- index(data)[1]
  
  DiagOMX <- ripsDiag(D, maxdimension, DiagLim, "arbitrary",printProgress = TRUE)
  plot_name <- paste("PH","_FX30_",idx, "_50d",".png", sep = "")
  png(plot_name)
  #tseq <- seq(0, 0.8, length = 1000) #domain
  #Land <- landscape(DiagOMX[["diagram"]], dimension = 1, KK = 1, tseq)
  #Sil <- silhouette(DiagOMX[["diagram"]], p = 1, dimension = 1, tseq)
  #plot(tseq, Land, type = "l", main = idx, ylim = c(0, 0.05))
  #plot(tseq, Sil, type = "l", main = idx) 
  takens = buildTakens(omx[,30][i:(50+i)], embedding.dim = 3, time.lag=1)
  #plot(takens)
  DiagOMX <- ripsDiag(takens, maxdimension, DiagLim, "arbitrary",printProgress = TRUE)
  plot(DiagOMX[["diagram"]], main = idx)
  #plot(DiagOMX[["diagram"]], rotated = TRUE, main = idx)
  #plot(DiagOMX[["diagram"]], barcode = TRUE, main = idx)
  dev.off()
  Sys.sleep(1)
}


#######################################################
# http://rstudio-pubs-static.s3.amazonaws.com/172765_27d85ecf84244916a22d3c0708502cc7.html

# Correlation matrix computation is better in pandas!
D = cor(omx)


##########################################################
#prcomp can be changed to princomp
data <- omx[3916:4422]
data <- omx[3000:4350]
data <- omx[1000:4423]
data <- as.data.frame(data)
data <- t(data)
pca <- prcomp(data, center = TRUE, scale. = TRUE)
pca <- prcomp(data)
biplot(pca)

plot(pca$x[,'PC1']-
       pca$x[,'PC1'][1]+
       data[,30][1], type = 'l')
lines(data[,30], col = 'red')
lines(pca$x[,'PC2']-
        pca$x[,'PC2'][1]+
        data[,30][1], type = 'l', col = 'green')
lines(pca$x[,'PC3']-
        pca$x[,'PC3'][1]+
        data[,30][1], type = 'l', col = 'green')

######################################################
library(TDAmapper)
require(fastcluster)
data <- FX[3000:4350]
data <- as.data.frame(data)
#D = as.dist((1- cor(data))/2) # Correlation distance
#D = dist((1-cor(data))/2)
D = dist(data)
pca <- prcomp(data, center = TRUE, scale. = TRUE)
filter <- pca$rotation[,'PC1']

#TDAmapper
m1 <- mapper1D(
  distance_matrix = D, 
  filter_values = filter,
  num_intervals = 10,
  percent_overlap = 50,
  num_bins_when_clustering = 10
)

m1 <- mapper1D(
  distance_matrix = D, 
  filter_values = filter,
  num_intervals = 10,
  percent_overlap = 50,
  num_bins_when_clustering = 5
)
#overlap 35

#Construct adjacency df
# set row names and col names after m1$adjacency in data.frame to try get correct labeling
#df_adjacency <- data.frame(m1$adjacency)

#colnames(m1$adjacency) <- c("names for resp column --> col names for node")
colnames(m1$adjacency) <- c(1:sqrt(length(m1$adjacency)))
#colnames(m1$adjacency)[1] <- specify colnames for specific col
#colnames(m1$adjacency)[2] <- "uber"

for (i in c(1:length(colnames(m1$adjacency)))){
  colnames(m1$adjacency)[i] <- names(m1$points_in_vertex[[i]][1])
}

library(igraph)
g1 <- graph.adjacency(m1$adjacency, mode = "undirected")
plot(g1, layout = layout.auto(g1))

######
# WORKING OMX CLUSTERING LINE - Color is return mean, size of node is number of asset in it
######
library(TDAmapper)
require(fastcluster)

data <- omx[3000:4422]
data <- as.data.frame(data)
D = dist(data)
pca <- prcomp(data, center = TRUE, scale. = TRUE)
#pca <- prcomp(data)
filter <- pca$rotation[,'PC1']

#TDAmapper
m1 <- mapper1D(
  distance_matrix = D, 
  filter_values = filter,
  num_intervals = 10,
  percent_overlap = 50,
  num_bins_when_clustering = 10
)

library(igraph)
g1 <- graph.adjacency(m1$adjacency, mode = "undirected")
plot(g1, layout = layout.auto(g1))


# mean return of each vertex
y.mean.vertex <- rep(0,m1$num_vertices)
for (i in 1:m1$num_vertices){
  points.in.vertex <- m1$points_in_vertex[[i]][!is.na(m1$points_in_vertex[[i]])] #!is.na() removes na valu
  #points.in.vertex <- First.Example.mapper$points_in_vertex[[i]]
  y.mean.vertex[i] <- mean(data[names(points.in.vertex)][!is.na(data[names(points.in.vertex)])])
  #y.mean.vertex[i] <-mean((First.Example.data$y[points.in.vertex]))
}
y.mean.vertex[is.na(y.mean.vertex)] <- 0 # fill NA with 0 --> 0 mean

# Vertex size
vertex.size <- rep(0,m1$num_vertices)
for (i in 1:m1$num_vertices){
  points.in.vertex <- m1$points_in_vertex[[i]][!is.na(m1$points_in_vertex[[i]])]
  vertex.size[i] <- length(m1$points_in_vertex[[i]][!is.na(m1$points_in_vertex[[i]])])
}


# Colored in function of $y and vertex size proportional to number of points inside
y.mean.vertex.grey <- grey(1-(y.mean.vertex - min(y.mean.vertex))/(max(y.mean.vertex) - min(y.mean.vertex) ))
y.mean.vertex.grey <- rainbow((y.mean.vertex - min(y.mean.vertex))/(max(y.mean.vertex) - min(y.mean.vertex) ))

## Use n equally spaced breaks to assign each value to n-1 equal sized bins
values <- y.mean.vertex
ii <- cut(values, breaks = seq(min(values), max(values), len = 100), 
          include.lowest = TRUE)
## Use bin indices, ii, to select color from vector of n-1 equally spaced colors
colors <- colorRampPalette(c("lightblue", "blue"))(99)[ii]

## This call then also produces the plot below
image(seq_along(values), 1, as.matrix(seq_along(values)), col = colors,
      axes = F)
y.mean.vertex.grey <- colors

#y.mean.vertex.grey <- grey(y.mean.vertex*50)
V(g1)$color <- y.mean.vertex.grey
V(g1)$size <- vertex.size
plot(g1,main ="Mapper Graph")
#legend(x=-2, y=-1, c("y small","y medium","large y"),pch=21,
#       col="#777777", pt.bg=grey(c(1,0.5,0)), pt.cex=2, cex=.8, bty="n", ncol=1)
plot(g1, 
     vertex.color = adjustcolor(colors, alpha.f = .5), 
     vertex.label.color = adjustcolor("black", 1),
     vertex.label.dist = 1.2)

##########
############

######
# WORKING OMX CLUSTERING LINE - size is return mean, color of node is number of asset in it
######


library(TDAmapper)
require(fastcluster)
set.seed(5)

# We need to fill NaN for 
omx = read.zoo("OMXS30_perc_diff_NaN0.csv", sep = ',', header = TRUE, format = "%Y-%m-%d") 
#DROP X.OMX
omx$X.OMX <- NULL
data <- omx[3916:4422]
#data <- omx[3000:4422]

data <- as.data.frame(data)
D = dist(data)
#D = dist(data, method = "minkowski")

pca <- prcomp(data, center = TRUE, scale. = TRUE)
#pca <- prcomp(data)
filter <- pca$rotation[,'PC1']

# Re-read without NaN-fill for TDA
omx = read.zoo("OMXS30_perc_diff.csv", sep = ',', header = TRUE, format = "%Y-%m-%d") 
omx$X.OMX <- NULL
data <- omx[3916:4422]

#TDAmapper
m1 <- mapper1D(
  distance_matrix = D, 
  filter_values = filter,
  num_intervals = 15, #15
  percent_overlap = 25,
  num_bins_when_clustering = 11 # 11
)

library(igraph)
g1 <- graph.adjacency(m1$adjacency, mode = "undirected")
plot(g1, layout = layout.auto(g1))


# mean return of each vertex
y.mean.vertex <- rep(0,m1$num_vertices)
for (i in 1:m1$num_vertices){
  points.in.vertex <- m1$points_in_vertex[[i]][!is.na(m1$points_in_vertex[[i]])] #!is.na() removes na valu
  #points.in.vertex <- First.Example.mapper$points_in_vertex[[i]]
  y.mean.vertex[i] <- mean(data[names(points.in.vertex)][!is.na(data[names(points.in.vertex)])])
  #y.mean.vertex[i] <-mean((First.Example.data$y[points.in.vertex]))
}
y.mean.vertex[is.na(y.mean.vertex)] <- 0 # fill NA with 0 --> 0 mean

# Vertex size
vertex.size <- rep(0,m1$num_vertices)
for (i in 1:m1$num_vertices){
  points.in.vertex <- m1$points_in_vertex[[i]][!is.na(m1$points_in_vertex[[i]])]
  vertex.size[i] <- length(m1$points_in_vertex[[i]][!is.na(m1$points_in_vertex[[i]])])
}


# Colored in function of $y and vertex size proportional to number of points inside
y.mean.vertex.grey <- grey(1-(y.mean.vertex - min(y.mean.vertex))/(max(y.mean.vertex) - min(y.mean.vertex) ))
y.mean.vertex.grey <- rainbow((y.mean.vertex - min(y.mean.vertex))/(max(y.mean.vertex) - min(y.mean.vertex) ))

## Use n equally spaced breaks to assign each value to n-1 equal sized bins
values <- vertex.size
ii <- cut(values, breaks = seq(min(values), max(values), len = 100), 
          include.lowest = TRUE)
## Use bin indices, ii, to select color from vector of n-1 equally spaced colors
colors <- colorRampPalette(c("lightblue", "blue"))(99)[ii]

## This call then also produces the plot below
image(seq_along(values), 1, as.matrix(seq_along(values)), col = colors,
      axes = F)
y.mean.vertex.grey <- colors

#y.mean.vertex.grey <- grey(y.mean.vertex*50)
V(g1)$color <- y.mean.vertex.grey
V(g1)$size <- (y.mean.vertex - min(y.mean.vertex))/(max(y.mean.vertex) - min(y.mean.vertex))*30
plot(g1,main ="Mapper Graph")
plot(g1, 
     vertex.color = adjustcolor(colors, alpha.f = .5), 
     vertex.label.color = adjustcolor("black", 1),
     vertex.label.dist = 1.2)
#legend(x=-2, y=-1, c("y small","y medium","large y"),pch=21,
#       col="#777777", pt.bg=grey(c(1,0.5,0)), pt.cex=2, cex=.8, bty="n", ncol=1)



#############
#############
data <- FX[3000:4350]
data <- as.data.frame(data)
#D = as.dist((1- cor(data))/2) # Correlation distance
D = dist((1-cor(data))/2)
D = dist(data)
pca <- prcomp(data, center = TRUE, scale. = TRUE)
filter <- pca$rotation[,'PC1']

dmap = diffuse(D, eps.val=0.7, t=1, neigen=2) ## just run with the standard default settings
rownames(dmap$X) <- colnames(FX)
DimRed2d(dmap, "Diffusion Map", 5, labels = colnames(FX))


m2 <- mapper2D(
  distance_matrix = D,
  filter_values = list(dmap$X[,1],dmap$X[,2]),
  num_intervals = c(3,3),
  percent_overlap = 50,
  num_bins_when_clustering = 6
)
library(igraph)
g2 <- graph.adjacency(m2$adjacency, mode = "undirected")
plot(g2, layout = layout.auto(g2))
##########