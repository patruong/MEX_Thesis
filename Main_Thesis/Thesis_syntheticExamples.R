"
Synthetic examples script

"

rm(list = ls())
par(mfrow=c(1,1))


library(TDA)
library(nonlinearTseries)
library(rgl) # plot3d

############################
### TDA Toy-examples #######
############################

# cirlce equation

x_c <- seq(-7, 7, length = 500)
y_c <- abs(sqrt(49-x_c^2))/7

# x-values
x <- seq(0, 16*pi, length = 1000)
x2 <- seq(0, pi, length = 10000)
x2 <- seq(0, pi, length = 1000)
x3 <- seq(0, 10, length = 500)

# y-values - Test equations
y = sin(x) #circle
y = 10*sin(10*x)+x2 #void
y = sin(4*x)+x2 #void
y = 4*sin(x2)*sin(32*x2)+ 4*x2
y = 10*y_c * sin(4*x) + 4*x2 #ball
y = 10*y_c * sin(4*x) + 4*x2 #ball
y = (4*sin(x2)*sin(32*x2) + 4*x2) *sin(4*x2)
y = sin(2*x2)

# Add jitter to y
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
max_y = max(complex_noiseRed[,3][2:length(complex_noiseRed[,3])])
max_x = max(complex_noiseRed[,2][1:length(complex_noiseRed[,2])])
max_xy = max(max_x, max_y)
max_y = max_x
plot(complex_noiseRed[,3] ~ complex_noiseRed[,2], 
     col =  ifelse(complex_noiseRed[,1] == 0, 'black', ifelse(complex_noiseRed[,1] == 1, 'red', 'blue')),
     pch = ifelse(complex_noiseRed[,1] == 0, 16, ifelse(complex_noiseRed[,1] == 1, 2, 5)),
     lwd = 2,
     xlim = c(0, 3), ylim = c(0, 3),#c(0, max_x), ylim=c(0,max_y),
     xlab = "Birth",
     ylab = "Death")#,
lines(x = c(0, max_xy), y = c(0, max_xy))
lines(x = c(0, max_x), y = c(0, max_x))
lines(x = c(0, 3), y = c(0, 3))

tseq <- seq(0, 10, length = 300)
Landscape <- landscape(DiagAlphaShape[["diagram"]], dimension = 1, KK = 1, tseq)
plot(tseq, Landscape, type = 'l', ylim = c(0, 0.4))



