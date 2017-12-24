######
# WORKING OMX CLUSTERING LINE - size is return mean, color of node is number of asset in it
######

rm(list = ls())
par(mfrow=c(1,1))
# READIN DATA
setwd("C:/Users/extra23/Desktop/Filer/MEX/Local laptop backup/Other Financial Data/TDA")

#raw-data
omx = read.zoo("OMXS30_log_ffill.csv", sep = ',', header = TRUE, format = "%Y-%m-%d") 

library(TDAmapper)
require(fastcluster)
set.seed(911117)

#DROP X.OMX
omx$X.OMX <- NULL
windowed <- window(omx, start = as.Date("2015-05-04"),end = as.Date("2017-05-04"))
data <- as.data.frame(windowed)

data <- as.data.frame(data)
D = dist(data)
#D = dist(data, method = "minkowski")

pca <- prcomp(data, center = TRUE, scale. = TRUE)
#pca <- prcomp(data)
filter <- pca$rotation[,'PC1']
#filter <- pca$rotation[,'PC2']


#TDAmapper
m1 <- mapper1D( # 12 nodes, 14
  distance_matrix = D, 
  filter_values = filter,
  num_intervals = 12, #16
  percent_overlap = 50, #40
  num_bins_when_clustering = 8#6
)

library(igraph)
g1 <- graph.adjacency(m1$adjacency, mode = "undirected")
plot(g1, layout = layout.auto(g1))
plot(simplify(g1), layout = layout.auto(g1))

# Create list of empty vertices
empty_vertices <- c()
for (i in 1:m1$num_vertices){
  if(length(m1$points_in_vertex[[i]][!is.na(m1$points_in_vertex[[i]])]) < 1){
    #g1 <- delete_vertices(g1, i) 
    empty_vertices <- c(empty_vertices, i)
  } 
}


# Delete all empty vertices
g1 <- delete_vertices(g1, empty_vertices)

plot(g1, layout = layout.auto(g1))
plot(simplify(g1), layout = layout.auto(g1))



# mean return of each vertex

# Create list with all non-empty vertices
all_vertices <- c(seq(1,m1$num_vertices))
non_empty_vertices <- all_vertices[!(all_vertices %in% empty_vertices)]

#y.mean.vertex <- rep(0,m1$num_vertices)
#y.mean.vertex <- rep(0, length(non_empty_vertices))
y.mean.vertex <- c()
for (i in 1:m1$num_vertices){
  points.in.vertex <- m1$points_in_vertex[[i]][!is.na(m1$points_in_vertex[[i]])] #!is.na() removes na valu
  #points.in.vertex <- First.Example.mapper$points_in_vertex[[i]]
  if (!is.nan(mean(data[names(points.in.vertex)][!is.na(data[names(points.in.vertex)])]))){
    y.mean.vertex <- c(y.mean.vertex, 
                       mean(data[names(points.in.vertex)][!is.na(data[names(points.in.vertex)])]))
  }
  #y.mean.vertex[i] <-mean((First.Example.data$y[points.in.vertex]))
}
#y.mean.vertex[is.na(y.mean.vertex)] <- 0 # fill NA with 0 --> 0 mean

# Vertex size
#vertex.size <- rep(0,m1$num_vertices)
vertex.size <- c()
for (i in 1:m1$num_vertices){
  points.in.vertex <- m1$points_in_vertex[[i]][!is.na(m1$points_in_vertex[[i]])]
  if (!is.nan(mean(data[names(points.in.vertex)][!is.na(data[names(points.in.vertex)])]))){
    vertex.size <- c(vertex.size, 
                     length(m1$points_in_vertex[[i]][!is.na(m1$points_in_vertex[[i]])]))
  }
  #vertex.size[i] <- length(m1$points_in_vertex[[i]][!is.na(m1$points_in_vertex[[i]])])
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

