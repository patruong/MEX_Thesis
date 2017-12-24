"
Function for removing noise from persistence diagrams

"

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
