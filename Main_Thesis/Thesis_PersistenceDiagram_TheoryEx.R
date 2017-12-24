"
Plots theory example persistence diagram
"


################################
######### PLOT EXAMPLE #########
################################

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
