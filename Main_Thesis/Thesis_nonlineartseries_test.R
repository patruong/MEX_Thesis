library("nonlinearTseries")

t = seq(0,200,0.1)


# Sine-wave
y = sin(t)
plot(t, y, type ='l', xlab = "time", ylab = "Sine wave")

# Cos-wave
y = cos(t)
plot(t, y, type ='l', xlab = "time", ylab = "Cosine wave")

# Sine + noise
y = sin(t) + runif(t)
plot(t, y, type ='l', xlab = "time", ylab = "Sine wave")

# Sine + Cosine
y = sin(t) + cos(t)
plot(t, y, type ='l', xlab = "time", ylab = "Cosine + Sine wave")

# Random
y = runif(t)
plot(t, y, type ='l', xlab = "time", ylab = "Cosine + Sine wave")

y=rnorm(200)
plot(y, type = 'l')

#ar
arma= arima.sim(n = 300, list(ar = c(0.5, 0)), sd = sqrt(0.1796))
plot(arma)
takens = buildTakens(arma, embedding.dim = 3, time.lag = 10)

#ma
arma= arima.sim(n = 300, list(ma = c(1)), sd = sqrt(0.1796))
plot(arma)
takens = buildTakens(arma, embedding.dim = 3, time.lag = 10)


#arima
arma= arima.sim(n = 300, list(ar = c(0.8897, -0.4858), ma = c(-0.2279, 02488)), sd = sqrt(0.1796))
plot(arma)
takens = buildTakens(arma, embedding.dim = 3, time.lag = 10)


#Henon
h = henon(n.sample = 3000, n.transient = 100, a = 1.4, b = 0.3, start = c(0.73954883, 0.04772637), do.plot = FALSE)
takens = buildTakens(h$x, embedding.dim = 2, time.lag = 1)
plot(h)
plot(takens)

#

takens = buildTakens(y, embedding.dim = 3, time.lag =10)
plot(takens)
plot3d(takens)


#################
library(forecast)
T <- seq(0, 20, length=200)
Y <- 1 + 3*cos(4*T+2) + .2*T^2 + rnorm(200)
plot(T,Y, type = 'l')

takens = buildTakens(Y, embedding.dim = 3, time.lag = 10)
plot(takens)

fit <- nnetar(Y)
fcast <- forecast(fit)
plot(fcast)


require(zoo)
TS <- zoo(c(4, 5, 7, 3, 9, 8))
rollapply(TS, width = 3, by = 2, FUN = mean, align = "right")

