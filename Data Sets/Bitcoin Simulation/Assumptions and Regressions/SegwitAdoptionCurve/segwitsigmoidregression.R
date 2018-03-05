## Performing sigmoid regression on Segwit adoption based on data thru 10/23/2017 from Segwit.party
# data points were averaged over 36 block periods (~6 hours)

# First, a polynomial regression (x^2) is used to forecast the date of 50% adoption
# The date of 50% adoption is then used in determining the final logistic growth forecast

setwd("C:/Users/Jad/Dropbox/Thesis/AnyLogic/Regressions")

library("dplyr")
library("lattice")
library("ggplot2")
library("rpart")
library("rpart.plot")
library("cluster")
library("rattle")
library("fpc")
library("vegan")


mysegwit<- read.csv("segwit_adoption_avg.csv", header= TRUE)


xdays = mysegwit$days
yfrac = mysegwit$adoptionfraction
x2 = seq(0,365,1)

#### POLYNOMIAL REGRESSION FIRST ####
# first step is to assume an aggressive x^2 growth rate for the adoption fraction
# we will fit y=x^2, and when the adoption fraction is at 0.5, that can be the maximum value used for the slope
# in the final logistic growth curve

# plot raw data
plot(xdays, yfrac,xlim=c(0,200),ylim=c(0,1))

k=rep(0,length(xdays))

model <- lm(yfrac ~ -1+xdays+I(xdays^2)+offset(k))
summary(model)

predicted.intervals <- predict(model,data.frame(x=xdays),interval='confidence',
                               level=0.99)
lines(xdays,predicted.intervals[,1],col='green',lwd=3)
lines(xdays,predicted.intervals[,2],col='black',lwd=1)
lines(xdays,predicted.intervals[,3],col='black',lwd=1)
legend("bottomright",c("Observ.","Signal","Predicted"), 
       col=c("deepskyblue4","red","green"), lwd=3)

# great, so an r-squared value of 0.9318
# we'll solve for xdays when 50% adoption occurs, so 0.5 =0.001003*xdays + 0.00001894*xdays^2
# our result is approximately 138 days
# so we will assume our maximum slope occurs at 138 days. 
# find the derivative of the function when x is 138
# our solution is 0.00623044 %/day is the maximum growth rate and will occur at 138 days


#### LOGISTIC GROWTH FORECAST SECOND ####

# a standard sigmoid function holds the form f(x) = L/(1+ e^(-k(x-x0)))
# in our case, L is known to be 1.0, x0 was determined as 138 days, which leaves us with just k to fit


# fitmodel2 <- nls(yfrac~SSlogis(xdays,1,138,1), data=xdays)
fitmodel2 <- nls(yfrac ~ 1/(1+exp(-k*(xdays-138))), start=list(k=0.00623), trace=TRUE)
summary(fitmodel2)

# found suitable model with k = 0.0264025
#predicted <- 1/(1+exp(-k*(xdays-138)))

sigmoid = function(x) {
  1/(1+exp(-0.0264025*(x-138)))
}
lines(x2, sigmoid(x2),col="blue")


# looks good! we will use a logistic distribution with 