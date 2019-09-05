par(mfrow=c(1,2))

path = "/tmp/context/"
title = "With context, 2 blocks"

loss.data = read.csv(file=paste(path,"loss.csv", sep=""))
val.data = read.csv(file=paste(path,"validation.csv", sep=""))

plot(x = loss.data$iteration, y = loss.data$loss, type="l", main = "training loss", xlab ="iterations", ylab = "loss")
plot(val.data$iteration, val.data$pckh_0.5, col="red", type="l", ylab="PCKh @ 0.5", xlab="iterations", main="validation accuracy")

mtext(title, outer = FALSE, cex = 1.5)