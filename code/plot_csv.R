par(mfrow=c(1,2))

path = "/tmp/har_test_bkp/"
title = "HAR, Without context, 4 blocks"
har = TRUE

loss.data = read.csv(file=paste(path,"loss.csv", sep=""))
val.data = read.csv(file=paste(path,"validation.csv", sep=""))

plot(x = loss.data$iteration, y = loss.data$loss, type="l", xlab ="iterations", ylab = "training loss")

if (har == TRUE){
  plot(val.data$iteration, val.data$accuracy, col="red", type="l", ylab="validation accuracy", xlab="iterations")
  
} else {
  plot(val.data$iteration, val.data$pckh_0.5, col="red", type="l", ylab="validation accuracy", xlab="iterations")
  points(val.data$iteration, val.data$pckh_0.2, col="blue", type="l")
  legend(x="bottomright", legend=c("PCKh @ 0.5", "PCKh @ 0.2"), col=c("red", "blue"), lty=1)
}

title(title, outer=TRUE, line=-2)
#mtext(title, side = 3, line = -2, outer = TRUE, cex = 1.5, font=2)