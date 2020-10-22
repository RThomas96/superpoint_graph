library(cowplot)
library(ggplot2)
library(ggpubr)

file <- "/home/thomas/Data/Cajun/Data/Evaluation/Methods/superpoint_graph/projects/metrics5M/reports/statsTraining.csv"

dataset <- read.csv2(file, sep=";", stringsAsFactors=FALSE, dec=".")

dataset <- subset(dataset,  Knn.geometric.features == 45)

plots <- list()

addList <- function(myList, value)
{
  myList[[length(myList)+1]] <- value
  return(myList)
}

plot1A <- function(xAxisName, xAxis, yAxisName, yAxis, legendName, yPercentage = FALSE) {
  # Plot 1
  p1 = ggplot() 
  
  colorValues <- c()
  i <- 1
  for(yAx in yAxis)
  {
    yname <- yAx[[1]]
    value <- yAx[[2]]
    color <- yAx[[3]]
    p1 = p1 + geom_line(data=dataset, aes(x=xAxis, y=yAxis[[i]][[2]], colour=yAxis[[i]][[1]]))
    colorValues[yAxis[[i]][[1]]] = yAxis[[i]][[3]]
    
    i <- i + 1
  }
  
  
  #p1 = p1 + geom_line(data=dataset, aes(x=xAxis, y=Total.accuracy, colour="TotalAcc"))
  #p1 = p1 + geom_line(data=dataset, aes(x=xAxis, y=Panneau, colour="Panneau"))
  #p1 = p1 + geom_line(data=dataset, aes(x=xAxis, y=Extincteur, colour="Extincteur"))
  #p1 = p1 + geom_line(data=dataset, aes(x=xAxis, y=Bac, colour="Bac"))
  #p1 = p1 + geom_line(data=dataset, aes(x=xAxis, y=Sol, colour="Sol"))
  #p1 = p1 + geom_line(data=dataset, aes(x=xAxis, y=Barriere, colour="Barriere"))
  
  #p1 = p1 + scale_color_manual(name = legendName, values = c('TotalAcc' = 'yellow','Panneau' = 'red','Extincteur' = 'darkblue','Bac' = 'green','Sol' = 'pink', 'Barriere' = 'brown')) + labs(color = 'Y series')
  p1 = p1 + scale_color_manual(name = legendName, values = colorValues) + labs(color = 'Y series')
  
  p1 = p1 + scale_x_continuous(xAxisName, breaks = seq(min(xAxis), max(xAxis), by = 0.02))
  p1 = p1 + scale_y_continuous(yAxisName, breaks = seq(0, 100, by = 10))
  
  if(yPercentage)
  {
    p1 = p1 + scale_y_continuous(yAxisName, breaks = seq(0, 100, by = 10), labels = function(x) paste0(x, "%"))
  }
  
  return(p1)
}

#plot1A()

plots <- addList(plots, plot1A(xAxisName = "Regularization strength", xAxis=dataset$Regularization.strength, yAxisName = "Accuracy", yAxis= list(list("TotalAcc", dataset$Total.accuracy, "yellow"), list("TotalAcc", dataset$Sol, "red"), list("Panneau", dataset$Panneau, "red")), legendName = "Objects", yPercentage = TRUE))

# Plot 3
dataset <- read.csv2(file, sep=";", stringsAsFactors=FALSE, dec=".")
dataset <- subset(dataset,  Knn.geometric.features == 45)

p3 = ggplot() + geom_line(data=dataset, aes(x=Regularization.strength, y=Number.of.superpoints, colour="Nb of spp")) 

p3 = p3 + scale_x_continuous("Regularisation strength", breaks = seq(min(dataset$Regularization.strength), max(dataset$Regularization.strength), by = 0.02)) 
p3 = p3 + scale_y_continuous("Total nb of spp", breaks = seq(min(dataset$Number.of.superpoints), max(dataset$Number.of.superpoints), by = 200000), trans='log10')

#plots <- addList(plots, p3)

print(p3)

# Plot 5
dataset <- read.csv2(file, sep=";", stringsAsFactors=FALSE, dec=".")
dataset <- subset(dataset, Knn.geometric.features == 45)

p5 = ggplot()
p5 = p5 + geom_line(data=dataset, aes(x=Regularization.strength, y=Panneau.1, colour="Panneau"))
p5 = p5 + geom_line(data=dataset, aes(x=Regularization.strength, y=Extincteur.1, colour="Extincteur"))
p5 = p5 + geom_line(data=dataset, aes(x=Regularization.strength, y=Bac.1, colour="Bac"))
#p5 = p5 + geom_line(data=dataset, aes(x=Regularization.strength, y=Sol.1, colour="Sol"))
p5 = p5 + geom_line(data=dataset, aes(x=Regularization.strength, y=Barriere.1, colour="Barriere"))

p5 = p5 + scale_color_manual(name = "Objets", values = c('TotalAcc' = 'yellow','Panneau' = 'red','Extincteur' = 'darkblue','Bac' = 'green','Sol' = 'pink', 'Barriere' = 'brown')) + labs(color = 'Y series')

p5 = p5 + scale_x_continuous("Regularisation strength", breaks = seq(min(dataset$Regularization.strength), max(dataset$Regularization.strength), by = 0.02)) 
p5 = p5 + scale_y_continuous("Nb spp", breaks = seq(0, 10000, by = 500), trans='log2') 

print(p5)

#plots <- addList(plots, p5)

ggarrange(plotlist=plots, labels = c("KnnGeo=45", "RegStrength=0.03"), ncol = 2, nrow = 2)
dataset <- read.csv2(file, sep=";", stringsAsFactors=FALSE, dec=".")
ggsave(filename="gg-default.png", device="png", dpi=300)