library(ggplot2)
library(ggpubr)

file <- "/home/thomas/Data/Cajun/Data/Evaluation/Methods/superpoint_graph/projects/metrics10M/reports/statsTraining.csv"

dataset <- read.csv2(file, sep=";", stringsAsFactors=FALSE, dec=".")

dataset <- subset(dataset, Regularization.strength <= 0.01& Knn.geometric.features == 45)
#dataset <- subset(dataset, Knn.geometric.features!=45)

#attach(mtcars)
#par(mfrow=c(2,2)) # For multiple layout

# Plot 1
p1 = ggplot() + geom_line(data=dataset, aes(x=Regularization.strength, y=Total.accuracy, colour="TotalAcc")) 

p1 = p1 + geom_line(data=dataset, aes(x=Regularization.strength, y=Panneau, colour="Panneau"))
p1 = p1 + geom_line(data=dataset, aes(x=Regularization.strength, y=Extincteur, colour="Extincteur"))
p1 = p1 + geom_line(data=dataset, aes(x=Regularization.strength, y=Bac, colour="Bac"))
p1 = p1 + geom_line(data=dataset, aes(x=Regularization.strength, y=Sol, colour="Sol"))
p1 = p1 + geom_line(data=dataset, aes(x=Regularization.strength, y=Barriere, colour="Barriere"))

p1 = p1 + scale_color_manual(name = "Objets", values = c('TotalAcc' = 'yellow','Panneau' = 'red','Extincteur' = 'darkblue','Bac' = 'green','Sol' = 'pink', 'Barriere' = 'brown')) + labs(color = 'Y series')

p1 = p1 + scale_x_continuous("Regularisation strength", breaks = seq(min(dataset$Regularization.strength), max(dataset$Regularization.strength), by = 0.001)) 
p1 = p1 + scale_y_continuous("Accuracy", breaks = seq(0, 100, by = 10), labels = function(x) paste0(x, "%"))

print(p1)

# Plot 3
dataset <- read.csv2(file, sep=";", stringsAsFactors=FALSE, dec=".")
dataset <- subset(dataset,  Regularization.strength <= 0.01&Knn.geometric.features == 45)

p3 = ggplot() + geom_line(data=dataset, aes(x=Regularization.strength, y=Number.of.superpoints, colour="Nb of spp")) 

p3 = p3 + scale_x_continuous("Regularisation strength", breaks = seq(min(dataset$Regularization.strength), max(dataset$Regularization.strength), by = 0.001)) 
p3 = p3 + scale_y_continuous("Total nb of spp", breaks = seq(min(dataset$Number.of.superpoints), max(dataset$Number.of.superpoints), by = 100000), trans='log10')

print(p3)

# Plot 5
dataset <- read.csv2(file, sep=";", stringsAsFactors=FALSE, dec=".")
dataset <- subset(dataset, Regularization.strength <= 0.01&Knn.geometric.features == 45)

p5 = ggplot()
p5 = p5 + geom_line(data=dataset, aes(x=Regularization.strength, y=Panneau.1, colour="Panneau"))
p5 = p5 + geom_line(data=dataset, aes(x=Regularization.strength, y=Extincteur.1, colour="Extincteur"))
p5 = p5 + geom_line(data=dataset, aes(x=Regularization.strength, y=Bac.1, colour="Bac"))
#p5 = p5 + geom_line(data=dataset, aes(x=Regularization.strength, y=Sol.1, colour="Sol"))
p5 = p5 + geom_line(data=dataset, aes(x=Regularization.strength, y=Barriere.1, colour="Barriere"))

p5 = p5 + scale_color_manual(name = "Objets", values = c('TotalAcc' = 'yellow','Panneau' = 'red','Extincteur' = 'darkblue','Bac' = 'green','Sol' = 'pink', 'Barriere' = 'brown')) + labs(color = 'Y series')

p5 = p5 + scale_x_continuous("Regularisation strength", breaks = seq(min(dataset$Regularization.strength), max(dataset$Regularization.strength), by = 0.001)) 
p5 = p5 + scale_y_continuous("Nb spp", breaks = seq(0, 4000, by = 100), trans='log10') 

print(p5)

print(p6)

ggarrange(p1, p5, p3, ncol = 2, nrow = 3)
dataset <- read.csv2(file, sep=";", stringsAsFactors=FALSE, dec=".")
ggsave(filename="gg-default.png", device="png", dpi=300)