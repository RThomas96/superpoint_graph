library(ggplot2)
#library(ggpubr)

dataset <- read.csv2("/home/thomas/Data/Cajun/Data/Evaluation/Methods/superpoint_graph/projects/metrics/reports/statsTesting.csv", sep=";", stringsAsFactors=FALSE, dec=".")

dataset <- subset(dataset, Regularization.strength>=0.01 & Knn.geometric.features == 45)
#dataset <- subset(dataset, Knn.geometric.features!=45)

attach(mtcars)
par(mfrow=c(2,2)) # For multiple layout

# Plot 1
p1 = ggplot() + geom_line(data=dataset, aes(x=Regularization.strength, y=Total.accuracy, colour="TotalAcc")) 

p1 = p1 + geom_line(data=dataset, aes(x=Regularization.strength, y=Panneau, colour="Panneau"))
p1 = p1 + geom_line(data=dataset, aes(x=Regularization.strength, y=Extincteur, colour="Extincteur"))
p1 = p1 + geom_line(data=dataset, aes(x=Regularization.strength, y=Bac, colour="Bac"))
p1 = p1 + geom_line(data=dataset, aes(x=Regularization.strength, y=Sol, colour="Sol"))
p1 = p1 + geom_line(data=dataset, aes(x=Regularization.strength, y=Sol, colour="Barriere"))

p1 = p1 + scale_color_manual(name = "name", values = c('TotalAcc' = 'yellow','Panneau' = 'red','Extincteur' = 'darkblue','Bac' = 'green','Sol' = 'pink', 'Barriere' = 'brown')) + labs(color = 'Y series')
p1 = p1 + scale_y_continuous(labels = function(x) paste0(x, "%")) # Add percent sign 

print(p)

# Plot 2
p2 = ggplot() + geom_line(data=dataset, aes(x=Regularization.strength, y=Total.accuracy, colour="TotalAcc")) 

p2 = p2 + geom_line(data=dataset, aes(x=Regularization.strength, y=Panneau, colour="Panneau"))
p2 = p2 + geom_line(data=dataset, aes(x=Regularization.strength, y=Extincteur, colour="Extincteur"))
p2 = p2 + geom_line(data=dataset, aes(x=Regularization.strength, y=Bac, colour="Bac"))
p2 = p2 + geom_line(data=dataset, aes(x=Regularization.strength, y=Sol, colour="Sol"))
p2 = p2 + geom_line(data=dataset, aes(x=Regularization.strength, y=Sol, colour="Barriere"))

p2 = p2 + scale_color_manual(name = "name", values = c('TotalAcc' = 'yellow','Panneau' = 'red','Extincteur' = 'darkblue','Bac' = 'green','Sol' = 'pink', 'Barriere' = 'brown')) + labs(color = 'Y series')
p2 = p2 + scale_y_continuous(labels = function(x) paste0(x, "%")) # Add percent sign 

print(p2)

ggarrange(p1, p2, 
          labels = c("A", "B"),
          ncol = 2, nrow = 1)