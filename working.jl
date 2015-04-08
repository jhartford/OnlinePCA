cd("/Users/jasonhartford/MediaFire/Documents/ComputerScience/UBC/536 - Randomised/Project")
X = readcsv("mnistHelper/mnist.csv")
include("OnlinePCA.jl")
@time y = PCA1(X[1:500,:], 30, 0.7);