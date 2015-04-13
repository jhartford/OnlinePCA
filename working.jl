X = readcsv("mnist.csv")
include("OnlinePCA.jl")
@time X_pca = PCA1(X, 30, 0.7);
