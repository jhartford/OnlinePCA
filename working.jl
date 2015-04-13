X = readcsv("mnist.csv")
using MultivariateStats, GLM
include("OnlinePCA.jl")
include("datagen.jl")

function run_tests(X, y, test_set_prop = 0.1)
  n, m = size(X);
  test = int(n*test_set_prop)
  try
    logit_mod = GLM.fit(GLM.GeneralizedLinearModel, X[1:n-test,:], y[1:n-test], GLM.Binomial(), maxIter = 30)
    y_test = [i > 0.5? 1: 0 for i in logistic(X[n-test:n,:] * GLM.coef(logit_mod))]

    return(mean(y[n-test:n] .== y_test))
  catch err
      if isa(err, ErrorException)
          print(err)
          return(0.0)
      else
          println(err)
      end
  end
end

function run_all_tests(X, y)
  print("Raw data, ")
  println(run_tests(X, y))

  print("Built in PCA, ")
  M = MultivariateStats.fit(MultivariateStats.PCA, X[1:n - test,:]'; method = :svd, maxoutdim = d);
  X1 = transform(M, X')';
  println(run_tests(X1, y))

 # print("Online PCA 1, ")
 # X2 = PCA1(X, d, 0.35)
 # println(run_tests(X2[:,1:d], y))

  print("SimpleSketch, ")
  X3 = SimpleSketch(X', d)'
  println(run_tests(X3[:,1:13], y))

  print("Online PCA 2, ")
  X4 = PCA2(X, d, 0.35, 100)
  println(run_tests(X4[:,1:d], y))
  return X1, X3, X4
end

n = 50000; m = 1000; d = 50; test = int(0.1*n);

X, y = gendata(n, m, d, 5.0);

println("Running tests on synthetic data")
run_all_tests(X,y)

print("Built in PCA, ")
@time M = MultivariateStats.fit(MultivariateStats.PCA, X[1:n - test,:]'; method = :svd, maxoutdim = d);
X1 = transform(M, X')';
println(run_tests(X1, y))

include("OnlinePCA.jl")

@time B = SimpleSketch(X, d)
u, s, v = svd(B);
X3 = X*v[:,1:d];
println(run_tests(X3, y))

X2 = PCA1(X, d, 0.35)

int(ceil(d/(0.3^3)))
X4 = PCA2(X, d, 0.1, vecnorm(X, 2)^2*3, true)
println(run_tests(X4[:,1:3], y))

println("Reading MNIST data")
X = readcsv("mnistHelper/mnist.csv");
y = readcsv("mnistHelper/mnist-label.csv");
println("Running tests on MNIST data")
run_all_tests(X,y)

