using MultivariateStats, GLM
using Distributions: Binomial
cd("/Users/jasonhartford/MediaFire/Documents/ComputerScience/UBC/536 - Randomised/Project")
#X = readcsv("mnistHelper/mnist.csv")
include("OnlinePCA.jl")


#@time y = PCA1(X[1:500,:], 30, 0.7);

logistic(X) = 1./(1 + exp(-X));

function randsignalmat(m, n, d)
  U = randn(m, d);
  V, = qr(rand(n, d));
  Σ = diagm(2.0.^ (-1:-1:-d));
  A = U * Σ * V';
  return A, U, V, Σ;
end

function gendata(n = 1000, m = 100, d = 20, ρ = 1.0)
  X, U, V, Σ = randsignalmat(n,m,d);
  β = vcat(rand(d,1), zeros(m-d,1));
  X = 50*X; # rescale X so that it covers a large part of it's space
  p = logistic(X * β);
  Xobs = X + rand(n,m)*ρ; # could also use rand(Normal(0,1), (n,m))
  y = [rand(Distributions.Binomial(1, i)) for i in p];
  return Xobs, y
end

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

  print("Online PCA 1, ")
  X2 = PCA1(X, d, 0.35)
  println(run_tests(X2[:,1:d], y))

  print("SimpleSketch, ")
  X3 = SimpleSketch(X', d)'
  println(run_tests(X3[:,1:13], y))

  print("Online PCA 2, ")
  X4 = PCA2(X, d, 0.35, 100)
  println(run_tests(X4[:,1:d], y))
end

n = 10000; m = 200; d = 20; test = int(0.1*n);

X, y = gendata(n, m, d, 5.0);

println("Running tests on synthetic data")
run_all_tests(X,y)

println("Reading MNIST data")
X = readcsv("mnistHelper/mnist.csv");
y = readcsv("mnistHelper/mnist-label.csv");
println("Running tests on MNIST data")
run_all_tests(X,y)
