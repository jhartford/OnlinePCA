using MultivariateStats, GLM
println("Reading Data")
X = readcsv("mnistHelper/mnist.csv");
y = readcsv("mnistHelper/mnist-label.csv");
println("Read Data, starting experiments")
include("OnlinePCA.jl");
logistic(X) = 1./(1 + exp(-X));
function run_tests(X, y, test_set_prop = 0.1)
  n, m = size(X);
  test = int(n*test_set_prop)
  try
    logit_mod = GLM.fit(GLM.GeneralizedLinearModel, X[1:n-test,:], y[1:n-test], GLM.Binomial(), maxIter = 100)
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


d = 50
n = 60000
test = 10000
m = 60000
n = 786

M = MultivariateStats.fit(MultivariateStats.PCA, X[1:(m - test),:]'; method = :svd, maxoutdim = 50);
X1 = transform(M, X')';
pca = run_tests(X1, y)

B = SimpleSketch(X[1:(m - test),:], d)
u, s, v = svd(B);
X3 = X*v[:,1:d];
fqrdir = run_tests(X3, y)

X2 = readcsv("mnist_OPCA.csv");
opca = run_tests(X2, y)
println("$m, $n, $d, $pca, $opca, $fqrdir")
