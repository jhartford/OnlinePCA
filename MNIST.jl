using MultivariateStats, GLM
X = readcsv("mnistHelper/mnist.csv");
y = readcsv("mnistHelper/mnist-label.csv");
include("OnlinePCA.jl");

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


#d = 20
n = 60000
test = 10000

for d = 20:10:110
  M = MultivariateStats.fit(MultivariateStats.PCA, X[1:(n - test),:]'; method = :svd, maxoutdim = d);
  X1 = transform(M, X')';
  println(d,", Built in PCA, ",run_tests(X1, y))
  B = SimpleSketch(X[1:(n - test),:], 2*d)
  u, s, v = svd(B);
  X3 = X*v[:,1:d];
  println(d,", FrequentDirections, ",run_tests(X3, y))
end