using MultivariateStats, GLM
include("OnlinePCA.jl");
#include("datagen.jl")
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
      #else
      #    println(err)
      end
  end
end

f = readdir("data/")
Xf = f[1:15]
Xtf = f[16:30]
yf = f[31:45]
m = 5000;
test = 500;
println("m, n, d, rho, PCA, OnlinePCA,FrequentDirections")
for i = 1:15
  yfi = yf[i];
  st = split(yfi, "-")
  n = int(st[2])
  d = int(st[3])
  rho = st[4][1:length(st[4])-4]
  Xfi = Xf[i];
  Xtfi = Xtf[i];
  X = readcsv("data/$Xfi");
  y = readcsv("data/$yfi");
  X2 = readcsv("data/$Xtfi");
  # PCA
  M = MultivariateStats.fit(MultivariateStats.PCA, X[1:(m - test),:]'; method = :svd, maxoutdim = d);
  X1 = transform(M, X')';
  pca = run_tests(X1, y)

  #Online PCA
  opca = run_tests(X2, y)

  #FrequentDirections
  B = SimpleSketch(X[1:(m - test),:], d)
  u, s, v = svd(B);
  X3 = X*v[:,1:d];
  fqrdir = run_tests(X3, y)
  println("$m, $n, $d, $rho, $pca, $opca, $fqrdir")
end

f
