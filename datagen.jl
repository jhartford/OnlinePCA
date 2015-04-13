using Distributions: Binomial;
logistic(X) = 1./(1 + exp(-X));

function randsignalmat(m, n, d)
  U = randn(m, d);
  V, = qr(rand(n, d));
  Σ = diagm(2.0.^ (-1:-1:-d));
  #Σ = diagm(1 - (0:d - 1)/d);
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
