function [Xobs, y] = gendata(n, m, d, rho, l)
  X = randsignalmat(n,m,d, l);
  Beta = [rand(d,1); zeros(m-d,1)];
  X = 50*X; %# rescale X so that it covers a large part of it's space
  p = 1./(1 + exp(-X * Beta));
  Xobs = X + rand(n,m)*rho; %# could also use rand(Normal(0,1), (n,m))
  y = binornd(1, p);
end