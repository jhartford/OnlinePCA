function A = randsignalmat(m, n, d)
  U = randn(m, d);
  [V,~] = qr(rand(n, d), 0);
  Sig = diag(2.0.^ (-1:-1:-d));
  A = U * Sig * V';
end