function A = randsignalmat(m, n, d, l)
  U = randn(m, d);
  [V,~] = qr(rand(n, d), 0);
  if l==1
      Sig = diag(2.0.^ (-1:-1:-d));
  else
      Sig = diag(1 - (0:d - 1)/d);
  end
  A = U * Sig * V';
end