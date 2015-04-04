function PCA1(X, k, ϵ, X_F)
  l = ceil(8*k/(ϵ^2))
  U = zeros(Float64, d, l)
  C = zeros(Float64, d, d)
  Y = zeros(Float64, n, l)
  idx = 1
  for t = 1:n
    x_t = X(t,:)
    r = x_t - U*U'*x_t
    while norm(C + r*r') ≥ 2*X_F/l
      λ, u = eigs(C, nev = 1, which = "LM")
      U[:, idx] = u
      idx = idx + 1
      C = λ*u*u'
      r = x_t - U*U'x_t
    end
    C = C + r*r'
    Y[:, t] = U'*x_t
  end
  return Y
end


