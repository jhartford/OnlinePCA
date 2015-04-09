normalize(X) = X - repmat(sum(X, 1)/size(X, 1), size(X, 1));

function genvud(n = 100, d = 20)
  v = qr(rand(d,d))[1];
  u = qr(rand(n,d)*reshape(linspace(1.0,100.0,d*d),(d,d)))[1];
  Σ = sort(exp(rand(d)*10), rev = true);
  return u, Σ, v
end

function PCA2(X, k::Int64, ϵ::Float64, w0 = 0.0)
  n, d = size(X);
  l = int(ceil(k/(ϵ^3)));
  I = eye(Float64, d);
  U = zeros(Float64, (d, l));
  l2 = int(ceil(k/(ϵ^2)));
  Z = zeros(Float64, (d, l2));
  w = 0;
  w_u = zeros(Float64, l);
  idx = 1;
  Y = zeros(Float64, (n, l));
  println("Starting loop with l = ",l," l2 = ",l2)
  for t = 1:n
    if t% 200 == 0
      println("Iteration - ",t)
    end
    x_t = X[t,:]';
    w = w + norm(x_t)^2;
    r = x_t - U*U'*x_t;
    c1 = (I - (U*U'));
    C = c1*(Z*Z')*c1;
#     println("C + r*r': ", norm(C + (r*r')))
#     println("C: ", norm(C))
#     println("r*r': ", norm(r*r'))
#     println("max(w0, w)*(k/(ϵ^2)): ", max(w0, w)*(k/(ϵ^2)))
#     println("w0: ", w0)
#     println("w: ", w)
#     println("k: ", k)
#     println("ϵ^2: ", ϵ^2)
#     println(" ")
    zero_row = 1;
    while norm(C + (r*r')) >= max(w0, w)*(k/(ϵ^2))
      #println("Inner loop")
      λ, u = eigs(C, nev = 1, which = :LM);
      if idx < l
        w_u[idx] = λ[1];
        U[:, idx] = u;
        idx = idx + 1;
      else
        val, idx = findmin(w_u);
        w_u[idx] = λ[1];
        U[:, idx] = u;
        idx = l;
      end

      c1 = (I - (U*U'));
      C = c1*(Z*Z')*c1;
      r = x_t - U*U'x_t;
      #print(t)
      #print(" - ")
      #println(idx)
    end

    Z, zero_row = sketch(r, Z', zero_row);
    Z = Z';

    for i = 1:idx
      u = U[:, i];
      w_u[i] = w_u[i] + ((u'*x_t)[1])^2;
    end
    Y[t, :] = U'*x_t;
  end
  return Y;
end

function SimpleSketch(A::DenseMatrix{Float64}, l)
  n, m = size(A);
  #A = A; #work col wise...
  zero_row = 1;
  B = zeros(Float64, l, m);
  for i = 1:n
    B, zero_row = Sketch(A[i,:], B, zero_row);
  end
  return B;
end

function sketch(rt::DenseMatrix{Float64}, B::DenseMatrix{Float64}, zero_row)
  l, m = size(B);
  if zero_row <= l
    B[zero_row, :] = rt;
    return B, (zero_row + 1);
  else
    U, Σ, V = svd(B);
    δ = Σ[int(l/2)];
    Sig = sqrt(max(Σ.^2 .- δ, 0));
    B = diagm(Sig)*V';
    zr = min_zero_idx(B); # could do this waaaay faster!
    return B, zr;
  end
end

function min_zero_idx(A)
  s = sum(A,2);
  n, m = size(A);
  for i = 1:n
    if abs(s[i]) < 1e-20
      return(i)
    end
  end
  return(n+1)
end

function PCA1(X, k::Int64, ϵ::Float64)
  println("Starting")
  X_F = vecnorm(X, 2);
  n, d = size(X);
  l = int(ceil(8*k/(ϵ^2)));
  println("l: ", l);
  U = zeros(Float64, (d, l));
  C = zeros(Float64, (d, d));
  Y = zeros(Float64, (n, l));
  idx = 1;
  for t = 1:n
    if t%500 == 0
      println("Iteration: ",t)
    end
    x_t = X[t,:]';
    r = x_t - U*U'*x_t;
    while norm(C + r*r') ≥ 2*(X_F^2)/l
      λ, u = eigs(C, nev = 1, which = :LM);
      U[:, idx] = real(u);
      idx = idx + 1;
      C = C - λ[1]*u*u';
      r = x_t - U*U'x_t;
    end
    C = C + r*r';
    Y[t, :] = U'*x_t;
  end
  return Y;
end
