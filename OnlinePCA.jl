normalize(X) = X - repmat(sum(X, 1)/size(X, 1), size(X, 1));

function symm_mat_norm(A::Array{Float64})
  tot = 0.0
  m, n = size(A)
  for i = 1:m
    for j = i:n
      if j == i
        tot += A[i,j]^2;
      else
        tot += A[i,j]^2;
        tot += A[i,j]^2;
      end
    end
  end
  return sqrt(tot)
end


function genvud(n = 100, d = 20)
  v = qr(rand(d,d))[1];
  u = qr(rand(n,d)*reshape(linspace(1.0,100.0,d*d),(d,d)))[1];
  Σ = sort(exp(rand(d)*10), rev = true);
  return u, Σ, v
end

function PCA2(X, k::Int64, ϵ::Float64, w0 = 0.0, debug = false)
  n, d = size(X);
  l = int(ceil(k/(ϵ^3)));
  I = eye(Float64, d);
  U = zeros(d, 0); # change l
  l2 = int(ceil(k/(ϵ^2)));
  Z = zeros(Float64, (d, 1)); #change l2
  w = 0;
  w_u = zeros(Float64, 0); # change l
  idx = 0;
  Y = zeros(0, k); #change l
  println("Starting loop with l = ",l," l2 = ",l2)
  for t = 1:n
    if t% 100 == 0
      println("Iteration - ",t)
    end
    x_t = X[t,:]';
    w = w + norm(x_t)^2;
    r = x_t - U*U'*x_t;
    c1 = (I - (U*U'));
    C = c1*(Z*Z')*c1;
    zero_row = 1;
    while norm(C + (r*r')) >= max(w0, w) / (k/(ϵ^2))
      println(idx, " ")
      λ, u = eigs(C, nev = 1, which = :LM);
      if idx < l
        w_u = vcat(w_u, real(λ[1])); #w_u[idx] = λ[1];
        U = hcat(U, real(u));
        idx = idx + 1;
      else
        val, idx = findmin(w_u);
        w_u[idx] = λ[1];
        U[:, idx] = u;
        idx = l;
      end
      c1 = (I - (U*U'));
      C = c1*(Z*Z')*c1;
      r = x_t - U*U'*x_t;
    end

    Z, zero_row = sketch(r, Z', zero_row);
    Z = Z';

    for i = 1:idx
      u = U[:, i];
      #println("line 72 ",size(w_u))
      w_u[i] = w_u[i] + ((u'*x_t)[1])^2;
    end
    y = zeros(1, k)
    if idx > 0
      #println(size(U'*x_t))
      #println(idx)
      #println(size(U))
      y[1, 1:idx] = U'*x_t;
    end
    Y = vcat(Y, y);
  end
  return Y;
end

function SimpleSketch(A::Array{Float64}, l::Int64)
  n, d = size(A);
  #A = A; #work col wise..
  zero_row = l+1;
  B = zeros(Float64, l, d);
  B[1:(l), :] = A[1:(l), :]
  for i = (l+1):n
    B, zero_row = sketch(A[i,:], B, zero_row, l);
  end
  return B;
end

function sketch(rt::Array{Float64}, B::Array{Float64}, zero_row::Int64, l::Int64)
  if zero_row <= l
    B[zero_row, :] = rt;
    return B, (zero_row + 1);
  else
    U, Σ, V = svd(B);
    δ = Σ[int(l/2)]^2;
    Sig = sqrt(max(Σ.^2 .- δ, 0));
    B = spdiagm(Sig)*V';
    #BLAS.symm!('l', 'u', 1.0, diagm(Sig), V', 0.0, B);
    zr = int(l/2);
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

function PCA1(X, k::Int64, ϵ::Float64, verbose = false)
  if verbose
    println("Starting")
  end
  X_F = vecnorm(X, 2);
  n, d = size(X);
  l = int(ceil(8*k/(ϵ^2)));
  println("l: ", l);
  U = zeros(Float64, (d, 0));
  C = zeros(Float64, (d, d));
  Y = zeros(Float64, (n,l));
  idx = 0;
  for t = 1:n
    if t%100== 0
      println("Iteration: ",t)
    end
    x_t = X[t,:]';
    r = x_t - U*U'*x_t;
    while norm(C + r*r') ≥ 2*(X_F^2)/l #norm(C + r*r')
      idx = idx + 1;
      λ, u = eigs(C, nev = 1, which = :LR);
      println(typeof(λ), typeof(u))
      U = hcat(U, real(u));
      C = C - λ[1]*u*u';
      r = x_t - U*U'*x_t;
    end

    C = C + r*r';
    y = zeros(1, l);
    if idx > 0
      y[1, 1:idx] = U'*x_t;
    end
    Y[t,:] = y;
  end
  return Y;
end
