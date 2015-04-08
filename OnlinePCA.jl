#using Distributions:
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
    println("Iteration - ",t)
    x_t = X[t,:]';
    w = w + norm(x_t)^2;
    r = x_t - U*U'*x_t;
    c1 = (I - (U*U'));
    C = c1*(Z*Z')*c1;
    println("C + r*r': ", norm(C + (r*r')))
    println("C: ", norm(C))
    println("r*r': ", norm(r*r'))
    println("max(w0, w)*(k/(ϵ^2)): ", max(w0, w)*(k/(ϵ^2)))
    println("w0: ", w0)
    println("w: ", w)
    println("k: ", k)
    println("ϵ^2: ", ϵ^2)
    println(" ")
    while norm(C + (r*r')) >= max(w0, w)*(k/(ϵ^2))
      println("Inner loop")
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
      print(t)
      print(" - ")
      println(idx)
    end

    if t <= l2
        Z[:, t] = r;
    else
        Z = Sketch(r, Z);
    end

    for i = 1:idx
      u = U[:, i];
      w_u[i] = w_u[i] + ((u'*x_t)[1])^2;
    end
    Y[t, :] = U'*x_t;
  end
  return Y;
end

function Sketch(rt, Z)
    d, l = size(Z);
    U, Σ, V = svd(Z);
    δ = Σ[int(l/2)]^2;
    Sig = sqrt(max(Σ.^2 - eye(l)*δ, 0));
    B = diag(Sig)*V';
    return B
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
    println("Iteration: ",t)
    x_t = X[t,:]';
    #println(size(x_t), size(U))
    #println("Calc r:")
    r = x_t - U*U'*x_t;
    #println(size(r))
    #println("norm(C + r*r'): ", norm(C + (r*r')))
    #println(" 2*X_F^2/l: ",  2*X_F^2/l)
    #println(" X_F^2: ",  X_F^2)
    while norm(C + r*r') ≥ 2*(X_F^2)/l
      println("Correcting U - idx:", idx)
      λ, u = eigs(C, nev = 1, which = :LM);
      U[:, idx] = real(u);
      idx = idx + 1;
      C = C - λ[1]*u*u';
      r = x_t - U*U'x_t;
      println(norm(r))
      #println(t, ' - ', idx)
    end
    #println("Calc C:")
    C = C + r*r';
    #println("Calc Y:")
    Y[t, :] = U'*x_t;
  end
  return Y;
end

#n = 1500;
#d = 300;
#k = 5;
#ϵ = 0.5;

#println("Generating X with dim ",n,"x",d,". Target dim: ",n,"x",int(ceil(8*k/(ϵ^2))));
#@time u, Σ, v = genvud(n, d)
#X = u*diagm(Σ)*v;
#X = X ./ maximum(X, 1);
#X = rand((100,1),(n, d));
#println("Size of X: ", size(X));
#println("Running PCA");
#@time PCA1(X, k, ϵ)

#l = int(ceil(k/(ϵ^2)))

#maximum([norm(X[i,:])^2 for i = 1:n])

#vecnorm(X, 2)^2 / 160

#A = rand(10, 10)
#norm(A)
#u, d, v = svd(A);
#maximum(d)
