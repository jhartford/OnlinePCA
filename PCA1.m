function Y = PCA1(X, k, epsilon)
  X_F = norm(X, 'fro');
  [n, d] = size(X);
  l = ceil(8*k/(epsilon^2));
  U = zeros(d, 0);
  C = zeros(d, d);
  Y = zeros(n,l);
  idx = 0;
  for t = 1:n
    if mod(t,100) == 0
      fprintf('Iteration: %d\n',t)
    end
    x_t = X(t,:)';
    r = x_t - U*U'*x_t;
    while norm(C + r*r') >= 2*(X_F^2)/l %norm(C + r*r')
      idx = idx + 1;
      [u, lamb] = eigs(C,1);
      U = [U, u];
      C = C - lamb*(u*u');
      r = x_t - U*U'*x_t;
    end

    C = C + (r*r');
    y = zeros(1, l);
    if idx > 0
      y(1, 1:idx) = U'*x_t;
    end
    Y(t,:) = y;
  end
end