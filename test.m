clear all;
rng(123)
m = 5000; n = 100; d = 50; test_set = ceil(0.1*n);

[X, y] = gendata(m, n, d, 5.0);
Xt = PCA1(X, d, 0.35);
csvwrite('X2.csv', Xt);
csvwrite('y.csv', y);
csvwrite('X.csv', X);
